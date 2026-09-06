"""
Streaming WebSocket TTS worker for Fish Audio S2 Pro (fine-tuned bn_bd) — RunPod Load Balancer endpoint.

Routes
  GET /ping            health (required by RunPod; served on PORT and, if different, on PORT_HEALTH)
  WS  /ws/tts          streaming text-in / audio-out session

Client -> server (JSON text frames)
  {"type":"start","reference_audio_b64":"...","reference_text":"...","seed":42}   # once per socket (reference optional)
  {"type":"text","text":"partial LLM output..."}                                    # as many as you like, any granularity
  {"type":"flush"}                                                                  # LLM reply finished: speak whatever is buffered
  {"type":"cancel"}                                                                 # user barged in: drop everything queued/in-flight
  {"type":"end"}                                                                    # close session

Server -> client
  JSON  {"type":"ready","sample_rate":44100}
  JSON  {"type":"chunk_start","turn":T,"seq":N,"text":"..."}
  BIN   12-byte header (uint32 turn, uint32 seq, uint32 part) + int16 LE mono PCM at sample_rate
  JSON  {"type":"chunk_end","turn":T,"seq":N}
  JSON  {"type":"turn_done","turn":T}      # everything flushed for this turn has been sent
  JSON  {"type":"error","message":"..."}

Serverless notes
  The model is loaded on a background thread so the HTTP server is listening (and /ping answering)
  from the first second of the cold start. /ping reports "starting" until the warm-up finishes and
  goes unhealthy — without exiting — if the checkpoint is missing or the GPU thread dies, so RunPod
  recycles the worker instead of routing traffic into a black hole.
"""
import asyncio, base64, json, os, queue, struct, sys, threading, time, traceback

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_codec
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

CKPT = os.environ.get("CKPT", "/runpod-volume/models/s2-pro-bn-bd")
PORT = int(os.environ.get("PORT", "8000"))
PORT_HEALTH = int(os.environ.get("PORT_HEALTH", str(PORT)))
COMPILE = os.environ.get("COMPILE", "1") == "1"
# RunPod kills workers that fail health checks. If your endpoint's startup grace period is shorter
# than the cold start (torch.compile can take minutes), set this to 1 so /ping returns 200 while
# loading; WebSocket connects are still refused until the model is actually ready.
PRECISION = torch.bfloat16

# ---------------------------------------------------------------- worker state
STATUS = "starting"          # starting | healthy | error
STATUS_DETAIL = "model not loaded yet"
engine = None
SAMPLE_RATE = None

# One GPU, one generation at a time. All sessions share this queue.
_gpu_jobs: "queue.Queue[tuple]" = queue.Queue()
_gpu_thread: "threading.Thread | None" = None


def _fail(detail: str):
    """Enter a permanent unhealthy state. Do NOT exit: a crash-looping worker is billed for every
    restart and shows up as a flapping endpoint instead of a diagnosable one."""
    global STATUS, STATUS_DETAIL
    STATUS, STATUS_DETAIL = "error", detail
    print(f"[worker] UNHEALTHY: {detail}", flush=True)


# ---------------------------------------------------------------- model (loaded once, off the main thread)
def _load_model():
    global STATUS, STATUS_DETAIL, engine, SAMPLE_RATE, _gpu_thread

    if not os.path.exists(f"{CKPT}/config.json"):
        for d in ("/runpod-volume", "/workspace"):
            print(f"[worker] {d}:", os.listdir(d) if os.path.exists(d) else "(not mounted)", flush=True)
        _fail(f"model not found at {CKPT}")
        return

    try:
        t0 = time.time()
        STATUS_DETAIL = "loading checkpoints"
        llama = launch_thread_safe_queue(checkpoint_path=CKPT, device="cuda", precision=PRECISION, compile=COMPILE)
        codec = load_codec("modded_dac_vq", f"{CKPT}/codec.pth", device="cuda")
        eng = TTSInferenceEngine(llama_queue=llama, decoder_model=codec, precision=PRECISION, compile=COMPILE)
        sr = codec.spec_transform.sample_rate if hasattr(codec, "spec_transform") else codec.sample_rate

        STATUS_DETAIL = "warming up (torch.compile)" if COMPILE else "warming up"
        for _ in eng.inference(ServeTTSRequest(text="আমি ঢাকায় থাকি।", format="wav", max_new_tokens=128, streaming=True)):
            pass

        engine, SAMPLE_RATE = eng, sr
        _gpu_thread = threading.Thread(target=_gpu_worker, name="gpu-worker", daemon=True)
        _gpu_thread.start()
        STATUS, STATUS_DETAIL = "healthy", ""
        print(f"[worker] model ready in {time.time()-t0:.1f}s, sr={sr}", flush=True)
    except Exception:
        traceback.print_exc()
        _fail(f"model load failed: {traceback.format_exc(limit=1).strip()}")


def _gpu_worker():
    """Never dies. A raised exception here used to kill the thread permanently, leaving a worker that
    still passed health checks but produced silence forever."""
    while True:
        session, turn, seq, text = _gpu_jobs.get()
        try:
            if session.closed or turn < session.cancel_floor:   # cancelled / stale
                continue
            session.run_chunk(turn, seq, text)
        except Exception:
            traceback.print_exc()
            try:
                session.send_error(f"generation failed: {sys.exc_info()[1]}")
            except Exception:
                pass


def ready() -> bool:
    return STATUS == "healthy" and _gpu_thread is not None and _gpu_thread.is_alive()


# ---------------------------------------------------------------- text chunker
SENT_END = "।?!.\n"
SOFT_BREAK = ",;:—-"


class Chunker:
    """Buffer LLM deltas; emit speakable chunks. First chunk short (fast first audio), later ones full sentences."""

    def __init__(self, first_min=25, soft_min=60, hard_max=180):
        self.buf, self.emitted = "", 0
        self.first_min, self.soft_min, self.hard_max = first_min, soft_min, hard_max

    def feed(self, delta: str):
        self.buf += delta
        out = []
        while True:
            c = self._take()
            if c is None:
                break
            out.append(c)
        return out

    def flush(self):
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []

    def _take(self):
        s = self.buf
        min_len = self.first_min if self.emitted == 0 else self.soft_min
        for i, ch in enumerate(s):
            if ch in SENT_END and i + 1 >= min_len // 2:
                return self._cut(i + 1)
            if ch in SOFT_BREAK and i + 1 >= min_len:
                return self._cut(i + 1)
        if len(s) >= self.hard_max:                      # runaway text with no punctuation
            sp = s.rfind(" ", 0, self.hard_max)
            return self._cut(sp if sp > 0 else self.hard_max)
        return None

    def _cut(self, n):
        chunk, self.buf = self.buf[:n].strip(), self.buf[n:]
        if not chunk:
            return None
        self.emitted += 1
        return chunk


# ---------------------------------------------------------------- session
class Session:
    """One WebSocket = one session. `turn` increases on every flush (new LLM reply);
    `cancel_floor` marks turns that were barged-in and must be silently dropped."""

    def __init__(self, ws: WebSocket, loop):
        self.ws, self.loop = ws, loop
        self.turn, self.seq = 0, 0
        self.cancel_floor = 0
        self.closed = False
        self.references: list[ServeReferenceAudio] = []
        self.seed = None
        self.chunker = Chunker()
        self.pending: dict[int, int] = {}     # turn -> chunks still generating
        self.flushed: set[int] = set()        # turns whose text is complete

    def alive(self, turn):
        return not self.closed and turn >= self.cancel_floor

    # ---- GPU thread
    def run_chunk(self, turn, seq, text):
        req = ServeTTSRequest(
            text=text, references=self.references, seed=self.seed,
            use_memory_cache="on",            # reference encoded once, reused for every chunk
            format="wav", streaming=True, chunk_length=300,
            temperature=0.8, top_p=0.8, repetition_penalty=1.1,
        )
        self._send_json({"type": "chunk_start", "turn": turn, "seq": seq, "text": text})
        part = 0
        try:
            for r in engine.inference(req):
                if not self.alive(turn):
                    break                      # barge-in: stop streaming this chunk
                if r.code == "segment":
                    _, audio = r.audio
                    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                    self._send_bytes(struct.pack("<III", turn, seq, part) + pcm)
                    part += 1
                elif r.code == "error":
                    self._send_json({"type": "error", "message": str(r.error)})
        finally:
            self.pending[turn] -= 1
            if self.alive(turn):
                self._send_json({"type": "chunk_end", "turn": turn, "seq": seq})
                if self.pending[turn] == 0 and turn in self.flushed:
                    self._send_json({"type": "turn_done", "turn": turn})

    # ---- event loop
    def feed(self, delta):
        for chunk in self.chunker.feed(delta):
            self._submit(chunk)

    def flush(self):
        for chunk in self.chunker.flush():
            self._submit(chunk)
        turn = self.turn
        self.flushed.add(turn)
        done_now = self.pending.get(turn, 0) == 0
        self.turn += 1; self.seq = 0; self.chunker = Chunker()
        return turn, done_now

    def cancel(self):
        self.cancel_floor = self.turn + 1      # everything up to and including current turn is dead
        self.turn += 1; self.seq = 0; self.chunker = Chunker()

    def _submit(self, text):
        self.pending[self.turn] = self.pending.get(self.turn, 0) + 1
        _gpu_jobs.put((self, self.turn, self.seq, text))
        self.seq += 1

    def send_error(self, message):
        self._send_json({"type": "error", "message": message})

    def _send_json(self, obj):
        if self.closed:
            return
        asyncio.run_coroutine_threadsafe(self.ws.send_text(json.dumps(obj, ensure_ascii=False)), self.loop)

    def _send_bytes(self, b):
        if self.closed:
            return
        asyncio.run_coroutine_threadsafe(self.ws.send_bytes(b), self.loop)


# ---------------------------------------------------------------- app
app = FastAPI()
health_app = FastAPI()          # only bound when PORT_HEALTH != PORT


def _ping():
    if ready():
        return {"status": "healthy", "sample_rate": SAMPLE_RATE}
    if STATUS == "healthy":     # loaded, but the GPU thread is gone — let RunPod replace this worker
        return JSONResponse({"status": "unhealthy", "detail": "gpu worker thread died"}, status_code=503)
    if STATUS == "error":
        return JSONResponse({"status": "unhealthy", "detail": STATUS_DETAIL}, status_code=503)
    body = {"status": "starting", "detail": STATUS_DETAIL}
    return JSONResponse(body, status_code=200)


app.get("/ping")(_ping)
health_app.get("/ping")(_ping)


@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    await ws.accept()
    if not ready():
        await ws.send_text(json.dumps({"type": "error", "message": f"worker not ready: {STATUS} {STATUS_DETAIL}".strip()}))
        await ws.close(code=1013)              # try again later
        return
    s = Session(ws, asyncio.get_running_loop())
    await ws.send_text(json.dumps({"type": "ready", "sample_rate": SAMPLE_RATE}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")
            if t == "start":
                s.seed = msg.get("seed")
                if msg.get("reference_audio_b64"):
                    s.references = [ServeReferenceAudio(
                        audio=base64.b64decode(msg["reference_audio_b64"]),
                        text=msg.get("reference_text", ""),
                    )]
                    # pre-encode the reference now so the first real chunk doesn't pay for it
                    await asyncio.to_thread(engine.load_by_hash, s.references, "on")
            elif t == "text":
                s.feed(msg.get("text", ""))
            elif t == "flush":
                turn, done_now = s.flush()
                if done_now:
                    await ws.send_text(json.dumps({"type": "turn_done", "turn": turn}))
            elif t == "cancel":
                s.cancel()
                await ws.send_text(json.dumps({"type": "cancelled", "next_turn": s.turn}))
            elif t == "end":
                break
    except WebSocketDisconnect:
        pass
    finally:
        s.closed = True


async def _serve():
    servers = [uvicorn.Server(uvicorn.Config(
        app, host="0.0.0.0", port=PORT, ws_ping_interval=20, ws_ping_timeout=20,
    )).serve()]
    if PORT_HEALTH != PORT:
        servers.append(uvicorn.Server(uvicorn.Config(
            health_app, host="0.0.0.0", port=PORT_HEALTH, log_level="warning",
        )).serve())
    await asyncio.gather(*servers)


if __name__ == "__main__":
    # Serve first, load second: /ping must answer during the whole cold start.
    threading.Thread(target=_load_model, name="model-loader", daemon=True).start()
    print(f"[worker] serving on :{PORT}" + (f", health on :{PORT_HEALTH}" if PORT_HEALTH != PORT else ""), flush=True)
    asyncio.run(_serve())
