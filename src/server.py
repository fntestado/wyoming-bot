# server.py
import os, re, sys, csv, subprocess, asyncio, json, threading, time
from datetime import datetime
from typing import Iterator, Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("WY_DATA_DIR", os.path.join(BASE_DIR, "data"))
OUT_DIR  = os.getenv("WY_OUT_DIR",  os.path.join(BASE_DIR, "output"))
UI_DIR   = os.getenv("WY_UI_DIR",   os.path.join(BASE_DIR, "ui"))

_RA_FIELD_KEYS = {
    "registered_agent_name",
    "registered agent name",
    "registered_agent",
    "registered agent",
    "agent_name",
    "agent name",
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(UI_DIR,   exist_ok=True)

RUNS_DIR = Path(OUT_DIR) / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CURRENT = {"run_id": None, "proc": None, "dir": None, "log": None, "csv": None}

ITEM_DEFAULTS = os.getenv("WY_ITEM_DEFAULTS", "Wyoming Filing Defaults")

app = FastAPI(title="Wyoming Filing Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Mount UI and output (mount once, using unified OUT_DIR)
app.mount("/ui",     StaticFiles(directory=UI_DIR,  html=True), name="ui")
app.mount("/output", StaticFiles(directory=OUT_DIR, html=False), name="output")

@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>UI not found</h1><p>Put index.html in /ui</p>", status_code=404)

def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def _sse(line: str) -> str:
    return f"data: {line.rstrip()}\n\n"

def _tee(line: str, logfile: Path):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")

def _broadcast_evt(payload: dict):
    """Append an [[EVT]] line to the active run's sse.log so UIs see it."""
    rid = CURRENT.get("run_id")
    if not rid:
        return
    logf = _run_dir(rid) / "sse.log"
    _tee(f'[[EVT]] {json.dumps(payload, ensure_ascii=False)}', logf)

def _bg_worker(cmd: list[str], env: dict, run_id: str):
    rdir = _run_dir(run_id)
    logf = rdir / "sse.log"

    # meta (lets UI recover csv_path on reload)
    meta = {
        "run_id": run_id,
        "csv_path": env.get("WY_FILINGS_CSV"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (rdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # first line tells UI the run id
    init = f'[[RUN]] {json.dumps({"id": run_id}, ensure_ascii=False)}'
    _tee(init, logf)
    _tee("Starting run…", logf)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env
    )
    CURRENT.update({"run_id": run_id, "proc": proc, "dir": str(rdir), "log": str(logf), "csv": meta["csv_path"]})

    assert proc.stdout is not None
    for raw in proc.stdout:
        _tee(raw.rstrip("\n"), logf)

    code = proc.wait()
    _tee(f"Run finished with exit code {code}.", logf)
    CURRENT.update({"run_id": None, "proc": None})

def _tail_log_stream(run_id: str):
    """Replay + live tail of runs/<id>/sse.log — independent of the worker."""
    logf = _run_dir(run_id) / "sse.log"

    # replay existing
    if logf.exists():
        for line in logf.read_text(encoding="utf-8").splitlines():
            yield _sse(line)

    # tail new lines
    last_size = logf.stat().st_size if logf.exists() else 0
    idle = 0
    while True:
        time.sleep(1.0)
        if not logf.exists():
            continue
        size = logf.stat().st_size
        if size > last_size:
            with logf.open("r", encoding="utf-8") as f:
                f.seek(last_size)
                chunk = f.read()
            last_size = size
            idle = 0
            for line in chunk.splitlines():
                yield _sse(line)
        else:
            idle += 1
            # if this is no longer the active run AND nothing changed for a bit, end this tail
            if CURRENT.get("run_id") != run_id and idle > 5:
                break

def _stream_process(cmd: list[str], env: dict, run_id: str):
    # init run folder + log file
    rdir = _run_dir(run_id)
    # ensure no stale abort flag
    try:
        (rdir / "abort.flag").unlink(missing_ok=True)
    except Exception:
        pass
    logf = rdir / "sse.log"

    # initial announcement (lets the UI learn the run_id)
    init = f'[[RUN]] {json.dumps({"id": run_id}, ensure_ascii=False)}'
    _tee(init, logf); yield _sse(init)

    yield _sse("Starting run…")
    _tee("Starting run…", logf)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env
    )

    # publish this as the active run
    CURRENT.update({"run_id": run_id, "proc": proc, "dir": str(rdir)})

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        _tee(line, logf)
        yield _sse(line)

    code = proc.wait()
    tail = f"Run finished with exit code {code}."
    _tee(tail, logf); yield _sse(tail)

    # clear active marker when done
    CURRENT.update({"run_id": None, "proc": None})

@lru_cache(maxsize=1)
def get_default_registered_agent_name() -> str:
    """
    Retrieve the default Registered Agent name from 1Password.

    Priority:
      1) 1Password CLI (op item get "<WY_ITEM_DEFAULTS>" --vault "<WY_VAULT>")
      2) Env fallback WY_DEFAULT_RA
      3) Empty string

    Required env:
      - OP_SERVICE_ACCOUNT_TOKEN  (service account token)
      - WY_VAULT                  (e.g., "Wyoming Filing Bot")
      - WY_ITEM_DEFAULTS          (e.g., "Wyoming Filing Defaults")
    Optional env:
      - WY_DEFAULT_RA             (fallback if 1P not available)
    """
    vault_name = _clean(os.getenv("WY_VAULT", "Wyoming Filing Bot"))
    item_title = _clean(os.getenv("WY_ITEM_DEFAULTS", "Wyoming Filing Defaults"))
    env_fallback = _clean(os.getenv("WY_DEFAULT_RA", ""))

    # Prefer 1Password CLI if present and token available
    token = _clean(os.getenv("OP_SERVICE_ACCOUNT_TOKEN", ""))
    op_path = shutil.which("op")

    if token and op_path:
        try:
            # Non-interactive auth via service account token
            env = os.environ.copy()
            env["OP_SERVICE_ACCOUNT_TOKEN"] = token

            out = subprocess.check_output(
                [op_path, "item", "get", item_title, "--vault", vault_name, "--format", "json"],
                env=env,
                text=True,
                stderr=subprocess.STDOUT,
                timeout=10,
            )
            data = json.loads(out)

            # 1Password v2 CLI returns fields as a flat list; each field can have id/label/purpose + value.
            fields = data.get("fields") or []
            for f in fields:
                name = _clean(f.get("id") or f.get("label") or f.get("purpose") or "")
                if name.lower() in _RA_FIELD_KEYS:
                    val = _clean(f.get("value", ""))
                    if val:
                        return val

            # Extra safety: try top-level keys (rare, but cheap)
            for k in _RA_FIELD_KEYS:
                if k in data and _clean(data[k]):
                    return _clean(data[k])

        except Exception:
            # Swallow and fall through to env fallback
            pass

    # Fallback to env if CLI not available or item/field missing
    return env_fallback

def _clean(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # strip optional surrounding quotes and normalize whitespace
    s = s.strip().strip('"').strip("'")
    s = s.replace("\u00A0", " ")
    s = " ".join(s.split())
    return s

@app.get("/defaults")
def read_defaults():
    return {
        "registered_agent_name": get_default_registered_agent_name()
    }

# -------- Upload CSV
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    # Always save to a single canonical filename to avoid duplicates
    dest_name = os.getenv("WY_FILINGS_CSV", "filings.csv").strip() or "filings.csv"
    if not dest_name.lower().endswith(".csv"):
        dest_name += ".csv"

    save_path = os.path.join(DATA_DIR, dest_name)

    data = await file.read()
    with open(save_path, "wb") as f:
        f.write(data)

    # Optional: quick preview for UI
    preview = []
    try:
        with open(save_path, newline="", encoding="utf-8-sig") as f:
            rdr = csv.DictReader(f)
            for i, row in enumerate(rdr):
                if i >= 50: break
                preview.append({
                    "entity_name": row.get("entity_name") or row.get("entity") or row.get("company_name") or "",
                    "registered_agent_name": row.get("registered_agent_name") or row.get("ra_name") or "",
                    "city": row.get("city") or "",
                    "state": row.get("state") or "",
                })
    except Exception:
        preview = []

    return {"ok": True, "csv_path": dest_name, "preview": preview}

# -------- Start run (SSE)
@app.get("/run")
def run(
    vault: str = "Wyoming Filing Bot",
    item_defaults: str = "Wyoming Filing Defaults",
    item_card: str = "Credit Card",
    item_2captcha: str = "2Captcha API Key",
    data_dir: str = "data",
    out_dir: str = "output",
    csv_path: str = "filings.csv",
    run_id: str | None = None,
):
    env = os.environ.copy()
    env.update({
        "WY_VAULT": vault,
        "WY_ITEM_DEFAULTS": item_defaults,
        "WY_ITEM_CARD": item_card,
        "WY_ITEM_2CAPTCHA": item_2captcha,
        "WY_DATA_DIR": data_dir,
        "WY_OUT_DIR": out_dir,
        "WY_FILINGS_CSV": csv_path,
    })

    # Pick a run id and expose it to the child
    rid = run_id or _make_run_id()
    env["WY_RUN_ID"] = rid

    # Let /active-run report csv_path (and anything else you want)
    CURRENT.update({"run_id": rid, "csv_path": csv_path})

    meta_path = _run_dir(rid) / "meta.json"
    meta_path.write_text(json.dumps({
        "id": rid,
        "csv_path": csv_path,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False), encoding="utf-8")

    cmd = ["python3", "main.py"]
    return StreamingResponse(
        _stream_process(cmd, env, rid),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )

@app.post("/stop-run")
def stop_run():
    """Soft stop: create abort.flag; main.py will finish current entity then exit."""
    rid  = CURRENT.get("run_id")
    rdir = CURRENT.get("dir")
    if not rid or not rdir:
        return {"ok": True}

    try:
        Path(rdir).mkdir(parents=True, exist_ok=True)
        (Path(rdir) / "abort.flag").touch()
    except Exception:
        pass

    _broadcast_evt({"type": "stopping", "run_id": rid})
    return {"ok": True}

@app.post("/stop-run-now")
def stop_run_now():
    """Hard stop: terminate the child process immediately."""
    rid  = CURRENT.get("run_id")
    proc = CURRENT.get("proc")
    if not rid or not proc:
        return {"ok": True}

    _broadcast_evt({"type": "stopping", "run_id": rid, "mode": "hard"})
    try:
        proc.terminate()
        time.sleep(2)
        if proc.poll() is None:
            proc.kill()
    finally:
        _broadcast_evt({"type": "aborted", "run_id": rid})
        CURRENT.update({"run_id": None, "proc": None, "dir": None})
    return {"ok": True}

# -------- Active run for reload
@app.get("/active-run")
def active_run():
    rid  = CURRENT.get("run_id")
    proc = CURRENT.get("proc")
    if rid and proc and (proc.poll() is None):
        return {"run_id": rid, "running": True, "csv_path": CURRENT.get("csv_path")}
    return {"run_id": None, "running": False}

@app.get("/run-meta/{run_id}")
def run_meta(run_id: str):
    p = _run_dir(run_id) / "meta.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="meta not found")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

# -------- Live tail with keep-alives (never auto-break)
@app.get("/stream/{run_id}")
async def stream_run(run_id: str):
    logf = _run_dir(run_id) / "sse.log"

    async def agen():
        # replay
        if logf.exists():
            for line in logf.read_text(encoding="utf-8").splitlines():
                yield _sse(line)

        # tail
        last_size = logf.stat().st_size if logf.exists() else 0
        while True:
            await asyncio.sleep(1.0)
            if not logf.exists():
                break
            size = logf.stat().st_size
            if size > last_size:
                with logf.open("r", encoding="utf-8") as f:
                    f.seek(last_size)
                    chunk = f.read()
                last_size = size
                for line in chunk.splitlines():
                    yield _sse(line)

    return StreamingResponse(agen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})

# -------- Queue snapshot (used on reload + light polling)
def _progress_from_log(run_id: str):
    finished, current = set(), None
    p = _run_dir(run_id) / "sse.log"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("[[EVT]] "):
                try:
                    ev = json.loads(line[8:])
                    if ev.get("type") == "start":
                        current = int(ev.get("idx") or 0) or None
                    elif ev.get("type") == "finish":
                        i = int(ev.get("idx") or 0)
                        if i > 0: finished.add(i)
                except:
                    pass
    return finished, current

@app.get("/queue-preview")
def queue_preview(csv_path: str = Query(...), run_id: str = Query(...), limit: int = 200):
    # resolve CSV full path
    csv_full = os.path.join(DATA_DIR, csv_path)
    if not os.path.isfile(csv_full):
        raise HTTPException(status_code=404, detail="csv not found")

    # load CSV rows
    rows = []
    with open(csv_full, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "entity_name": row.get("entity_name") or row.get("entity") or row.get("company_name") or "",
                "registered_agent_name": row.get("registered_agent_name") or row.get("ra_name") or "",
                "city": row.get("city") or "",
                "state": row.get("state") or "",
            })

    total = len(rows)

    # parse sse.log for latest start/finish indexes
    logf = _run_dir(run_id) / "sse.log"
    last_start = 0
    last_finish = 0
    if logf.exists():
        for line in logf.read_text(encoding="utf-8").splitlines():
            if line.startswith("[[EVT]] "):
                try:
                    ev = json.loads(line[8:])
                    if ev.get("type") == "start":
                        last_start = max(last_start, int(ev.get("idx", 0)))
                    elif ev.get("type") == "finish":
                        last_finish = max(last_finish, int(ev.get("idx", 0)))
                except Exception:
                    pass

    # remaining starts after the last finished index
    start_idx = max(last_finish, 0)  # 0-based slice anchor
    preview = rows[start_idx : start_idx + limit]
    remaining = max(total - last_finish, 0)

    return {
        "preview": preview,
        "remaining": remaining,
        "current_idx": last_start if last_start else None,  # 1-based for UI highlight
        "total": total,
    }

# -------- Latest results + download
@app.get("/latest-results")
def latest_results():
    try:
        files = sorted(
            [f for f in os.listdir(OUT_DIR) if f.startswith("results_") and f.endswith(".csv")],
            reverse=True
        )
        if not files:
            return {"ok": True, "file": None}
        return {"ok": True, "file": f"/download/{files[0]}"}
    except FileNotFoundError:
        return {"ok": True, "file": None}

@app.get("/download/{fname}")
def download_result(fname: str):
    path = os.path.join(OUT_DIR, fname)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, filename=fname)