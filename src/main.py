import os, re, sys, json, time, subprocess, tempfile, pathlib, csv, urllib.parse, urllib.request, textwrap, shutil
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote
from playwright.sync_api import Playwright, sync_playwright, expect, Error
from twocaptcha import TwoCaptcha
from difflib import SequenceMatcher
from contextlib import contextmanager
from pathlib import Path

# =========================
# --- Pretty logging + user notes ---
# =========================

def _truthy(s: str | None) -> bool:
    return str(s or "").strip().lower() in ("1", "true", "yes", "on")

EVT_PREFIX = "[[EVT]] "

# === event emission (top of file) ===
RUN_ID = os.getenv("WY_RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
RUNS_ROOT = Path(os.getenv("WY_OUT_DIR", "output")) / "runs"
RUN_DIR = RUNS_ROOT / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_PATH = RUN_DIR / "events.jsonl"
TRACING_ENABLED = _truthy(os.getenv("WY_TRACING", "0"))  # default OFF

DAYS_KEEP = int(os.getenv("WY_DAYS_KEEP", "7"))
PRUNE_FOLDERS = [
    s.strip() for s in os.getenv("WY_PRUNE_FOLDERS", "receipts,registrations,cgs,gsc").split(",")
    if s.strip()
]

_DESIGNATION_END_RE = re.compile(
    r"(?:L\.?L\.?C\.?|L\.?C\.?|LC|Limited Liability(?:\s+Co\.?|\s+Company)|"
    r"Limited Company|Ltd\. Liability(?:\s+Co\.?|\s+Company))\.?\s*$",
    re.IGNORECASE,
)


# =========================
# --- CONSTANTS ---
# =========================
FIELD_ALIASES = {
    "entity": "entity_name",
    "entityname": "entity_name",
    "company_name": "entity_name",

    "registered_agent": "registered_agent_name",
    "ra_name": "registered_agent_name",

    "addr": "physical_address",
    "physical_addr": "physical_address",
    "mailing_addr": "mailing_address",

    "zip": "postal_code",
    "postal": "postal_code",

    "phone_number": "phone",
    "email_address": "email",

    "organizer_addr": "organizer_address",

    "firstname": "signature_first_name",
    "first_name": "signature_first_name",
    "lastname": "signature_last_name",
    "last_name": "signature_last_name",
    "title": "signature_title",
}

ABORT_FLAG = RUN_DIR / "abort.flag"

# --- load .env early ---
try:
    from dotenv import load_dotenv
    from pathlib import Path

    # try a few likely locations
    for p in [Path.cwd()/".env", Path(__file__).parent/".env", Path(__file__).parent.parent/".env"]:
        if p.exists():
            load_dotenv(p, override=False)
            print(f">>> .env loaded from {p}")
            break
except Exception as _e:
    print(">>> .env not loaded (python-dotenv missing or unreadable).")

class SkipEntity(Exception):
    """Raised to indicate we should skip this entity without treating it as an error."""
    pass

class PaymentDeclined(Exception):
    """Raised when the hosted-payments page shows a decline dialog."""
    pass

def _now_hhmmss():
    return datetime.now().strftime("%H:%M:%S")

def log_info(msg): print(f"[{_now_hhmmss()}] {msg}", flush=True)
def log_ok(msg):   print(f"[{_now_hhmmss()}] {msg}", flush=True)
def log_warn(msg): print(f"[{_now_hhmmss()}] {msg}", flush=True)
def log_err(msg):  print(f"[{_now_hhmmss()}] {msg}", flush=True)

@contextmanager
def step(title: str):
    log_info(f"{title}…")
    try:
        yield
        log_ok(f"{title} — done")
    except Exception as e:
        # Attach the step title so we can humanize later
        e._wy_step = title  # type: ignore[attr-defined]
        log_err(f"{title} — failed")
        raise

def _extract_timeout_target(detail: str) -> tuple[str, str] | None:
    """
    Tries to extract what Playwright was waiting for when it timed out.
    Returns (role, name) if it can parse get_by_role(...), else None.
    """
    m = re.search(r'get_by_role\("([^"]+)",\s*name="?(.+?)"?\)', detail)
    if m:
        role, name = m.group(1), m.group(2)
        # unescape doubled quotes that Playwright prints
        name = name.replace('""', '"')
        return role, name
    return None

def humanize_error(e: Exception) -> str:
    """
    Convert low-level exceptions into concise, friendly notes for CSV/UI.
    """
    detail = str(e)
    step_name = getattr(e, "_wy_step", None)
    # 0) Navigation timeout ("Page.goto: Timeout ... navigating to ... waiting until ...")
    m = re.search(
        r'Page\.goto:\s*Timeout\s+(\d+)ms.*?navigating to "([^"]+)",\s*waiting until "([^"]+)"',
        detail,
        re.S,
    )
    if m:
        ms   = int(m.group(1))
        secs = ms // 1000
        url  = m.group(2)
        phase= m.group(3)  # "load" or "domcontentloaded"

        # Map a few known endpoints to friendlier names
        page_name = url
        mapping = {
            "RegistrationInstr.aspx": "Registration Instructions page",
            "RegistrationType.aspx":  "Registration Type page",
            "hosted-payments":        "Hosted Payments page",
            "FilingSearch.aspx":      "Filing Search page",
        }
        for needle, label in mapping.items():
            if needle in url:
                page_name = label
                break

        where = f" during “{step_name}”" if step_name else ""
        tip = "Site may be slow. Try again in a minute or switch IP/VPN."
        return f"Timed out loading {page_name} (waited for “{phase}”) after {secs}s{where}. Tip: {tip}"

    # A) Visibility / Hidden assertions
    if ("Locator expected to be visible" in detail) or ("Locator expected to be hidden" in detail):
        # Try to extract target info
        role_name = _extract_timeout_target(detail)  # ("checkbox", "Pursuant to …") if present
        m_css = re.search(r'locator\("([^"]+)"\)', detail)  # e.g. locator("#txtNameConfirm")
        target_txt = None
        if role_name:
            target_txt = f'{role_name[0]} “{role_name[1]}”'
        elif m_css:
            target_txt = f'element {m_css.group(1)}'

        # Detect nav URL if Playwright logged it
        m_nav = re.search(r'navigated to "([^"]+)"', detail)
        nav_hint = f" (landed on {m_nav.group(1)})" if m_nav else ""

        which = "be visible" if "expected to be visible" in detail else "be hidden"
        where = f" during “{step_name}”" if step_name else ""
        tgt = f" {target_txt}" if target_txt else ""
        tip = "The site may have changed screens or the control text changed. Try reloading or retrying."
        return f"Expected{tgt} to {which}{where}{nav_hint} but it wasn’t. Tip: {tip}"

    # B) Playwright Timeout (keep your existing nicer path)
    if "Timeout" in detail and "Locator" in detail and "get_by_role" in detail:
        target = _extract_timeout_target(detail)
        secs_m = re.search(r"Timeout\s+(\d+)ms", detail)
        secs = int(secs_m.group(1)) // 1000 if secs_m else 30
        if target:
            role, name = target
            nice = f"Couldn’t find the {role} “{name}” within {secs}s"
        else:
            nice = f"Timed out waiting for the page within {secs}s"
        where = f" during “{step_name}”" if step_name else ""
        tip = "The site may be slow or showing a blank page. Try a different IP/VPN or refresh and retry."
        return f"{nice}{where}. Tip: {tip}"

    # C) Network-ish
    if any(x in detail for x in ("net::ERR_", "SSL", "Navigation failed")):
        where = f" during “{step_name}”" if step_name else ""
        return f"Network issue{where}. Tip: Check internet/proxy and try again."

    # D) reCAPTCHA shorteners
    if "ERROR_ZERO_BALANCE" in detail:
        return "2Captcha balance is zero. Add funds and rerun."
    if "ERROR_WRONG_USER_KEY" in detail:
        return "2Captcha API key is invalid. Update the key in 1Password or env."
    if "ERROR_CAPTCHA_UNSOLVABLE" in detail:
        return "reCAPTCHA could not be solved after several attempts. Will need a retry."

    # E) Generic one-liner fallback
    base = step_name + ": " if step_name else ""
    brief = detail.splitlines()[0].strip()
    return f"{base}{brief}"

# ---------------------------
# Results CSV / paths helpers
# ---------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def _union_input_headers(rows: list[dict]) -> list[str]:
    """
    Build an ordered list of input headers:
    - Start with the first row's keys (keeps a stable, human-friendly order)
    - Append any new keys seen later
    """
    if not rows:
        return []
    headers = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    return headers

def write_results_csv(results: list[dict], input_headers: list[str], out_dir: str = "output") -> str:
    """
    Write a timestamped results CSV (e.g., results_YYYY-MM-DD_HHMMSS.csv),
    then prune old results and old run folders.
    """
    _ensure_dir(out_dir)
    ts = _timestamp()
    out_path = os.path.join(out_dir, f"results_{ts}.csv")

    extra_cols = [
        "status", "notes", "filing_id",
        "receipt_file", "registration_file",
        "cgs_file", "gsc_file",
        "started_at", "finished_at", "elapsed_s",
    ]
    fieldnames = input_headers + [c for c in extra_cols if c not in input_headers]

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in results:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    # NEW: age-based pruning (last 7 days by default; override via WY_DAYS_KEEP)
    prune_results_by_age(out_dir, DAYS_KEEP)
    prune_documents_by_age(out_dir, DAYS_KEEP, PRUNE_FOLDERS)
    prune_run_dirs_by_age(RUNS_ROOT, DAYS_KEEP, exclude=(RUN_ID,))
    prune_media_by_age(DAYS_KEEP)

    print(f">>> Results written to: {out_path}")
    return out_path

def maybe_pause(page, where: str = ""):
    """
    Pause in Playwright Inspector when WY_PAUSE is truthy or PWDEBUG=1 is set.
    Use WY_PAUSE=1 for all pauses, or WY_PAUSE=after_payment to only pause there.
    """
    flag = (os.getenv("PWDEBUG") == "1") or bool(os.getenv("WY_PAUSE"))
    scoped = os.getenv("WY_PAUSE", "").lower()  # e.g., "after_payment"
    # if flag and (scoped in ("1", "true", "yes", "", "after_payment")):
    print(f">>> [PAUSE] {where} — open Inspector and click Resume to continue.")
    page.pause()

# =========================
# --- CSV helpers ---
# =========================
def load_filings_csv(path: str = "filings.csv", allowed_keys: set[str] | None = None) -> list[dict]:
    """
    Read filings.csv and return a list of row dicts with normalized keys.
    - Requires 'entity_name'
    - Trims whitespace
    - Drops empty values
    - If allowed_keys is provided, only keeps keys in that set (plus 'entity_name')
    """
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("filings.csv has no header row.")
        headers = [_norm_key(h) for h in reader.fieldnames]
        if "entity_name" not in headers:
            raise RuntimeError("filings.csv must include a header column named 'entity_name'.")

        # map original header -> normalized header
        hmap = {orig: _norm_key(orig) for orig in reader.fieldnames}

        for i, raw in enumerate(reader, start=1):
            # normalize + trim
            cleaned = {}
            for orig_k, v in raw.items():
                k = hmap[orig_k]
                v = (v or "").strip()
                if v == "":
                    continue
                cleaned[k] = v

            if not cleaned.get("entity_name"):
                print(f"!!! Skipping row {i}: missing 'entity_name'")
                continue

            if allowed_keys:
                keep = set(allowed_keys) | {"entity_name"}
                cleaned = {k: v for k, v in cleaned.items() if k in keep}

            rows.append(cleaned)

    return rows

def _norm_key(k: str) -> str:
    k = (k or "").strip()
    lk = k.lower().replace(" ", "").replace("-", "").replace("_", "")
    return FIELD_ALIASES.get(lk, k)  # fall back to original if no alias

def _clean_value(key: str, val):
    if val is None:
        return None
    if isinstance(val, str):
        val = val.strip()
    # light normalization (safe, non-destructive)
    if key == "state" and isinstance(val, str):
        return val.upper()
    if key in ("postal_code", "phone") and isinstance(val, str):
        return re.sub(r"\D", "", val)
    return val

def merge_filing_row(row: dict, defaults_profile: dict) -> dict:
    """
    Merge priority: CSV row (non-empty) > 1Password defaults.
    No hard-coded DEFAULTS are used.
    """
    # Start from 1Password defaults (drop empties)
    merged = {k: v for k, v in (defaults_profile or {}).items() if v not in ("", None)}

    # Normalize CSV keys, clean values, and overlay on top
    for raw_k, raw_v in (row or {}).items():
        k = _norm_key(raw_k)
        v = _clean_value(k, raw_v)
        if v not in ("", None):
            merged[k] = v

    # Guard: entity_name is required by your runner
    if not (merged.get("entity_name") or "").strip():
        raise ValueError("Missing required field 'entity_name' after merge.")

    return merged

# =========================
# --- reCAPTCHA helpers ---
# =========================
def _extract_recaptcha_info(page):
    """
    Detect reCAPTCHA on the page.
    Returns:
      {
        'sitekey': str,
        'enterprise': bool,
        'data_s': str|None,
        'invisible': bool,
        'action': str|None,
        'api_domain': 'www.google.com' | 'www.recaptcha.net' | 'recaptcha.google.cn' | None,
      }
    """
    info = {"sitekey": None, "enterprise": False, "data_s": None,
            "invisible": False, "action": None, "api_domain": None}

    # 1) Prefer the visible container
    try:
        sitekey = page.evaluate("""() => document.querySelector('[data-sitekey]')?.getAttribute('data-sitekey') || null""")
        if sitekey:
            info["sitekey"] = sitekey
            info["invisible"] = bool(page.evaluate(
                """() => document.querySelector('[data-sitekey]')?.getAttribute('data-size') === 'invisible'"""
            ))
    except Exception:
        pass

    # 2) Parse iframes — prefer ANCHOR over BFRAME
    def _maybe_take_from_url(url_str):
        try:
            u = urlparse(url_str)
            if "recaptcha" not in u.path and "recaptcha" not in u.netloc:
                return
            if "anchor" not in u.path and "bframe" not in u.path:
                return
            qs = parse_qs(u.query)
            k = qs.get("k", [None])[0]
            if k and not info["sitekey"]:
                info["sitekey"] = unquote(k)
            s = qs.get("s", [None])[0]
            if s:
                info["data_s"] = s
            if "/enterprise/" in u.path:
                info["enterprise"] = True
            if "size=invisible" in u.query:
                info["invisible"] = True
            # capture the host so we can set 2Captcha domain
            info["api_domain"] = u.netloc or info["api_domain"]
        except Exception:
            pass

    # Pass 1: anchor first
    for f in page.frames:
        if "anchor" in (f.url or "") and "recaptcha" in (f.url or ""):
            _maybe_take_from_url(f.url)

    # Pass 2: take bframe only if we still lack sitekey
    if not info["sitekey"]:
        for f in page.frames:
            if "bframe" in (f.url or "") and "recaptcha" in (f.url or ""):
                _maybe_take_from_url(f.url)

    # 3) Optional: action (rare on v2 invisible)
    try:
        action = page.evaluate("""() => document.querySelector('[data-action]')?.getAttribute('data-action') || null""")
        if action:
            info["action"] = action
    except Exception:
        pass

    return info

def _solve_recaptcha_with_2captcha(page, api_key, info=None):
    """
    Finds the REAL sitekey (from the .g-recaptcha submit button or anchor iframe),
    solves it with 2Captcha, and returns the token.
    """
    # Prefer widget attribute
    btn = page.locator('#id_recaptcha.g-recaptcha, .g-recaptcha').first
    btn.wait_for(state='visible', timeout=10000)
    sitekey = btn.get_attribute('data-sitekey')

    # Fallback: parse anchor iframe
    if not sitekey:
        anchor_iframe = page.locator('iframe[src*="recaptcha/api2/anchor"]').first
        if anchor_iframe.count() > 0:
            src = anchor_iframe.get_attribute('src') or ''
            m = re.search(r'[?&]k=([^&]+)', src)
            if m:
                sitekey = m.group(1)

    if not sitekey:
        raise RuntimeError("Could not find reCAPTCHA sitekey on this page.")

    payload = {
        "sitekey": sitekey,
        "url": page.url,
    }
    if info:
        if info.get("invisible"):  payload["invisible"]  = 1
        if info.get("enterprise"): payload["enterprise"] = 1
        if info.get("data_s"):     payload["data_s"]     = info["data_s"]
        if info.get("action"):     payload["action"]     = info["action"]
        if info.get("api_domain"): payload["domain"]     = info["api_domain"]  # www.google.com / www.recaptcha.net / recaptcha.google.cn

    print(f">>> Solving reCAPTCHA with sitekey={sitekey[:8]}… (opts={ {k:v for k,v in payload.items() if k!='sitekey' and k!='url'} })")
    solver = TwoCaptcha(api_key)
    result = solver.recaptcha(**payload)
    return result['code']

def _finalize_recaptcha_and_submit(page, token: str):
    # 1) Put token into every expected field and fire input/change
    page.evaluate("""
    (tok) => {
      const fire = el => { el.dispatchEvent(new Event('input',{bubbles:true})); el.dispatchEvent(new Event('change',{bubbles:true})); };
      const setVal = el => { el.value = tok; fire(el); };
      const ids = ['g-recaptcha-response','g-recaptcha-response-100000','g-recaptcha-response-1'];
      ids.forEach(id => { const el = document.getElementById(id); if (el) setVal(el); });
      document.querySelectorAll('textarea[name="g-recaptcha-response"]').forEach(setVal);
    }
    """, token)

    # 2) Make grecaptcha.getResponse() return our token
    page.evaluate("""
    (tok) => {
      try {
        if (!window.grecaptcha) window.grecaptcha = {};
        const orig = window.grecaptcha.getResponse;
        window.grecaptcha.getResponse = function(){ return tok; };
        const clients = (window.___grecaptcha_cfg && window.___grecaptcha_cfg.clients) || {};
        for (const c of Object.values(clients)) {
          for (const obj of Object.values(c || {})) {
            const candidates = [];
            if (obj && typeof obj === 'object') {
              for (const v of Object.values(obj)) {
                if (typeof v === 'function') candidates.push(v);
                if (v && typeof v.callback === 'function') candidates.push(v.callback);
                if (v && v.V && typeof v.V.callback === 'function') candidates.push(v.V.callback);
                if (v && v.W && typeof v.W.callback === 'function') candidates.push(v.W.callback);
                if (v && v.Y && typeof v.Y.callback === 'function') candidates.push(v.Y.callback);
                if (v && v.Z && typeof v.Z.callback === 'function') candidates.push(v.Z.callback);
              }
            }
            for (const fn of candidates) { try { fn(tok); } catch(e){} }
          }
        }
      } catch(e) {}
    }
    """, token)

    # 3) Programmatic click + form submit
    page.evaluate("""
    () => {
      const btn = document.getElementById('id_recaptcha') || document.querySelector('.g-recaptcha, button[type="submit"]');
      if (btn) btn.dispatchEvent(new MouseEvent('click', {bubbles:true, cancelable:true}));
      const form = btn ? btn.closest('form') : document.querySelector('form');
      if (form && form.requestSubmit) form.requestSubmit(btn || undefined);
    }
    """)

    try:
        page.locator('iframe[src*="recaptcha/api2/bframe"]').wait_for(state='detached', timeout=5000)
    except:
        pass

def _inject_recaptcha_token_and_submit(page, token):
    page.evaluate("""
    (tok) => {
      const fire = el => { el.dispatchEvent(new Event('input',{bubbles:true})); el.dispatchEvent(new Event('change',{bubbles:true})); };
      const setVal = el => { el.value = tok; fire(el); };
      const ids = ['g-recaptcha-response','g-recaptcha-response-100000','g-recaptcha-response-1'];
      ids.forEach(id => { const el = document.getElementById(id); if (el) setVal(el); });
      document.querySelectorAll('textarea[name="g-recaptcha-response"]').forEach(setVal);
    }
    """, token)

    page.wait_for_function("""
      () => Array.from(document.querySelectorAll('textarea[name="g-recaptcha-response"]'))
                 .some(el => el.value && el.value.length > 0)
    """, timeout=5000)

    submitted = page.evaluate("""
    () => {
      const btn = document.querySelector('button#id_recaptcha, button.g-recaptcha, button[type="submit"]');
      const form = btn ? btn.closest('form') : document.querySelector('form');
      if (form && form.requestSubmit) { form.requestSubmit(btn || undefined); return true; }
      if (btn) { btn.click(); return true; }
      return false;
    }
    """)

    if submitted:
      return

    try:
        page.get_by_role("button", name=re.compile(r"Submit", re.I)).click(force=True, timeout=3000)
        return
    except:
        page.evaluate("""() => document.querySelectorAll('iframe[src*="recaptcha"]').forEach(f => f.style.pointerEvents = 'none')""")
        page.get_by_role("button", name=re.compile(r"Submit", re.I)).click(force=True)

def handle_payment_submit_and_captcha(page, captcha_api_key: str, entity: str):
    emit_ux("Payment", "Submitting payment", "info", entity=entity)

    # First click (accept "Submit Payment" or generic "Submit")
    try:
        page.get_by_role("button", name=re.compile(r"Submit Payment", re.I)).click()
    except Exception:
        page.get_by_role("button", name=re.compile(r"\bSubmit\b", re.I)).click()

    # If the decline dialog appears quickly, stop here
    _fail_if_declined(page, entity=entity, timeout_ms=2000)

    # Let invisible v2 render if needed
    page.wait_for_timeout(1200)

    # Detect widget
    info = _extract_recaptcha_info(page)
    if info.get("sitekey"):
        emit_ux("CAPTCHA", "reCAPTCHA detected", "start", entity=entity)
        try:
            max_attempts = int(os.getenv("WY_MAX_CAPTCHA_ATTEMPTS", "5"))
            solve_recaptcha_and_submit(page, captcha_api_key, max_attempts=max_attempts, base_sleep=3.0, entity=entity)
            emit_ux("CAPTCHA", "reCAPTCHA solved", "success", entity=entity)
        except Exception as e:
            emit_ux("CAPTCHA", f"reCAPTCHA error: {humanize_error(e)}", "error", entity=entity)

    else:
        print(">>> No reCAPTCHA widget found; proceeding…")

    # Check for a decline dialog after CAPTCHA/submit
    _fail_if_declined(page, entity=entity, timeout_ms=2000)

    # Try to detect success nav; otherwise one more submit poke
    try:
        page.wait_for_url(re.compile(r"(RegistrationComplete|Receipt|/Print)", re.I), timeout=8000)
    except Exception:
        try:
            page.get_by_role("button", name=re.compile(r"Submit Payment|Submit", re.I)).click()
        except Exception:
            pass

        # Final check before returning
        _fail_if_declined(page, entity=entity, timeout_ms=4000)

def _fail_if_declined(page, *, entity: str = "", timeout_ms: int = 0):
    """
    If the Angular Material decline dialog is present, extract its text,
    optionally click 'Cancel Filing', emit UX, and raise PaymentDeclined.
    """
    rx = re.compile(r"Your payment is declined", re.I)

    def _find_dialog():
        # Prefer a dialog that contains the header text
        dlg = page.locator('md-dialog[role="dialog"]').filter(has_text=rx).first
        if dlg.count():
            return dlg
        # Fallback: dialog with the known retry button
        btn_retry = page.locator("#id_retry").first
        if btn_retry.count():
            return btn_retry.locator("xpath=ancestor::md-dialog[1]").first
        return None

    dlg = _find_dialog()
    if timeout_ms and (dlg is None or not dlg.count()):
        try:
            page.locator('md-dialog[role="dialog"]').filter(has_text=rx).first.wait_for(
                state="visible", timeout=timeout_ms
            )
            dlg = _find_dialog()
        except Exception:
            dlg = _find_dialog()

    if dlg and dlg.count():
        # Pull some readable text for notes
        try:
            msg = dlg.locator("md-dialog-content").inner_text().strip()
        except Exception:
            msg = "Your payment is declined."
        emit_ux("Payment", tidy_note(msg), "error", entity=entity)

        # Best-effort: click "Cancel Filing" to unwind the page
        try:
            page.locator("#id_cancel").click(timeout=1000)
        except Exception:
            pass

        raise PaymentDeclined(tidy_note(msg))

def _reset_recaptcha_widget(page) -> bool:
    """
    Try to reset the reCAPTCHA widget so a fresh challenge can be issued.
    Returns True if we think we reset something; False otherwise.
    """
    try:
        return page.evaluate("""
        () => {
          try {
            let reset = false;
            const gr = (window.grecaptcha && (window.grecaptcha.enterprise || window.grecaptcha)) || null;

            // Try generic reset
            if (gr && typeof gr.reset === 'function') { gr.reset(); reset = true; }

            // Poke into clients (covers multiple widgets on page)
            const cfg = window.___grecaptcha_cfg && window.___grecaptcha_cfg.clients;
            if (cfg && typeof cfg === 'object') {
              for (const c of Object.values(cfg)) {
                if (!c) continue;
                for (const v of Object.values(c)) {
                  if (v && typeof v.reset === 'function') { v.reset(); reset = true; }
                  if (v && v.V && typeof v.V.reset === 'function') { v.V.reset(); reset = true; }
                  if (v && v.W && typeof v.W.reset === 'function') { v.W.reset(); reset = true; }
                }
              }
            }
            return reset;
          } catch(e) { return false; }
        }
        """)
    except Exception:
        return False

def solve_recaptcha_and_submit(page, api_key: str, max_attempts: int = 5, base_sleep: float = 3.0, entity: str | None = None):
    """
    Re-solves reCAPTCHA up to max_attempts with exponential backoff.
    Between attempts:
      - reset the widget
      - re-click the Submit button to trigger a fresh challenge
    """
    for attempt in range(1, max_attempts + 1):
        try:
            info = _extract_recaptcha_info(page)
            if not info.get("sitekey"):
                emit_ux("CAPTCHA", "No reCAPTCHA widget found; skipping solver.", "info", entity=entity)
                return

            emit_ux("CAPTCHA", f"Attempt {attempt}/{max_attempts}: solving…", "info", entity=entity)
            token = _solve_recaptcha_with_2captcha(page, api_key, info)

            emit_ux("CAPTCHA", "Token received; submitting…", "info", entity=entity)
            _finalize_recaptcha_and_submit(page, token)

            emit_ux("CAPTCHA", "reCAPTCHA solved", "success", entity=entity)
            return  # success

        except Exception as e:
            msg = str(e)
            # Non-retryable
            if "ERROR_ZERO_BALANCE" in msg or "ERROR_WRONG_USER_KEY" in msg:
                emit_ux("CAPTCHA", humanize_error(e), "error", entity=entity)
                raise

            if attempt < max_attempts:
                emit_ux("CAPTCHA", "Solve failed — resetting widget and re-submitting…", "info", entity=entity)
                try:
                    _reset_recaptcha_widget(page)
                except Exception:
                    pass
                # Re-trigger the challenge
                try:
                    page.get_by_role("button", name=re.compile(r"Submit Payment|Submit", re.I)).click()
                except Exception:
                    pass

                sleep_s = round(base_sleep * (1.5 ** (attempt - 1)), 2)
                page.wait_for_timeout(int(sleep_s * 1000))
                continue

            # Out of attempts
            emit_ux("CAPTCHA", f"Failed after {max_attempts} attempts: {humanize_error(e)}", "error", entity=entity)
            raise

# =======================================
# --- 1Password helpers  ---
# =======================================
def validate_2captcha_key(key: str):
    import urllib.request, urllib.parse
    key = (key or "").strip()
    if not key:
        raise RuntimeError("2Captcha key is empty.")

    # allow skipping if env asks
    if os.getenv("WY_SKIP_2CAPTCHA_BALANCE_CHECK", "").lower() in ("1","true","yes","on"):
        print(">>> Skipping 2Captcha balance check (WY_SKIP_2CAPTCHA_BALANCE_CHECK set).")
        return

    qs = urllib.parse.urlencode({"key": key, "action": "getbalance"})
    url = f"https://2captcha.com/res.php?{qs}"

    body = None
    err  = None

    # Try urllib with certifi-backed context
    try:
        import ssl, certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=15, context=ctx) as r:
            body = r.read().decode().strip()
    except Exception as e:
        err = e
        # Optional fallback via requests (also uses certifi)
        try:
            import requests, certifi as _c
            resp = requests.get(url, timeout=15, verify=_c.where())
            resp.raise_for_status()
            body = resp.text.strip()
        except Exception as e2:
            raise RuntimeError(
                f"2Captcha key check failed due to network/SSL error: {err} ; fallback also failed: {e2}. "
                "Set WY_SKIP_2CAPTCHA_BALANCE_CHECK=1 to bypass."
            ) from err

    if body.startswith("ERROR_"):
        raise RuntimeError(f"2Captcha key check failed: {body}")
    print(f">>> 2Captcha balance: {body}")

def _clean_key(s: str) -> str:
    # remove quotes/whitespace/newlines just in case
    s = (s or "").strip().strip("'").strip('"')
    s = s.replace(" ", "").replace("\n", "").replace("\r", "")
    return s

def get_2captcha_api_key(vault_name: str, item_title: str) -> str:
    # 0) ENV override for quick isolation
    env_key = os.getenv("WY_2CAPTCHA_KEY")
    if env_key:
        return _clean_key(env_key)

    print(">>> Fetching 2Captcha API key from 1Password...")
    # Grab full JSON and pick the right concealed field
    cmd = ["op", "item", "get", item_title, "--vault", vault_name, "--format", "json"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        item = json.loads(res.stdout)
    except subprocess.CalledProcessError as e:
        print(f"!!! Error fetching 2Captcha API key: {e.stderr}")
        raise

    candidates = []
    for f in item.get("fields", []):
        label = (f.get("label") or f.get("id") or "").lower()
        val   = _clean_key(f.get("value") or "")
        ftype = (f.get("type") or "").upper()  # "CONCEALED" for secrets
        if not val:
            continue
        # Prefer obvious labels
        if any(tok in label for tok in ("api key", "apikey", "2captcha")):
            candidates.append(val)
        # Otherwise any concealed secret is a candidate
        elif ftype == "CONCEALED":
            candidates.append(val)

    if not candidates:
        raise RuntimeError("No suitable field found in 1Password item for 2Captcha key.")

    key = candidates[0]

    # Sanity hint if it looks suspicious
    if not re.fullmatch(r"[A-Za-z0-9]{32}", key):
        print(f"!!! Warning: 2Captcha key shape looks odd: {repr(key)} (len={len(key)})")

    print(">>> Successfully fetched API key.")
    return key

def get_payment_details_from_1password(vault_name: str, item_title: str) -> dict:
    """
    Fetch payment details from 1Password using a service account token (OP_SERVICE_ACCOUNT_TOKEN).
    This version REQUIRES the 1Password CLI ("op") and a valid token; it does not fall back.
    """
    print(">>> Fetching payment details from 1Password (service account)…", flush=True)

    op = shutil.which("op")
    if not op:
        raise RuntimeError(
            "1Password CLI ('op') is not installed. Install it on the server and try again."
        )

    token = (os.getenv("OP_SERVICE_ACCOUNT_TOKEN") or "").strip()
    token_file = (os.getenv("OP_SERVICE_ACCOUNT_TOKEN_FILE") or "").strip()
    if not token and token_file:
        try:
            token = open(token_file, "r", encoding="utf-8").read().strip()
        except Exception:
            pass
    if not token:
        raise RuntimeError(
            "OP_SERVICE_ACCOUNT_TOKEN (or OP_SERVICE_ACCOUNT_TOKEN_FILE) is not set. "
            "Provide a 1Password service account token."
        )

    env = os.environ.copy()
    env["OP_SERVICE_ACCOUNT_TOKEN"] = token

    # Optional: verify auth
    try:
        subprocess.run([op, "whoami", "--format", "json"],
                       env=env, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("1Password CLI authentication failed. Check your service account token.") from e

    # Prefer an exact-title match in the given vault
    try:
        lst = subprocess.run([op, "item", "list", "--vault", vault_name, "--format", "json"],
                             env=env, capture_output=True, text=True, check=True)
        items = json.loads(lst.stdout) or []
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to list items in vault '{vault_name}': {e.stderr.strip()}") from e

    exact = [it for it in items if (it.get("title") or "") == item_title]
    if len(exact) == 1:
        selector = exact[0]["id"]
        print(f">>> Using exact title match: {item_title} (id={selector})")
    elif len(exact) > 1:
        env_id = (os.getenv("WY_ITEM_CARD_ID") or "").strip()
        if not env_id:
            ids = ", ".join(it.get("id","") for it in exact[:5])
            raise RuntimeError(
                f"Multiple items titled '{item_title}' in vault '{vault_name}'. "
                f"Set WY_ITEM_CARD_ID to the desired item id. Candidates: {ids}…"
            )
        selector = env_id
        print(f">>> Multiple matches; using WY_ITEM_CARD_ID={selector}")
    else:
        # allow explicit ID or fall back to title lookup (may be fuzzy on 1P side)
        selector = (os.getenv("WY_ITEM_CARD_ID") or "").strip() or item_title
        if selector == item_title:
            print(">>> No exact-title match; using title lookup via 1Password.")

    try:
        res = subprocess.run([op, "item", "get", selector, "--vault", vault_name, "--format", "json"],
                             env=env, capture_output=True, text=True, check=True)
        item_json = json.loads(res.stdout)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or "").lower()
        if "more than one item matches" in msg:
            raise RuntimeError(
                f"1Password returned multiple matches for '{selector}'. "
                f"Set WY_ITEM_CARD_ID to the exact item id."
            ) from e
        raise RuntimeError(f"Failed to fetch 1Password item '{selector}': {(e.stderr or '').strip()}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError("Failed to parse 1Password response JSON.") from e

    fields = item_json.get("fields") or []

    def _get(*labels):
        wants = {lb.strip().lower() for lb in labels}
        for f in fields:
            lab = (f.get("label") or f.get("id") or f.get("purpose") or "").strip().lower()
            if lab in wants:
                return (f.get("value") or "").strip()
        return ""

    def _norm_exp(v, out_style="MMYY"):
        s = str(v or "").strip()
        digits = re.sub(r"\D", "", s)
        mm = None; yy2 = None
        m = re.match(r"^\s*(\d{1,4})\s*[\/\-\s]\s*(\d{1,4})\s*$", s)
        if m:
            a, b = m.group(1), m.group(2)
            if a.isdigit() and 1 <= int(a) <= 12:
                mm = int(a); yy2 = b[-2:]
            elif len(a) == 4 and b.isdigit() and 1 <= int(b) <= 12:
                mm = int(b); yy2 = a[-2:]
        else:
            if len(digits) == 4:
                mm = int(digits[:2]); yy2 = digits[2:]
            elif len(digits) == 6:
                first_two = int(digits[:2]); last_two = int(digits[4:6]); first_four = int(digits[:4])
                if (first_two > 12 and 1 <= last_two <= 12) or (1900 <= first_four <= 2099 and 1 <= last_two <= 12):
                    mm = last_two; yy2 = digits[2:4]
                else:
                    mm = first_two; yy2 = digits[4:6]
            else:
                return ""
        mm = max(1, min(int(mm), 12))
        return f"{mm:02d}{yy2}" if out_style.upper() == "MMYY" else f"{mm:02d}/{yy2}"

    exp_raw   = _get("expiry date", "expiration date", "exp", "expires")
    exp_mm_yy = _norm_exp(exp_raw, out_style="MMYY")
    details = {
        "card_name":   _get("cardholder name", "name on card", "card name", "name"),
        "card_number": _get("number", "card number", "cardnumber"),
        "exp_date":    exp_mm_yy,
        "exp_mm":      exp_mm_yy[:2] if len(exp_mm_yy) == 4 else "",
        "exp_yy":      exp_mm_yy[2:] if len(exp_mm_yy) == 4 else "",
        "cvv":         _get("verification number", "cvv", "cvc", "security code"),
        "company":     _get("billing company", "company"),
        "first_name":  _get("billing first name", "first name"),
        "last_name":   _get("billing last name", "last name"),
        "address":     _get("billing address", "address 1", "address", "street address"),
        "city":        _get("billing city", "city"),
        "state":       _get("billing state", "state"),
        "zip":         _get("billing zip", "postal code", "zip"),
        "email":       _get("billing email", "email"),
        "_source":     "1password",
    }

    missing = [k for k in ("card_number", "exp_date", "cvv") if not details.get(k)]
    if missing:
        raise RuntimeError(f"Missing required card fields in 1Password item: {', '.join(missing)}")

    print(">>> Payment details fetched from 1Password.")
    return details

def get_defaults_profile_from_1password(vault: str, item_title: str) -> dict:
    """
    Load the default filing profile from 1Password.
    Priority:
      1) WY_DEFAULTS_JSON env (for local dev quick overrides)
      2) JSON blob in notesPlain of the item
      3) Field labels/values in the item (label -> normalized key)
    No hard-coded DEFAULTS are used.
    """
    profile: dict = {}

    # Skip if 'op' not installed
    if not shutil.which("op"):
        print(">>> 1Password CLI ('op') not installed; skipping defaults from 1Password.")
        return {}

    # Skip if no service account token available
    if not os.getenv("OP_SERVICE_ACCOUNT_TOKEN", "").strip():
        print(">>> OP_SERVICE_ACCOUNT_TOKEN not set; skipping defaults from 1Password.")
        return {}

    # local key normalizer (keeps this function self-contained)
    def _norm_key_local(k: str) -> str:
        if not k:
            return ""
        k = k.strip().lower()
        # collapse spaces/dashes to underscores
        k = re.sub(r"[\s\-]+", "_", k)
        return k

    # 1) Optional local override for dev: export WY_DEFAULTS_JSON='{"email":"..."}'
    env_json = os.getenv("WY_DEFAULTS_JSON")
    if env_json:
        try:
            data = json.loads(env_json)
            profile.update({ _norm_key_local(k): v for k, v in (data or {}).items() if v not in ("", None) })
        except Exception:
            print("!!! WY_DEFAULTS_JSON is not valid JSON; ignoring.")

    # 2) Pull item from 1Password
    cmd = ["op", "item", "get", item_title, "--vault", vault, "--format", "json"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        item = json.loads(res.stdout)

        # Preferred: JSON stored in notesPlain
        notes = (item.get("notesPlain") or "").strip()
        if notes.startswith("{"):
            try:
                data = json.loads(notes)
                profile.update({ _norm_key_local(k): v for k, v in (data or {}).items() if v not in ("", None) })
                return profile
            except Exception:
                print("!!! notesPlain JSON parse failed; falling back to fields.")

        # Fallback: use item fields (label -> value)
        # We don't filter by a DEFAULTS list; we just normalize and include non-empty.
        for f in item.get("fields", []):
            label = f.get("label") or f.get("id") or ""
            val = f.get("value")
            k = _norm_key_local(label)
            if k and val not in ("", None):
                profile[k] = val

    except subprocess.CalledProcessError as e:
        print(f"!!! Could not load 1Password defaults: {e.stderr.strip()}")

    return profile

def require_1password_service_token():
    if not (os.getenv("OP_SERVICE_ACCOUNT_TOKEN") or os.getenv("OP_SERVICE_ACCOUNT_TOKEN_FILE")):
        print("!!! 1Password: No service account token found. Set OP_SERVICE_ACCOUNT_TOKEN "
              "or OP_SERVICE_ACCOUNT_TOKEN_FILE so the bot can read items headlessly.", flush=True)
        sys.exit(1)

# =======================================
# --- Simple image CAPTCHA (CGS)      ---
# =======================================
ENTITY_NAME = "Q. Neal Holdings LLC"
SEARCH_URL  = "https://wyobiz.wyo.gov/Business/FilingSearch.aspx"

def _two_captcha() -> TwoCaptcha:
    # Keep your existing logic (hardcoded key) for the simple image CAPTCHA section
    api_key = "6372ee21fa7744c7cc04fcc250f06f14"
    if not api_key:
        raise RuntimeError("Set TWO_CAPTCHA_API_KEY in your environment.")
    return TwoCaptcha(api_key)

def _gate_is_present(page, timeout: int = 8000) -> bool:
    try:
        page.wait_for_function(
            """
            () => {
              const t = (document.body && (document.body.innerText || "")) || "";
              if (/What code is in the image/i.test(t)) return true;
              if (/Your support ID is/i.test(t)) return true;
              if (/This question is for testing whether you are a human/i.test(t)) return true;
              if (document.querySelector("#ans")) return true;
              if (document.querySelector("img[alt='Red dot'], img[alt*='dot' i], img[alt*='bottle' i], img[src^='data:image']")) return true;
              return false;
            }
            """,
            timeout=timeout,
        )
        return True
    except Exception:
        return False

def _locate_gate_image(page):
    candidates = [
        "img[alt='Red dot']",
        "img[alt*='dot' i]",
        "img[alt*='bottle' i]",
        "img[src^='data:image']",
        "form:has(#ans) img",
        "form:has-text('What code is in the image') img",
        "img[alt*='captcha' i], img[src*='captcha' i]",
        "div[id*='captcha' i] img, div[class*='captcha' i] img",
    ]
    for sel in candidates:
        loc = page.locator(sel).first
        if loc.count():
            return loc
    try:
        prompt = page.get_by_text("What code is in the image?", exact=False).first
        if prompt.count():
            loc = prompt.locator("xpath=following::img[1]")
            if loc.count():
                return loc.first
    except Error:
        pass
    return page.locator("img").first  # last resort

def _solve_simple_image_gate(page, max_attempts=10) -> bool:
    if not _gate_is_present(page, timeout=8000):
        return False
    print(">>> Simple image CAPTCHA detected. Solving via 2Captcha…")

    img = _locate_gate_image(page)
    if not img.count():
        out = pathlib.Path("captcha_gate_fullpage.png").resolve()
        page.screenshot(path=str(out), full_page=True)
        raise RuntimeError("Could not find CAPTCHA image (saved full-page screenshot).")

    solver = _two_captcha()

    for attempt in range(1, max_attempts + 1):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = tmp.name
        img.screenshot(path=img_path)

        try:
            result = solver.normal(img_path, caseSensitive=1)
            code = result["code"].strip()
            print(f">>> 2Captcha (attempt {attempt}) => {code}")
        finally:
            try: os.remove(img_path)
            except OSError: pass

        target_input = None
        for sel in [
            "#ans", "input[name='answer']", "input[name='code']",
            "input[name*='captcha' i]", "input[aria-label*='code' i]", "input[type='text']",
        ]:
            loc = page.locator(sel).first
            if loc.count():
                target_input = loc
                break
        if not target_input:
            raise RuntimeError("Could not find CAPTCHA answer input.")
        target_input.fill(code)

        submitted = False
        for sel in ["#jar", "button:has-text('submit')", "input[type='submit']", "button[type='submit']"]:
            btn = page.locator(sel).first
            if btn.count():
                try:
                    btn.click()
                    submitted = True
                    break
                except Error:
                    pass
        if not submitted:
            page.get_by_role("button", name=re.compile(r"(submit|verify|continue)", re.I)).click()

        try:
            expect(page.locator("#MainContent_txtFilingName")).to_be_visible(timeout=4000)
            print(">>> Simple gate passed.")
            return True
        except (AssertionError, Error):
            if attempt < max_attempts:
                print(">>> CAPTCHA answer rejected; retrying…")
                continue
            raise RuntimeError("Failed simple image CAPTCHA after multiple attempts.")

def _search_entity(page, entity_name: str) -> bool:
    # Ensure the form is ready
    expect(page.locator("#MainContent_txtFilingName")).to_be_visible(timeout=8000)

    # Fill the name
    page.locator("#MainContent_txtFilingName").fill(entity_name)

    # Select "Contains" robustly
    contains_radio = page.locator("#MainContent_chkSearchIncludes")
    smart_check_radio(page, contains_radio)

    # Click Search
    page.get_by_role("button", name=re.compile(r"search", re.I)).click()

    # Look for the result link
    link = page.get_by_role("link", name=re.compile(re.escape(entity_name), re.I)).first
    try:
        expect(link).to_be_visible(timeout=5000)
        # Your existing logic
        return False
    except (AssertionError, Error):
        return False

def _click_entity_result(page, entity_name: str) -> None:
    link = page.get_by_role("link", name=re.compile(re.escape(entity_name), re.I)).first
    expect(link).to_be_visible(timeout=10000)
    link.click()

def has_valid_designation(name: str) -> bool:
    return bool(_DESIGNATION_END_RE.search((name or "").strip()))

def add_default_designation(name: str) -> str:
    base = (name or "").strip().rstrip(",.;")
    return base if has_valid_designation(base) else f"{base} LLC"

def designation_error_text(page, timeout_ms: int = 2000) -> str:
    """
    Returns the page's designation error text if present (e.g. 'The Entity Name must contain one of the following designations: ...'),
    else empty string.
    """
    try:
        el = page.locator("#lblErrorMessage")
        el.wait_for(state="visible", timeout=timeout_ms)
        txt = (el.inner_text() or "").strip()
        if "must contain one of the following designations" in txt.lower():
            return txt
    except Exception:
        pass
    return ""

def _extract_filing_id(page) -> str:
    try:
        label = page.get_by_text(re.compile(r"Filing\s*ID", re.I)).first
        if label.count():
            fid = page.evaluate(
                """
                (el) => {
                  const root = el.closest('tr,div,section,dl,li') || el.parentElement || document.body;
                  const m = (root.innerText || '').match(/\\b\\d{4}-\\d{9}\\b/);
                  return m ? m[0] : null;
                }
                """,
                label
            )
            if fid:
                print(f">>> Filing ID (label-nearby): {fid}")
                return fid
    except Error:
        pass

    try:
        fid = page.evaluate("""() => {
            const t = (document.body && document.body.innerText) || '';
            const m = t.match(/\\b\\d{4}-\\d{9}\\b/);
            return m ? m[0] : null;
        }""")
        if fid:
            print(f">>> Filing ID (page-scan): {fid}")
            return fid
    except Error:
        pass

    try:
        any_match = page.get_by_text(re.compile(r"\\b\\d{4}-\\d{9}\\b")).first
        if any_match.count():
            fid = any_match.inner_text().strip()
            print(f">>> Filing ID (exact-text elem): {fid}")
            return fid
    except Error:
        pass

    raise RuntimeError("Could not find Filing ID on the entity page.")

def _download_cgs_or_print(page, entity_name: str, out_dir: str, prefer_cgs=True) -> pathlib.Path:
    ts   = _ts_short()
    slug = _slug(entity_name)
    cgs_dir = os.path.join(out_dir, "cgs")
    _ensure_dir(cgs_dir)

    link = (page.get_by_role("link", name=re.compile(r"Certificate of Good Standing|CGS", re.I)).first
            if prefer_cgs else
            page.get_by_role("link", name=re.compile(r"Print", re.I)).first)
    if not link.count():
        link = page.get_by_role("link", name=re.compile(r"Print", re.I)).first
    expect(link).to_be_visible(timeout=10000)

    try:
        with page.expect_download(timeout=8000) as dl_info:
            link.click()
        download = dl_info.value
    except Exception:
        # likely hit a gate first
        if _gate_is_present(page, timeout=8000):
            _solve_simple_image_gate(page)
        with page.expect_download(timeout=20000) as dl_info:
            link.click()
        download = dl_info.value

    # _download_cgs_or_print(...)
    target = pathlib.Path(cgs_dir) / f"CGS_{slug}_{RUN_ID}_{ts}.pdf"
    download.save_as(str(target))
    print(f">>> Downloaded CGS: {target}")
    return target

def _download_good_standing_by_id(page, filing_id: str, entity_name: str, out_dir: str) -> pathlib.Path:
    ts   = _ts_short()
    slug = _slug(entity_name)
    gsc_dir = os.path.join(out_dir, "gsc")
    _ensure_dir(gsc_dir)

    print(">>> Step 8: Navigating to 'Good Standing Certificates'…")
    link = page.get_by_role("link", name=re.compile(r"Good Standing Certificates", re.I)).first
    expect(link).to_be_visible(timeout=10000)
    link.click()

    page.wait_for_load_state("domcontentloaded")
    if _gate_is_present(page, timeout=8000):
        _solve_simple_image_gate(page)

    print(f">>> Filling Filing ID: {filing_id}")
    expect(page.locator("#MainContent_txtFilingID")).to_be_visible(timeout=10000)
    page.locator("#MainContent_txtFilingID").fill(filing_id)

    trigger = page.get_by_title(re.compile(r"Get a Certificate of Good", re.I)).first
    if trigger.count():
        trigger.click()
    else:
        page.get_by_role("button", name=re.compile(r"Get.*Certificate", re.I)).click()

    page.wait_for_load_state("domcontentloaded")
    if _gate_is_present(page, timeout=5000):
        _solve_simple_image_gate(page)

    print(">>> Printing Good Standing Certificate…")
    btn_print = page.get_by_role("button", name=re.compile(r"Print", re.I)).first
    expect(btn_print).to_be_visible(timeout=10000)

    try:
        with page.expect_download(timeout=20000) as dl_info:
            btn_print.click()
        download = dl_info.value
    except Exception:
        if _gate_is_present(page, timeout=8000):
            _solve_simple_image_gate(page)
            with page.expect_download(timeout=20000) as dl_info:
                btn_print.click()
            download = dl_info.value
        else:
            raise RuntimeError("Failed to download GSC after printing.")

    target = pathlib.Path(gsc_dir) / f"GSC_{slug}_{filing_id}_{RUN_ID}_{ts}.pdf"
    download.save_as(str(target))
    print(f">>> GSC downloaded: {target}")
    return target
    
# =========================
# --- Main flow helpers  ---
# =========================

def prune_old_results(out_dir: str,
                      keep: int = int(os.getenv("WY_RESULTS_KEEP", "3")),
                      pattern: str = "results_*.csv") -> None:
    """
    Keep only the newest `keep` results CSVs in out_dir, delete the rest.
    Default keep is 3 (current + two previous). Override via WY_RESULTS_KEEP.
    """
    try:
        p = Path(out_dir)
        files = sorted(p.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        for old in files[keep:]:
            try:
                old.unlink()
                print(f">>> Deleted old results: {old}")
            except Exception as e:
                print(f">>> Could not delete {old}: {e}")
    except Exception as e:
        print(f">>> prune_old_results failed: {e}")


def prune_old_runs(runs_root: str,
                   keep: int = int(os.getenv("WY_RUNS_KEEP", "3")),
                   exclude: tuple[str, ...] = ()) -> None:
    """
    Keep only the newest `keep` subdirectories in runs_root (run workdirs), delete the rest.
    `exclude` can list run_ids (folder names) that must never be deleted (e.g., the current run).
    Default keep is 3. Override via WY_RUNS_KEEP.
    """
    try:
        root = Path(runs_root)
        if not root.exists():
            return
        # sort folders by last modified desc
        dirs = [d for d in root.iterdir() if d.is_dir()]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

        protected = set(exclude or ())
        for d in dirs[keep:]:
            if d.name in protected:
                continue
            try:
                shutil.rmtree(d)
                print(f">>> Deleted old run folder: {d}")
            except Exception as e:
                print(f">>> Could not delete run folder {d}: {e}")
    except Exception as e:
        print(f">>> prune_old_runs failed: {e}")

# ---- keep near other prune helpers ----
def _kept_run_ids(runs_root: str,
                  keep_prev: int,
                  exclude: tuple[str, ...] = ()) -> tuple[set[str], float | None]:
    """
    Return ({run_ids_to_keep}, min_mtime_of_kept_runs). We keep the last `keep_prev`
    previous runs (excluding current), plus anything in `exclude` (e.g., current RUN_ID).
    """
    root = Path(runs_root)
    if not root.exists():
        return (set(exclude or ()), None)

    dirs = [d for d in root.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    # exclude current (and any extra excludes) from the "previous runs" list
    exclude_set = set(exclude or ())
    prev_ids = [d.name for d in dirs if d.name not in exclude_set]
    keep_ids = set(prev_ids[:keep_prev]) | exclude_set

    # min mtime among the kept run folders (for legacy file pruning)
    kept_mtimes = [d.stat().st_mtime for d in dirs if d.name in keep_ids]
    min_kept_mtime = min(kept_mtimes) if kept_mtimes else None
    return keep_ids, min_kept_mtime


def prune_doc_buckets_by_run(out_dir: str,
                             runs_root: str,
                             keep_prev: int = int(os.getenv("WY_DOC_RUNS_KEEP", "3")),
                             exclude: tuple[str, ...] = (RUN_ID,),
                             subdirs: tuple[str, ...] = ("cgs", "gsc", "receipts", "registrations")) -> None:
    """
    Delete PDFs in the doc buckets unless their filename contains a RUN_ID that is
    in the last `keep_prev` previous runs, plus any `exclude` ids (current run).
    For legacy files (no RUN_ID in filename), delete if older than the oldest kept run.
    """
    try:
        keep_ids, min_kept_mtime = _kept_run_ids(runs_root, keep_prev, exclude)
        rid_rx = re.compile(r'_(\d{8}-\d{6})(?:_|\.|$)')  # matches RUN_ID like 20250101-123456

        for bucket in subdirs:
            folder = Path(out_dir) / bucket
            if not folder.exists():
                continue

            for f in folder.iterdir():
                if not f.is_file():
                    continue
                name = f.name
                m = rid_rx.search(name)
                if m:
                    rid = m.group(1)
                    if rid not in keep_ids:
                        try:
                            f.unlink()
                            print(f">>> Deleted old {bucket} file (run {rid} not kept): {f}")
                        except Exception as e:
                            print(f">>> Could not delete {f}: {e}")
                else:
                    # Legacy file with no RUN_ID tag — prune by timestamp vs kept runs
                    if min_kept_mtime is not None and f.stat().st_mtime < min_kept_mtime:
                        try:
                            f.unlink()
                            print(f">>> Deleted legacy {bucket} file (older than kept runs): {f}")
                        except Exception as e:
                            print(f">>> Could not delete {f}: {e}")
    except Exception as e:
        print(f">>> prune_doc_buckets_by_run failed: {e}")

def _within(parent: Path, child: Path) -> bool:
    try:
        parent = parent.resolve()
        child  = child.resolve()
        return str(child).startswith(str(parent))
    except Exception:
        return False

def _delete_files_older_than(dirpath: Path, days: int, patterns=("**/*",)):
    cut = time.time() - days * 86400
    deleted = 0
    if not dirpath.exists():
        return 0
    for pat in patterns:
        for p in dirpath.glob(pat):
            try:
                if p.is_file() and p.stat().st_mtime < cut:
                    p.unlink()
                    deleted += 1
            except Exception as e:
                print(f">>> Could not delete {p}: {e}")
    return deleted

def prune_results_by_age(out_dir: str, days: int = DAYS_KEEP) -> None:
    """
    Delete results_*.csv older than `days`.
    """
    base = Path(out_dir)
    if not base.exists(): return
    cut = time.time() - days * 86400
    for f in base.glob("results_*.csv"):
        try:
            if f.stat().st_mtime < cut:
                f.unlink()
                print(f">>> Deleted old results CSV: {f}")
        except Exception as e:
            print(f">>> Could not delete {f}: {e}")

def prune_documents_by_age(out_dir: str, days: int = DAYS_KEEP, folders: list[str] = None) -> None:
    """
    Delete doc files older than `days` inside selected subfolders.
    Also prunes stray PDFs in the out_dir root (safe/no-op if none).
    """
    base = Path(out_dir)
    if not base.exists(): return
    folders = folders or PRUNE_FOLDERS

    # Subfolders (receipts, registrations, cgs, gsc)
    for sub in folders:
        d = base / sub
        if _within(base, d):
            n = _delete_files_older_than(d, days, patterns=("**/*.pdf", "**/*.png"))
            if n: print(f">>> Deleted {n} old file(s) in {d}")

    # Root PDFs (just in case any old flow saved here)
    n_root = _delete_files_older_than(base, days, patterns=("*.pdf",))
    if n_root:
        print(f">>> Deleted {n_root} old PDF(s) in {base}")

def prune_run_dirs_by_age(runs_root: str, days: int = DAYS_KEEP, exclude: tuple[str, ...] = ()) -> None:
    """
    Delete run folders older than `days`, except those in `exclude`.
    """
    root = Path(runs_root)
    if not root.exists(): return
    cut = time.time() - days * 86400
    protected = set(exclude or ())
    for d in root.iterdir():
        if not d.is_dir(): continue
        if d.name in protected: continue
        try:
            if d.stat().st_mtime < cut:
                shutil.rmtree(d)
                print(f">>> Deleted old run folder: {d}")
        except Exception as e:
            print(f">>> Could not delete run folder {d}: {e}")

def prune_media_by_age(days: int = DAYS_KEEP) -> None:
    """
    Optional: delete videos & traces older than `days` when enabled.
    """
    # Videos (when WY_VIDEO_DIR is set AND WY_PRUNE_VIDEOS is truthy)
    if _truthy(os.getenv("WY_PRUNE_VIDEOS", "1")):
        vdir = os.getenv("WY_VIDEO_DIR")
        if vdir:
            n = _delete_files_older_than(Path(vdir), days, patterns=("**/*",))
            if n: print(f">>> Deleted {n} old video file(s) in {vdir}")

    # Traces (only if a trace dir is configured)
    tdir = os.getenv("WY_TRACE_DIR")
    if tdir:
        n = _delete_files_older_than(Path(tdir), days, patterns=("**/*",))
        if n: print(f">>> Deleted {n} old trace file(s) in {tdir}")

def _retry_nav(fn, attempts=6, sleep_s=0.25):
    for i in range(attempts):
        try:
            return fn()
        except Error as e:
            msg = str(e)
            if "Execution context was destroyed" in msg or "Target closed" in msg or "detached" in msg:
                time.sleep(sleep_s)
                continue
            raise
    raise

def _find_confirmation_page(context, timeout_ms: int = 120000):
    """
    Returns the page that shows the Registration Complete screen (with Receipt/Registration PDF links),
    scanning all open pages and handling tabs closing/navigating during payment handoff.
    """
    deadline = time.time() + timeout_ms/1000.0
    last_err = None

    def _looks_like_done(pg) -> bool:
        try:
            # Quick URL check helps when they use a distinct route
            if re.search(r"(RegistrationComplete|Receipt|/Print)", pg.url or "", re.I):
                return True
            # DOM check (resists minor URL changes)
            return pg.evaluate("""() => {
                const t = (document.body && document.body.innerText) || '';
                if (/Registration Complete/i.test(t)) return true;
                if (document.querySelector('#linkbtnPrintReceipt')) return true;
                if ([...document.querySelectorAll('a')].some(a => /Receipt PDF|Registration PDF/i.test(a.textContent||''))) return true;
                return false;
            }""")
        except Error:
            return False

    while time.time() < deadline:
        for pg in context.pages:
            if pg.is_closed():
                continue
            try:
                pg.wait_for_load_state("domcontentloaded", timeout=1500)
            except Error:
                pass
            if _looks_like_done(pg):
                return pg
        time.sleep(0.3)

    raise TimeoutError("Could not find confirmation page with Receipt/Registration links after payment.")

def _name_unavailable_reason(page, timeout_ms: int = 4000) -> str | None:
    """
    After clicking 'Next >>' on the name page, the site shows a banner when the
    name already exists or is deceptively similar. Detect that and return a reason.
    """
    patterns = [
        r"\bnot available\b",
        r"\bdeceptively similar\b",
        r"\balready exists?\b",
        r"\bname.*unavailable\b",
        r"helpful search tips",  # phrase from their message
    ]
    joined = "(" + "|".join(patterns) + ")"

    try:
        page.wait_for_function(
            f"""() => new RegExp({json.dumps(joined)}, 'i')
                  .test((document.body && document.body.innerText) || '')""",
            timeout=timeout_ms
        )
        # grab a short snippet to include in notes
        txt = page.evaluate("() => (document.body && document.body.innerText) || ''")[:2000]
        # find first matching line for a nicer note
        import re
        for line in (txt or "").splitlines():
            if re.search(joined, line, re.I):
                return line.strip()
        return "Business name not available in Wyoming."
    except Exception:
        return None

def abort_requested() -> bool:
    try: return ABORT_FLAG.exists()
    except: return False
    
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _select_registered_agent_from_dialog(page, agent_name: str, min_similarity: float = 0.70) -> None:
    """
    Assumes you've already clicked 'Search'. This waits for the result dialog,
    handles 'No Results Found.', logs the count, and clicks the best match.
    It prefers exact match, then contains, then fuzzy.
    """
    # Wait for the modal/dialog that contains either results or "No Results Found."
    dialog = page.locator("div.ui-dialog-content.ui-widget-content").filter(
        has_text=re.compile(r"(Results Found|No Results Found\.)", re.I)
    ).first
    expect(dialog).to_be_visible(timeout=10000)

    # Quick no-results check
    if dialog.get_by_text(re.compile(r"^\s*No Results Found\.\s*$", re.I)).count():
        raise RuntimeError(f"No Results Found for registered agent: '{agent_name}'")

    # (Optional) log the results count if present
    try:
        txt = dialog.inner_text()
        m = re.search(r"(\d+)\s+Results\s+Found", txt, re.I)
        if m:
            print(f">>> Agent search: {m.group(1)} result(s) found.")
    except Exception:
        pass

    # Collect candidates
    items = dialog.locator("ol.search-results > li")
    n = items.count()
    if n == 0:
        raise RuntimeError("Agent search dialog rendered without list items.")

    # Inspect each <li>: name lives in .resFile1 .resultField (the 2nd span)
    best_idx, best_score = -1, 0.0
    best_name, best_id = None, None

    target_norm = _norm(agent_name)

    for i in range(n):
        li = items.nth(i)
        # Extract display name
        name_loc = li.locator(".resFile1 .resultField").nth(1)
        name_text = (name_loc.inner_text().strip() if name_loc.count() else "")

        # Extract the RA id from onclick="Load_AgentFields('0203370');"
        onclick = li.locator("a").first.get_attribute("onclick") or ""
        m = re.search(r"Load_AgentFields\('(\d+)'\)", onclick)
        ra_id = m.group(1) if m else None

        # Match strategy: exact > contains > fuzzy (SequenceMatcher)
        if name_text.strip().lower() == agent_name.strip().lower():
            print(f">>> Agent exact match: {name_text} (ID {ra_id or '?'})")
            li.locator("a").click()
            return

        if re.search(re.escape(agent_name), name_text, re.I):
            print(f">>> Agent contains match: {name_text} (ID {ra_id or '?'})")
            li.locator("a").click()
            return

        score = SequenceMatcher(None, _norm(name_text), target_norm).ratio()
        if score > best_score:
            best_score = score
            best_idx = i
            best_name = name_text
            best_id = ra_id

    # Fuzzy fallback if nothing exact/contains matched
    if best_idx >= 0 and best_score >= min_similarity:
        print(f">>> Agent fuzzy match ({best_score:.2f}): {best_name} (ID {best_id or '?'})")
        items.nth(best_idx).locator("a").click()
        return

    raise RuntimeError(f"Could not match registered agent '{agent_name}' to any of {n} result(s).")

def robust_goto(page, url, anchor_locator=None, tries=2):
    """
    Go to url, wait for an 'anchor' locator. If anchor not visible or DOM looks empty,
    reload once and try again.
    """
    last_err = None
    for attempt in range(1, tries + 1):
        page.goto(url, wait_until="domcontentloaded")
        # blank-page quick check
        try:
            body_len = page.evaluate("() => (document.body && document.body.innerText || '').length")
        except Exception:
            body_len = 0
        if body_len < 20:
            page.wait_for_timeout(1000)

        if anchor_locator:
            try:
                page.locator(anchor_locator).first.wait_for(state="visible", timeout=8000)
                return
            except Exception as e:
                last_err = e
                if attempt < tries:
                    page.reload(wait_until="domcontentloaded")
                    continue
                raise last_err
        else:
            return
        
def _ts_short() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", (s or "").strip()).strip("_")

def _safe_join_dir(dirpath: str, *names: str) -> str:
    _ensure_dir(dirpath)
    return os.path.join(dirpath, *names)

def _wait_for_registration_complete(page, timeout_ms: int = 90000):
    """
    Wait robustly for the confirmation screen. Retries if the page navigates while evaluating.
    """
    deadline = time.time() + timeout_ms/1000.0
    last = None

    while time.time() < deadline:
        try:
            page.wait_for_load_state("domcontentloaded", timeout=2000)
            page.wait_for_function(
                """
                () => {
                  const hasLinks =
                    !!document.querySelector('#linkbtnPrintReceipt') ||
                    Array.from(document.querySelectorAll('a')).some(a =>
                      /Receipt PDF/i.test(a.textContent || '') || /Registration PDF/i.test(a.textContent || '')
                    );
                  const doneHdr = /Registration Complete/i.test((document.body && document.body.innerText) || '');
                  return hasLinks || doneHdr;
                }
                """,
                timeout=3000
            )
            # One more small guard to ensure at least one link is actionable
            try:
                page.locator("#linkbtnPrintReceipt, a:has-text('Receipt PDF')").first.wait_for(state="visible", timeout=2000)
            except Error:
                pass
            return
        except Error as e:
            txt = str(e)
            # If the page navigated mid-eval, just retry quickly
            if "Execution context was destroyed" in txt or "Target closed" in txt:
                continue
            last = e
    # If we got here, surface the last error or a generic timeout
    if last:
        raise last
    raise TimeoutError("Timed out waiting for confirmation screen.")

def _all_contexts(page):
    """Return [page, *frames] so we can search both top doc and iframes."""
    return [page, *page.frames]

def _clear_input(loc):
    try:
        loc.fill("")
    except Error:
        try:
            loc.press("Control+a"); loc.press("Backspace")
        except Error:
            pass
    
def _type_or_fill(loc, text, slow_type=False):
    if slow_type:
        loc.type(str(text), delay=35)
    else:
        loc.fill(str(text))

def _fill_textbox_anywhere(page, name_patterns, value, slow_type=False, timeout=10000):
    """
    name_patterns: str | re.Pattern | list of those
    Tries: role(textbox,name=pattern) -> label(pattern) -> placeholder(plain) -> [aria-label="plain"]
    Never puts regex into CSS.
    """
    if isinstance(name_patterns, (str, re.Pattern)):
        patterns = [name_patterns]
    else:
        patterns = list(name_patterns)

    last_err = None

    # 1) Role(name=regex or string)
    for ctx in _all_contexts(page):
        for pat in patterns:
            try:
                loc = ctx.get_by_role("textbox", name=pat).first
                if loc.count():
                    loc.wait_for(state="visible", timeout=timeout)
                    loc.click()
                    _clear_input(loc)
                    _type_or_fill(loc, value, slow_type=slow_type)
                    return True
            except Error as e:
                last_err = e

    # 2) By label (regex or string)
    for ctx in _all_contexts(page):
        for pat in patterns:
            try:
                loc = ctx.get_by_label(pat).first
                if loc.count():
                    loc.wait_for(state="visible", timeout=timeout)
                    loc.click()
                    _clear_input(loc)
                    _type_or_fill(loc, value, slow_type=slow_type)
                    return True
            except Error as e:
                last_err = e

    # 3) Placeholder (must be plain string)
    for ctx in _all_contexts(page):
        for pat in patterns:
            if isinstance(pat, re.Pattern):
                continue
            try:
                loc = ctx.get_by_placeholder(pat).first
                if loc.count():
                    loc.wait_for(state="visible", timeout=timeout)
                    loc.click()
                    _clear_input(loc)
                    _type_or_fill(loc, value, slow_type=slow_type)
                    return True
            except Error as e:
                last_err = e

    # 4) Plain CSS aria-label equals (convert regex to a plain-ish guess)
    def _to_plain(p):
        if isinstance(p, re.Pattern):
            s = p.pattern
            s = re.sub(r"\\s\*", " ", s)   # \s* -> space
            s = s.replace("\\", "").replace("^", "").replace("$", "").strip()
            return s
        return p

    for ctx in _all_contexts(page):
        for pat in patterns:
            plain = _to_plain(pat)
            if not plain:
                continue
            try:
                loc = ctx.locator(f'[aria-label="{plain}"]').first
                if loc.count():
                    loc.wait_for(state="visible", timeout=timeout)
                    loc.click()
                    _clear_input(loc)
                    _type_or_fill(loc, value, slow_type=slow_type)
                    return True
            except Error as e:
                last_err = e

    if last_err:
        print(f">>> _fill_textbox_anywhere fallback note: {type(last_err).__name__}: {last_err}")
    return False

def _select_country_any(page, label_regex=r"^Country$", country_label="United States"):
    """
    Handle either a native <select> (combobox) or a custom dropdown (button + listbox).
    """
    rx = re.compile(label_regex, re.I)

    # 1) Native select via role=combobox or by label
    for ctx in _all_contexts(page):
        try:
            # try by label (most robust)
            sel = ctx.get_by_label(rx).first
            if sel.count():
                try:
                    sel.select_option(label=country_label)
                    return True
                except Error:
                    pass
            # try role=combobox
            combo = ctx.get_by_role("combobox", name=rx).first
            if combo.count():
                try:
                    combo.select_option(label=country_label)
                    return True
                except Error:
                    pass
        except Error:
            pass

    # 2) Custom button-triggered list (e.g., Angular Material, MUI)
    for ctx in _all_contexts(page):
        try:
            btn = ctx.get_by_role("button", name=rx).first
            if btn.count():
                btn.click()
                # options can appear in the same doc or in an overlay frame
                for sub in _all_contexts(page):
                    opt = sub.get_by_role("option", name=re.compile(rf"^{re.escape(country_label)}$", re.I)).first
                    if opt.count():
                        opt.click()
                        return True
        except Error:
            pass

    # 3) Fallback: click button "Country" then use a generic list item
    try:
        page.get_by_role("button", name=rx).first.click(timeout=2000)
        page.get_by_text(country_label, exact=True).first.click(timeout=2000)
        return True
    except Error:
        return False

def _click_and_download(page, selector_or_text, timeout=30000):
    # Selector can be css or a ('role','name') tuple for fallback
    def _do():
        # wait for either id or the text fallback to be attached & visible
        try:
            page.wait_for_selector(selector_or_text, state="visible", timeout=8000)
            with page.expect_download(timeout=timeout) as dl_info:
                page.click(selector_or_text)
            return dl_info.value
        except Exception:
            # fallback to text role if provided like ('link', r'Receipt PDF')
            if isinstance(selector_or_text, tuple):
                role, name_rx = selector_or_text
                loc = page.get_by_role(role, name=re.compile(name_rx, re.I)).first
                expect(loc).to_be_visible(timeout=8000)
                with page.expect_download(timeout=timeout) as dl_info:
                    loc.click()
                return dl_info.value
            raise

    return _retry_nav(_do)

def _download_receipt_and_registration(page, entity_name: str, out_dir: str) -> tuple[pathlib.Path, pathlib.Path]:
    ts   = _ts_short()
    slug = _slug(entity_name)

    rec_dir = os.path.join(out_dir, "receipts")
    reg_dir = os.path.join(out_dir, "registrations")
    _ensure_dir(rec_dir); _ensure_dir(reg_dir)

    # --- Receipt ---
    # Try id first, then text fallback, but never do .count(); always wait+click in a retry wrapper.
    rec_dl = _click_and_download(page, "#linkbtnPrintReceipt")
    rec_path = pathlib.Path(rec_dir) / f"Receipt_{slug}_{RUN_ID}_{ts}.pdf"
    rec_dl.save_as(str(rec_path))
    print(f">>> Receipt saved: {rec_path}")

    # The click above can cause a partial postback; give the DOM a beat & re-assert we're on the same page.
    _retry_nav(lambda: page.wait_for_load_state("domcontentloaded", timeout=3000))
    _retry_nav(lambda: page.wait_for_function(
        "()=>document.querySelector('#linkbtnPrintForm') || [...document.querySelectorAll('a')].some(a=>/Registration PDF/i.test(a.textContent||''))",
        timeout=5000
    ))

    # --- Registration ---
    reg_dl = _click_and_download(page, "#linkbtnPrintForm")
    reg_path = pathlib.Path(reg_dir) / f"Registration_{slug}_{RUN_ID}_{ts}.pdf"
    reg_dl.save_as(str(reg_path))
    print(f">>> Registration saved: {reg_path}")

    return rec_path, reg_path

def _download_link_pdf(page, link_regex: str, save_to: str, timeout_ms: int = 20000) -> str | None:
    link = page.get_by_role("link", name=re.compile(link_regex, re.I)).first
    if not link.count():
        return None
    with page.expect_download(timeout=timeout_ms) as dl_info:
        link.click()
    dl = dl_info.value
    dl.save_as(save_to)
    log_ok(f"Saved: {save_to}")
    return save_to

def download_post_payment_docs(page, entity_name: str, out_dir: str = None) -> dict:
    """
    On the payment confirmation page, grab Registration PDF + Receipt PDF if present.
    Returns dict with keys 'registration_pdf' and 'receipt_pdf' (paths or '').
    """
    out_dir = out_dir or os.getenv("WY_OUT_DIR", "output")
    _ensure_dir(out_dir)

    # Wait briefly for the confirmation page/links to render
    try:
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(800)  # small settle
    except Exception:
        pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _slug(entity_name)
    reg_path = _safe_join_dir(out_dir, f"{base}_Registration_{stamp}.pdf")
    rct_path = _safe_join_dir(out_dir, f"{base}_Receipt_{stamp}.pdf")

    saved_reg = _download_link_pdf(page, r"Registration\s*PDF", reg_path) or ""
    saved_rct = _download_link_pdf(page, r"Receipt\s*PDF",     rct_path) or ""

    if not saved_reg and not saved_rct:
        log_warn("No Registration/Receipt PDF links found on confirmation page (may vary by result).")

    return {"registration_pdf": saved_reg, "receipt_pdf": saved_rct}

def build_doc_path(entity_name: str, label: str, filing_id: str | None = None, out_dir: str | None = None) -> str:
    """
    Make a pretty filename like:
      output/Bright_Skies_Holdings_LLC_CGS_20250908_124109.pdf
      output/Bright_Skies_Holdings_LLC_GSC_2025-000123456_20250908_124201.pdf
    """
    out_dir = out_dir or os.getenv("WY_OUT_DIR", "output")
    _ensure_dir(out_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base  = _slug(entity_name)
    parts = [base, label]
    if filing_id:
        parts.append(filing_id)
    fname = "_".join(parts) + f"_{stamp}.pdf"
    return os.path.join(out_dir, fname)

def smart_check_radio(page, radio_locator):
    # 1) If already checked, we’re done
    try:
        if radio_locator.is_checked():
            return
    except Exception:
        pass

    # 2) Normal path: use Playwright's check()
    try:
        radio_locator.check(timeout=3000)
        if radio_locator.is_checked():
            return
    except Exception:
        pass

    # 3) Try clicking the associated <label for="...">
    try:
        rid = radio_locator.get_attribute("id")
        if rid:
            lbl = page.locator(f'label[for="{rid}"]').first
            if lbl.count():
                lbl.click()
                # allow any partial postback to settle
                page.wait_for_timeout(300)
                if radio_locator.is_checked():
                    return
    except Exception:
        pass

    # 4) Last-resort: set it with JS and fire events
    try:
        radio_locator.evaluate("""
            el => {
              el.checked = true;
              el.dispatchEvent(new Event('input', { bubbles: true }));
              el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        """)
        # give the page a tick to react
        page.wait_for_timeout(100)
    except Exception:
        # As an extreme fallback, try a direct click (forced)
        try:
            radio_locator.click(force=True)
        except Exception:
            pass

# === Structured SSE helpers (place near other helpers) ===
def emit_event(event_type: str, **kwargs):
    print(f'[[EVT]] {json.dumps({"type": event_type, **kwargs}, ensure_ascii=False)}', flush=True)

def emit_ux(stage: str, message: str, level: str = "info", *, entity: str = "", idx: int | None = None):
    """
    level: start | info | success | error
    """
    payload = {"stage": stage, "message": message, "level": level}
    if entity: payload["entity"] = entity
    if idx is not None: payload["idx"] = idx
    print(f'[[UX]] {json.dumps(payload, ensure_ascii=False)}', flush=True)

@contextmanager
def step(title: str, stage: str | None = None, entity: str = "", idx: int | None = None):
    log_info(f"{title}…")
    if stage:
        emit_ux(stage, title, "start", entity=entity, idx=idx)
    try:
        yield
        log_ok(f"{title} — done")
        if stage:
            emit_ux(stage, title, "success", entity=entity, idx=idx)
    except Exception as e:
        e._wy_step = title  # keep
        log_err(f"{title} — failed")
        if stage:
            emit_ux(stage, humanize_error(e), "error", entity=entity, idx=idx)
        raise

def tidy_note(s: str) -> str:
    """
    Strip leading/trailing whitespace, remove common indentation,
    and collapse internal whitespace/newlines to single spaces.
    """
    if s is None:
        return ""
    s = textwrap.dedent(str(s)).strip()
    s = re.sub(r"\s+", " ", s)  # collapse runs of spaces/newlines/tabs
    return s

# =========================
# --- Main merged flow  ---
# =========================
def run(playwright: Playwright, filing_data: dict, payment_details: dict, captcha_api_key: str) -> dict:
    # --- Browser ---
    browser = playwright.chromium.launch(headless=False, args=["--disable-dev-shm-usage"])
    context = browser.new_context(
        accept_downloads=True,
        viewport={"width": 1366, "height": 900},
        user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
        locale="en-US",
        record_video_dir=os.getenv("WY_VIDEO_DIR", "videos"),
    )

    # Only set up video if you actually want it (optional but tidy)
    ctx_kwargs = dict(
        accept_downloads=True,
        viewport={"width": 1366, "height": 900},
        user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
        locale="en-US",
    )
    _video_dir = os.getenv("WY_VIDEO_DIR")  # leave empty/undefined to disable videos
    if _video_dir:
        ctx_kwargs["record_video_dir"] = _video_dir

    context = browser.new_context(**ctx_kwargs)

    # --- Tracing (env-gated) ---
    trace_dir = None
    if TRACING_ENABLED:
        trace_dir = Path(os.getenv("WY_TRACE_DIR", "traces"))
        trace_dir.mkdir(parents=True, exist_ok=True)
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    page.set_default_timeout(30000)

    # things the finally block might need
    entity = filing_data.get("entity_name", "unknown")
    receipt_file = ""
    registration_file = ""
    filing_id = ""
    cgs_path = None
    gsc_path = None

    try:

        # 1) Registration: type + agree
        with step("Select business type & agree", "Registration", entity=entity):
            page.goto("https://wyobiz.wyo.gov/Business/RegistrationInstr.aspx")
            page.get_by_role("link", name="Form or Register a New").click()
            page.locator("#MainContent_slctBusType").select_option("RegistrationLLC.aspx")
            expect(page.get_by_role("checkbox", name="Pursuant to W.S. 16-4-201 et")).to_be_visible()
            page.get_by_role("checkbox", name="Pursuant to W.S. 16-4-201 et").check()
            page.get_by_role("button", name="NEXT >>").click()

        # 2) Registration: name entry + availability check
        with step("Enter entity name", "Registration", entity=entity):
            expect(page.locator("#txtName")).to_be_visible()
            expect(page.locator("#txtNameConfirm")).to_be_visible()

            # Do not auto-append LLC; fail fast if designation missing
            if not has_valid_designation(entity):
                msg = (
                    "Entity name missing required designation. "
                    "The Entity Name must contain one of the following designations: "
                    "L.C., L.L.C., LC, Limited Company, Limited Liability Co., "
                    "Limited Liability Company, LLC, Ltd. Liability Co., "
                    "Ltd. Liability Company"
                )
                emit_ux("Validation", msg, "error", entity=entity)
                raise SkipEntity(msg)

            # Fill as-is (no modification)
            page.locator("#txtName").fill(entity)
            page.locator("#txtNameConfirm").fill(entity)
            emit_ux("Registration", "Entity name entered", "info", entity=entity)

            # First Next (prefer explicit ID; fallback to role)
            clicked = False
            try:
                page.locator("#ContinueButton").click(timeout=2000)
                clicked = True
            except Exception:
                pass
            if not clicked:
                page.get_by_role("button", name=re.compile(r"^Next\s*>>$", re.I)).click()

            # If the page still throws a designation error (edge cases), stop this entity
            msg = designation_error_text(page, timeout_ms=2500)
            if msg:
                emit_ux("Validation", f"Name designation invalid: {msg}", "error", entity=entity)
                raise SkipEntity(f"Name designation invalid. {msg}")

            # Some flows need a second Next
            try:
                page.get_by_role("button", name=re.compile(r"^Next\s*>>$", re.I)).click(timeout=2000)
            except Exception:
                pass

            # Final quick availability check (your existing helper)
            reason = _name_unavailable_reason(page, timeout_ms=2500)
            if reason:
                emit_ux("Availability", f"Name not available. {reason}", "skipped", entity=entity)
                raise SkipEntity(reason)

        # 3) Registration: Registered Agent search & select
        with step("Registered agent", "Registration", entity=entity):
            expect(page.locator("#txtOrgNameSearch")).to_be_visible(timeout=10000)
            page.locator("#txtOrgNameSearch").fill(filing_data["registered_agent_name"])
            page.get_by_role("button", name="Search").click()
            _select_registered_agent_from_dialog(page, filing_data["registered_agent_name"])
            emit_ux("Registration", "Registered agent selected", "info", entity=entity)
            page.get_by_role("checkbox", name="I have obtained a signed and").check()
            page.get_by_role("button", name="Next >>").click()

        # 4) Registration: Business address & contact
        with step("Business address & contact", "Registration", entity=entity):
            expect(page.locator('input[name="ctl00$MainContent$ucAddress$txtAddr1"]')).to_be_visible()
            page.locator('input[name="ctl00$MainContent$ucAddress$txtAddr1"]').fill(filing_data["physical_address"])
            page.locator("#txtCity").fill(filing_data["city"])
            page.locator("#txtState").fill(filing_data["state"])
            page.locator("#txtPostal").fill(filing_data["postal_code"])
            page.locator("#txtPhone").fill(filing_data["phone"])
            page.locator("#txtEmail").fill(filing_data["email"])

            page.locator('input[name="ctl00$MainContent$ucAddress$txtAddr1Mail"]').fill(filing_data["mailing_address"])
            page.locator("#txtCityMail").fill(filing_data["city"])
            page.locator("#txtStateMail").fill(filing_data["state"])
            page.locator("#txtPostalMail").fill(filing_data["postal_code"])

            page.get_by_role("button", name="Next >>").click()

        # 5) Registration: Organizer
        with step("Organizer", "Registration", entity=entity):
            expect(page.locator("#txtFirstName")).to_be_visible()
            page.locator("#txtFirstName").fill(filing_data["signature_first_name"])
            page.locator("#txtLastName").fill(filing_data["signature_last_name"])
            page.locator('input[name="ctl00$MainContent$ucParties$txtMail1"]').fill(filing_data["organizer_address"])
            page.get_by_role("button", name="Add").click()
            expect(page.locator("#OfficerList")).to_contain_text(
                f'{filing_data["signature_first_name"]} {filing_data["signature_last_name"]}'
            )
            page.get_by_role("button", name="Next >>").click()
            expect(page.locator("#OfficerList")).to_be_hidden()

        # 6) Registration: Declarations & contact confirmations
        with step("Declarations", "Registration", entity=entity):
            page.get_by_role("button", name="Next >>").click()
            expect(page.get_by_role("button", name="Continue")).to_be_visible()
            page.get_by_role("button", name="Continue").click()

            expect(page.get_by_role("checkbox", name="I am the person whose")).to_be_visible()
            page.get_by_role("checkbox", name="I am the person whose").check()
            page.get_by_role("checkbox", name="I am filing in accordance").check()
            page.get_by_role("checkbox", name="I understand that the").check()
            page.get_by_role("checkbox", name="I intend and agree that the").check()
            page.get_by_role("checkbox", name="I have conducted the").check()
            page.get_by_role("checkbox", name="I consent on behalf of the").check()
            page.get_by_role("checkbox", name="I acknowledge having read W.S").check()
            page.get_by_label("An Organization").check()

            page.locator("#txtFirstName").fill(filing_data["signature_first_name"])
            page.locator("#txtLastName").fill(filing_data["signature_last_name"])
            page.locator("#txtTitle").fill(filing_data["signature_title"])
            page.locator("#txtPhone").fill(filing_data["phone"])
            page.locator("#txtPhoneConfirm").fill(filing_data["phone"])
            page.locator("#txtEmail").fill(filing_data["email"])
            page.locator("#txtEmailConfirm").fill(filing_data["email"])
            page.get_by_role("button", name="Next >>").click()

        # 8) Payment: open hosted checkout & fill cardholder details
        with step("Handling Payment", stage="Payment", entity=entity):
            # --- open hosted checkout (same-page or new tab; handle both) ---
            emit_ux("Payment", "Opening hosted checkout", "start", entity=entity)

            # Button sometimes sits below the fold — scroll, then click
            pay_btn = page.get_by_role(
                "button",
                name=re.compile(r"Click here to (enter|make) payment", re.I)
            )
            if not pay_btn.count():
                # exact text fallback (original)
                pay_btn = page.get_by_role("button", name="Click here to enter payment")

            # Make sure we really see it
            expect(pay_btn).to_be_visible(timeout=15000)
            try:
                pay_btn.scroll_into_view_if_needed()
            except Exception:
                pass

            # It may open in the SAME page or a NEW page; race both paths
            new_pay_page = None
            try:
                with context.expect_page(timeout=2500) as pg_info:
                    pay_btn.click()
                new_pay_page = pg_info.value
            except Exception:
                # likely same-page nav
                pass

            if new_pay_page:
                page = new_pay_page  # continue on the hosted-payments page
            else:
                page.wait_for_url("**/hosted-payments/**", timeout=20000)

            # --- enter checkout form (use your original accessible names) ---
            expect(page.get_by_role("button", name="Checkout")).to_be_visible()
            page.get_by_role("button", name="Checkout").click()
            emit_ux("Payment", "Hosted checkout page loaded", "info", entity=entity)

            # Your original locators — unchanged
            expect(page.get_by_role("textbox", name="Card Number *")).to_be_visible()
            page.get_by_role("textbox", name="Card Number *").fill(payment_details["card_number"])
            page.get_by_role("textbox", name="Expiration Date (MMYY) *").fill(payment_details["exp_date"])
            page.get_by_role("textbox", name="CVV2 *").fill(payment_details["cvv"])
            page.get_by_role("textbox", name="Company").fill(payment_details.get("company", ""))

            page.get_by_role("textbox", name="First Name *").fill(payment_details["first_name"])
            page.get_by_role("textbox", name="Last Name *").fill(payment_details["last_name"])
            page.get_by_role("textbox", name="Address 1 *").fill(payment_details["address"])
            page.get_by_role("textbox", name="City *").fill(payment_details["city"])

            # Keep your exact name first; add tiny fallback if label changes slightly
            try:
                page.get_by_role("textbox", name="State/Province *").fill(payment_details["state"])
            except Exception:
                page.get_by_role("textbox", name=re.compile(r"State\s*/?\s*Province\s*\*", re.I)).fill(payment_details["state"])

            page.get_by_role("textbox", name="Postal Code (No Dash) *").fill(payment_details["zip"])

            # Country can be a custom menu; your original path first, then a generic fallback
            try:
                page.get_by_role("button", name="Country").click()
                page.get_by_role("option", name="United States", exact=True).click()
            except Exception:
                try:
                    page.get_by_role("combobox", name=re.compile(r"Country", re.I)).select_option("United States")
                except Exception:
                    pass

            # Email + phone as in your draft
            page.get_by_role("textbox", name="Email Address *").fill(payment_details["email"])
            page.get_by_role("textbox", name="Phone").fill(filing_data["phone"])

            emit_ux("Payment", "Payment form filled", "info", entity=entity)
            maybe_pause(page, "after filling card form")  # optional recording point

            # --- submit + CAPTCHA (handles retries & resubmits) ---
            handle_payment_submit_and_captcha(page, captcha_api_key, entity)
            emit_ux("Payment", "Payment submitted", "success", entity=entity)
            # Re-attach to whichever tab actually shows the confirmation
            page = _find_confirmation_page(context, timeout_ms=120000)
            _wait_for_registration_complete(page, timeout_ms=90000)
            # maybe_pause(page, "after Submit Payment click")

        # 10) Documents: receipt + registration PDFs
        out_dir = os.getenv("WY_OUT_DIR", "output")
        with step("Capture receipt & registration PDFs", "Documents", entity=entity):
            emit_ux("Documents", "Waiting for confirmation & links", "start", entity=entity)
            try:
                _wait_for_registration_complete(page, timeout_ms=90000)
                # FIX: pass page as the first arg
                rec_path, reg_path = _download_receipt_and_registration(page, filing_data["entity_name"], out_dir)
                receipt_file      = str(rec_path)
                registration_file = str(reg_path)
                emit_ux("Documents", "Receipt & Registration saved", "success", entity=entity)
            except Exception as e:
                print(f"!!! Could not capture Receipt/Registration PDFs: {e}")
                emit_ux("Documents", f"Could not capture PDFs: {humanize_error(e)}", "error", entity=entity)

        # 11) CGS: search
        entity_for_cgs = entity
        with step("Search business record", "CGS", entity=entity):
            page.goto(SEARCH_URL)
            got_results_fast = _search_entity(page, entity_for_cgs)
            if not got_results_fast:
                emit_ux("CAPTCHA", "Quick human check detected (image gate)", "start", entity=entity)
                if page.get_by_role("link", name="Search for Business Names/").count():
                    page.get_by_role("link", name="Search for Business Names/").click()
                elif page.get_by_role("link", name="Search for Business Names").count():
                    page.get_by_role("link", name="Search for Business Names").click()
                else:
                    page.get_by_text("Search for Business Names").first.click()

                page.wait_for_load_state("domcontentloaded")
                if _gate_is_present(page, timeout=10000):
                    _solve_simple_image_gate(page)
                emit_ux("CAPTCHA", "CAPTCHA verified", "success", entity=entity)
                _search_entity(page, entity_for_cgs)

            _click_entity_result(page, entity_for_cgs)

        # 12) CGS download
        with step("Download CGS", "CGS", entity=entity):
            cgs_path = _download_cgs_or_print(page, entity_for_cgs, out_dir, prefer_cgs=True)
        # 13) GSC download
        with step("Download GSC via Filing ID", "GSC", entity=entity):
            filing_id = _extract_filing_id(page)
            gsc_path = _download_good_standing_by_id(page, filing_id, entity_for_cgs, out_dir)

        emit_ux("Wrap-up", "All documents saved; finishing", "success", entity=entity)

        return {
            "filing_id": filing_id,
            "cgs_path": str(cgs_path) if cgs_path else "",
            "gsc_path": str(gsc_path) if gsc_path else "",
            "receipt_file": receipt_file,
            "registration_file": registration_file,
        }
    finally:
        # --- Tracing off + cleanup (always runs) ---
        try:
            if TRACING_ENABLED and trace_dir is not None:
                trace_file = trace_dir / f"trace_{_slug(entity)}_{_ts_short()}.zip"
                context.tracing.stop(path=str(trace_file))
                print(f">>> Trace saved to: {trace_file}")
        except Exception as e:
            print(f">>> Trace stop failed: {e}")

        try:
            context.close()
        except Exception as e:
            print(f">>> Context close failed: {e}")
        try:
            browser.close()
        except Exception as e:
            print(f">>> Browser close failed: {e}")


if __name__ == "__main__":
    require_1password_service_token()

    # ---------- Directories & env ----------
    DATA_DIR         = os.getenv("WY_DATA_DIR", "data")
    OUT_DIR          = os.getenv("WY_OUT_DIR", "output")
    VAULT            = os.getenv("WY_VAULT", "Wyoming Filing Bot")
    ITEM_DEFAULTS    = os.getenv("WY_ITEM_DEFAULTS", "Wyoming Filing Defaults")
    ITEM_CARD        = os.getenv("WY_ITEM_CARD", "Credit Card")
    ITEM_2CAPTCHA    = os.getenv("WY_ITEM_2CAPTCHA", "2Captcha API Key")
    FILINGS_BASENAME = os.getenv("WY_FILINGS_CSV", "filings.csv")

    # Resolve the CSV path (respect absolute paths too)
    filings_csv_path = (
        FILINGS_BASENAME if os.path.isabs(FILINGS_BASENAME)
        else os.path.join(DATA_DIR, FILINGS_BASENAME)
    )

    # ---------- Load defaults & input ----------
    defaults_profile = get_defaults_profile_from_1password(VAULT, ITEM_DEFAULTS)
    print(">>> Loaded defaults profile keys:", ", ".join(sorted(defaults_profile.keys())) or "(none)")

    filings = load_filings_csv(filings_csv_path)
    if not filings:
        raise SystemExit(f"No valid rows found in {filings_csv_path} (need at least one 'entity_name').")

    input_headers = _union_input_headers(filings)

    # ---------- Run ----------
    results_rows: list[dict] = []
    aborted = False

    with sync_playwright() as p:
        payment_details = get_payment_details_from_1password(VAULT, ITEM_CARD)

        required = ["card_name", "card_number", "exp_mm", "exp_yy", "cvv"]
        if not payment_details or not all(payment_details.get(k) for k in required):
            print(">>> Missing payment details. Either install 1Password CLI+token, or set WY_CARD_* in .env", flush=True)
            sys.exit(1)
        
        captcha_api_key = get_2captcha_api_key(VAULT, ITEM_2CAPTCHA)
        print(f">>> Using 2Captcha key: {captcha_api_key[:4]}…{captcha_api_key[-4:]} (len={len(captcha_api_key)})  repr={repr(captcha_api_key)}")
        validate_2captcha_key(captcha_api_key)

        total = len(filings)
        for idx, raw in enumerate(filings, start=1):
            # Stop before starting the next entity if requested
            if abort_requested():
                log_warn("Abort requested — stopping before next entity.")
                aborted = True
                break

            entity = (raw.get("entity_name") or "").strip()

            # Let the UI know we started this entity
            emit_event("start", idx=idx, total=total, entity=entity)

            if not entity:
                # Defensive: should not happen because load_filings_csv skips, but keep UI in sync
                started_at  = datetime.now().isoformat(timespec="seconds")
                finished_at = started_at
                emit_event(
                    "finish",
                    idx=idx,
                    entity=f"(row {idx})",
                    status="ERROR",
                    notes="Missing 'entity_name' in CSV row",
                    filing_id="",
                    cgs_file="",
                    gsc_file="",
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_s=0.0,
                )
                print(f"!!! Skipping row {idx}: missing 'entity_name'")
                continue

            # Merge CSV row (overrides) with defaults from 1Password
            try:
                filing_data = merge_filing_row(raw, defaults_profile)
            except Exception as e:
                started_at  = datetime.now().isoformat(timespec="seconds")
                finished_at = started_at
                note = f"MergeError: {e}"
                results_rows.append({
                    **raw,
                    "status": "ERROR",
                    "notes": note,
                    "filing_id": "",
                    "cgs_file": "",
                    "gsc_file": "",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_s": 0.0,
                })
                # tell UI
                emit_event(
                    "finish",
                    idx=idx,
                    entity=entity,
                    status="ERROR",
                    notes=note,
                    filing_id="",
                    cgs_file="",
                    gsc_file="",
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_s=0.0,
                )
                continue

            print("\n" + "=" * 80)
            print(f">>> Filing {idx}/{total} — {entity}")
            print("=" * 80)

            started_at = datetime.now().isoformat(timespec="seconds")
            t0 = time.perf_counter()

            status    = "SUCCESS"
            notes     = []
            filing_id = ""
            cgs_file  = ""
            gsc_file  = ""
            receipt_file = ""         # <-- NEW
            registration_file = ""    # <-- NEW

            try:
                info = run(p, filing_data, payment_details, captcha_api_key)
                if isinstance(info, dict):
                    filing_id = info.get("filing_id", "") or ""
                    cgs_file  = info.get("cgs_path", "") or ""
                    gsc_file  = info.get("gsc_path", "") or ""
                    receipt_file      = info.get("receipt_file", "") or ""        # <-- NEW
                    registration_file = info.get("registration_file", "") or ""   # <-- NEW
                notes.append(tidy_note("Completed registration + CGS flow"))

            except SkipEntity as e:
                status = "SKIPPED"
                notes.append(tidy_note(f"Skipped: {e}"))

            except Exception as e:
                status = "ERROR"
                notes.append(tidy_note(f"{type(e).__name__}: {e}"))

            notes_str   = " • ".join(tidy_note(n) for n in notes if n).strip()
            elapsed_s   = round(time.perf_counter() - t0, 2)
            finished_at = datetime.now().isoformat(timespec="seconds")

            row_out = {
                **raw,
                "status": status,
                "notes": notes_str,
                "filing_id": filing_id,
                "receipt_file": receipt_file,           # <-- NEW
                "registration_file": registration_file, # <-- NEW
                "cgs_file": cgs_file,
                "gsc_file": gsc_file,
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_s": elapsed_s,
            }
            results_rows.append(row_out)

            # Notify UI that this entity has finished (success/error/skipped)
            emit_event(
                "finish",
                idx=idx,
                entity=entity,
                status=status,
                notes=notes_str,
                filing_id=filing_id,
                receipt_file=receipt_file,           # <-- NEW
                registration_file=registration_file, # <-- NEW
                cgs_file=cgs_file,
                gsc_file=gsc_file,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_s=elapsed_s,
            )

            # Stop as soon as we finish the current entity if requested mid-run
            if abort_requested():
                log_warn("Abort requested — stopping after current entity.")
                aborted = True
                break

    # ---------- Write results ----------
    results_path = write_results_csv(results_rows, input_headers=input_headers, out_dir=OUT_DIR)

    # keep last 3 *previous* runs' docs + this current run
    prune_doc_buckets_by_run(OUT_DIR, RUNS_ROOT, keep_prev=int(os.getenv("WY_DOC_RUNS_KEEP", "3")), exclude=(RUN_ID,))

    emit_event("batch_done", results_path=results_path, aborted=aborted)