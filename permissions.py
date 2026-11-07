"""
permissions.py - permission-controlled, READ-ONLY laptop access for Alex.

Design rules (all enforced here in Python, never by the LLM prompt):
  * Deny by default. Nothing is reachable until you list it in
    security_config.json. Alex has no tool that can edit that file, so
    changing the allowlist is always a separate, manual step.
  * Only these operations exist: read a text file, list a folder, open an
    allowlisted website, launch an allowlisted app (no arguments).
    There is no write / delete / move / run / send / install / download code.
  * Every action needs a fresh, exact, single-use "yes" to the specific
    question Alex asked. Approvals expire and never carry over.
  * Drive C: is always blocked. Paths with "..", UNC/device paths, NTFS
    streams, symlinks and junctions are refused. Credential-like names
    (.ssh, .env, cookies, passwords, tokens, browser profiles ...) are blocked.
  * Nothing scans or indexes the disk: Alex only touches the exact path you
    approved. The audit log (data/audit.log) stores paths, never contents.
"""

import itertools
import json
import os
import re
import stat
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "security_config.json"
AUDIT_FILE = BASE_DIR / "data" / "audit.log"

APPROVAL_TTL = 60            # seconds an approval question stays valid
MAX_READ_BYTES = 256 * 1024  # never read more than this from one file
MAX_LIST_ITEMS = 50
SPOKEN_PREVIEW_CHARS = 400

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".csv", ".log", ".html", ".css", ".js", ".ts",
    ".yaml", ".yml", ".ini", ".toml", ".xml", ".java", ".c", ".cpp", ".h", ".cs",
}

BLOCKED_DRIVES = {"C:", os.environ.get("SystemDrive", "C:").upper()}

# Fail safe: this may over-block (e.g. "tokenizer.py"), never under-block.
BLOCKED_NAME = re.compile(
    r"^(appdata|user data|mozilla|\.ssh|\.aws|\.gnupg|\.azure|\.kube|\.docker|\.codex|\.agents|\.git|\.venv|venv|virtualenv|__pycache__|\.env.*"
    r"|.*\.(pem|key|pfx|p12|kdbx|ppk|crt|cer)|id_(rsa|ed25519|ecdsa|dsa).*"
    r"|ntuser\.dat.*|cookies.*|login data.*|web data|local state"
    r"|.*password.*|.*passwd.*|.*credential.*|.*secret.*|.*token.*|.*wallet.*"
    r"|\$recycle\.bin|system volume information)$",
    re.I,
)

_DEVICE = re.compile(r"(^|[\\/])(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.[^\\/]*)?([\\/]|$)", re.I)
_KNOWN_APPS = {"chrome", "vscode", "whatsapp", "notepad", "calculator"}
_SHELLS = {"terminal", "command prompt", "cmd", "powershell", "windows terminal", "shell", "console"}

# NOTE: the old dict had "chrome" twice, so the second entry silently replaced
# the first. Merged into one entry here.
_APP_ALIASES = {
    "vscode": ("vs code", "vscode", "visual studio code", "vs core", "vs coat", "vs cord",
               "vs quote", "vs cold", "bs code", "vis code", "v s code", "vs cove"),
    "chrome": ("chrome", "google chrome", "crome", "my browser", "browser", "web browser"),
    "whatsapp": ("whatsapp", "whats app", "what's app"),
    "notepad": ("notepad", "notepad app"),
    "calculator": ("calculator", "calc", "windows calculator"),
}

_APP_LABELS = {
    "chrome": "Google Chrome",
    "vscode": "Visual Studio Code",
    "notepad": "Notepad",
    "calculator": "Calculator",
}

# Spoken shortcut -> website. Say "open g" for ChatGPT, "open c" for Claude.
_SITE_SHORTCUTS = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com/mail/u/0/#inbox",
    "g": "https://chatgpt.com",
    "gee": "https://chatgpt.com",
    "gpt": "https://chatgpt.com",
    "chatgpt": "https://chatgpt.com",
    "chat gpt": "https://chatgpt.com",
    "c": "https://claude.ai",
    "see": "https://claude.ai",
    "claude": "https://claude.ai",
}


def _canon_app(name):
    """Normalise spoken app names, including common Whisper mishearings."""
    n = re.sub(r"\s+", " ", str(name).strip().lower())
    for canon, variants in _APP_ALIASES.items():
        if n in variants:
            return canon
    return n


_lock = threading.RLock()
_pending = {}   # channel -> (action dict, expires_at)
_ROOTS = {}     # alias -> resolved Path
_APPS = {}      # alias -> resolved exe Path
_SITES = []     # allowed host names


class Denied(Exception):
    """An action was refused by policy. str(exc) is safe to speak."""


# ------------------------------------------------------------------
# Audit log (paths and decisions only, never file contents)
# ------------------------------------------------------------------

def _audit(event, kind, target, note=""):
    rec = {"time": datetime.now().isoformat(timespec="seconds"),
           "event": event, "kind": kind, "target": target, "note": note}
    try:
        with _lock:
            AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ------------------------------------------------------------------
# Path safety
# ------------------------------------------------------------------

def _is_link(path):
    """True for symlinks, junctions and any other reparse point."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def _check_text(s):
    if not s or "\x00" in s or len(s) > 400:
        raise Denied("That path isn't valid.")
    if s.startswith(("\\\\", "//")):
        raise Denied("Network and device paths are blocked.")
    if ".." in re.split(r"[\\/]", s):
        raise Denied("Paths containing '..' are blocked.")
    head = s[:2]
    if ":" in s[2:] or (":" in head and not re.match(r"[A-Za-z]:", head)):
        raise Denied("Alternate data streams are blocked.")
    if _DEVICE.search(s):
        raise Denied("Device names are blocked.")


def _clean_resolve(raw, allow_blocked_drive=False):
    """Resolve a path strictly, refusing links, C:, and anything odd."""
    s = os.fspath(raw)
    _check_text(s)
    p = Path(os.path.abspath(s))
    drive = os.path.splitdrive(str(p))[0].upper()
    if not allow_blocked_drive and drive in BLOCKED_DRIVES:
        raise Denied("Local Disk C: is blocked.")
    for q in (p, *p.parents):
        if q != Path(q.anchor) and _is_link(q):
            raise Denied("Links and junctions are not followed.")
    try:
        r = p.resolve(strict=True)
    except OSError:
        raise Denied("I couldn't find that.")
    if os.path.normcase(str(r)) != os.path.normcase(str(p)):
        raise Denied("That path changes when resolved, so I won't use it.")
    if not allow_blocked_drive and r.drive.upper() in BLOCKED_DRIVES:
        raise Denied("Local Disk C: is blocked.")
    return r


def _authorize(raw):
    """Return (resolved, alias, relative) only if inside an approved folder."""
    r = _clean_resolve(raw)
    for alias, root in _ROOTS.items():
        try:
            rel = r.relative_to(root)
        except ValueError:
            continue
        for part in rel.parts:
            if BLOCKED_NAME.match(part):
                raise Denied("That looks like a private or credential item, so it is blocked.")
        return r, alias, rel
    raise Denied("That isn't in an approved folder. Approved folders can only be "
                 "changed by editing security_config.json.")


# ------------------------------------------------------------------
# Config (read once at start; Alex cannot modify it)
# ------------------------------------------------------------------

def _load():
    global _ROOTS, _APPS, _SITES
    _ROOTS, _APPS, _SITES = {}, {}, []
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[Permissions: {CONFIG_FILE.name} not found - all laptop access is denied]")
        return
    except (OSError, ValueError) as exc:
        print(f"[Permissions: could not read config ({exc}) - all laptop access is denied]")
        return

    for alias, p in (data.get("allowed_folders") or {}).items():
        try:
            r = _clean_resolve(p)
            if not r.is_dir() or r == Path(r.anchor):
                raise Denied("must be a real folder, not a whole drive")
            if any(BLOCKED_NAME.match(part) for part in r.parts):
                raise Denied("private and credential folders cannot be allowlisted")
            _ROOTS[str(alias).strip().lower()] = r
        except Denied as exc:
            print(f"[Permissions: skipped folder '{alias}': {exc}]")

    # The only exception to the C: rule: exact .exe paths YOU list, launched with no arguments.
    for alias, p in (data.get("allowed_apps") or {}).items():
        try:
            r = _clean_resolve(p, allow_blocked_drive=True)
            if not r.is_file() or r.suffix.lower() != ".exe":
                raise Denied("must be an .exe file")
            _APPS[_canon_app(alias)] = r
        except Denied as exc:
            print(f"[Permissions: skipped app '{alias}': {exc}]")

    _SITES = [str(h).strip().lower() for h in (data.get("allowed_websites") or []) if str(h).strip()]
    print(f"[Permissions: {len(_ROOTS)} folder(s), {len(_APPS)} app(s), {len(_SITES)} site(s) approved; read-only]")


# ------------------------------------------------------------------
# The only operations that exist (all read-only)
# ------------------------------------------------------------------

def _read_file(raw):
    r, alias, rel = _authorize(raw)          # re-checked at execution time
    if not r.is_file():
        raise Denied("That isn't a file.")
    if r.suffix.lower() not in TEXT_EXTENSIONS:
        raise Denied("I can only read plain text and code files.")
    with open(r, "rb") as f:
        if not os.path.samestat(os.fstat(f.fileno()), os.stat(r)):
            raise Denied("The file changed while I was opening it.")
        data = f.read(MAX_READ_BYTES + 1)
    truncated = len(data) > MAX_READ_BYTES
    text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
    if "\x00" in text:
        raise Denied("That looks like a binary file.")
    print(f"\n----- {rel.as_posix()} ({alias}) -----\n{text}\n----- end -----")
    preview = re.sub(r"\s+", " ", text).strip()[:SPOKEN_PREVIEW_CHARS]
    note = " It was longer than my limit, so I read only the start." if truncated else ""
    return f"Done. {r.name} has {len(text)} characters.{note} It begins: {preview}"


def _list_folder(raw):
    r, alias, rel = _authorize(raw)
    if not r.is_dir():
        raise Denied("That isn't a folder.")
    names = []
    with os.scandir(r) as it:
        for e in itertools.islice(it, 2000):
            if BLOCKED_NAME.match(e.name) or _is_link(e.path):
                continue
            names.append(e.name + ("/" if e.is_dir(follow_symlinks=False) else ""))
    names.sort(key=str.lower)
    names = names[:MAX_LIST_ITEMS]
    print(f"\n----- {rel.as_posix() or alias} -----\n" + "\n".join(names) + "\n----- end -----")
    if not names:
        return "That folder is empty."
    spoken = ", ".join(names[:8]).replace("/", "")
    more = f", and {len(names) - 8} more are on your screen" if len(names) > 8 else ""
    return f"It has {len(names)} items: {spoken}{more}."


def _open_site(url):
    # Opening a URL is delegated to the OS default browser. A True return means
    # the browser accepted the request; do not claim completion otherwise.
    if not webbrowser.open(url, new=2):
        return "The website request was approved, but the browser did not confirm it opened."
    return "Approved. The browser accepted the request to open the website."


def _open_app(alias):
    exe = _APPS.get(alias)
    if not exe:
        raise Denied("That app isn't approved.")
    if _clean_resolve(exe, allow_blocked_drive=True) != exe:
        raise Denied("The app path changed, so I won't launch it.")
    starter = getattr(os, "startfile", None)
    if starter is None:
        raise Denied("Launching apps is only supported on Windows.")
    starter(str(exe))                      # no arguments, no shell
    label = _APP_LABELS.get(alias, alias)
    return f"Completed. Windows accepted the launch request for {label}."


# ------------------------------------------------------------------
# Turning words into a specific, approvable action
# ------------------------------------------------------------------

def _spec_to_path(spec):
    spec = spec.strip().strip("\"'").rstrip(".")
    low = spec.lower()
    m = re.match(r"^(?P<name>.+?)\s+(?:in|from|inside)\s+(?:the\s+)?(?P<alias>[\w -]+?)(?:\s+folder)?$", spec, re.I)
    if m and m["alias"].strip().lower() in _ROOTS:
        return os.path.join(str(_ROOTS[m["alias"].strip().lower()]), m["name"].strip())
    if re.match(r"^[A-Za-z]:[\\/]", spec):
        return spec
    bare = re.sub(r"\s+folder$", "", low)
    if bare in _ROOTS:
        return str(_ROOTS[bare])
    if len(_ROOTS) == 1:
        return os.path.join(str(next(iter(_ROOTS.values()))), spec)
    raise Denied("Tell me which approved folder, like: read file notes.txt in projects.")


def _display(alias, rel):
    return f"{rel.as_posix()} in {alias}" if rel.parts else f"the {alias} folder"


def _site_url(spec):
    spec = spec.strip()
    if len(spec) > 2048 or re.search(r"\s|\\", spec) or any(ord(c) < 32 or ord(c) == 127 for c in spec):
        raise Denied("That web address isn't valid.")
    if "://" not in spec:
        spec = "https://" + spec
    try:
        u = urlparse(spec)
        port = u.port
    except ValueError:
        raise Denied("That web address isn't valid.")
    if u.scheme not in {"http", "https"} or u.username or u.password or port:
        raise Denied("I only open HTTP or HTTPS website addresses without credentials or custom ports.")
    host = (u.hostname or "").lower()
    try:
        host_ascii = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise Denied("That website domain isn't valid.")
    if (not host_ascii or len(host_ascii) > 253 or
            not re.fullmatch(r"(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)(?:\.(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?))+$", host_ascii)):
        raise Denied("That website domain isn't valid.")
    # Canonicalize the domain while retaining the exact requested page, query,
    # and fragment for display and one-time approval.
    url = f"{u.scheme.lower()}://{host_ascii}{u.path}"
    if u.query:
        url += f"?{u.query}"
    if u.fragment:
        url += f"#{u.fragment}"
    return url, host_ascii


_P = r"^(?:(?:please|alex|can you|could you|will you)\s+)*"
_READ = re.compile(_P + r"(?:read|open|show)\s+(?:me\s+)?(?:the\s+)?file\s+(?P<t>.+)$", re.I)
_LIST = re.compile(_P + r"(?:list|open|show|what(?:'s| is) in)\s+(?:me\s+)?(?:the\s+)?(?:folder|directory)\s+(?P<t>.+)$", re.I)
_SITE = re.compile(_P + r"(?:(?:open|go to|visit)\s+(?:mail|gmail)\s+then\s+)?(?:open|go to|visit)\s+(?:(?:the\s+)?(?:website|site|url|page|link|this)\s+|(?=https?://))(?P<t>.+)$", re.I)
_DOMAIN_SITE = re.compile(_P + r"(?:open|go to|visit)\s+(?:the\s+)?(?P<t>(?:[a-z0-9-]+\.)+[a-z]{2,})(?:\s+(?:in|using)\s+(?:chrome|my browser|the browser))?$", re.I)

# Built from _SITE_SHORTCUTS so "open g" / "open c" / "open claude" etc. all match.
_SHORTCUT_NAMES = "|".join(re.escape(k) for k in sorted(_SITE_SHORTCUTS, key=len, reverse=True))
_NAMED_SITE = re.compile(
    _P + r"(?:open|go to|visit)\s+(?:the\s+)?(?P<t>" + _SHORTCUT_NAMES + r")"
    r"(?:\s+(?:in|using)\s+(?:chrome|my browser|the browser))?$",
    re.I,
)
_GMAIL = re.compile(
    r"(?:please\s+|alex\s+)?(?:open|launch|go to)\s+(?:my\s+)?(?:gmail|mail|gmail inbox|mail inbox)",
    re.I,
)
_SITE_CLARIFY = re.compile(_P + r"(?:open|go to|visit)\s+(?:a|the)\s+(?:website|site)\s+(?:in|using)\s+(?:chrome|my browser|the browser)$", re.I)
_APP = re.compile(_P + r"(?:open|launch|start)\s+(?:the\s+)?(?P<t>[a-z0-9 .+-]{2,30}?)(?:\s+app)?$", re.I)
_COMM = re.compile(_P + r"(?:open|read|check|show|go through)\s+(?:me\s+)?(?:my\s+)?"
                   r"(?:e-?mails?|inbox|gmail|outlook|whatsapp|chats?|messages|telegram|slack)\b", re.I)
_DENY = re.compile(
    _P + r"(?:(?:delete|remove|rename|move|copy|overwrite|edit|modify|create|write|save)\s+"
    r"(?:the\s+|this\s+|that\s+|a\s+|my\s+)?(?:file|folder|directory|document)s?\b"
    r"|send\s+(?:an?\s+|the\s+)?(?:e-?mail|mail|whatsapp|message|text)\b"
    r"|(?:run|execute)\s+(?:a\s+|the\s+|this\s+)?(?:command|code|script|program|terminal)\b"
    r"|(?:install|uninstall)\s+\w+|(?:upload|download|share)\s+(?:the\s+|this\s+|a\s+)?(?:file|folder)\b"
    r"|change\s+(?:the\s+)?(?:system\s+)?settings\b|(?:make|do)\s+a\s+payment\b|pay\s+for\b|purchase\b)",
    re.I,
)
_YES = re.compile(r"^yes$", re.I)

_NO_WRITE = ("I can't do that. I'm read-only: I never edit, delete, move, send, run, install, "
             "upload, download or pay for anything.")
_NO_COMM = ("I can't read email or chat content. There is no way to make that read-only and "
            "safe yet, so I refuse it.")


def _ask(channel, kind, target, question, run):
    with _lock:
        _pending[channel] = ({"kind": kind, "target": target, "run": run}, time.time() + APPROVAL_TTL)
    _audit("requested", kind, target)
    return f"{question} Say yes to approve or no to cancel."


def handle(text, channel="voice"):
    """If `text` is a laptop-access request, return Alex's reply (usually an approval
    question). Returns None when it isn't one, so normal chat continues."""
    t = re.sub(r"\s+", " ", text or "").strip().rstrip(" .!?,")
    if not t:
        return None

    if _DENY.match(t):
        _audit("refused", "write-like", "-")
        return _NO_WRITE

    try:
        m = _READ.match(t)
        if m:
            raw = _spec_to_path(m["t"])
            r, alias, rel = _authorize(raw)          # refuse before even asking
            return _ask(channel, "read_file", str(r),
                        f"May I read the file {_display(alias, rel)}?", lambda: _read_file(raw))

        m = _LIST.match(t)
        if m:
            raw = _spec_to_path(m["t"])
            r, alias, rel = _authorize(raw)
            return _ask(channel, "list_folder", str(r),
                        f"May I list the contents of {_display(alias, rel)}?", lambda: _list_folder(raw))

        if _SITE_CLARIFY.match(t):
            return ("Which website domain should I open? I won't navigate until you "
                    "name it and approve the exact domain.")

        if _GMAIL.fullmatch(t):
            url = _SITE_SHORTCUTS["gmail"]
            return _ask(channel, "open_site", url,
                        f"May I open your Gmail inbox at {url}?",
                        lambda: _open_site(url))

        m = _SITE.match(t) or _DOMAIN_SITE.match(t) or _NAMED_SITE.match(t)
        if m:
            requested = m["t"].strip().lower()
            requested = _SITE_SHORTCUTS.get(requested, requested)   # g -> ChatGPT, c -> Claude
            url, _host = _site_url(requested)
            return _ask(channel, "open_site", url,
                        f"May I open {url} in your default browser?",
                        lambda: _open_site(url))

        m = _APP.match(t)
        if m:
            name = _canon_app(m["t"])
            label = _APP_LABELS.get(name, name)
            if name in _APPS:
                caveat = " This may expose your already logged-in browser profile." if name == "chrome" else ""
                return _ask(channel, "open_app", str(_APPS[name]),
                            f"May I open {label}?{caveat}", lambda: _open_app(name))
            if name in _SHELLS:
                _audit("refused", "shell", name)
                return ("I can't open a terminal or command prompt. I'm read-only and never "
                        "run commands.")
            if name in _KNOWN_APPS:
                caveat = " Launching Chrome can expose its already logged-in browser profile." if name == "chrome" else ""
                return (f"{label} is blocked because it isn't approved yet. Add its verified exact executable path "
                        f"under allowed_apps in security_config.json, then restart me.{caveat}")
            # Any other "open <thing>": answer here so the language model can't pretend it worked.
            if len(name.split()) <= 3 and not re.match(r"(?:a|an|some|any)\b", name):
                _audit("refused", "unknown-app", name)
                return f"I can't open {name}. It isn't in my approved apps list."

        if _COMM.match(t):
            _audit("refused", "mail-chat", "-")
            return _NO_COMM
    except Denied as exc:
        _audit("denied", "request", t[:120], str(exc))
        return f"I can't do that. {exc}"
    return None


def handle_pending(channel, text):
    """If an approval question is waiting on this channel, consume the next message as
    the answer. Only an explicit 'yes' runs the action; anything else cancels it.
    The approval is single-use and expires after APPROVAL_TTL seconds."""
    with _lock:
        item = _pending.pop(channel, None)
    if not item:
        return None
    action, expires = item
    if time.time() >= expires:
        _audit("expired", action["kind"], action["target"])
        return "That approval request expired. Please ask again."
    answer = re.sub(r"[^\w\s']", "", text or "").strip()
    if not _YES.fullmatch(answer):
        _audit("declined", action["kind"], action["target"])
        return "Cancelled. I won't perform that action."
    try:
        out = action["run"]()
    except Denied as exc:
        _audit("denied", action["kind"], action["target"], str(exc))
        return f"I can't do that. {exc}"
    except Exception as exc:      # fail closed
        _audit("error", action["kind"], action["target"], type(exc).__name__)
        return "Something went wrong, so I stopped."
    _audit("approved", action["kind"], action["target"])
    return out


_load()


