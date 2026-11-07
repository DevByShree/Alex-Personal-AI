"""
Task reminders for Alex.

- Tasks are saved to  <this folder>/data/tasks.json  so they survive restarts.
- handle(text) understands spoken / typed commands and returns Alex's reply
  (or None if the text is not about tasks, so the normal LLM can answer).
- start_scheduler(callback) watches the clock and calls callback(message)
  when a task is due.  Tasks that came due while Alex was off are announced
  as soon as Alex starts.
"""

import json
import os
import re
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
TASKS_FILE = DATA_DIR / "tasks.json"

KEEP_DONE_DAYS = 14      # finished tasks are forgotten after this many days
DEFAULT_HOUR = 9         # "remind me tomorrow to ..." -> 9 AM
SNOOZE_MINUTES = 10

_lock = threading.RLock()
_tasks = None
_last_id = None          # last task added or announced ("done", "snooze", "delete that")
_stop = threading.Event()

# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------


def _load():
    global _tasks
    with _lock:
        if _tasks is not None:
            return
        _tasks = []
        try:
            data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _tasks = [t for t in data if isinstance(t, dict) and t.get("text")]
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            print(f"[Reminders: could not read {TASKS_FILE}: {exc}]")
            try:  # keep the unreadable file instead of overwriting it
                TASKS_FILE.replace(TASKS_FILE.with_suffix(".bad"))
            except OSError:
                pass
        _purge()


def _purge():
    cutoff = (datetime.now() - timedelta(days=KEEP_DONE_DAYS)).isoformat(timespec="seconds")
    before = len(_tasks)
    _tasks[:] = [t for t in _tasks if not (t.get("done") and (t.get("done_at") or "") < cutoff)]
    if len(_tasks) != before:
        _save()


def _save():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = TASKS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_tasks, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, TASKS_FILE)
    except OSError as exc:
        print(f"[Reminders: could not save tasks: {exc}]")


def _pending():
    with _lock:
        _load()
        items = [t for t in _tasks if not t.get("done")]
    return sorted(items, key=lambda t: (t.get("due") is None, t.get("due") or ""))


def _get(task_id):
    for t in _tasks or []:
        if t["id"] == task_id:
            return t
    return None


# ------------------------------------------------------------------
# Numbers and time words
# ------------------------------------------------------------------

_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}


def _to_int(s):
    s = (s or "").strip().lower().replace("-", " ")
    if s.isdigit():
        return int(s)
    if s in ("a", "an"):
        return 1
    total, ok = 0, False
    for w in s.split():
        if w not in _NUMS:
            return None
        total += _NUMS[w]
        ok = True
    return total if ok else None


_UNITS_W = "one|two|three|four|five|six|seven|eight|nine"
_TENS_W = "twenty|thirty|forty|fifty"
_NUM = (
    rf"(?:\d+|(?:{_TENS_W})[\s-](?:{_UNITS_W})|(?:{_TENS_W})|ten|eleven|twelve|"
    rf"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|{_UNITS_W}|an?)"
)
_REL = re.compile(
    rf"\b(?:in|after)\s+(?:(?P<half>half an? hour)|(?P<n>{_NUM})\s*"
    r"(?P<u>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?))\b",
    re.I,
)
_CLOCK = re.compile(
    r"\b(?:(?P<at>at|by)\s+)?(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>am|pm)?\b", re.I
)
_NOON = re.compile(r"\b(?:at\s+)?(noon|midnight)\b", re.I)
_DAY = re.compile(r"\b(today|tonight|tomorrow|day after tomorrow)\b", re.I)
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY = re.compile(r"\b(?:(?:on|next|this)\s+)?(" + "|".join(_WEEKDAYS) + r")\b", re.I)
_UNIT_NAME = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


_HW = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_MW = (rf"(?:(?:twenty|thirty|forty|fifty)(?:[\s-](?:{_UNITS_W}))?|ten|eleven|twelve|thirteen|fourteen|"
       rf"fifteen|sixteen|seventeen|eighteen|nineteen|(?:oh|o)[\s-](?:{_UNITS_W}))")


def _normalize(text):
    """Turn how Whisper writes spoken times into plain clock times.

    "3 o'clock, 15 minutes" -> 3:15   "at three fifteen" -> at 3:15
    "half past five" -> 5:30          "at 3.15" / "at 3 15" -> at 3:15
    """
    t = re.sub(r"\bo['\u2019]?\s?clock\b", "o'clock", text, flags=re.I)

    def num(x):
        return _to_int(re.sub(r"^(?:oh|o)[\s-]", "", x.strip().lower()))

    def sub(pattern, fn):
        nonlocal t

        def safe(m):
            try:
                return fn(m)
            except (TypeError, ValueError):
                return m.group(0)

        t = re.sub(pattern, safe, t, flags=re.I)

    sub(rf"\b(at|by)\s+({_HW})\s+({_MW})\b(?:\s*(am|pm))?",
        lambda m: f"{m[1]} {num(m[2])}:{num(m[3]):02d}{m[4] or ''}")
    sub(rf"\b({_HW})\s*(am|pm|o'clock)", lambda m: f"{num(m[1])} {m[2]}")
    sub(rf"\b(at|by)\s+({_HW})\b", lambda m: f"{m[1]} {num(m[2])}")
    sub(rf"\bquarter past ({_HW}|\d{{1,2}})\b", lambda m: f"{num(m[1])}:15")
    sub(rf"\bhalf past ({_HW}|\d{{1,2}})\b", lambda m: f"{num(m[1])}:30")
    sub(rf"\bquarter to ({_HW}|\d{{1,2}})\b", lambda m: f"{(num(m[1]) - 2) % 12 + 1}:45")
    sub(r"\b(\d{1,2})\s*o'clock,?\s*(\d{1,2})\s*(?:minutes?|mins?)\b",
        lambda m: f"{int(m[1])}:{int(m[2]):02d}" if int(m[2]) < 60 else m[0])
    sub(r"\b(\d{1,2})\s*o'clock", lambda m: f"{int(m[1])}:00")
    sub(r"\b(at|by)\s+(\d{1,2})[.\s](\d{2})\b(?!\s*(?:minutes?|mins?|hours?))",
        lambda m: f"{m[1]} {int(m[2])}:{int(m[3]):02d}" if int(m[3]) < 60 else m[0])
    sub(r"\b(\d{1,2})[.\s](\d{2})\s*(am|pm)\b",
        lambda m: f"{int(m[1])}:{int(m[2]):02d}{m[3]}" if int(m[2]) < 60 else m[0])
    return t


_JUNK_TAIL = re.compile(
    r"[,\s]*\b(?:and\s+)?(?:the\s+)?(?:\d+(?:st|nd|rd|th)?|first|second|third|next|another|other)\s+"
    r"(?:task|reminder|one)s?\s*(?:is|are|will be)?\s*$", re.I)

_TIME_HEAD = re.compile(
    rf"^(?:(?:in|after)\s+(?:half an? hour|{_NUM}\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?))"
    r"|(?:at|by)\s+\d{1,2}|\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|tomorrow\b|tonight\b|today\b"
    r"|(?:(?:on|next|this)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)", re.I)


def _strip(text):
    text = _JUNK_TAIL.sub("", text)
    t = re.sub(r"\s+", " ", text).strip(" \t,.;:!?-")
    for _ in range(4):
        t2 = re.sub(r"^(?:to|that|about|and|at|on|in|for|by|please)\b\s*", "", t, flags=re.I)
        t2 = re.sub(r"\s*\b(?:to|and|at|on|by)$", "", t2, flags=re.I)
        t2 = t2.strip(" \t,.;:!?-")
        if t2 == t:
            break
        t = t2
    return t


def parse_when(text, now=None):
    """Find a time in `text`. Returns (datetime or None, text without the time words)."""
    now = now or datetime.now()
    text = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", text, flags=re.I)  # 5 p.m. -> 5 pm
    text = _normalize(text)

    m = _REL.search(text)
    if m:
        if m.group("half"):
            delta = timedelta(minutes=30)
        else:
            n = _to_int(m.group("n")) or 1
            delta = timedelta(**{_UNIT_NAME[m.group("u").lower()[0]]: n})
        return now + delta, _strip(text[:m.start()] + " " + text[m.end():])

    spans = []
    day_off, tonight, weekday_mode = None, False, False

    dm = _DAY.search(text)
    wm = _WEEKDAY.search(text)
    if dm:
        spans.append(dm.span())
        w = dm.group(1).lower()
        day_off = {"today": 0, "tonight": 0, "tomorrow": 1, "day after tomorrow": 2}[w]
        tonight = w == "tonight"
    elif wm:
        spans.append(wm.span())
        day_off = (_WEEKDAYS.index(wm.group(1).lower()) - now.weekday()) % 7
        weekday_mode = True

    hh = mm = None
    guessed = False
    nm = _NOON.search(text)
    if nm:
        spans.append(nm.span())
        hh, mm = (12, 0) if nm.group(1).lower() == "noon" else (0, 0)
    else:
        for cm in _CLOCK.finditer(text):
            at, h, mi, ap = cm.group("at"), int(cm.group("h")), cm.group("m"), (cm.group("ap") or "").lower()
            if not (at or ap or mi):
                continue
            mi = int(mi) if mi else 0
            if mi > 59:
                continue
            if ap:
                if not 1 <= h <= 12:
                    continue
                h = h % 12 + (12 if ap == "pm" else 0)
            elif h > 23:
                continue
            elif 1 <= h <= 12:
                guessed = True            # "at 5" -> assume 5 PM, "at 9" -> 9 AM
                if 1 <= h <= 6:
                    h += 12
            hh, mm = h, mi
            spans.append(cm.span())
            break

    if hh is None and day_off is None:
        return None, _strip(text)

    if hh is None:
        hh, mm = (20, 0) if tonight else (DEFAULT_HOUR, 0)

    base = now.replace(second=0, microsecond=0)
    due = (base + timedelta(days=day_off or 0)).replace(hour=hh, minute=mm)

    if due <= now and guessed and day_off in (None, 0) and hh < 12:
        alt = due + timedelta(hours=12)
        if alt > now:
            due = alt
    if due <= now:
        if day_off is None:
            due += timedelta(days=1)
        elif weekday_mode:
            due += timedelta(days=7)
        else:
            due = now + timedelta(minutes=1)

    for a, b in sorted(spans, reverse=True):
        text = text[:a] + " " + text[b:]
    return due, _strip(text)


# ------------------------------------------------------------------
# Wording
# ------------------------------------------------------------------


def _clock(dt):
    c = dt.strftime("%I:%M %p").lstrip("0")
    return c.replace(":00", "") if dt.minute == 0 else c


def _you(text):
    """Task text is stored as the user said it; Alex speaks it back in second person."""
    for pat, new, flags in ((r"\bmyself\b", "yourself", re.I), (r"\bmine\b", "yours", re.I),
                            (r"\bmy\b", "your", re.I), (r"\bI'm\b", "you're", 0),
                            (r"\bI am\b", "you are", 0), (r"\bI\b", "you", 0), (r"\bme\b", "you", re.I)):
        text = re.sub(pat, new, text, flags=flags)
    return text


def _when(dt, now=None):
    now = now or datetime.now()
    clock = _clock(dt)
    d = (dt.date() - now.date()).days
    if d == 0:
        day = "today"
    elif d == 1:
        day = "tomorrow"
    elif d == -1:
        day = "yesterday"
    elif 1 < d < 7:
        day = dt.strftime("%A")
    else:
        day = dt.strftime("%d %B").lstrip("0")
    return f"{day} at {clock}"


def _friendly(dt, now=None):
    now = now or datetime.now()
    secs = (dt - now).total_seconds()
    if 0 <= secs < 3600:
        mins = max(1, int(secs // 60 + (1 if secs % 60 else 0)))
        return f"in {mins} minute{'s' if mins != 1 else ''}, at {_clock(dt)}"
    return _when(dt, now)


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

_STOP = {"the", "a", "an", "my", "to", "of", "task", "tasks", "reminder", "reminders",
         "about", "that", "it", "this", "please", "number"}
_AGAIN = {"again", "it", "that", "this", "later", "again later"}


def _words(s):
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in _STOP}


def _resolve(ref):
    pend = _pending()
    ref = (ref or "").strip(" .,!?").lower()
    if ref in ("", "it", "that", "this", "last", "that one", "this one", "the last one"):
        return _get(_last_id) if _last_id and not (_get(_last_id) or {}).get("done") else None
    n = _to_int(re.sub(r"^(?:number|no\.?|#)\s*", "", ref))
    if n is not None:
        return pend[n - 1] if 1 <= n <= len(pend) else None
    rw = _words(ref)
    if not rw:
        return None
    best, score = None, 0.0
    for t in pend:
        s = len(rw & _words(t["text"])) / len(rw)
        if s > score:
            best, score = t, s
    return best if score >= 0.5 else None


def _lower_first(s):
    return s[:1].lower() + s[1:] if s and not re.match(r"I\b", s) else s


def _store(text, due, now):
    """Save a task. If the same task is already saved (e.g. without a time), update it."""
    global _last_id
    text = text[:1].upper() + text[1:]
    iso = due.isoformat(timespec="seconds") if due else None
    for t in _pending():
        if t["text"].lower() == text.lower() and (t.get("due") is None or t.get("due") == iso):
            if iso:
                t["due"], t["notified"] = iso, False
            _last_id = t["id"]
            _save()
            return t
    task = {
        "id": uuid.uuid4().hex[:8],
        "text": text,
        "due": iso,
        "created": now.isoformat(timespec="seconds"),
        "done": False,
        "done_at": None,
        "notified": False,
    }
    _tasks.append(task)
    _last_id = task["id"]
    _save()
    return task


def _split_reminders(rest):
    """"after 1 hour do X and after 2 hours call Y" -> ["after 1 hour do X", "after 2 hours call Y"]"""
    rest = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", rest, flags=re.I)
    rest = _normalize(rest)
    out = []
    for chunk in re.split(r"\bremind me\b[,:]?", rest, flags=re.I):
        if not chunk.strip(" ,.;"):
            continue
        parts = [p.strip() for p in re.split(
            r"(?:,\s*)?\b(?:and then|and also|and|then|also|plus)\b[,\s]*|,\s*", chunk, flags=re.I)
            if p and p.strip()]
        cur = ""
        for p in parts:
            if not cur:
                cur = p
            elif (re.match(r"to\s", p, re.I) or _TIME_HEAD.match(p)
                  or (parse_when(p)[0] and parse_when(cur)[0])):
                out.append(cur)          # p starts a new reminder
                cur = p
            else:
                cur += " and " + p        # same reminder ("buy milk and eggs at 5")
        if cur:
            out.append(cur)
    return out


def _add_many(rest):
    items = _split_reminders(rest)
    if len(items) <= 1:
        return _add(items[0] if items else rest, need_time=True)

    now = datetime.now()
    lines, missing = [], None
    for item in items:
        due, text = parse_when(item, now)
        if not text:
            continue
        _store(text, due, now)
        spoken = _lower_first(_you(text[:1].upper() + text[1:]))
        if due:
            lines.append(f"{len(lines) + 1}: {spoken} {_friendly(due, now)}.")
        else:
            lines.append(f"{len(lines) + 1}: {spoken}, with no time yet.")
            missing = missing or text
    if not lines:
        return "What should I remind you about?"
    out = f"Okay, I saved {len(lines)} reminders. " + " ".join(lines)
    if missing:
        out += f" Tell me a time for the last one, like: remind me to {_lower_first(missing)} at 5 pm."
    return out


def _add(rest, need_time=False):
    now = datetime.now()
    about = bool(re.match(r"\s*about\b", rest, re.I))
    due, text = parse_when(rest, now)

    if text.lower() in _AGAIN or (not text and due):
        t = _get(_last_id)
        if t and not t.get("done") and due:
            t["due"], t["notified"] = due.isoformat(timespec="seconds"), False
            _save()
            return f"Okay, I'll remind you again {_friendly(due, now)}."
    if not text:
        return "What should I remind you about?"

    if need_time and not due:
        return (f"I didn't catch the time. Say it again like: "
                f"remind me to {_lower_first(text)} at 5 pm.")

    text = text[0].upper() + text[1:]
    _store(text, due, now)

    if due:
        link = "about" if about else "to"
        return f"Okay, I'll remind you {link} {_lower_first(_you(text))} {_friendly(due, now)}."
    return f"Added to your tasks: {_you(text)}."


def _list():
    pend = _pending()
    if not pend:
        return "You have no pending tasks."
    now = datetime.now()
    n = len(pend)
    parts = [f"You have {n} {'task' if n == 1 else 'tasks'}."]
    for i, t in enumerate(pend[:5], 1):
        s = f"Number {i}: {_you(t['text'])}"
        if t.get("due"):
            s += f", {_when(datetime.fromisoformat(t['due']), now)}"
        parts.append(s + ".")
    if n > 5:
        parts.append(f"And {n - 5} more.")
    return " ".join(parts)


def _done(ref):
    t = _resolve(ref)
    if not t:
        return "I couldn't find that task."
    t["done"], t["done_at"] = True, datetime.now().isoformat(timespec="seconds")
    _save()
    return f"Marked as done: {_you(t['text'])}."


def _delete(ref):
    t = _resolve(ref)
    if not t:
        return "I couldn't find that task."
    _tasks.remove(t)
    _save()
    return f"Removed: {_you(t['text'])}."


def _clear_all():
    n = len(_pending())
    _tasks[:] = [t for t in _tasks if t.get("done")]
    _save()
    return "All tasks cleared." if n else "You have no pending tasks."


def _snooze(rest):
    t = _get(_last_id)
    if not t or t.get("done"):
        return "There is nothing to snooze."
    now = datetime.now()
    rest = re.sub(r"^(?:for|by|until|till)\s+", "", (rest or "").strip(), flags=re.I)
    due = None
    if rest:
        due, _ = parse_when(rest, now)
        if not due:
            due, _ = parse_when("in " + rest, now)
    due = due or now + timedelta(minutes=SNOOZE_MINUTES)
    t["due"], t["notified"] = due.isoformat(timespec="seconds"), False
    _save()
    return f"Okay, I'll remind you again {_friendly(due, now)}."


_R = re.I
_CLEAR_ALL = re.compile(
    r"^(?:delete|remove|clear)\s+(?:(?:all|every)\s+(?:of\s+)?(?:my\s+|the\s+)?(?:tasks?|reminders?|to-?dos?)"
    r"|(?:my\s+|the\s+)?(?:tasks|reminders|to-?dos))(?:\s+list)?$", _R)
_SNOOZE = re.compile(r"^snooze(?:\s+(?:it|that|this))?(?:\s+(?P<rest>.+))?$", _R)
_DONE_A = re.compile(
    r"^(?:mark|set)\s+(?:task\s+|reminder\s+)?(?P<ref>.+?)\s+(?:as\s+)?(?:done|complete|completed|finished)$", _R)
_DONE_B = re.compile(
    r"^(?:complete|finish|finished|done with|i(?:'ve| have)?\s+(?:done|finished|completed)|tick off|check off)"
    r"\s+(?:task\s+|reminder\s+)?(?P<ref>.+)$", _R)
_DELETE = re.compile(
    r"^(?:delete|remove|cancel)\s+(?:my\s+|the\s+)?(?:task|reminder)s?\s*(?P<ref>.*)$", _R)
_ADD_REMIND = re.compile(r"\bremind me\b[,:]?\s*(?P<rest>.*)$", _R)
_ADD_SET = re.compile(
    r"\b(?:set|create|make|add)\s+(?:a\s+|an\s+)?(?:new\s+)?reminder\b(?:\s+for|\s+to|\s+about)?\s*(?P<rest>.*)$", _R)
_ADD_TASK = re.compile(
    r"^(?:add|create|make|new|note)\s+(?:a\s+|an\s+)?(?:new\s+)?(?:task|todo|to-do|to do)\b[:,]?\s*"
    r"(?:to\s+|for\s+)?(?P<rest>.*)$", _R)
_ADD_LIST = re.compile(
    r"^add\s+(?P<rest>.+?)\s+to\s+(?:my\s+)?(?:task|to-?do|reminder)s?(?:\s+list)?$", _R)
_LIST_NOUN = re.compile(r"\b(?:tasks?|reminders?|to-?do(?:s| list)?|agenda)\b", _R)
_LIST_VERB = re.compile(r"\b(?:what|list|show|read|tell|check|do i have|any|my|pending|upcoming)\b", _R)


def handle(text):
    """Return Alex's reply for a task/reminder command, or None if it isn't one."""
    t = re.sub(r"\s+", " ", text or "").strip().rstrip(" .!?,")
    if not t:
        return None

    with _lock:
        _load()

        if _CLEAR_ALL.match(t):
            return _clear_all()
        m = _SNOOZE.match(t)
        if m:
            return _snooze(m.group("rest"))
        m = _DONE_A.match(t) or _DONE_B.match(t)
        if m and (_to_int(m.group("ref")) is not None or _resolve(m.group("ref"))
                  or re.match(r"(?:task|reminder|it|that|this)\b", t.split(" ", 1)[-1], _R)
                  or re.match(r"^(?:mark|set)\s+(?:task|reminder)", t, _R)):
            return _done(m.group("ref"))
        m = _DELETE.match(t)
        if m:
            return _delete(m.group("ref"))
        if re.match(r"(?:did|do|does|have|has|why|how|are|is)\b", t, _R) and _ADD_REMIND.search(t):
            return _list()  # a question about reminders, not a new reminder
        m = _ADD_REMIND.search(t) or _ADD_SET.search(t)
        if m:
            return _add_many(m.group("rest"))
        for pat in (_ADD_TASK, _ADD_LIST):
            m = pat.match(t)
            if m:
                return _add(m.group("rest"))
        if _LIST_NOUN.search(t) and _LIST_VERB.search(t) and len(t.split()) <= 10:
            return _list()
        if re.search(r"\bremind(?:er)?s?\b", t, _R):
            # never let the small LLM pretend it set a reminder
            return "Sorry, I didn't catch that. Try: remind me to call mom at 5 pm."
    return None


# ------------------------------------------------------------------
# Public helpers + scheduler
# ------------------------------------------------------------------


def pending_count():
    return len(_pending())


def all_tasks():
    now = datetime.now()
    out = []
    for i, t in enumerate(_pending(), 1):
        due = datetime.fromisoformat(t["due"]) if t.get("due") else None
        out.append({"id": t["id"], "number": i, "text": t["text"], "due": t.get("due"),
                    "when": _when(due, now) if due else None})
    return out


def recent_done(limit=3):
    """Tasks finished in the last 24 hours, newest first."""
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat(timespec="seconds")
    with _lock:
        _load()
        items = [t for t in _tasks if t.get("done") and (t.get("done_at") or "") >= cutoff]
    items.sort(key=lambda t: t["done_at"], reverse=True)
    return [{"id": t["id"], "text": t["text"]} for t in items[:limit]]


def _due_now():
    now = datetime.now().isoformat(timespec="seconds")
    with _lock:
        _load()
        out = [dict(t) for t in _tasks
               if not t.get("done") and t.get("due") and not t.get("notified") and t["due"] <= now]
        for t in out:
            _get(t["id"])["notified"] = True
        if out:
            _save()
    return out


def _announcement(t):
    now = datetime.now()
    due = datetime.fromisoformat(t["due"])
    if (now - due).total_seconds() > 120:
        return f"You missed a reminder from {_when(due, now)}: {_you(t['text'])}."
    return f"Reminder: {_you(t['text'])}."


def start_scheduler(on_due, interval=5):
    """Call on_due(message) from a background thread whenever a task is due."""
    global _last_id
    _load()
    _stop.clear()

    def loop():
        global _last_id
        while not _stop.is_set():
            try:
                for t in _due_now():
                    _last_id = t["id"]
                    on_due(_announcement(t))
            except Exception as exc:  # never let the scheduler die
                print(f"[Reminders: scheduler error: {exc}]")
            _stop.wait(interval)

    threading.Thread(target=loop, daemon=True, name="reminder-scheduler").start()


def stop_scheduler():
    _stop.set()