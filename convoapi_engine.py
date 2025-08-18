# =========================
# ConvoAPI v1 — Engine (stateless handler + external state store)
# =========================
# Public API:
#   - class StateAdapter (interface), class InMemoryAdapter (simple impl)
#   - class ConvoEngine(spec, store).handle(conversation_id, message) -> str
#   - default_spec() -> ConvoSpec with example functions
#   - fuzzy_command(token: str, commands: list[str]) -> FuzzyMatch
#
# Notes:
#   * All user-facing strings and params are defined at the top (constants).
#   * The engine is message-in / text-out; it persists state via StateAdapter.
#   * The command router maps canonical commands to handler functions.
#   * Fuzzy command recognition is robust (Levenshtein), with guardrails.
#   * Mobile-first parsing: spaces, commas, periods, newlines are separators.
#   * Labeled strings require no quotes; run to end-of-segment.
#   * Unlabeled strings only when exactly one string arg remains unmet.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

# =========================
# CONSTANTS (UX strings, routing, fuzzy thresholds)
# =========================

# Canonical command names (router keys)
COMMANDS: List[str] = [
    "help", "list", "open", "switch", "home", "back",
    "undo", "rewind", "edit", "again", "repeat",
    "confirm", "cancel", "over", "end", "answer"  # "answer" is internal
]

# Aliases that should map to canonical commands before fuzzy matching
COMMAND_ALIASES: Dict[str, str] = {
    "?": "help",
    "y": "confirm",
    "go": "confirm"
}

# Presentation strings
MSG_WELCOME = "Welcome. Type list or open NAME."
MSG_HOME = "Home. Type list or open NAME."
MSG_GOODBYE = "Goodbye."
MSG_HELP = (
    "Help\n- list, open NAME, switch NAME, home\n"
    "- answer as pairs: arg value (or shortkey value)\n"
    "- unlabeled values fill by precedence\n"
    "- strings: after the arg name, value runs to comma or period\n"
    "- broadcast: over arg v1, v2. other w1\n"
    "- edit, undo N, rewind to ARGNAME\n"
    "- confirm y go, cancel\n- end to exit"
)
ERR_OPEN_FIRST = "Open a function first. Type list or open NAME."
ERR_FUNC_NOT_FOUND = "Function {name} not found."
ERR_ARG_NOT_FOUND = "Argument {name} not found."
ERR_INDEX_RANGE = "Index out of range."
ERR_AMBIGUOUS_STRING = (
    "Ambiguous string: multiple string fields unmet. "
    "Please label, for example: to Shira. message Happy day."
)
HINT_USE_UNDO_OR_REWIND = "Use undo or rewind to ARGNAME."
HINT_INTERPRETED = "[interpreted {src} → {dst}]"
HINT_UPDATED = "Updated."
HINT_CANCELED = "Canceled."

TPL_PREVIEW_HEADER = "Preview"
TPL_PLAN_HEADER = "Plan"
TPL_CONFIRM_HINT_SINGLE = "(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)"
TPL_CONFIRM_HINT_PLAN = "Confirm? (y go confirm, edit INDEX ARGNAME VALUE, cancel)"
TPL_RECEIPT_LINE = "Receipt: {pairs}"
TPL_CREATED_RANGE = "✅ {func} {first}..{last} created"
TPL_CREATED_SINGLE = "✅ {func} {cid}"

# Fuzzy command recognition thresholds & guardrails
FUZZY_MAX_ABS = 2          # maximum absolute edit distance
FUZZY_MAX_REL = 0.5       # maximum relative distance (d/len <= this)
FUZZY_MIN_LEN = 2          # ignore fuzzy if token length < 2 (unless alias)
FUZZY_REQUIRE_ALPHA = True # only fuzz tokens containing letters
FUZZY_SKIP_WHEN_AWAITING_VALUE = True  # while filling broadcast prompt, don't fuzzy commands

# Parsing separators
SEPARATORS_PRIMARY = [",", "\n"]  # segment boundaries
SEPARATOR_PERIOD = "."            # segment boundary unless decimal
WHITESPACE = " "


# =========================
# Spec dataclasses (Function, Args)
# =========================

@dataclass
class EnumVal:
    value: str
    aliases: List[str] = field(default_factory=list)

@dataclass
class ArgSpec:
    name: str
    type: str  # enum,int,float,bool,string
    required: bool
    constraints: Dict[str, Any] = field(default_factory=dict)
    enum: List[EnumVal] = field(default_factory=list)
    shortkey: Optional[str] = None
    prompt_beginner: str = ""
    prompt_power: str = ""

@dataclass
class FuncSpec:
    name: str
    display: str
    callable: Callable[..., Any]
    args: List[ArgSpec]

@dataclass
class ConvoSpec:
    functions: List[FuncSpec]
    fuzzy_enabled: bool = True


# =========================
# State Adapter interface
# =========================

class StateAdapter:
    """Abstract persistence adapter."""
    def load(self, conversation_id: str) -> Dict[str, Any]:
        raise NotImplementedError
    def save(self, conversation_id: str, state: Dict[str, Any]) -> None:
        raise NotImplementedError

class InMemoryAdapter(StateAdapter):
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
    def load(self, conversation_id: str) -> Dict[str, Any]:
        return self._store.get(conversation_id, {}).copy()
    def save(self, conversation_id: str, state: Dict[str, Any]) -> None:
        self._store[conversation_id] = dict(state)


# =========================
# Utilities: Levenshtein, fuzzy, segmentation, tokenization
# =========================

@dataclass
class FuzzyMatch:
    matched: bool
    value: Optional[str] = None
    distance: int = 0

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]

def fuzzy_command(token: str, commands: List[str]) -> FuzzyMatch:
    """Map a possibly misspelled token to a canonical command via Levenshtein."""
    t = token.lower()
    # Exact first
    if t in commands:
        return FuzzyMatch(True, t, 0)
    # Guardrails
    if FUZZY_REQUIRE_ALPHA and not any(c.isalpha() for c in t):
        return FuzzyMatch(False, None, 0)
    if len(t) < FUZZY_MIN_LEN:
        return FuzzyMatch(False, None, 0)
    # Compute best
    best = None
    best_d = 10**9
    for cmd in commands:
        d = levenshtein(t, cmd)
        if d < best_d:
            best = cmd
            best_d = d
    if best is None:
        return FuzzyMatch(False, None, 0)
    rel = best_d / max(1, len(best))
    if best_d <= FUZZY_MAX_ABS and rel <= FUZZY_MAX_REL:
        # Extra guard to avoid matches like 'zzswitch' -> 'switch'
        first_alpha = next((c for c in t if c.isalpha()), '')
        best_first_alpha = next((c for c in best if c.isalpha()), '')
        if best_d >= 2 and first_alpha and best_first_alpha and first_alpha != best_first_alpha:
            return FuzzyMatch(False, None, best_d)
        return FuzzyMatch(True, best, best_d)
    return FuzzyMatch(False, None, best_d)

def split_segments(line: str) -> List[str]:
    """Split on commas, periods, newlines; but keep decimal numbers intact."""
    s = line.replace("\r\n", "\n").replace("\r", "\n")
    segments: List[str] = []
    cur: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in SEPARATORS_PRIMARY:
            seg = "".join(cur).strip()
            if seg:
                segments.append(seg)
            cur = []
            i += 1
            continue
        if ch == SEPARATOR_PERIOD:
            prev_ch = s[i - 1] if i > 0 else ""
            next_ch = s[i + 1] if i + 1 < len(s) else ""
            if prev_ch.isdigit() and next_ch.isdigit():
                cur.append(ch)
                i += 1
                continue
            else:
                seg = "".join(cur).strip()
                if seg:
                    segments.append(seg)
                cur = []
                i += 1
                continue
        cur.append(ch)
        i += 1
    if cur:
        seg = "".join(cur).strip()
        if seg:
            segments.append(seg)
    return segments

def tokenize_segment(seg: str) -> List[str]:
    return [t for t in seg.strip().split() if t]


# =========================
# Context helper (wraps state + spec; emits responses)
# =========================

def NEW_STATE() -> Dict[str, Any]:
    return {
        "current": None,             # {"func": str, "values": dict, "notes": list}
        "plan": None,                # [{"values": dict}, ...]
        "last_executed": None,       # {"func": str, "values": dict}
        "scratch": {},               # {"last buy_fruit": {...}, ...}
        "history": [],               # [{"type": "set"/"edit_plan", "payload": {...}}, ...]
        "await_plan_fill": None      # {"index": int, "argname": str} or None
    }

class Context:
    def __init__(self, spec: ConvoSpec, state: Dict[str, Any]):
        self.spec = spec
        self.state = state if state else NEW_STATE()
        self._hints: List[str] = []  # lines to prepend

    # Convenience accessors
    @property
    def current(self) -> Optional[Dict[str, Any]]:
        return self.state.get("current")
    @current.setter
    def current(self, val: Optional[Dict[str, Any]]):
        self.state["current"] = val

    @property
    def plan(self) -> Optional[List[Dict[str, Any]]]:
        return self.state.get("plan")
    @plan.setter
    def plan(self, val: Optional[List[Dict[str, Any]]]):
        self.state["plan"] = val

    @property
    def last_executed(self) -> Optional[Dict[str, Any]]:
        return self.state.get("last_executed")
    @last_executed.setter
    def last_executed(self, val: Optional[Dict[str, Any]]):
        self.state["last_executed"] = val

    def awaiting_value(self) -> bool:
        return bool(self.state.get("await_plan_fill"))

    def set_await_fill(self, idx: int, argname: str):
        self.state["await_plan_fill"] = {"index": idx, "argname": argname}

    def clear_await_fill(self):
        self.state["await_plan_fill"] = None

    def add_hint(self, msg: str):
        self._hints.append(msg)

    # Spec helpers
    def func_map(self) -> Dict[str, FuncSpec]:
        return {f.name: f for f in self.spec.functions}

    def get_func(self, name: str) -> Optional[FuncSpec]:
        return self.func_map().get(name)

    # Response merging
    def render(self, lines: List[str]) -> str:
        if self._hints:
            return "\n".join(self._hints + lines)
        return "\n".join(lines)

    # Prompting
    def next_arg(self) -> Optional[ArgSpec]:
        if not self.current:
            return None
        func = self.get_func(self.current["func"])
        for a in func.args:
            if a.name not in self.current["values"]:
                return a
        return None

    def unmet_required(self) -> List[ArgSpec]:
        if not self.current:
            return []
        func = self.get_func(self.current["func"])
        return [a for a in func.args if a.required and a.name not in self.current["values"]]

    def preview(self) -> List[str]:
        if self.plan:
            lines = [TPL_PLAN_HEADER]
            func = self.get_func(self.current["func"])
            for i, d in enumerate(self.plan, 1):
                parts = []
                for a in func.args:
                    v = d["values"].get(a.name, None)
                    parts.append(f"{a.name} {v if v is not None else 'none'}")
                lines.append(f"{i}) " + ", ".join(parts))
            lines.append(TPL_CONFIRM_HINT_PLAN)
            return lines
        if self.current:
            func = self.get_func(self.current["func"])
            lines = [TPL_PREVIEW_HEADER]
            for a in func.args:
                v = self.current["values"].get(a.name, None)
                lines.append(f"- {a.name} {v if v is not None else 'none'}")
            lines.append(TPL_CONFIRM_HINT_SINGLE)
            return lines
        return [MSG_HOME]

    def prompt_next_or_preview(self) -> str:
        if not self.current:
            return MSG_HOME
        nxt = self.next_arg()
        if not nxt:
            return self.render(self.preview())
        # Beginner + Power prompts
        return self.render([
            f"Beginner: {nxt.prompt_beginner}",
            f"Power: {nxt.prompt_power}"
        ])


# =========================
# Argument lookup, typing and coercion
# =========================

def arg_lookup(func: FuncSpec, name: str, fuzzy_ok: bool = True) -> Optional[ArgSpec]:
    """Match arg by name or shortkey; optional fuzzy."""
    candidates: Dict[str, ArgSpec] = {}
    for a in func.args:
        candidates[a.name.lower()] = a
        if a.shortkey:
            candidates[a.shortkey.lower()] = a
    key = name.lower()
    if key in candidates:
        return candidates[key]
    if not fuzzy_ok:
        return None
    # fuzzy over arg names and shortkeys
    opts = list(candidates.keys())
    fm = fuzzy_command(key, opts)
    if fm.matched:
        return candidates[fm.value]
    return None

def enum_resolve(a: ArgSpec, token: str) -> Tuple[Optional[str], bool]:
    """Return (canonical enum value, was_fuzzy)."""
    t = token.lower()
    mapping: Dict[str, str] = {}
    for ev in a.enum:
        mapping[ev.value.lower()] = ev.value
        for al in ev.aliases:
            mapping[al.lower()] = ev.value
    if t in mapping:
        return mapping[t], False
    # fuzzy over canonical + aliases
    fm = fuzzy_command(t, list(mapping.keys()))
    if fm.matched and fm.value in mapping:
        return mapping[fm.value], True
    return None, False

def is_int_token(tok: str) -> bool:
    if tok.startswith("+"):
        tok2 = tok[1:]
    else:
        tok2 = tok
    return tok2.isdigit() or (tok2.startswith("-") and tok2[1:].isdigit())

def is_float_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False

def is_bool_token(tok: str) -> bool:
    return tok.lower() in ["true", "false", "yes", "no", "y", "n", "1", "0"]

def normalize_bool(tok: str) -> bool:
    return tok.lower() in ["true", "yes", "y", "1"]

def type_accepts(a: ArgSpec, token: str) -> bool:
    t = a.type
    if t == "int":
        return is_int_token(token)
    if t == "float":
        return is_float_token(token)
    if t == "bool":
        return is_bool_token(token)
    if t == "enum":
        v, _ = enum_resolve(a, token)
        return v is not None
    if t == "string":
        return True
    return False

def coerce_value(a: ArgSpec, raw: str) -> Tuple[Any, Optional[str]]:
    """Return (value, hint) where hint may be a fuzzy interpretation message."""
    if a.type == "int":
        val = int(raw)
        if "min" in a.constraints and val < a.constraints["min"]:
            raise ValueError(f"{a.name} {val} invalid, use {a.constraints['min']} to {a.constraints.get('max','∞')}")
        if "max" in a.constraints and val > a.constraints["max"]:
            raise ValueError(f"{a.name} {val} invalid, use {a.constraints['min']} to {a.constraints.get('max','∞')}")
        return val, None
    if a.type == "float":
        val = float(raw)
        if "min" in a.constraints and val < a.constraints["min"]:
            raise ValueError(f"{a.name} {val} invalid, must be at least {a.constraints['min']}")
        return val, None
    if a.type == "bool":
        return normalize_bool(raw), None
    if a.type == "enum":
        v, fuzzy = enum_resolve(a, raw)
        if v is None:
            raise ValueError(f"{a.name} {raw} not recognized")
        return v, (HINT_INTERPRETED.format(src=raw, dst=v) if fuzzy else None)
    if a.type == "string":
        return raw, None
    return raw, None


# =========================
# Router and handlers
# =========================

def _choose_command_token(ctx: Context, first_segment: str) -> Tuple[str, str]:
    """Return (command, args_in_first_segment) or ('answer', whole_segment)."""
    tokens = tokenize_segment(first_segment)
    if not tokens:
        return "answer", first_segment

    token0 = tokens[0].lower()
    token0 = COMMAND_ALIASES.get(token0, token0)

    if token0 in COMMANDS:
        return token0, first_segment[len(tokens[0]):].strip()

    # Fuzzy, with guardrails
    can_fuzz = True
    if FUZZY_SKIP_WHEN_AWAITING_VALUE and ctx.awaiting_value():
        can_fuzz = False
    if FUZZY_REQUIRE_ALPHA and not any(c.isalpha() for c in token0):
        can_fuzz = False
    if token0 in COMMANDS:
        can_fuzz = False

    if can_fuzz:
        fm = fuzzy_command(token0, COMMANDS)
        if fm.matched:
            ctx.add_hint(HINT_INTERPRETED.format(src=token0, dst=fm.value))
            return fm.value, first_segment[len(tokens[0]):].strip()

    return "answer", first_segment


def handle_help(ctx: Context, args: str, segments: List[str]) -> str:
    return MSG_HELP

def handle_list(ctx: Context, args: str, segments: List[str]) -> str:
    lines = []
    for f in ctx.spec.functions:
        args_text = ", ".join([
            f"{a.name}{'' if a.required else ' (optional)'}" for a in f.args
        ])
        lines.append(f"- {f.name} ({args_text})")
    return "\n".join(lines)

def _fuzzy_func(spec: ConvoSpec, name: str) -> Optional[FuncSpec]:
    fmap = {f.name: f for f in spec.functions}
    if name in fmap:
        return fmap[name]
    fm = fuzzy_command(name, list(fmap.keys()))
    if fm.matched and fm.value in fmap:
        return fmap[fm.value]
    return None

def handle_open(ctx: Context, args: str, segments: List[str]) -> str:
    name = args.strip().lower()
    if not name:
        return ERR_FUNC_NOT_FOUND.format(name=name)
    f = _fuzzy_func(ctx.spec, name)
    if not f:
        return ERR_FUNC_NOT_FOUND.format(name=name)
    ctx.current = {"func": f.name, "values": {}, "notes": []}
    ctx.plan = None
    return ctx.prompt_next_or_preview()

def handle_switch(ctx: Context, args: str, segments: List[str]) -> str:
    if ctx.current:
        ctx.state["scratch"][f"last {ctx.current['func']}"] = ctx.current
    return handle_open(ctx, args, segments)

def handle_home(ctx: Context, args: str, segments: List[str]) -> str:
    ctx.current = None
    ctx.plan = None
    ctx.clear_await_fill()
    return MSG_HOME

def handle_back(ctx: Context, args: str, segments: List[str]) -> str:
    return HINT_USE_UNDO_OR_REWIND

def handle_undo(ctx: Context, args: str, segments: List[str]) -> str:
    hist = ctx.state["history"]
    n = 1
    toks = tokenize_segment(args)
    if toks and toks[0].isdigit():
        n = int(toks[0])
    while n > 0 and hist:
        entry = hist.pop()
        typ = entry["type"]
        p = entry["payload"]
        if typ == "set" and ctx.current:
            key = p["key"]
            old = p["old"]
            if old is None:
                ctx.current["values"].pop(key, None)
            else:
                ctx.current["values"][key] = old
        elif typ == "edit_plan" and ctx.plan:
            idx = p["idx"]
            key = p["key"]
            old = p["old"]
            if 0 <= idx < len(ctx.plan):
                if old is None:
                    ctx.plan[idx]["values"].pop(key, None)
                else:
                    ctx.plan[idx]["values"][key] = old
        n -= 1
    return "\n".join(ctx.preview())

def handle_rewind(ctx: Context, args: str, segments: List[str]) -> str:
    if not ctx.current:
        return "No draft to rewind."
    toks = tokenize_segment(args)
    # Expect "to ARGNAME"
    if len(toks) >= 2 and toks[0].lower() == "to":
        argname = " ".join(toks[1:]).lower()
    else:
        return "Say: rewind to ARGNAME"
    func = ctx.get_func(ctx.current["func"])
    a = arg_lookup(func, argname)
    if not a:
        return ERR_ARG_NOT_FOUND.format(name=argname)
    # Keep values before a; drop a and after
    newvals = {}
    for spec_a in func.args:
        if spec_a.name == a.name:
            break
        if spec_a.name in ctx.current["values"]:
            newvals[spec_a.name] = ctx.current["values"][spec_a.name]
    ctx.current["values"] = newvals
    return ctx.prompt_next_or_preview()

def handle_edit(ctx: Context, args: str, segments: List[str]) -> str:
    if ctx.plan:
        # edit INDEX ARGNAME VALUE
        toks0 = tokenize_segment(args)
        if len(toks0) < 3 or not toks0[0].isdigit():
            return "Say: edit INDEX ARGNAME VALUE"
        idx = int(toks0[0]) - 1
        if not (0 <= idx < len(ctx.plan)):
            return ERR_INDEX_RANGE
        # try two-word arg, then one-word
        func = ctx.get_func(ctx.current["func"])
        a = arg_lookup(func, " ".join(toks0[1:3]), fuzzy_ok=True)
        pos = 3
        if not a:
            a = arg_lookup(func, toks0[1], fuzzy_ok=True)
            pos = 2
        if not a:
            return ERR_ARG_NOT_FOUND.format(name=" ".join(toks0[1:]))
        value_str = " ".join(toks0[pos:])
        try:
            if a.type == "string":
                val, hint = coerce_value(a, value_str)
            else:
                # take first token only for non-strings
                if not value_str.strip():
                    return f"Missing value for {a.name}."
                first_tok = tokenize_segment(value_str)[0]
                val, hint = coerce_value(a, first_tok)
        except Exception as e:
            return str(e)
        old = ctx.plan[idx]["values"].get(a.name)
        ctx.plan[idx]["values"][a.name] = val
        ctx.state["history"].append({"type": "edit_plan", "payload": {"idx": idx, "key": a.name, "old": old}})
        lines = []
        if hint:
            lines.append(hint)
        lines.append(HINT_UPDATED)
        lines.extend(ctx.preview())
        return "\n".join(lines)
    if not ctx.current:
        return "No draft to edit."
    # edit ARGNAME VALUE
    toks = tokenize_segment(args)
    if len(toks) < 2:
        return "Say: edit ARGNAME VALUE"
    func = ctx.get_func(ctx.current["func"])
    a = arg_lookup(func, " ".join(toks[:2]), fuzzy_ok=True)
    pos = 2
    if not a:
        a = arg_lookup(func, toks[0], fuzzy_ok=True)
        pos = 1
    if not a:
        return ERR_ARG_NOT_FOUND.format(name=" ".join(toks))
    value_str = " ".join(toks[pos:])
    try:
        if a.type == "string":
            val, hint = coerce_value(a, value_str)
        else:
            if not value_str.strip():
                return f"Missing value for {a.name}."
            first_tok = tokenize_segment(value_str)[0]
            val, hint = coerce_value(a, first_tok)
    except Exception as e:
        return str(e)
    old = ctx.current["values"].get(a.name)
    ctx.current["values"][a.name] = val
    ctx.state["history"].append({"type": "set", "payload": {"key": a.name, "old": old}})
    lines = []
    if hint:
        lines.append(hint)
    lines.extend(ctx.preview())
    return "\n".join(lines)

def handle_again(ctx: Context, args: str, segments: List[str]) -> str:
    if not ctx.last_executed:
        return "No previous call to repeat."
    # clone last_executed
    last = ctx.last_executed
    ctx.current = {"func": last["func"], "values": dict(last["values"]), "notes": []}
    ctx.plan = None
    # apply diffs from remaining segments as answers
    return handle_answer(ctx, args="", segments=segments[1:])

def handle_confirm(ctx: Context, args: str, segments: List[str]) -> str:
    if ctx.plan:
        # ensure completeness
        func = ctx.get_func(ctx.current["func"])
        for i, d in enumerate(ctx.plan):
            for a in func.args:
                if a.required and d["values"].get(a.name) is None:
                    return f"Missing {a.name} for item {i+1}."
        # execute all
        ids = []
        receipts = []
        for d in ctx.plan:
            cid, rec = _execute_and_receipt(func, d["values"])
            ids.append(cid)
            receipts.append(rec)
        ctx.last_executed = {"func": ctx.current["func"], "values": ctx.plan[-1]["values"]}
        ctx.plan = None
        ctx.current = None
        lines = receipts
        if ids:
            lines.append(TPL_CREATED_RANGE.format(func=func.name.upper(), first=ids[0], last=ids[-1]))
        return "\n".join(lines)
    if not ctx.current:
        return "No draft to confirm."
    # single call
    func = ctx.get_func(ctx.current["func"])
    missing = [a.name for a in func.args if a.required and a.name not in ctx.current["values"]]
    if missing:
        return "Missing required: " + ", ".join(missing)
    cid, rec = _execute_and_receipt(func, ctx.current["values"])
    ctx.last_executed = {"func": func.name, "values": dict(ctx.current["values"])}
    ctx.current = None
    lines = [rec, TPL_CREATED_SINGLE.format(func=func.name.upper(), cid=cid)]
    return "\n".join(lines)

def _execute_and_receipt(func: FuncSpec, values: Dict[str, Any]) -> Tuple[str, str]:
    ordered = [values.get(a.name) for a in func.args]
    try:
        func.callable(*ordered)
    except Exception:
        pass
    cid = (func.name[:2] + "-" + uuid.uuid4().hex[:6]).upper()
    pairs = ", ".join([f"{a.name} {values.get(a.name, None)}" for a in func.args])
    rec = TPL_RECEIPT_LINE.format(pairs=pairs)
    return cid, rec

def handle_cancel(ctx: Context, args: str, segments: List[str]) -> str:
    ctx.current = None
    ctx.plan = None
    ctx.clear_await_fill()
    return HINT_CANCELED

def _classify_token(func: FuncSpec, tok: str) -> str:
    # Priority: enum > int > float > bool > string
    # Exact enum/alias first; fuzzy only for alpha tokens
    t = tok.lower()
    for a in func.args:
        if a.type == "enum":
            for ev in a.enum:
                if t == ev.value.lower():
                    return "enum"
                for al in ev.aliases:
                    if t == al.lower():
                        return "enum"
    if any(ch.isalpha() for ch in t):
        # allow fuzzy enum classification across all enum options
        opts = []
        for a in func.args:
            if a.type == "enum":
                for ev in a.enum:
                    opts.append(ev.value.lower())
                    for al in ev.aliases:
                        opts.append(al.lower())
        fm = fuzzy_command(t, opts)
        if fm.matched:
            return "enum"
    if is_int_token(tok):
        return "int"
    if is_float_token(tok):
        return "float"
    if is_bool_token(tok):
        return "bool"
    return "string"

def _apply_positional_tokens(ctx: Context, func: FuncSpec, draft_values: Dict[str, Any], tokens: List[str]) -> Optional[str]:
    # Assign unlabeled tokens by precedence: earliest unmet required, then earliest unmet optional
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        category = _classify_token(func, tok)
        # Check ambiguous unlabeled string case
        if category == "string":
            string_unmet = [a for a in func.args if a.type == "string" and a.name not in draft_values]
            if len(string_unmet) > 1:
                req_unmet = [a for a in string_unmet if a.required]
                if len(req_unmet) == 1:
                    chosen = req_unmet[0]
                    rest = " ".join(tokens[i:])
                    val, hint = coerce_value(chosen, rest)
                    draft_values[chosen.name] = val
                    if hint:
                        ctx.add_hint(hint)
                    ctx.state["history"].append({"type": "set", "payload": {"key": chosen.name, "old": None}})
                    return None
                elif len(req_unmet) >= 2:
                    return ERR_AMBIGUOUS_STRING

            string_unmet = [a for a in func.args if a.type == "string" and a.name not in draft_values]
            if len(string_unmet) > 1:
                return ERR_AMBIGUOUS_STRING
        # Build candidate pool for this token
        candidates = []
        for a in func.args:
            if a.name in draft_values:
                continue
            if a.type == category and type_accepts(a, tok):
                candidates.append(a)
        if not candidates:
            # If exactly one string slot remains, consume rest into it
            only = next((a for a in func.args if a.type == "string" and a.name not in draft_values), None)
            if only and category == "string":
                rest = " ".join(tokens[i:])
                val, hint = coerce_value(only, rest)
                draft_values[only.name] = val
                if hint:
                    ctx.add_hint(hint)
                return None
            i += 1
            continue
        reqs = [a for a in candidates if a.required]
        pool = reqs if reqs else candidates
        chosen = None
        for a in func.args:
            if a in pool:
                chosen = a
                break
        try:
            if chosen.type == "string":
                rest = " ".join(tokens[i:])
                val, hint = coerce_value(chosen, rest)
                draft_values[chosen.name] = val
                if hint:
                    ctx.add_hint(hint)
                return None
            else:
                val, hint = coerce_value(chosen, tok)
        except Exception as e:
            return str(e)
        draft_values[chosen.name] = val
        if hint:
            ctx.add_hint(hint)
        # history (we only record that something was set; for positional, old is None)
        ctx.state["history"].append({"type": "set", "payload": {"key": chosen.name, "old": None}})
        i += 1
    return None

def _scan_labeled_pairs(ctx: Context, func: FuncSpec, segment: str, draft_values: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Scan a segment for labeled pairs anywhere; return (error_or_None, leftover_unlabeled_tokens)."""
    tokens = tokenize_segment(segment)
    i = 0
    leftover: List[str] = []
    while i < len(tokens):
        # Two-word label
        a2 = None
        if i + 1 < len(tokens):
            a2 = arg_lookup(func, f"{tokens[i]} {tokens[i+1]}", fuzzy_ok=False)
        if a2:
            if a2.type == "string":
                val = " ".join(tokens[i+2:]).strip()
                v, hint = coerce_value(a2, val)
                draft_values[a2.name] = v
                if hint:
                    ctx.add_hint(hint)
                # strings consume the rest of segment
                return None, []
            else:
                if i + 2 >= len(tokens):
                    return f"Missing value for {a2.name}.", []
                v, hint = coerce_value(a2, tokens[i+2])
                draft_values[a2.name] = v
                if hint:
                    ctx.add_hint(hint)
                ctx.state["history"].append({"type": "set", "payload": {"key": a2.name, "old": None}})
                i += 3
                continue
        # One-word label
        a1 = arg_lookup(func, tokens[i], fuzzy_ok=False)
        if a1:
            if a1.type == "string":
                val = " ".join(tokens[i+1:]).strip()
                v, hint = coerce_value(a1, val)
                draft_values[a1.name] = v
                if hint:
                    ctx.add_hint(hint)
                return None, []
            else:
                if i + 1 >= len(tokens):
                    return f"Missing value for {a1.name}.", []
                v, hint = coerce_value(a1, tokens[i+1])
                draft_values[a1.name] = v
                if hint:
                    ctx.add_hint(hint)
                ctx.state["history"].append({"type": "set", "payload": {"key": a1.name, "old": None}})
                i += 2
                continue
        # Neither label: treat as unlabeled
        leftover.append(tokens[i])
        i += 1
    return None, leftover

def handle_over(ctx: Context, args: str, segments: List[str]) -> str:
    if not ctx.current:
        return ERR_OPEN_FIRST
    func = ctx.get_func(ctx.current["func"])
    vectors: Dict[str, List[Any]] = {}
    last_arg: Optional[ArgSpec] = None

    for seg in segments:
        st = seg.strip()
        if not st:
            continue
        if st.lower().startswith("over"):
            st = st[4:].strip()
            if not st:
                continue
        tokens = tokenize_segment(st)
        if not tokens:
            continue
        # try two-word, then one-word arg
        a = None
        rest = []
        if len(tokens) >= 2:
            a = arg_lookup(func, f"{tokens[0]} {tokens[1]}", fuzzy_ok=False)
            if a:
                rest = tokens[2:]
        if not a:
            a = arg_lookup(func, tokens[0], fuzzy_ok=False)
            if a:
                rest = tokens[1:]
        if not a:
            # continuation for last_arg
            if last_arg:
                a = last_arg
                rest = tokens
            else:
                return f"Argument not recognized in segment: {seg}"
        last_arg = a
        value_str = " ".join(rest).strip()
        inner = split_segments(value_str) if value_str else []
        if not inner and rest:
            inner = [" ".join(rest)]
        if not inner and not rest and tokens:
            inner = [" ".join(tokens)]

        vals = []
        for val_seg in inner:
            vs = val_seg.strip()
            if not vs:
                continue
            if a.type == "string":
                vals.append(vs)
            else:
                first_tok = tokenize_segment(vs)[0]
                coerced, hint = coerce_value(a, first_tok)
                if hint:
                    ctx.add_hint(hint)
                vals.append(coerced)
        vectors.setdefault(a.name, []).extend(vals)

    # Determine target count
    n_targets = 1
    for vs in vectors.values():
        if len(vs) > 1:
            n_targets = max(n_targets, len(vs))
    if n_targets == 1:
        return "Provide a list with over to create multiple calls (e.g., over fruit apple, banana)."

    items = [{"values": {}} for _ in range(n_targets)]
    for a in func.args:
        if a.name in vectors:
            vs = vectors[a.name]
            if len(vs) == 1:
                for d in items:
                    d["values"][a.name] = vs[0]
            elif len(vs) == n_targets:
                for i in range(n_targets):
                    items[i]["values"][a.name] = vs[i]
            else:
                for i in range(n_targets):
                    items[i]["values"][a.name] = vs[i] if i < len(vs) else None

    ctx.plan = items
    # Check first missing required
    for i, d in enumerate(ctx.plan):
        for a in func.args:
            if a.required and d["values"].get(a.name) is None:
                # Prompt for missing
                ctx.set_await_fill(i, a.name)
                desc = _describe_item(func, d["values"])
                return "\n".join([
                    f"{n_targets} targets; {a.name} has missing values. Need {a.name} for item {i+1} {desc}.",
                    "Reply a value, or fill last, or cancel."
                ])
    return "\n".join(ctx.preview())

def _describe_item(func: FuncSpec, values: Dict[str, Any]) -> str:
    for a in func.args:
        if a.type == "enum" and values.get(a.name) is not None:
            return str(values.get(a.name))
    return func.name

def _fill_plan_value(ctx: Context, line: str) -> str:
    """Handle the special mode where we await a single value for broadcast."""
    awaitd = ctx.state["await_plan_fill"]
    if not awaitd or not ctx.plan or not ctx.current:
        ctx.clear_await_fill()
        return ctx.prompt_next_or_preview()
    idx = awaitd["index"]
    argname = awaitd["argname"]
    func = ctx.get_func(ctx.current["func"])
    a = next((x for x in func.args if x.name == argname), None)
    if not a:
        ctx.clear_await_fill()
        return ctx.prompt_next_or_preview()

    segs = split_segments(line)
    if not segs:
        return f"Provide a value for {argname}."
    seg0 = segs[0].strip().lower()
    if seg0 == "fill last":
        last = None
        for d in ctx.plan:
            v = d["values"].get(argname)
            if v is not None:
                last = v
        if last is None:
            return "No last value to fill."
        ctx.plan[idx]["values"][argname] = last
    else:
        # use first token for non-string, entire first segment for string
        try:
            if a.type == "string":
                val, hint = coerce_value(a, segs[0])
            else:
                tok = tokenize_segment(segs[0])[0]
                val, hint = coerce_value(a, tok)
        except Exception as e:
            return str(e)
        ctx.plan[idx]["values"][argname] = val
        if hint:
            ctx.add_hint(hint)

    ctx.clear_await_fill()
    # Next missing?
    for j, d in enumerate(ctx.plan):
        for ax in func.args:
            if ax.required and d["values"].get(ax.name) is None:
                ctx.set_await_fill(j, ax.name)
                desc = _describe_item(func, d["values"])
                return "\n".join([
                    f"Need {ax.name} for item {j+1} {desc}.",
                    "Reply a value, or fill last, or cancel."
                ])
    return "\n".join(ctx.preview())

def handle_answer(ctx: Context, args: str, segments: List[str]) -> str:
    # If awaiting broadcast value, fill that first.
    if ctx.awaiting_value():
        return _fill_plan_value(ctx, segments[0])

    # If no current draft, try to interpret as 'open NAME'
    if not ctx.current and not ctx.plan:
        # attempt fuzzy open: first token as function name
        tokens = tokenize_segment(segments[0]) if segments else []
        if tokens:
            f = _fuzzy_func(ctx.spec, tokens[0].lower())
            if f:
                ctx.current = {"func": f.name, "values": {}, "notes": []}
                # apply rest of segments as answers
                segs = segments[1:] if segments else []
                if segs:
                    _apply_segments_to_current(ctx, segs)
                return ctx.prompt_next_or_preview()
        return ERR_OPEN_FIRST

    # Normal answer mode: apply segments to current draft
    _apply_segments_to_current(ctx, segments)
    # If all required filled, show preview; else prompt next
    if not ctx.unmet_required():
        return "\n".join(ctx.preview())
    return ctx.prompt_next_or_preview()

def _apply_segments_to_current(ctx: Context, segments: List[str]) -> None:
    if not ctx.current:
        return
    func = ctx.get_func(ctx.current["func"])
    for seg in segments:
        # Labeled scan first
        err, leftovers = _scan_labeled_pairs(ctx, func, seg, ctx.current["values"])
        if err:
            # Emit error immediately into hints area
            ctx.add_hint(err)
            continue
        # Positional fallback
        if leftovers:
            err2 = _apply_positional_tokens(ctx, func, ctx.current["values"], leftovers)
            if err2:
                ctx.add_hint(err2)

def handle_end(ctx: Context, args: str, segments: List[str]) -> str:
    # Do not clear state here; caller can choose to keep it around.
    return MSG_GOODBYE


# =========================
# Engine entry
# =========================

class ConvoEngine:
    def __init__(self, spec: ConvoSpec, store: StateAdapter):
        self.spec = spec
        self.store = store
        # Build router map
        self.handlers = {
            "help": handle_help,
            "list": handle_list,
            "open": handle_open,
            "switch": handle_switch,
            "home": handle_home,
            "back": handle_back,
            "undo": handle_undo,
            "rewind": handle_rewind,
            "edit": handle_edit,
            "again": handle_again,
            "repeat": handle_again,   # alias
            "confirm": handle_confirm,
            "cancel": handle_cancel,
            "over": handle_over,
            "end": handle_end,
            "answer": handle_answer
        }

    def handle(self, conversation_id: str, message: str) -> str:
        state = self.store.load(conversation_id) or NEW_STATE()
        ctx = Context(self.spec, state)

        # Segment input and choose command based on the first segment
        segments = split_segments(message or "")
        if not segments:
            out = ctx.prompt_next_or_preview()
            self.store.save(conversation_id, ctx.state)
            return out

        command, first_args = _choose_command_token(ctx, segments[0])
        handler = self.handlers.get(command, handle_answer)

        # For "over", we should pass all segments as is (for multi-arg vectors)
        if command == "over":
            out = handler(ctx, first_args, segments)
        else:
            out = handler(ctx, first_args, segments)

        self.store.save(conversation_id, ctx.state)
        return out


# =========================
# Example functions + default spec
# =========================

def buy_fruit_callable(fruit, qty, price_cap, note, priority):
    return {"ok": True}

def send_gift_callable(item, to, qty, wrap, speed, insurance, budget, message):
    return {"ok": True}

def default_spec() -> ConvoSpec:
    buy_fruit = FuncSpec(
        name="buy_fruit",
        display="Buy Fruit",
        callable=buy_fruit_callable,
        args=[
            ArgSpec(
                name="fruit", type="enum", required=True,
                enum=[EnumVal("apple", ["a"]), EnumVal("banana", ["ba"]), EnumVal("berries", ["be"]), EnumVal("cherries", ["c"])],
                prompt_beginner="Fruit? apple, banana, berries, cherries.",
                prompt_power="fruit apple  or  a  .  over fruit apple, banana."
            ),
            ArgSpec(
                name="qty", type="int", required=True,
                constraints={"min": 1, "max": 20}, shortkey="q",
                prompt_beginner="Quantity 1 to 20?",
                prompt_power="qty 4  or  q 4"
            ),
            ArgSpec(
                name="price cap", type="float", required=False,
                constraints={"min": 0}, shortkey="cap",
                prompt_beginner="Price cap optional, zero or more.",
                prompt_power="price cap 10.5  or  cap 10.5"
            ),
            ArgSpec(
                name="note", type="string", required=False,
                prompt_beginner="Note optional.",
                prompt_power="note for office"
            ),
            ArgSpec(
                name="priority", type="int", required=True,
                constraints={"min": 1, "max": 6}, shortkey="p",
                prompt_beginner="Priority 1 to 6?",
                prompt_power="priority 3  or  p 3"
            ),
        ]
    )

    send_gift = FuncSpec(
        name="send_gift",
        display="Send Gift",
        callable=send_gift_callable,
        args=[
            ArgSpec(name="item", type="enum", required=True,
                enum=[EnumVal("mug"), EnumVal("book"), EnumVal("flowers")],
                prompt_beginner="Item? mug, book, flowers.",
                prompt_power="item flowers"),
            ArgSpec(name="to", type="string", required=True,
                prompt_beginner="Recipient name?",
                prompt_power="to Noa Levi"),
            ArgSpec(name="qty", type="int", required=True,
                constraints={"min": 1, "max": 20}, shortkey="q",
                prompt_beginner="Quantity 1 to 20?",
                prompt_power="qty 2  or  q 2"),
            ArgSpec(name="wrap", type="enum", required=True,
                enum=[EnumVal("none"), EnumVal("paper"), EnumVal("foil")],
                prompt_beginner="Wrap? none, paper, foil.",
                prompt_power="wrap paper"),
            ArgSpec(name="speed", type="enum", required=True,
                enum=[EnumVal("standard"), EnumVal("express"), EnumVal("overnight")],
                prompt_beginner="Speed? standard, express, overnight.",
                prompt_power="speed express"),
            ArgSpec(name="insurance", type="bool", required=True,
                prompt_beginner="Insurance? yes or no.",
                prompt_power="insurance yes"),
            ArgSpec(name="budget", type="float", required=False,
                constraints={"min": 0},
                prompt_beginner="Budget optional, zero or more.",
                prompt_power="budget 120"),
            ArgSpec(name="message", type="string", required=False,
                prompt_beginner="Message optional.",
                prompt_power="message Happy day"),
        ]
    )

    return ConvoSpec(functions=[buy_fruit, send_gift], fuzzy_enabled=True)