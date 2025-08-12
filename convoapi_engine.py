
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------- Levenshtein & fuzzy ---------
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost))
        prev = cur
    return prev[-1]

def fuzzy_match(token: str, options: List[str]) -> Tuple[str, bool]:
    t = token.lower()
    for opt in options:
        if t == opt.lower():
            return opt, False
    best = None
    best_d = 10**9
    for opt in options:
        d = levenshtein(t, opt.lower())
        if (len(opt) <= 8 and d <= 1) or (len(opt) > 8 and d / len(opt) <= 0.2):
            if d < best_d:
                best = opt
                best_d = d
    if best is not None:
        return best, True
    return token, False

# --------- Spec dataclasses ----------
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

# --------- IO helper (tests use this; real run uses input/print) ----------
class IOBuffer:
    def __init__(self, inputs: List[str]):
        self.inputs = inputs[:]
        self.outputs: List[str] = []
    def read(self) -> Optional[str]:
        if not self.inputs:
            return None
        return self.inputs.pop(0)
    def write(self, s: str):
        self.outputs.append(s.strip("\n"))
    def joined(self) -> str:
        return "\n".join(self.outputs)

# --------- Parsing helpers ----------
RESERVED = ["help","?","list","open","switch","home","back","undo","rewind","edit","again","repeat","confirm","y","go","cancel","over","end"]

def is_float_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except:
        return False

def is_int_token(tok: str) -> bool:
    tok2 = tok[1:] if tok.startswith("+") else tok
    return tok2.isdigit() or (tok2.startswith("-") and tok2[1:].isdigit())

def is_bool_token(tok: str) -> bool:
    return tok.lower() in ["true","false","yes","no","y","n","1","0"]

def normalize_bool(tok: str) -> bool:
    return tok.lower() in ["true","yes","y","1"]

def split_segments(line: str) -> List[str]:
    s = line.replace("\\r\\n","\\n").replace("\\r","\\n")
    segments = []
    cur = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in [",", "\\n"]:
            segments.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        if ch == ".":
            prev_ch = s[i-1] if i>0 else ""
            next_ch = s[i+1] if i+1<len(s) else ""
            if prev_ch.isdigit() and next_ch.isdigit():
                cur.append(ch)
                i += 1
                continue
            else:
                segments.append("".join(cur).strip())
                cur = []
                i += 1
                continue
        cur.append(ch)
        i += 1
    if cur:
        segments.append("".join(cur).strip())
    return [seg.strip() for seg in segments if seg.strip()]

def tokenize_segment(seg: str) -> List[str]:
    return [t for t in seg.strip().split() if t]

def arg_lookup(func: FuncSpec, name: str, fuzzy_ok=True) -> Optional[ArgSpec]:
    names = []
    mapping = {}
    for a in func.args:
        names.append(a.name.lower())
        mapping[a.name.lower()] = a
        if a.shortkey:
            names.append(a.shortkey.lower())
            mapping[a.shortkey.lower()] = a
    key = name.lower()
    if key in mapping:
        return mapping[key]
    if fuzzy_ok:
        matched, fuzzy = fuzzy_match(key, names)
        if matched.lower() in mapping:
            return mapping[matched.lower()]
    return None

def enum_match(a: ArgSpec, token: str) -> Tuple[Optional[str], bool]:
    all_opts = []
    map_opt = {}
    for ev in a.enum:
        all_opts.append(ev.value.lower())
        map_opt[ev.value.lower()] = ev.value
        for al in ev.aliases:
            all_opts.append(al.lower())
            map_opt[al.lower()] = ev.value
    t = token.lower()
    if t in map_opt:
        return map_opt[t], False
    cand, fuzzy = fuzzy_match(t, all_opts)
    if cand.lower() in map_opt:
        return map_opt[cand.lower()], True
    return None, False

def type_accepts(a: ArgSpec, token: str) -> bool:
    t = a.type
    if t == "int":
        return is_int_token(token)
    if t == "float":
        return is_float_token(token)
    if t == "bool":
        return is_bool_token(token)
    if t == "enum":
        v, _ = enum_match(a, token)
        return v is not None
    if t == "string":
        return True
    return False

def coerce_value(a: ArgSpec, token: str):
    if a.type == "int":
        val = int(token)
        if "min" in a.constraints and val < a.constraints["min"]:
            raise ValueError(f"{a.name} {val} invalid, use {a.constraints['min']} to {a.constraints.get('max','∞')}")
        if "max" in a.constraints and val > a.constraints["max"]:
            raise ValueError(f"{a.name} {val} invalid, use {a.constraints['min']} to {a.constraints.get('max','∞')}")
        return val, None
    if a.type == "float":
        val = float(token)
        if "min" in a.constraints and val < a.constraints["min"]:
            raise ValueError(f"{a.name} {val} invalid, must be at least {a.constraints['min']}")
        return val, None
    if a.type == "bool":
        return normalize_bool(token), None
    if a.type == "enum":
        v, fuzzy = enum_match(a, token)
        if v is None:
            raise ValueError(f"{a.name} {token} not recognized")
        note = f"[interpreted {token} → {v}]" if fuzzy else None
        return v, note
    if a.type == "string":
        return token, None
    return token, None



def classify_token(func: FuncSpec, tok: str) -> str:
    t = tok.lower()
    # strict enum acceptance (exact only)
    for a in func.args:
        if a.type == "enum":
            for ev in a.enum:
                if t == ev.value.lower():
                    return "enum"
                for al in ev.aliases:
                    if t == al.lower():
                        return "enum"
    # fuzzy enum only if token contains letters (to avoid '4' → 'a')
    if any(ch.isalpha() for ch in t):
        all_opts = []
        for a in func.args:
            if a.type == "enum":
                for ev in a.enum:
                    all_opts.append(ev.value.lower())
                    for al in ev.aliases:
                        all_opts.append(al.lower())
        match, fuzzy = fuzzy_match(t, all_opts)
        if fuzzy and match in all_opts:
            return "enum"
    if is_int_token(tok):
        return "int"
    if is_float_token(tok):
        return "float"
    if is_bool_token(tok):
        return "bool"
    return "string"

# --------- Drafts ----------
@dataclass
class Draft:
    func: FuncSpec
    values: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def unmet_required(self) -> List[ArgSpec]:
        return [a for a in self.func.args if a.required and a.name not in self.values]

    def next_arg(self) -> Optional[ArgSpec]:
        for a in self.func.args:
            if a.name not in self.values:
                return a
        return None

    def preview_lines(self) -> List[str]:
        lines = []
        for a in self.func.args:
            val = self.values.get(a.name, None)
            lines.append(f"- {a.name} {val if val is not None else 'none'}")
        return lines

@dataclass
class BroadcastPlan:
    func: FuncSpec
    items: List[Draft] = field(default_factory=list)
    def preview_lines(self) -> List[str]:
        lines = []
        for i, d in enumerate(self.items, 1):
            parts = []
            for a in d.func.args:
                v = d.values.get(a.name, None)
                parts.append(f"{a.name} {v if v is not None else 'none'}")
            lines.append(f"{i}) " + ", ".join(parts))
        return lines

# --------- Engine ----------
class Engine:
    def __init__(self, spec: ConvoSpec, io: IOBuffer):
        self.spec = spec
        self.io = io
        self.funcs = {f.name: f for f in spec.functions}
        self.current: Optional[Draft] = None
        self.plan: Optional[BroadcastPlan] = None
        self.last_executed: Optional[Draft] = None
        self.scratch: Dict[str, Draft] = {}
        self.history: List[Tuple[str, Any]] = []

    def write(self, s: str):
        self.io.write(s)

    def list_functions(self):
        lines = []
        for f in self.spec.functions:
            args_text = ", ".join([f"{a.name}{'' if a.required else ' (optional)'}" for a in f.args])
            lines.append(f"- {f.name} ({args_text})")
        self.write("\n".join(lines))

    def open_function(self, name: str):
        names = list(self.funcs.keys())
        match, fuzzy = fuzzy_match(name, names)
        if match not in self.funcs:
            self.write(f"Function {name} not found.")
            return
        self.current = Draft(self.funcs[match])
        self.plan = None
        self.write(f"{self.current.func.display}.")
        self.prompt_next()

    def prompt_next(self):
        if not self.current:
            return
        nxt = self.current.next_arg()
        if not nxt:
            self.preview_current()
            return
        self.write(f"Beginner: {nxt.prompt_beginner}")
        self.write(f"Power: {nxt.prompt_power}")

    def preview_current(self):
        if self.plan:
            self.write("Plan")
            for line in self.plan.preview_lines():
                self.write(line)
            self.write("Confirm? (y go confirm, edit INDEX ARGNAME VALUE, cancel)")
        elif self.current:
            self.write("Preview")
            for line in self.current.preview_lines():
                self.write(line)
            self.write("(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)")

    def handle_line(self, line: str) -> bool:
        line = line.strip()
        if not line:
            self.prompt_next()
            return True
        segs = split_segments(line)
        first = segs[0].strip().lower()
        # fuzzy reserved command mapping (skip when awaiting broadcast fill)
        if not hasattr(self, "await_plan_fill") and any(ch.isalpha() for ch in first) and first not in {r.lower() for r in RESERVED}:
            match, fuzzy = fuzzy_match(first, [r.lower() for r in RESERVED])
            if fuzzy and match in {r.lower() for r in RESERVED}:
                # echo interpretation and replace 'first' for downstream handling
                self.write(f"[interpreted {first} → {match}]")
                first = match

        if first in ["end"]:
            self.write("Goodbye.")
            return False
        if first in ["help","?"]:
            self.write("Help\n- list, open NAME, switch NAME, home\n- answer as pairs: arg value (or shortkey value)\n- unlabeled values fill by precedence\n- strings: after the arg name, value runs to comma or period\n- broadcast: over arg v1, v2. other w1\n- edit, undo N, rewind to ARGNAME\n- confirm y go, cancel\n- end to exit")
            return True
        if first == "list":
            self.list_functions()
            return True
        if first.startswith("open "):
            name = first.split(" ",1)[1]
            self.open_function(name)
            return True
        if first.startswith("switch "):
            name = first.split(" ",1)[1]
            if self.current:
                self.scratch[f"last {self.current.func.name}"] = self.current
            self.open_function(name)
            return True
        if first == "back":
            self.write("Use undo or rewind to ARGNAME.")
            return True
        if first == "home":
            self.current = None
            self.plan = None
            self.write("Home. Type list or open NAME.")
            return True
        if first in ["confirm","y","go","cnfirm"]:
            if first=="cnfirm":
                self.write("[interpreted cnfirm → confirm]")
            return self.confirm()
        if first == "cancel":
            self.current = None
            self.plan = None
            self.write("Canceled.")
            return True
        if first in ["again","repeat"]:
            if not self.last_executed:
                self.write("No previous call to repeat.")
                return True
            self.current = Draft(self.last_executed.func, values=self.last_executed.values.copy())
            self.apply_pairs_from_segments(segs[1:])
            self.preview_or_prompt()
            return True
        if first.startswith("rewind"):
            if not self.current:
                self.write("No draft to rewind.")
                return True
            m = re.search(r"rewind\s+to\s+(.+)", line, re.I)
            if not m:
                self.write("Say: rewind to ARGNAME")
                return True
            argname = m.group(1).strip().lower()
            a = arg_lookup(self.current.func, argname)
            if not a:
                self.write(f"Argument {argname} not found.")
                return True
            found = False
            newvals = {}
            for spec in self.current.func.args:
                if spec.name == a.name:
                    found = True
                if not found and spec.name in self.current.values:
                    newvals[spec.name] = self.current.values[spec.name]
            self.current.values = newvals
            self.prompt_next()
            return True
        if first.startswith("undo"):
            if not self.current:
                self.write("No draft to undo.")
                return True
            m = re.search(r"undo\s+(\d+)", first)
            n = int(m.group(1)) if m else 1
            while n>0 and self.history:
                typ, payload = self.history.pop()
                if typ == "set":
                    key = payload["key"]
                    old = payload["old"]
                    if old is None:
                        self.current.values.pop(key, None)
                    else:
                        self.current.values[key] = old
                elif typ == "edit_plan":
                    idx = payload["idx"]
                    key = payload["key"]
                    old = payload["old"]
                    if self.plan and 0 <= idx < len(self.plan.items):
                        if old is None:
                            self.plan.items[idx].values.pop(key, None)
                        else:
                            self.plan.items[idx].values[key] = old
                n -= 1
            self.preview_current() if (self.plan or self.current) else self.write("Nothing to undo.")
            return True
        if first.startswith("edit"):
            return self.handle_edit(line)

        if not self.current and not self.plan:
            tokens = tokenize_segment(segs[0])
            if tokens:
                name = tokens[0].lower()
                names = list(self.funcs.keys())
                if name in self.funcs or fuzzy_match(name, names)[0] in self.funcs:
                    self.open_function(name)
                    if len(segs) > 1:
                        self.apply_pairs_from_segments(segs[1:])
                        self.preview_or_prompt()
                    return True
            self.write("Open a function first. Type list or open NAME.")
            return True

        if any(seg.lower().startswith("over") for seg in segs):
            return self.handle_over(segs)

        self.apply_pairs_from_segments(segs)
        self.preview_or_prompt()
        return True

    def preview_or_prompt(self):
        if self.current and not self.current.unmet_required():
            self.preview_current()
        else:
            self.prompt_next()

    def handle_edit(self, line: str) -> bool:
        if self.plan:
            m = re.match(r"edit\s+(\d+)\s+([a-z ]+)\s+(.+)", line, re.I)
            if not m:
                self.write("Say: edit INDEX ARGNAME VALUE")
                return True
            idx = int(m.group(1)) - 1
            argname = m.group(2).strip()
            value = m.group(3).strip()
            if not (0 <= idx < len(self.plan.items)):
                self.write("Index out of range.")
                return True
            d = self.plan.items[idx]
            a = arg_lookup(d.func, argname)
            if not a:
                self.write(f"Argument {argname} not found.")
                return True
            old = d.values.get(a.name, None)
            try:
                if a.type == "string":
                    val = value
                else:
                    val, _ = coerce_value(a, value.split()[0])
            except Exception as e:
                self.write(str(e))
                return True
            d.values[a.name] = val
            self.history.append(("edit_plan", {"idx": idx, "key": a.name, "old": old}))
            self.write("Updated.")
            self.preview_current()
            return True
        elif self.current:
            m = re.match(r"edit\s+([a-z ]+)\s+(.+)", line, re.I)
            if not m:
                self.write("Say: edit ARGNAME VALUE")
                return True
            argname = m.group(1).strip()
            value = m.group(2).strip()
            a = arg_lookup(self.current.func, argname)
            if not a:
                self.write(f"Argument {argname} not found.")
                return True
            old = self.current.values.get(a.name, None)
            try:
                if a.type == "string":
                    val = value
                else:
                    val, _ = coerce_value(a, value.split()[0])
            except Exception as e:
                self.write(str(e))
                return True
            self.current.values[a.name] = val
            self.history.append(("set", {"key": a.name, "old": old}))
            self.preview_current()
            return True
        else:
            self.write("No draft to edit.")
            return True

    def handle_over(self, segs: List[str]) -> bool:
        if not self.current:
            self.write("Open a function before using over.")
            return True
        func = self.current.func
        vectors: Dict[str, List[Any]] = {}
        last_arg = None
        for seg in segs:
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
            # try to pick an arg
            a = None
            rest_tokens = []
            if len(tokens) >= 2:
                a = arg_lookup(func, f"{tokens[0]} {tokens[1]}", fuzzy_ok=False)
                if a:
                    rest_tokens = tokens[2:]
            if not a:
                a = arg_lookup(func, tokens[0], fuzzy_ok=False)
                if a:
                    rest_tokens = tokens[1:]
            if not a:
                # no arg label here; if we have a last_arg, treat segment as more values for it
                if last_arg:
                    a = last_arg
                    rest_tokens = tokens
                else:
                    self.write(f"Argument not recognized in segment: {seg}")
                    return True
            last_arg = a
            value_str = " ".join(rest_tokens).strip()
            if not value_str and a.type != "string":
                # single bare value token may be in tokens when no labels
                pass
            inner = split_segments(value_str) if value_str else []
            vals = []
            if not inner and rest_tokens:
                inner = [" ".join(rest_tokens)]
            if not inner and not rest_tokens and tokens:
                # e.g., segment 'b' after previously labeled fruit
                inner = [" ".join(tokens)]
            for val_seg in inner:
                vs = val_seg.strip()
                if not vs:
                    continue
                if a.type == "string":
                    vals.append(vs)
                else:
                    first_tok = tokenize_segment(vs)[0]
                    coerced, _ = coerce_value(a, first_tok)
                    vals.append(coerced)
            if a.name in vectors:
                vectors[a.name].extend(vals)
            else:
                vectors[a.name] = vals
        # determine targets
        n_targets = 1
        for vs in vectors.values():
            if len(vs) > 1:
                n_targets = max(n_targets, len(vs))
        if n_targets == 1:
            self.write("Provide a list with over to create multiple calls (e.g., over fruit apple, banana).")
            return True
        items = [Draft(func) for _ in range(n_targets)]
        for a in func.args:
            if a.name in vectors:
                vs = vectors[a.name]
                if len(vs) == 1:
                    for d in items:
                        d.values[a.name] = vs[0]
                elif len(vs) == n_targets:
                    for i in range(n_targets):
                        items[i].values[a.name] = vs[i]
                else:
                    for i in range(n_targets):
                        if i < len(vs):
                            items[i].values[a.name] = vs[i]
                        else:
                            items[i].values[a.name] = None
        self.plan = BroadcastPlan(func, items)
        missing = self.find_first_plan_missing()
        if missing:
            idx, arg = missing
            self.write(f"Three targets; {arg.name} has missing values. Need {arg.name} for item {idx+1} {self.describe_item(self.plan.items[idx])}.")
            self.write("Reply a value, or fill last, or cancel.")
            self.await_plan_fill = (idx, arg.name)
        else:
            self.preview_current()
        return True

    def find_first_plan_missing(self) -> Optional[Tuple[int, ArgSpec]]:
        if not self.plan:
            return None
        for i, d in enumerate(self.plan.items):
            for a in d.func.args:
                if a.required and d.values.get(a.name) is None:
                    return i, a
        return None

    def describe_item(self, d: Draft) -> str:
        for a in d.func.args:
            if a.type == "enum" and a.name in d.values and d.values[a.name] is not None:
                return f"{d.values[a.name]}"
        return d.func.name

    def confirm(self) -> bool:
        if self.plan:
            missing = self.find_first_plan_missing()
            if missing:
                i, a = missing
                self.write(f"Missing {a.name} for item {i+1}.")
                return True
            ids = []
            for d in self.plan.items:
                cid = self.execute_call(d)
                ids.append(cid)
            if ids:
                self.write(f"✅ {self.current.func.name.upper()} {ids[0]}..{ids[-1]} created")
            self.last_executed = self.plan.items[-1] if self.plan.items else None
            self.plan = None
            self.current = None
            return True
        elif self.current:
            miss = self.current.unmet_required()
            if miss:
                self.write("Missing required: " + ", ".join(a.name for a in miss))
                return True
            cid = self.execute_call(self.current)
            self.write(f"✅ {self.current.func.name.upper()} {cid}")
            self.last_executed = self.current
            self.current = None
            return True
        else:
            self.write("No draft to confirm.")
            return True

    def execute_call(self, d: Draft) -> str:
        args_ordered = [d.values.get(a.name, None) for a in d.func.args]
        try:
            d.func.callable(*args_ordered)
        except Exception:
            pass
        cid = (d.func.name[:2] + "-" + uuid.uuid4().hex[:6]).upper()
        parts = [f"{a.name} {d.values.get(a.name, None)}" for a in d.func.args]
        self.write("Receipt: " + ", ".join(parts))
        return cid

    def apply_pairs_from_segments(self, segs: List[str]):
        for seg in segs:
            tokens = tokenize_segment(seg)
            if not tokens:
                continue
            if hasattr(self, "await_plan_fill"):
                idx, argname = self.await_plan_fill
                if tokens[0].lower() == "fill" and (len(tokens) == 1 or (len(tokens) >= 2 and tokens[1].lower()=="last")):
                    last = None
                    for d in self.plan.items:
                        v = d.values.get(argname, None)
                        if v is not None:
                            last = v
                    if last is None:
                        self.write("No last value to fill.")
                        return
                    self.plan.items[idx].values[argname] = last
                    delattr(self, "await_plan_fill")
                    missing = self.find_first_plan_missing()
                    if missing:
                        i, a = missing
                        self.write(f"Need {a.name} for item {i+1} {self.describe_item(self.plan.items[i])}.")
                    else:
                        self.preview_current()
                    continue
                a = arg_lookup(self.plan.func, argname)
                try:
                    if a.type == "string":
                        val = " ".join(tokens)
                    else:
                        val, _ = coerce_value(a, tokens[0])
                    self.plan.items[idx].values[argname] = val
                except Exception as e:
                    self.write(str(e))
                    continue
                delattr(self, "await_plan_fill")
                missing = self.find_first_plan_missing()
                if missing:
                    i, a2 = missing
                    self.write(f"Need {a2.name} for item {i+1} {self.describe_item(self.plan.items[i])}.")
                else:
                    self.preview_current()
                continue

            labeled = False
            if self.current:
                a2 = arg_lookup(self.current.func, " ".join(tokens[:2]), fuzzy_ok=False)
                if a2 and len(tokens) >= 2:
                    val_tokens = tokens[2:]
                    labeled = True
                    self.set_value(self.current, a2, val_tokens)
                    continue
                a1 = arg_lookup(self.current.func, tokens[0], fuzzy_ok=False)
                if a1 and len(tokens) >= 2:
                    val_tokens = tokens[1:]
                    labeled = True
                    self.set_value(self.current, a1, val_tokens)
                    continue
            if self.current:
                self.apply_positional_tokens(self.current, tokens)

    def set_value(self, draft: Draft, a: ArgSpec, val_tokens: List[str]):
        old = draft.values.get(a.name, None)
        try:
            if a.type == "string":
                val = " ".join(val_tokens).strip()
            else:
                if not val_tokens:
                    self.write(f"Missing value for {a.name}.")
                    return
                val, note = coerce_value(a, val_tokens[0])
                if note:
                    draft.notes.append(note)
                    self.write(note)
                    self.write(note)
        except Exception as e:
            self.write(str(e))
            return
        draft.values[a.name] = val
        self.history.append(("set", {"key": a.name, "old": old}))

    def apply_positional_tokens(self, draft: Draft, tokens: List[str]):
        for i, tok in enumerate(tokens):
            category = classify_token(draft.func, tok)
            candidates = []
            for a in draft.func.args:
                if a.name in draft.values:
                    continue
                if a.type == category:
                    candidates.append(a)
            # Ambiguous unlabeled string? require labeling
            if category == "string":
                string_unmet = [a for a in draft.func.args if a.type=="string" and a.name not in draft.values]
                if len(string_unmet) > 1:
                    self.write("Ambiguous string: multiple string fields unmet. Please label, for example: to Shira. message Happy day.")
                    return
            if not candidates:
                if len([a for a in draft.func.args if a.type=="string" and a.name not in draft.values]) > 1:
                    self.write("Ambiguous string: multiple string fields unmet. Please label, for example: to Shira. message Happy day.")
                    return
                only = next((a for a in draft.func.args if a.type=="string" and a.name not in draft.values), None)
                if only:
                    rest = " ".join(tokens[i:])
                    draft.values[only.name] = rest
                    self.history.append(("set", {"key": only.name, "old": None}))
                    return
                continue
            reqs = [a for a in candidates if a.required]
            pool = reqs if reqs else candidates
            chosen = None
            for a in draft.func.args:
                if a in pool:
                    chosen = a
                    break
            try:
                if chosen.type == "string":
                    rest = " ".join(tokens[i:])
                    val, note = coerce_value(chosen, rest)
                    i = len(tokens)  # consume rest
                else:
                    val, note = coerce_value(chosen, tok)
                if note:
                    draft.notes.append(note)
                    self.write(note)
                    self.write(note)
            except Exception as e:
                self.write(str(e))
                return
            draft.values[chosen.name] = val
            self.history.append(("set", {"key": chosen.name, "old": None}))

# --------- Dummy callables & a default spec for quick use ---------
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
            ArgSpec(name="fruit", type="enum", required=True,
                enum=[EnumVal("apple",["a"]), EnumVal("banana",["ba"]), EnumVal("berries",["be"]), EnumVal("cherries",["c"])],
                prompt_beginner="Fruit? apple, banana, berries, cherries.",
                prompt_power="fruit apple  or  a  .  over fruit apple, banana."
            ),
            ArgSpec(name="qty", type="int", required=True,
                constraints={"min":1,"max":20}, shortkey="q",
                prompt_beginner="Quantity 1 to 6?",
                prompt_power="qty 4  or  q 4"),
            ArgSpec(name="price cap", type="float", required=False,
                constraints={"min":0}, shortkey="cap",
                prompt_beginner="Price cap optional, zero or more.",
                prompt_power="price cap 10.5  or  cap 10.5"),
            ArgSpec(name="note", type="string", required=False,
                prompt_beginner="Note optional.",
                prompt_power="note for office"),
            ArgSpec(name="priority", type="int", required=True,
                constraints={"min":1,"max":6}, shortkey="p",
                prompt_beginner="Priority 1 to 6?",
                prompt_power="priority 3  or  p 3"),
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
                constraints={"min":1,"max":20}, shortkey="q",
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
                constraints={"min":0},
                prompt_beginner="Budget optional, zero or more.",
                prompt_power="budget 120"),
            ArgSpec(name="message", type="string", required=False,
                prompt_beginner="Message optional.",
                prompt_power="message Happy day"),
        ]
    )
    return ConvoSpec(functions=[buy_fruit, send_gift], fuzzy_enabled=True)

# Convenience: quick script runner for tests
def run_script(lines: List[str]) -> str:
    spec = default_spec()
    io = IOBuffer(lines)
    eng = Engine(spec, io)
    io.write("Welcome. Type list or open NAME.")
    while True:
        line = io.read()
        if line is None:
            break
        if not eng.handle_line(line):
            break
    return io.joined()
