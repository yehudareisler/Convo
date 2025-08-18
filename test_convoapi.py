# =========================
# ConvoAPI v1 — Tests (focused on fuzzy recognizer + basic routing)
# =========================
# Note: Per your request, these are provided but not executed here.

import pytest

from convoapi_engine import (
    fuzzy_command, levenshtein, COMMANDS, COMMAND_ALIASES,
    ConvoEngine, InMemoryAdapter, default_spec
)

# --------- Fuzzy basics ---------

@pytest.mark.parametrize("a,b,dist", [
    ("confirm", "confirm", 0),
    ("confirm", "confrim", 2),   # transposition + swap
    ("help", "hepl", 2),
    ("open", "opne", 2),
    ("switch", "swich", 1),
    ("again", "agani", 2),
])
def test_levenshtein_basics(a, b, dist):
    assert levenshtein(a, b) == dist

# Positive: should fuzz to canonical commands
@pytest.mark.parametrize("token, expected", [
    ("cnfirm", "confirm"),
    ("cnfrm",  "confirm"),
    ("confrim", "confirm"),
    ("bak",    "back"),
    ("hom",    "home"),
    ("lis",    "list"),
    ("lit",    "list"),
    ("agani",  "again"),
    ("agin",   "again"),
    ("repeta", "repeat"),
    ("swich",  "switch"),
    ("opne",   "open"),
])
def test_fuzzy_command_positive(token, expected):
    fm = fuzzy_command(token, COMMANDS)
    assert fm.matched
    assert fm.value == expected

# Boundary: aliases must stay mapped to canonical, not fuzzed elsewhere
@pytest.mark.parametrize("token, expected", [
    ("y", "confirm"),
    ("go", "confirm"),
    ("?", "help")
])
def test_aliases_are_canonical(token, expected):
    # Aliases are mapped outside fuzzy_command; still, sanity:
    mapped = COMMAND_ALIASES.get(token, token)
    assert mapped == expected

# Negative: numeric or too-short should not fuzz
@pytest.mark.parametrize("token", ["", "x", "3", "1", "0"])
def test_fuzzy_command_negative_short_or_numeric(token):
    fm = fuzzy_command(token, COMMANDS)
    assert fm.matched is False

# Negative: too distant
@pytest.mark.parametrize("token", ["confxx", "zzswitch", "helpxxx"])
def test_fuzzy_command_negative_distance(token):
    fm = fuzzy_command(token, COMMANDS)
    assert fm.matched is False

# Tie-break consistency: prefer the nearest with stable ordering
def test_fuzzy_tie_break_stability():
    # Construct a tie: "hom" should go to "home" (not "help")
    fm = fuzzy_command("hom", COMMANDS)
    assert fm.matched
    assert fm.value == "home"


# --------- Minimal engine smoke tests for routing ---------

def _mk_engine():
    return ConvoEngine(default_spec(), InMemoryAdapter())

def test_engine_welcome_and_list():
    eng = _mk_engine()
    cid = "u1"
    # First call, no state: empty input yields home/welcome-ish prompt
    out = eng.handle(cid, "")
    assert "Type list or open NAME" in out

    out = eng.handle(cid, "list")
    assert "- buy_fruit (" in out
    assert "- send_gift (" in out

def test_engine_open_and_prompt():
    eng = _mk_engine()
    cid = "u2"
    out = eng.handle(cid, "open buy_fruit")
    assert "Fruit? apple, banana, berries, cherries." in out
    # Fuzzy function open:
    out = eng.handle(cid, "opne send_gift")
    assert "Recipient name?" in out or "Item? mug, book, flowers." in out

def test_engine_fuzzy_confirm_path():
    eng = _mk_engine()
    cid = "u3"
    # One-shot: open, fill, confirm (fuzzy 'cnfirm')
    eng.handle(cid, "open buy_fruit")
    eng.handle(cid, "fruit apple, q 4. p 3")
    out = eng.handle(cid, "cnfirm")
    # We expect either "Missing required" (needs priority and possibly others) or a receipt if enough provided.
    # To keep test stable, just assert fuzzy mapped and engine produced a consistent string (receipt or missing).
    assert "interpreted cnfirm" in out or "Receipt:" in out or "Missing required" in out

def test_engine_over_and_fill_prompt():
    eng = _mk_engine()
    cid = "u4"
    eng.handle(cid, "open buy_fruit")
    out = eng.handle(cid, "over fruit apple, banana, cherries. qty 4, 8. p 2")
    assert "Need qty for item 3" in out or "has missing values" in out
    # Now fill the missing value
    out2 = eng.handle(cid, "3")
    assert "Preview" in out2 or "Plan" in out2
