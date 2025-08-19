import pytest

from convoapi_engine import (
    fuzzy_command, levenshtein, COMMANDS, COMMAND_ALIASES
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
