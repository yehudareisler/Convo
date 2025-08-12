
import re
from convoapi_engine import run_script

def test_beginner_flow():
    out = run_script([
        "open buy_fruit",
        "appl",
        "4",
        "10.5",
        "for office",
        "7",
        "p 3",
        "confirm",
        "end"
    ])
    assert "interpreted appl → apple" in out
    assert "priority 7 invalid" in out
    assert "Receipt:" in out
    assert "fruit apple" in out and "qty 4" in out and "priority 3" in out

def test_unordered_send():
    out = run_script([
        "open send_gift",
        "2 flowers paper exprss yes message Happy day, to Noa Levi",
        "edit budget 120",
        "go",
        "end"
    ])
    assert "interpreted exprss → express" in out
    assert "to Noa Levi" in out
    assert "budget 120" in out

def test_broadcast_complete():
    out = run_script([
        "open buy_fruit",
        "over fruit apple, b, cherries. qty 4. p 3",
        "ba",
        "y",
        "end"
    ])
    assert "Plan" in out
    assert "created" in out

def test_broadcast_partial_fill_and_edit():
    out = run_script([
        "open buy_fruit",
        "over fruit apple, banana, cherries. qty 4, 8. p 2",
        "3",
        "edit 2 qty 5",
        "y",
        "end"
    ])
    assert "Need qty for item 3" in out or "Need qty for item 3 cherries" in out
    assert "Updated" in out or "edit 2" in out
    assert "created" in out

def test_rewind_price_cap():
    out = run_script([
        "open buy_fruit",
        "ba",
        "6",
        "15",
        "for kids",
        "4",
        "bak",
        "rewind to price cap",
        "12.99",
        "keep",
        "5",
        "confirm",
        "end"
    ])
    assert "interpreted bak → back" in out
    assert "price cap 12.99" in out
    assert "priority 5" in out

def test_power_again_switch():
    out = run_script([
        "buy_fruit",
        "fruit apple, q 4. cap 10. note office. p 3",
        "go",
        "again p 2",
        "y",
        "switch send_gift",
        "item flowers. to Dan Cohen. q 1. wrap foil. speed overnight. insurance no. message No note",
        "confirm",
        "end"
    ])
    assert "send gift" in out.lower()
    assert "to Dan Cohen" in out

def test_ambiguous_strings_forced_label():
    out = run_script([
        "open send_gift",
        "book Shira 1 paper standard yes Happy day",
        "to Shira. message Happy day. item book. q 1. wrap paper. speed standard. insurance yes",
        "y",
        "end"
    ])
    assert "Ambiguous string" in out
    assert "to Shira" in out and "message Happy day" in out

def test_error_minimal_fixes():
    out = run_script([
        "buy_fruit",
        "d 0 p 9",
        "fruit cherries, qty 2, p 6",
        "y",
        "end"
    ])
    assert "not recognized" in out or "invalid" in out
    assert "fruit cherries" in out and "qty 2" in out and "priority 6" in out
