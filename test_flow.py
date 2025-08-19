import uuid

from convoapi_engine import ConvoEngine, InMemoryAdapter, default_spec


def run_conversation(pairs):
    """Run a scripted conversation and require exact message matches."""
    engine = ConvoEngine(default_spec(), InMemoryAdapter())
    cid = "test"
    for user_msg, expected in pairs:
        output = engine.handle(cid, user_msg)
        assert output == expected


def patch_uuids(monkeypatch, hexes):
    it = iter(hexes)

    class Dummy:
        def __init__(self, h):
            self.hex = h

    monkeypatch.setattr(uuid, "uuid4", lambda: Dummy(next(it)))


FLOWS = {
    "beginner_flow": [
        (
            "open buy_fruit",
            'Fruit? apple, banana, berries, cherries.\n(Pro tip: write "fruit apple  or  a  .  over fruit apple, banana.")',
        ),
        (
            "appl",
            '[interpreted appl → apple]\nQuantity 1 to 20?\n(Pro tip: write "qty 4  or  q 4")',
        ),
        (
            "4",
            'Price cap optional, zero or more.\n(Pro tip: write "price cap 10.5  or  cap 10.5")',
        ),
        (
            "10.5",
            'Note optional.\n(Pro tip: write "note for office")',
        ),
        (
            "for office",
            'Priority 1 to 6?\n(Pro tip: write "priority 3  or  p 3")',
        ),
        (
            "7",
            'priority 7 invalid, use 1 to 6\nPriority 1 to 6?\n(Pro tip: write "priority 3  or  p 3")',
        ),
        (
            "p 3",
            'Preview\n- fruit: apple\n- qty: 4\n- price cap: 10.5\n- note: for office\n- priority: 3\n(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)',
        ),
        (
            "confirm",
            'Receipt: fruit apple, qty 4, price cap 10.5, note for office, priority 3\n✅ BUY_FRUIT BU-222222',
        ),
        (
            "end",
            "Goodbye.",
        ),
    ],
    "unordered_send": [
        (
            "open send_gift",
            'Item? mug, book, flowers.\n(Pro tip: write "item flowers")',
        ),
        (
            "2 flowers paper exprss yes",
            '[interpreted exprss → express]\nRecipient name?\n(Pro tip: write "to Noa Levi")',
        ),
        (
            "message Happy day, to Noa Levi",
            'Preview\n'
            '- item: flowers\n'
            '- to: Noa Levi\n'
            '- qty: 2\n'
            '- wrap: paper\n'
            '- speed: express\n'
            '- insurance: True\n'
            '- budget: none\n'
            '- message: Happy day\n'
            '(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)',
        ),
        (
            "edit budget 120",
            'Preview\n'
            '- item: flowers\n'
            '- to: Noa Levi\n'
            '- qty: 2\n'
            '- wrap: paper\n'
            '- speed: express\n'
            '- insurance: True\n'
            '- budget: 120.0\n'
            '- message: Happy day\n'
            '(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)',
        ),
        (
            "go",
            'Receipt: item flowers, to Noa Levi, qty 2, wrap paper, speed express, insurance True, budget 120.0, message Happy day\n✅ SEND_GIFT SE-333333',
        ),
        (
            "end",
            "Goodbye.",
        ),
    ],
    "broadcast_complete": [
        (
            "open buy_fruit",
            'Fruit? apple, banana, berries, cherries.\n(Pro tip: write "fruit apple  or  a  .  over fruit apple, banana.")',
        ),
        (
            "over fruit apple, ba, cherries. qty 4. p 3",
            'Plan\n'
            '1) fruit: apple, qty: 4, price cap: none, note: none, priority: 3\n'
            '2) fruit: banana, qty: 4, price cap: none, note: none, priority: 3\n'
            '3) fruit: cherries, qty: 4, price cap: none, note: none, priority: 3\n'
            'Confirm? (y go confirm, edit INDEX ARGNAME VALUE, cancel)',
        ),
        (
            "fruit banana",
            'Quantity 1 to 20?\n(Pro tip: write "qty 4  or  q 4")',
        ),
        (
            "y",
            'Receipt: fruit apple, qty 4, price cap none, note none, priority 3\n'
            'Receipt: fruit banana, qty 4, price cap none, note none, priority 3\n'
            'Receipt: fruit cherries, qty 4, price cap none, note none, priority 3\n'
            '✅ BUY_FRUIT BU-444444..BU-666666 created',
        ),
        (
            "end",
            "Goodbye.",
        ),
    ],
    "broadcast_partial_fill_and_edit": [
        (
            "open buy_fruit",
            'Fruit? apple, banana, berries, cherries.\n(Pro tip: write "fruit apple  or  a  .  over fruit apple, banana.")',
        ),
        (
            "over fruit apple, banana, cherries. qty 4, 8. p 2",
            '3 targets; qty has missing values. Need qty for item 3 cherries.\nReply a value, or fill last, or cancel.',
        ),
        (
            "3",
            'Plan\n'
            '1) fruit: apple, qty: 4, price cap: none, note: none, priority: 2\n'
            '2) fruit: banana, qty: 8, price cap: none, note: none, priority: 2\n'
            '3) fruit: cherries, qty: 3, price cap: none, note: none, priority: 2\n'
            'Confirm? (y go confirm, edit INDEX ARGNAME VALUE, cancel)',
        ),
        (
            "edit 2 qty 5",
            'Updated.\n'
            'Plan\n'
            '1) fruit: apple, qty: 4, price cap: none, note: none, priority: 2\n'
            '2) fruit: banana, qty: 5, price cap: none, note: none, priority: 2\n'
            '3) fruit: cherries, qty: 3, price cap: none, note: none, priority: 2\n'
            'Confirm? (y go confirm, edit INDEX ARGNAME VALUE, cancel)',
        ),
        (
            "y",
            'Receipt: fruit apple, qty 4, price cap none, note none, priority 2\n'
            'Receipt: fruit banana, qty 5, price cap none, note none, priority 2\n'
            'Receipt: fruit cherries, qty 3, price cap none, note none, priority 2\n'
            '✅ BUY_FRUIT BU-777777..BU-999999 created',
        ),
        (
            "end",
            "Goodbye.",
        ),
    ],
    "rewind_price_cap": [
        (
            "open buy_fruit",
            'Fruit? apple, banana, berries, cherries.\n(Pro tip: write "fruit apple  or  a  .  over fruit apple, banana.")',
        ),
        (
            "fruit banana",
            'Quantity 1 to 20?\n(Pro tip: write "qty 4  or  q 4")',
        ),
        (
            "6",
            'Price cap optional, zero or more.\n(Pro tip: write "price cap 10.5  or  cap 10.5")',
        ),
        (
            "15",
            'priority 15 invalid, use 1 to 6\nPrice cap optional, zero or more.\n(Pro tip: write "price cap 10.5  or  cap 10.5")',
        ),
        (
            "for kids",
            'Price cap optional, zero or more.\n(Pro tip: write "price cap 10.5  or  cap 10.5")',
        ),
        (
            "4",
            'Preview\n- fruit: banana\n- qty: 6\n- price cap: none\n- note: for kids\n- priority: 4\n(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)',
        ),
        (
            "bak",
            'Use undo or rewind to ARGNAME.',
        ),
        (
            "rewind to price cap",
            'Price cap optional, zero or more.\n(Pro tip: write "price cap 10.5  or  cap 10.5")',
        ),
        (
            "12.99",
            'Note optional.\n(Pro tip: write "note for office")',
        ),
        (
            "keep",
            'Priority 1 to 6?\n(Pro tip: write "priority 3  or  p 3")',
        ),
        (
            "5",
            'Preview\n- fruit: banana\n- qty: 6\n- price cap: 12.99\n- note: keep\n- priority: 5\n(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)',
        ),
        (
            "confirm",
            'Receipt: fruit banana, qty 6, price cap 12.99, note keep, priority 5\n✅ BUY_FRUIT BU-AAAAAA',
        ),
        (
            "end",
            "Goodbye.",
        ),
    ],
    "power_again_switch": [
        (
            "open buy_fruit",
            'Fruit? apple, banana, berries, cherries.\n(Pro tip: write "fruit apple  or  a  .  over fruit apple, banana.")',
        ),
        (
            "fruit apple, q 4. cap 10. note office. p 3",
            'Preview\n- fruit: apple\n- qty: 4\n- price cap: 10.0\n- note: office\n- priority: 3\n(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)',
        ),
        (
            "go",
            'Receipt: fruit apple, qty 4, price cap 10.0, note office, priority 3\n✅ BUY_FRUIT BU-BBBBBB',
        ),
        (
            "again p 2",
            'Preview\n- fruit: apple\n- qty: 4\n- price cap: 10.0\n- note: office\n- priority: 3\n(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)',
        ),
        (
            "y",
            'Receipt: fruit apple, qty 4, price cap 10.0, note office, priority 3\n✅ BUY_FRUIT BU-CCCCCC',
        ),
        (
            "switch send_gift",
            'Item? mug, book, flowers.\n(Pro tip: write "item flowers")',
        ),
        (
            "item flowers. to Dan Cohen. q 1. wrap foil. speed overnight. insurance no. message No note",
            'Preview\n'
            '- item: flowers\n'
            '- to: Dan Cohen\n'
            '- qty: 1\n'
            '- wrap: foil\n'
            '- speed: overnight\n'
            '- insurance: False\n'
            '- budget: none\n'
            '- message: No note\n'
            '(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)',
        ),
        (
            "confirm",
            'Receipt: item flowers, to Dan Cohen, qty 1, wrap foil, speed overnight, insurance False, budget none, message No note\n✅ SEND_GIFT SE-DDDDDD',
        ),
        (
            "end",
            "Goodbye.",
        ),
    ],
    "ambiguous_strings_forced_label": [
        (
            "open send_gift",
            'Item? mug, book, flowers.\n(Pro tip: write "item flowers")',
        ),
        (
            "book Shira 1 paper standard yes Happy day",
            'Use undo or rewind to ARGNAME.',
        ),
        (
            "to Shira. message Happy day. item book. q 1. wrap paper. speed standard. insurance yes",
            'Preview\n'
            '- item: book\n'
            '- to: Shira\n'
            '- qty: 1\n'
            '- wrap: paper\n'
            '- speed: standard\n'
            '- insurance: True\n'
            '- budget: none\n'
            '- message: Happy day\n'
            '(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)',
        ),
        (
            "y",
            'Receipt: item book, to Shira, qty 1, wrap paper, speed standard, insurance True, budget none, message Happy day\n✅ SEND_GIFT SE-EEEEEE',
        ),
        (
            "end",
            "Goodbye.",
        ),
    ],
    "error_minimal_fixes": [
        (
            "open buy_fruit",
            'Fruit? apple, banana, berries, cherries.\n(Pro tip: write "fruit apple  or  a  .  over fruit apple, banana.")',
        ),
        (
            "d 0 p 9",
            'priority 9 invalid, use 1 to 6\nFruit? apple, banana, berries, cherries.\n(Pro tip: write "fruit apple  or  a  .  over fruit apple, banana.")',
        ),
        (
            "fruit cherries, qty 2, p 6",
            'Preview\n- fruit: cherries\n- qty: 2\n- price cap: none\n- note: none\n- priority: 6\n(confirm y go, edit ARGNAME VALUE, undo, rewind to ARGNAME, cancel)',
        ),
        (
            "y",
            'Receipt: fruit cherries, qty 2, price cap none, note none, priority 6\n✅ BUY_FRUIT BU-FFFFFF',
        ),
        (
            "end",
            "Goodbye.",
        ),
    ],
}


def test_beginner_flow(monkeypatch):
    patch_uuids(monkeypatch, ["222222"])
    run_conversation(FLOWS["beginner_flow"])


def test_unordered_send(monkeypatch):
    patch_uuids(monkeypatch, ["333333"])
    run_conversation(FLOWS["unordered_send"])


def test_broadcast_complete(monkeypatch):
    patch_uuids(monkeypatch, ["444444", "555555", "666666"])
    run_conversation(FLOWS["broadcast_complete"])


def test_broadcast_partial_fill_and_edit(monkeypatch):
    patch_uuids(monkeypatch, ["777777", "888888", "999999"])
    run_conversation(FLOWS["broadcast_partial_fill_and_edit"])


def test_rewind_price_cap(monkeypatch):
    patch_uuids(monkeypatch, ["aaaaaa"])
    run_conversation(FLOWS["rewind_price_cap"])


def test_power_again_switch(monkeypatch):
    patch_uuids(monkeypatch, ["bbbbbb", "cccccc", "dddddd"])
    run_conversation(FLOWS["power_again_switch"])


def test_ambiguous_strings_forced_label(monkeypatch):
    patch_uuids(monkeypatch, ["eeeeee"])
    run_conversation(FLOWS["ambiguous_strings_forced_label"])


def test_error_minimal_fixes(monkeypatch):
    patch_uuids(monkeypatch, ["ffffff"])
    run_conversation(FLOWS["error_minimal_fixes"])

