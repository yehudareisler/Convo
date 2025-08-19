from convoapi_engine import ConvoEngine, InMemoryAdapter, default_spec

eng = ConvoEngine(default_spec(), InMemoryAdapter())
cid = "cli"  # any string; this is your conversation id

while True:
    try:
        msg = input("> ")
    except EOFError:
        break
    print(eng.handle(cid, msg))
