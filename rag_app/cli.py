"""Interactive chat in the terminal.  Run:  python -m rag_app.cli [--debug]"""

import argparse

from rag_app.assistant import ChatSession, RAGAssistant
from rag_app.ingest import build_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="show rewritten query and retrieved chunks")
    args = parser.parse_args()

    build_index()
    session = ChatSession(RAGAssistant())
    print("Orbitly assistant. Type 'exit' to quit, 'reset' to start a new conversation.\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"exit", "quit"}:
            break
        if msg.lower() == "reset":
            session.history.clear()
            continue
        if not msg:
            continue
        resp = session.send(msg)
        if args.debug:
            print(f"  [query] {resp.standalone_query}")
            for c in resp.chunks:
                print(f"  [chunk] {c.chunk_id}  score={c.score:.3f}")
            print(f"  [{resp.latency_s:.2f}s, {resp.prompt_tokens}+{resp.completion_tokens} tokens]")
        print(f"orbi> {resp.answer}\n")


if __name__ == "__main__":
    main()
