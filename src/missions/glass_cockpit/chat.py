"""Minimal terminal chat loop backed by an OpenAI model."""

from dotenv import load_dotenv

from missions.glass_cockpit.llm_client import LLMClient, LLMInitialisationError, LLMRequestError

EXIT_KEYWORDS = ["exit", "quit", "bye"]


def chat() -> int:
    load_dotenv()
    print("Glass Cockpit — type a message. Ctrl+C or 'exit' to quit.")

    try:
        client = LLMClient()
    except LLMInitialisationError as exc:
        print(f"Could not initialise the LLM client: {exc}")
        return 1

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input.lower() in EXIT_KEYWORDS:
            return 0

        try:
            reply = client.send(user_input)
        except LLMRequestError as exc:
            print(f"error: {exc}")
            continue

        print(reply)


if __name__ == "__main__":
    raise SystemExit(chat())
