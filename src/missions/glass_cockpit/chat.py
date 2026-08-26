"""Minimal terminal chat loop."""

EXIT_KEYWORDS = ["exit", "quit", "bye"]


def chat() -> int:
    print("Glass Cockpit — type a message. Ctrl+C or 'exit' to quit.")

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

        print(f"you said: {user_input}")


if __name__ == "__main__":
    chat()
