# Glass Cockpit

A minimal terminal-based conversational LLM client.

## Current Status

The project currently contains a basic terminal chat loop with no LLM wired
up yet — it just echoes back what you type. It:

- Accepts user input from the terminal
- Exits on `exit`, `quit`, `bye`, `Ctrl+C`, or `Ctrl+D`

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose (optional, only needed to run in a container)

## Setup

```bash
make setup
# or: uv sync
```

## Running the app

```bash
make run
# or: uv run python src/missions/glass_cockpit/chat.py
```

### Running with Docker

```bash
make docker
# or: docker compose run --rm chat
```

Use `docker compose run`, not `up` — `up` streams container logs but doesn't
forward your terminal's stdin into the container, so an interactive app like
this one won't see your input. `run` attaches your terminal properly.

## Running tests

```bash
make test
# or: uv run pytest -q
```

Tests are located under `tests/` (e.g., `test_chat.py` testing terminal chat loop behavior).

## Linting

```bash
make lint
# or: uv run ruff check . && uv run ruff format --check .
```

To automatically format code:

```bash
make format
# or: uv run ruff format .
```

## Project Structure

```text
.
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── README.md
├── src/
│   └── missions/
│       ├── __init__.py
│       └── glass_cockpit/
│           ├── __init__.py
│           └── chat.py
└── tests/
    └── test_chat.py
```
