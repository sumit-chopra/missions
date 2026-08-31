.PHONY: setup install-hooks test eval-vault run run-chat run-vault run-copilot docker docker-chat docker-vault lint format

PROMPT ?= Draft a follow-up plan for application \#A-1423, stuck in verification for 4 days

setup:
	uv sync --all-extras

install-hooks:
	uv run pre-commit install

test:
	uv run pytest -q

eval-vault:
	uv run python src/missions/the_vault/eval/run.py

run-chat:
	uv run --quiet python src/missions/glass_cockpit/chat.py

run-vault:
	uv run --quiet uvicorn missions.the_vault.app:create_app --factory --reload --port 8000

run-copilot:
	uv run --quiet python -m missions.co_pilot.pilot "$(PROMPT)"

docker-chat:
	docker compose run --build --rm chat

docker-vault:
	docker compose up --build vault

run:
	docker compose up --build

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .