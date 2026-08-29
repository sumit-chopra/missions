.PHONY: setup install-hooks test run run-chat run-vault docker docker-chat docker-vault lint format

setup:
	uv sync --all-extras

install-hooks:
	uv run pre-commit install

test:
	uv run pytest -q

run-chat:
	uv run --quiet python src/missions/glass_cockpit/chat.py

run-vault:
	uv run --quiet uvicorn missions.the_vault.app:create_app --factory --reload --port 8000

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