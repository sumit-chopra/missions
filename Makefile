.PHONY: setup test run docker

setup:
	uv sync

install-hooks:
	uv run pre-commit install

test:
	uv run pytest -q

run:
	uv run python src/missions/glass_cockpit/chat.py

docker:
	docker compose run --rm chat

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .