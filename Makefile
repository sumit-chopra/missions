.PHONY: help setup install-hooks test eval eval-vault \
        run up down logs run-chat run-vault run-copilot \
        docker-chat docker-vault docker-copilot lint format

PROMPT ?= Draft a follow-up plan for application \#A-1423, stuck in verification for 4 days

.DEFAULT_GOAL := help

help:               ## list targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sort \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- core targets -------------------------------------------

setup:              ## install every extra + dev tools into .venv
	uv sync --all-extras

test:               ## run the unit suite (mocked, no network, no API key)
	uv run pytest -q

eval:               ## grade The Vault: raw vs. rag + citation recall (needs OPENAI_API_KEY)
	uv run --quiet python src/missions/the_vault/eval/run.py

run: up             ## docker compose up: chat + vault + prometheus, one command

# --- docker ----------------------------------------------------------------

up:                 ## docker compose up --build (chat + vault + prometheus)
	docker compose up --build

down:               ## stop the stack and drop the vault-data volume
	docker compose down -v

logs:               ## follow compose logs
	docker compose logs -f

docker-chat:        ## interactive chat in its container (compose run)
	docker compose run --build --rm chat

docker-vault:       ## just the vault service in a container
	docker compose up --build vault

docker-copilot:     ## run the co-pilot in its container against PROMPT="..."
	docker compose run --build --rm --no-deps copilot "$(PROMPT)"

# --- run a single mission locally (no docker) -----------------------------

run-chat:           ## Mission 1 — terminal chat REPL
	uv run --quiet python -m missions.glass_cockpit.chat

run-vault:          ## Mission 2 — RAG service on :8000 (--reload)
	uv run --quiet uvicorn missions.the_vault.app:create_app --factory --reload --port 8000

run-copilot:        ## Mission 3 — agent against PROMPT="..."
	uv run --quiet python -m missions.co_pilot.pilot "$(PROMPT)"

# --- aliases / extras ----------------------------------------------------

eval-vault: eval

install-hooks:      ## install the ruff pre-commit hooks
	uv run pre-commit install

lint:               ## ruff check + format --check
	uv run ruff check .
	uv run ruff format --check .

format:             ## ruff format
	uv run ruff format .
