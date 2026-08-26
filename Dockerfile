FROM python:3.11-slim

WORKDIR /app

COPY src ./src

ENTRYPOINT ["python", "src/missions/glass_cockpit/chat.py"]