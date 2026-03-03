FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir . && rm -rf /root/.cache
COPY bot.py embeds.py ./
RUN useradd -r bot
USER bot
CMD ["python", "bot.py"]
