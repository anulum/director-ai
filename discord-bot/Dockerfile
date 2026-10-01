FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --require-hashes --no-deps -r requirements.txt && pip install --no-deps --no-cache-dir . && rm -rf /root/.cache
COPY bot.py embeds.py ./
RUN useradd -r bot
USER bot
CMD ["python", "bot.py"]
