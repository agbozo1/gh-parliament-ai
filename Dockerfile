FROM python:3.11-slim

WORKDIR /app

# Selenium needs a Firefox binary + geckodriver for the scraper. Firefox
# ESR is used instead of Chrome/chromedriver because it keeps supporting
# older host OS versions (e.g. macOS Catalina, for local dev outside
# Docker) for much longer.
ENV GECKODRIVER_VERSION=0.35.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    firefox-esr \
    curl \
    && curl -sL "https://github.com/mozilla/geckodriver/releases/download/v${GECKODRIVER_VERSION}/geckodriver-v${GECKODRIVER_VERSION}-linux64.tar.gz" \
    | tar -xz -C /usr/local/bin \
    && chmod +x /usr/local/bin/geckodriver \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV FIREFOX_BIN=/usr/bin/firefox-esr \
    GECKODRIVER_PATH=/usr/local/bin/geckodriver \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
