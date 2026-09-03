# Multi-stage only to grab the Caddy binary without adding an apt repo --
# simpler and more reliable at build time than installing Caddy via apt.
FROM caddy:2-alpine AS caddy

FROM python:3.11-slim
WORKDIR /app

COPY --from=caddy /usr/bin/caddy /usr/local/bin/caddy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the classifier at build time so the image is self-contained and the
# artifact is reproducible from the committed labeled dataset -- nothing
# binary needs to be committed to git for the app to work out of the box.
RUN python scripts/train_classifier.py

# SQLite lives on a mounted volume so request history survives redeploys;
# the classifier artifact does not need to (it's rebuilt into the image).
ENV AUTOPILOT_DB_PATH=/data/autopilot.db
RUN mkdir -p /data

COPY deploy/Caddyfile /etc/caddy/Caddyfile
RUN chmod +x deploy/start.sh

EXPOSE 8080
CMD ["deploy/start.sh"]
