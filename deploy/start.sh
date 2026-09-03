#!/bin/bash
# Starts both processes on internal-only ports, then runs Caddy in the
# foreground on Railway's $PORT as the single public entry point. If either
# background process dies, the container should die too rather than serve
# half a system silently -- `wait -n` plus the trap makes that happen.
set -e

uvicorn api.main:app --host 127.0.0.1 --port 8001 &
API_PID=$!

streamlit run dashboard/app.py \
	--server.port 8501 \
	--server.address 127.0.0.1 \
	--server.headless true \
	--server.enableCORS false \
	--server.enableXsrfProtection false &
DASHBOARD_PID=$!

trap 'kill $API_PID $DASHBOARD_PID 2>/dev/null' TERM INT

caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &
CADDY_PID=$!

wait -n $API_PID $DASHBOARD_PID $CADDY_PID
exit_code=$?
kill $API_PID $DASHBOARD_PID $CADDY_PID 2>/dev/null
exit $exit_code
