#!/usr/bin/env bash
#
# Installs the bridge as a systemd user service.
#
# It never touches the configuration of whatever agent or client will use the
# bridge: pointing that at the bridge stays a separate, deliberate step, so a
# failed install cannot take your assistant offline.
#
#   ./scripts/install.sh            install and start the service
#   ./scripts/install.sh --check    check prerequisites and change nothing
#   ./scripts/install.sh --uninstall  stop and remove the service

set -euo pipefail

SERVICE_NAME="claude-model-bridge"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_FILE="$UNIT_DIR/$SERVICE_NAME.service"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${CLAUDE_BRIDGE_HOST:-127.0.0.1}"
PORT="${CLAUDE_BRIDGE_PORT:-8765}"

red() { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
note() { printf '%s\n' "$*"; }

fail() {
  red "FAILED: $*"
  exit 1
}

check_prerequisites() {
  local problems=0

  if [ "$(id -u)" -eq 0 ]; then
    red "Running as root. This is a user service, and the CLI login belongs to your user."
    problems=$((problems + 1))
  fi

  if command -v uv >/dev/null 2>&1; then
    green "uv found: $(command -v uv)"
  else
    red "uv not found. See https://docs.astral.sh/uv/"
    problems=$((problems + 1))
  fi

  if command -v claude >/dev/null 2>&1; then
    green "claude found: $(command -v claude)"
    if claude auth status 2>/dev/null | grep -q '"loggedIn": *true'; then
      green "claude is logged in"
    else
      red "claude is not logged in. Run: claude"
      problems=$((problems + 1))
    fi
  else
    red "claude not found on PATH"
    problems=$((problems + 1))
  fi

  # The bridge refuses to start with either of these set, so that it can only
  # ever spend the subscription rather than paid API billing.
  local credential
  for credential in ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN; do
    if [ -n "${!credential:-}" ]; then
      red "$credential is set. The bridge refuses to run while it is."
      problems=$((problems + 1))
    fi
  done

  if systemctl --user is-system-running >/dev/null 2>&1; then
    green "systemd user session available"
  else
    red "no systemd user session"
    problems=$((problems + 1))
  fi

  if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "yes" ]; then
    green "lingering enabled, so the service starts at boot"
  else
    note "NOTE: lingering is off, so the service stops when you log out."
    note "      Enable it with: sudo loginctl enable-linger $USER"
  fi

  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    note "NOTE: something already listens on port $PORT."
    note "      If that is an older copy of this service, installing will replace it."
  fi

  return "$problems"
}

write_unit() {
  # The service runs the console script from the project's own virtual
  # environment, and needs the directory holding `claude` on PATH.
  local claude_dir
  claude_dir="$(dirname "$(command -v claude)")"

  mkdir -p "$UNIT_DIR"
  if [ -f "$UNIT_FILE" ]; then
    cp "$UNIT_FILE" "$UNIT_FILE.backup"
    note "Existing unit backed up to $UNIT_FILE.backup"
  fi

  cat > "$UNIT_FILE" <<UNIT
[Unit]
Description=Claude Code model bridge
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
ExecStart=$REPO_DIR/.venv/bin/claude-code-model-bridge
WorkingDirectory=$REPO_DIR
Environment="PATH=$claude_dir:/usr/local/bin:/usr/bin:/bin"
Environment="CLAUDE_BRIDGE_HOST=$HOST"
Environment="CLAUDE_BRIDGE_PORT=$PORT"
Restart=always
RestartSec=5
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
UNIT
  green "Wrote $UNIT_FILE"
}

wait_until_serving() {
  local attempt
  for attempt in $(seq 1 30); do
    if curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

install_service() {
  check_prerequisites || fail "$? prerequisite(s) unmet. Fix the items above and run again."

  note "Building the environment..."
  (cd "$REPO_DIR" && uv sync --quiet)
  [ -x "$REPO_DIR/.venv/bin/claude-code-model-bridge" ] ||
    fail "uv sync did not produce $REPO_DIR/.venv/bin/claude-code-model-bridge"

  write_unit
  systemctl --user daemon-reload
  systemctl --user enable --now "$SERVICE_NAME.service"

  if wait_until_serving; then
    green "The bridge is serving on http://$HOST:$PORT/v1"
  else
    red "The service did not answer within 30 seconds. Recent log:"
    journalctl --user -u "$SERVICE_NAME.service" -n 20 --no-pager || true
    exit 1
  fi

  note ""
  note "Next, point a client at it. Nothing has changed outside this service."
  note "  Base URL: http://$HOST:$PORT/v1"
  note "  API key:  any string; the bridge authenticates as whoever claude is logged in as"
  note "  Models:   curl -s http://$HOST:$PORT/v1/models"
  note ""
  note "Back up whatever configuration file you edit before changing it, so"
  note "you can put it back exactly as it was."
  note ""
  note "Useful afterwards:"
  note "  systemctl --user status $SERVICE_NAME"
  note "  journalctl --user -u $SERVICE_NAME -f"
  note "  ./scripts/install.sh --uninstall"
}

uninstall_service() {
  systemctl --user disable --now "$SERVICE_NAME.service" 2>/dev/null || true
  if [ -f "$UNIT_FILE" ]; then
    rm "$UNIT_FILE"
    green "Removed $UNIT_FILE"
  else
    note "No unit file at $UNIT_FILE"
  fi
  systemctl --user daemon-reload
  green "Service removed. The repository and your client configuration are untouched."
}

case "${1:-}" in
  --check)
    if check_prerequisites; then
      green "Ready to install."
    else
      fail "$? prerequisite(s) unmet."
    fi
    ;;
  --uninstall) uninstall_service ;;
  "") install_service ;;
  *)
    note "Usage: $0 [--check | --uninstall]"
    exit 2
    ;;
esac
