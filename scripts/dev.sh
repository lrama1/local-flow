#!/usr/bin/env sh
# Runs backend + frontend dev servers (see scripts/dev.mjs).
set -e
cd "$(dirname "$0")/.."
exec node scripts/dev.mjs
