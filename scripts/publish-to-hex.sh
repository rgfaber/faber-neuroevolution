#!/usr/bin/env bash
set -euo pipefail

# Publish faber_neuroevolution to hex.pm
# Usage: ./scripts/publish-to-hex.sh
#
# Requires: ~/.config/rebar3/hex.config with api_key set

cd "$(dirname "$0")/.."

VERSION=$(grep -oP '{vsn,\s*"\K[^"]+' src/faber_neuroevolution.app.src)
echo "==> Publishing faber_neuroevolution v${VERSION} to hex.pm..."

echo "==> Building faber_neuroevolution..."
rebar3 compile

echo "==> Running tests..."
rebar3 eunit

echo "==> Building docs..."
rebar3 ex_doc

echo "==> Publishing to hex.pm..."
rebar3 hex publish --yes

echo "==> Done! faber_neuroevolution v${VERSION} published to hex.pm"
echo "==> View at: https://hex.pm/packages/faber_neuroevolution"
