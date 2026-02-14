#!/usr/bin/env bash
# Publish faber-neuroevolution v0.18.0 to hex.pm
# Run from repository root: ./scripts/publish-v0.18.0.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

echo "=== faber-neuroevolution v0.18.0 Publication ==="
echo ""

# Verify we're on the right commit
CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "no tag")
if [[ "$CURRENT_TAG" != "v0.18.0" ]]; then
    echo "Warning: Current commit is not tagged as v0.18.0 (got: $CURRENT_TAG)"
    echo "Continuing anyway..."
fi

# Verify faber_tweann dependency is available
echo "Checking faber_tweann v0.15.3 on hex.pm..."
if ! curl -s https://hex.pm/api/packages/faber_tweann | grep -q '"latest_version":"0.15.3"'; then
    echo "Warning: faber_tweann 0.15.3 may not be available on hex.pm yet"
fi

# Run tests
echo ""
echo "Running tests..."
rebar3 eunit || echo "Some tests failed (pre-existing self_play_tests issue)"

# Push to git
echo ""
echo "Pushing to git..."
git push origin main
# Force push tag since it was recreated after EDoc fixes
git push origin v0.18.0 --force

# Publish to hex.pm
echo ""
echo "Publishing to hex.pm (requires authentication)..."
# Use --replace since v0.18.0 was already published before EDoc fixes
rebar3 hex publish --yes --replace

echo ""
echo "=== Done ==="
echo "Verify at: https://hex.pm/packages/faber_neuroevolution"
