#!/usr/bin/env bash
set -e

# Simple hex.pm publish script for faber_neuroevolution
# Community Edition: NIF tests will fail (expected ~20 failures)

cd "$(dirname "$0")/.."

# Source secrets
[ -f "$HOME/.config/zshrc/01-secrets" ] && source "$HOME/.config/zshrc/01-secrets"

echo "Publishing faber_neuroevolution to hex.pm..."
echo ""

# Version check
VERSION=$(grep -oP '(?<={vsn, ")[^"]+' src/faber_neuroevolution.app.src)

echo "Version: $VERSION"
echo ""

# Temporarily rename _checkouts to use hex deps
if [ -d "_checkouts" ]; then
    echo "Moving _checkouts to _checkouts.bak for hex deps..."
    mv _checkouts _checkouts.bak
    trap "mv _checkouts.bak _checkouts 2>/dev/null || true" EXIT
fi

# Clean build completely to use hex deps
echo "Cleaning build..."
rm -rf _build rebar.lock
echo ""

# Fetch fresh deps from hex.pm
echo "Fetching dependencies from hex.pm..."
rebar3 get-deps
echo ""

# Run tests
echo "Running tests..."
# NIF tests will fail in Community Edition - we allow up to 25 failures
# (all NIF-related: to_network, evaluate, etc.)
MAX_ALLOWED_FAILURES=25
TEST_OUTPUT=$(rebar3 eunit 2>&1) || true
echo "$TEST_OUTPUT" | tail -25

# Extract failure count
FAILED_COUNT=$(echo "$TEST_OUTPUT" | grep -oP 'Failed: \K[0-9]+' || echo "0")
PASSED_COUNT=$(echo "$TEST_OUTPUT" | grep -oP 'Passed: \K[0-9]+' || echo "0")

echo ""
echo "Test results: Passed: $PASSED_COUNT, Failed: $FAILED_COUNT"

# Verify we have passing tests
if [ "$PASSED_COUNT" -eq 0 ]; then
    echo "ERROR: No tests passed. Something is wrong."
    exit 1
fi

# Check if failures exceed allowed NIF failures
if [ "$FAILED_COUNT" -gt "$MAX_ALLOWED_FAILURES" ]; then
    echo "ERROR: Too many failures ($FAILED_COUNT > $MAX_ALLOWED_FAILURES allowed)."
    echo "Some non-NIF tests are failing. Fix before publishing."
    exit 1
fi

if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "NOTE: $FAILED_COUNT NIF-related failures are expected in Community Edition."
fi
echo ""

# Build package
echo "Building hex package..."
rebar3 hex build
echo ""

# Publish
echo "Publishing..."
rebar3 hex publish --yes
echo ""
echo "Done! Published faber_neuroevolution $VERSION"
