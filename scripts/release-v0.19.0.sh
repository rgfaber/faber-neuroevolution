#!/bin/bash
# Release script for faber_neuroevolution v0.19.0
# This script publishes to hex.pm and tags the release in git

set -e

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"
VERSION="0.19.0"

echo "=== Faber Neuroevolution Release v${VERSION} ==="
echo ""

# Step 1: Temporarily move checkout to use hex dependency
if [ -L "_checkouts/faber_tweann" ]; then
    echo "Moving _checkouts/faber_tweann to use hex dependency..."
    mv _checkouts/faber_tweann _checkouts/faber_tweann.bak
    CHECKOUT_MOVED=1
fi

# Cleanup function
cleanup() {
    if [ "$CHECKOUT_MOVED" = "1" ] && [ -e "_checkouts/faber_tweann.bak" ]; then
        echo "Restoring _checkouts/faber_tweann..."
        mv _checkouts/faber_tweann.bak _checkouts/faber_tweann
    fi
}
trap cleanup EXIT

# Step 2: Clean and rebuild
echo "Cleaning and rebuilding..."
rebar3 clean
rebar3 unlock --all
rebar3 get-deps
rebar3 compile

# Step 3: Run tests
echo ""
echo "Running tests..."
rebar3 eunit

# Step 4: Check hex authentication
echo ""
echo "Checking hex.pm authentication..."
if ! rebar3 hex user whoami 2>/dev/null; then
    echo ""
    echo "You need to authenticate with hex.pm first:"
    echo "  rebar3 hex user auth"
    echo ""
    exit 1
fi

# Step 5: Publish to hex.pm
echo ""
echo "Publishing to hex.pm..."
rebar3 hex publish --yes

# Step 6: Tag and push
echo ""
echo "Creating git tag v${VERSION}..."
git add -A
git commit -m "Release v${VERSION}: Complete 13-Silo Liquid Conglomerate Architecture" || true
git tag -a "v${VERSION}" -m "Release v${VERSION}"
git push origin main
git push origin "v${VERSION}"

echo ""
echo "=== Release v${VERSION} Complete ==="
echo ""
echo "Verify at:"
echo "  - https://hex.pm/packages/faber_neuroevolution"
echo "  - https://hexdocs.pm/faber_neuroevolution"
