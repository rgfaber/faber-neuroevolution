#!/bin/bash
# Publish faber-neuroevolution to hex.pm
# Run this script from the repository root

set -e

echo "=== Publishing faber-neuroevolution v0.18.0 to hex.pm ==="
echo ""

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Check current version
VERSION=$(grep 'vsn' src/faber_neuroevolution.app.src | sed 's/.*"\(.*\)".*/\1/')
echo "Version: $VERSION"
echo ""

# Verify tests pass (lineage tests)
echo "Running lineage tests..."
rebar3 eunit --module=neuroevolution_lineage_events_tests
echo ""

# Build the package
echo "Building hex package..."
rebar3 hex build
echo ""

# Publish (will prompt for password)
echo "Publishing to hex.pm..."
echo "You will be prompted for your hex.pm password."
echo ""
rebar3 hex publish

# Tag the release
echo ""
echo "Tagging release v$VERSION..."
git tag -a "v$VERSION" -m "Release v$VERSION - Lineage Tracking (CQRS)"
git push origin "v$VERSION"

echo ""
echo "=== Published successfully! ==="
echo "View at: https://hex.pm/packages/faber_neuroevolution"
