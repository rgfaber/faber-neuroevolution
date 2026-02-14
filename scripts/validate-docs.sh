#!/bin/bash
# Validate documentation links for faber_neuroevolution
#
# Checks:
# 1. Internal markdown links (*.md) exist
# 2. SVG asset references exist
# 3. Reports broken links

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

ERRORS=0

echo "=== Validating Documentation Links ==="
echo "Project root: $PROJECT_ROOT"
echo

# Function to check if file exists
check_file() {
    local base_dir="$1"
    local link="$2"
    local source_file="$3"

    # Handle relative paths
    if [[ "$link" == /* ]]; then
        # Absolute path from project root
        target="$PROJECT_ROOT${link}"
    else
        # Relative path from source file's directory
        target="$base_dir/$link"
    fi

    # Normalize path (remove ./ and handle ../)
    target=$(cd "$(dirname "$target")" 2>/dev/null && pwd)/$(basename "$target") 2>/dev/null || echo "$target"

    if [[ ! -f "$target" ]]; then
        echo "BROKEN LINK in $source_file"
        echo "  -> $link"
        echo "  (expected at: $target)"
        echo
        return 1
    fi
    return 0
}

echo "--- Checking Markdown Links ---"

# Check markdown links from guides/*.md
for md_file in guides/*.md; do
    [[ -f "$md_file" ]] || continue
    base_dir=$(dirname "$md_file")

    # Extract markdown links: [text](file.md)
    while IFS= read -r link; do
        # Skip external links
        [[ "$link" == http* ]] && continue
        [[ "$link" == "#"* ]] && continue

        if ! check_file "$base_dir" "$link" "$md_file"; then
            ((ERRORS++))
        fi
    done < <(grep -oE '\]\([^)]+\.md\)' "$md_file" 2>/dev/null | sed 's/](\(.*\))/\1/' | sed 's/#.*//')
done

# Check markdown links from README.md
if [[ -f "README.md" ]]; then
    while IFS= read -r link; do
        [[ "$link" == http* ]] && continue
        [[ "$link" == "#"* ]] && continue

        if ! check_file "." "$link" "README.md"; then
            ((ERRORS++))
        fi
    done < <(grep -oE '\]\([^)]+\.md\)' "README.md" 2>/dev/null | sed 's/](\(.*\))/\1/' | sed 's/#.*//')
fi

echo
echo "--- Checking SVG Asset Links ---"

# Check SVG links from guides/*.md
for md_file in guides/*.md; do
    [[ -f "$md_file" ]] || continue
    base_dir=$(dirname "$md_file")

    # Extract image links: ![alt](path.svg)
    while IFS= read -r link; do
        # Skip external links
        [[ "$link" == http* ]] && continue

        if ! check_file "$base_dir" "$link" "$md_file"; then
            ((ERRORS++))
        fi
    done < <(grep -oE '!\[[^]]*\]\([^)]+\.svg\)' "$md_file" 2>/dev/null | sed 's/!\[[^]]*\](\([^)]*\))/\1/')
done

# Check SVG links from README.md
if [[ -f "README.md" ]]; then
    while IFS= read -r link; do
        [[ "$link" == http* ]] && continue

        if ! check_file "." "$link" "README.md"; then
            ((ERRORS++))
        fi
    done < <(grep -oE '!\[[^]]*\]\([^)]+\.svg\)' "README.md" 2>/dev/null | sed 's/!\[[^]]*\](\([^)]*\))/\1/')
fi

echo
echo "=== Summary ==="
if [[ $ERRORS -eq 0 ]]; then
    echo "All documentation links are valid!"
    exit 0
else
    echo "Found $ERRORS broken link(s)"
    exit 1
fi
