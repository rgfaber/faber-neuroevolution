#!/usr/bin/env bash
# Remove backtick code blocks from EDoc comments (EDoc doesn't support them)

cd "$(dirname "$0")/.."

FILES=$(grep -rl '```' src/)

for file in $FILES; do
    echo "Fixing: $file"
    # Remove lines that are just %% ``` or %% ```erlang or %% ```elixir
    sed -i '/^%% ```/d' "$file"
done

echo "Done. Fixed $(echo "$FILES" | wc -l) files."
