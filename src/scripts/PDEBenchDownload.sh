#!/bin/bash

CSV_FILE="src/scripts/pdebench_data_urls.csv"
SAVE_DIR="datasets/prjs1359/PDE_TEMP"

# Create target directory
mkdir -p "$SAVE_DIR"

# Print Hello World
echo "Hello, World!"

# Check if at least one index is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 index1 [index2 ... indexN]"
    exit 1
fi

adjusted_indices=()
for arg in "$@"; do
    adjusted_indices+=($((arg - 1)))
done

tail -n +0 "$CSV_FILE" | nl -v 0 -w 1 -s ',' | while IFS=',' read -r idx _ filename url _ _; do
    for adj in "${adjusted_indices[@]}"; do
        if [ "$idx" -eq "$adj" ]; then
            echo "Downloading: $filename"
            wget -c "$url" -O "$SAVE_DIR/$filename"
        fi
    done
done