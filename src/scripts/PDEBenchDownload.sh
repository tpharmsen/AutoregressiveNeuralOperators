#!/bin/bash

CSV_FILE="src/scripts/pdebench_data_urls.csv"
SAVE_DIR="data/prjs1359/NS_incom_inhom"

# Create target directory
mkdir -p "$SAVE_DIR"

# Print Hello World
echo "Hello, World!"

# Check if at least one index is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 index1 [index2 ... indexN]"
    exit 1
fi

# Read the CSV file and loop over the provided indices
tail "$CSV_FILE" | nl -v 0 -w 1 -s ',' | while IFS=',' read -r idx _ filename url _ _; do
    for arg in "$@"; do
        if [ "$idx" -eq "$arg" ]; then
            echo "Downloading: $filename"
            wget -c "$url" -O "$SAVE_DIR/$filename"
        fi
    done
done