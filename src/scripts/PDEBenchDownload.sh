#!/bin/bash

CSV_FILE="src/scripts/pdebench_data_urls.csv"
#SAVE_DIR = "datasets/prjs1359/PDEBench-Comp"
SAVE_DIR="datasets/prjs1359/PDEBench-Incomp"

# Create target directory
mkdir -p "$SAVE_DIR"

# Print Hello World
echo "Hello, World!"

# Check if at least one index is provided
#if [ $# -eq 0 ]; then
#    echo "Usage: $0 index1 [index2 ... indexN]"
#    exit 1
#fi
#mylist=($(seq 19 5 250))
#adjusted_indices=()
adjusted_indices=($(seq 19 5 290))
for arg in "$@"; do
    adjusted_indices+=($((arg - 1)))
done

tail -n +0 "$CSV_FILE" | nl -v 0 -w 1 -s ',' | while IFS=',' read -r idx _ filename url _ _; do
    for adj in "${adjusted_indices[@]}"; do
        if [ "$idx" -eq "$adj" ]; then
            file_path="$SAVE_DIR/$filename"
            if [[ -f "$file_path" ]]; then
                echo "File already exists: $file_path — skipping."
            else
                echo "Downloading: $filename"
                wget -c "$url" -O "$file_path"
            fi
        fi
    done
done
