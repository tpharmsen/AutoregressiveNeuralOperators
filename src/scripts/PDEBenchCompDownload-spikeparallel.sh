#!/bin/bash

CSV_FILE="src/scripts/pdebench_data_urls.csv"
SAVE_DIR="/data/fluidgpt/PDEBench-Comp"

# Create target directory
mkdir -p "$SAVE_DIR"


# Adjusted indices (zero-based): use arguments or define a sequence
adjusted_indices=($(seq 1 1 8))
for arg in "$@"; do
    adjusted_indices+=($((arg - 1)))
done

# Create a temporary file to hold selected download entries
TMP_FILE=$(mktemp)

# Process CSV and select only desired rows
tail -n +0 "$CSV_FILE" | nl -v 0 -w 1 -s ',' | while IFS=',' read -r idx _ filename url _ _; do
    for adj in "${adjusted_indices[@]}"; do
        if [ "$idx" -eq "$adj" ]; then
            echo "$filename,$url" >> "$TMP_FILE"
        fi
    done
done

# Use xargs to download in parallel (adjust -P for number of parallel jobs)
cat "$TMP_FILE" | xargs -P 6 -n 1 -I {} bash -c '
    IFS="," read -r filename url <<< "{}"
    SAVE_PATH="'$SAVE_DIR'/$filename"
    echo filedir for saving: $SAVE_PATH
    if [[ -f "$SAVE_PATH" ]]; then
        echo "File already exists: $SAVE_PATH — skipping."
    else
        echo "Downloading: $filename"
        echo "URL: $url"
        wget "$url" -O "$SAVE_PATH"
    fi
'

# Cleanup
rm "$TMP_FILE"