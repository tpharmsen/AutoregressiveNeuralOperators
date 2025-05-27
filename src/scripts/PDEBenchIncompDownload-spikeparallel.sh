#!/bin/bash

CSV_FILE="src/scripts/pdebench_data_urls.csv"
SAVE_DIR="../../data/fluidgpt/PDEBench-Incomp"
MAX_JOBS=8  # Limit of parallel downloads

mkdir -p "$SAVE_DIR"
echo "Starting parallel download..."

adjusted_indices=($(seq 19))
for arg in "$@"; do
    adjusted_indices+=($((arg - 1)))
done

download_file() {
    idx="$1"
    filename="$2"
    url="$3"
    file_path="$SAVE_DIR/$filename"

    if [[ -f "$file_path" ]]; then
        echo "File already exists: $file_path — skipping."
    else
        echo "Downloading: $filename"
        wget --show-progress -c "$url" -O "$file_path"
    fi
}

# Export function for subshells
export -f download_file
export SAVE_DIR

# Read and dispatch jobs
tail -n +0 "$CSV_FILE" | nl -v 0 -w 1 -s ',' | while IFS=',' read -r idx _ filename url _ _; do
    for adj in "${adjusted_indices[@]}"; do
        if [ "$idx" -eq "$adj" ]; then
            download_file "$idx" "$filename" "$url" &
            ((++num_jobs >= MAX_JOBS)) && wait && num_jobs=0
        fi
    done
done

wait  # Wait for any remaining jobs to finish
echo "All downloads complete."