#!/bin/bash

cd ../../data/fluidgpt || exit
mkdir -p AmiraSet

n=500  # number of elements to select
max=7999

# Compute interval step
step=$((max / (n - 1)))

# Function to download a single file
download_file() {
    val=$1
    fname=$(printf "%04d" "$val")
    filepath="AmiraSet/${fname}.am"

    if [[ -f "$filepath" ]]; then
        echo "File $filepath already exists. Skipping."
    else
        echo "Downloading ${fname}.am..."
        wget -q -O "$filepath" "https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=${fname}.am"
    fi
}

export -f download_file

# Generate values and run in parallel with 8 concurrent downloads
seq 0 $((n - 1)) | \
    awk -v step="$step" '{ print int($1 * step) }' | \
    sort -nu | \
    xargs -n1 -P8 -I{} bash -c 'download_file "$@"' _ {}