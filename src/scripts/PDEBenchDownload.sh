
#!/bin/bash

CSV_FILE="src/scripts/pdebench_data_urls.csv"
mkdir -p data/prjs1359/PDEBenchSet

# Print Hello World
echo "Hello, World!"

# Check if at least one index is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 index1 [index2 ... indexN]"
    exit 1
fi

# Read the CSV file and loop over the provided indices
tail -n +2 "$CSV_FILE" | nl -v 0 -w 1 -s ',' | while IFS=',' read -r idx _ filename url path _; do
    for arg in "$@"; do
        if [ "$idx" -eq "$arg" ]; then
            echo "Downloading: $filename"
            wget -c "$url" -O "data/prjs1359/$path/$filename"
        fi
    done
done