
CSV_FILE="../../src/scripts/pdebench_data_urls.csv"

cd data/prjs1359
mkdir -p PDEBenchSet 

# print hellow world
echo "Hello, World!"

# Read the CSV and download the files
#tail -n +2 "$CSV_FILE" | while IFS=',' read -r _ filename url path _; do
    # print the file url
    #echo "$url"
    #mkdir -p "$path"
    #wget -c "$url" -O "downloads/$path/$filename"
#done

# Check if at least one index is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 index1 [index2 ... indexN]"
    exit 1
fi

# Read the CSV file and loop over the provided indices
tail -n +2 "$CSV_FILE" | nl -v 0 -w 1 -s ',' | while IFS=',' read -r idx _ filename url path _; do
    for arg in "$@"; do
        if [[ "$idx" -eq "$arg" ]]; then
            echo "Downloading: $filename"
            mkdir -p "downloads/$path"
            wget -c "$url" -O "downloads/$path/$filename"
        fi
    done
done