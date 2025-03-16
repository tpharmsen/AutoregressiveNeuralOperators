
CSV_FILE="pdebench_data_urls.csv"

cd data/prjs1359
mkdir -p PDEBenchSet 

# Read the CSV and download the files
tail -n +2 "$CSV_FILE" | while IFS=',' read -r _ filename url path _; do
    # print the file url
    echo "$url"
    #mkdir -p "$path"
    #wget -c "$url" -O "downloads/$path/$filename"
done