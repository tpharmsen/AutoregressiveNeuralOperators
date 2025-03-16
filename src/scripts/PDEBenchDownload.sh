
CSV_FILE="pdebench_data_urls.csv"

# Function to download a file given its URL and target path
download_file() {
    url="$1"
    filename="$2"
    path="$3"
    md5_expected="$4"
    
    # Create the target directory if it doesn't exist
    mkdir -p "$path"
    
    # Download the file
    echo "Downloading $filename..."
    wget -q --show-progress -O "$path/$filename" "$url"
    
    # Verify MD5 checksum
    md5_actual=$(md5sum "$path/$filename" | awk '{ print $1 }')
    if [[ "$md5_actual" == "$md5_expected" ]]; then
        echo "MD5 checksum verified for $filename"
    else
        echo "Warning: MD5 checksum mismatch for $filename!"
    fi
}

# If no arguments provided, download all files
if [[ $# -eq 0 ]]; then
    echo "Downloading all files..."
    tail -n +2 "$CSV_FILE" | while IFS="," read -r _ filename url path md5; do
        download_file "$url" "$filename" "$path" "$md5"
    done
else
    echo "Downloading selected files: $@"
    for index in "$@"; do
        line=$(sed -n "$((index+1))p" "$CSV_FILE")
        IFS="," read -r _ filename url path md5 <<< "$line"
        download_file "$url" "$filename" "$path" "$md5"
    done
fi