#wget https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=0000.am 0000 t/m 7999
#wget https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F"&"files=0000.am

cd ../../data/fluidgpt
mkdir -p AmiraSet 

n=500  # number of elements to select
max=7999

# Compute interval step
step=$((max / (n - 1)))

for ((i = 0; i < n; i++)); do
    val=$((i * step))
    fname=$(printf "%04d" $val)
    filepath="AmiraSet/${fname}.am"

    if [[ -f "$filepath" ]]; then
        echo "File $filepath already exists. Skipping."
    else
        echo "Downloading ${fname}.am..."
        wget -O "$filepath" "https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=${fname}.am"
    fi
done