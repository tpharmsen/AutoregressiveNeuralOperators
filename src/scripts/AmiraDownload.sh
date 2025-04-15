#wget https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=0000.am 0000 t/m 7999
#wget https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F"&"files=0000.am

cd datasets/prjs1359
mkdir -p AmiraSet 

n=2  # number of elements to select
max=7999

# Compute interval step
step=$((max / (n - 1)))

for ((i = 0; i < n; i++)); do
    val=$((i * step))
    fname=$(printf "%04d" $val)
    wget -O "AmiraSet/${fname}.am" "https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=${fname}.am"
done