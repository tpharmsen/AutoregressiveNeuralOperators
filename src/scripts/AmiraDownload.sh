#wget https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=0000.am
#wget https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F"&"files=0000.am

mkdir -p AmiraSet 

for i in {0,10,100}; do
    wget -O "AmiraSet/$(printf "%04d" $i).am" "https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=$(printf "%04d" $i).am"
done