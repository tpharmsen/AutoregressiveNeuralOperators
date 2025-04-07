#wget https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=0000.am 0000 t/m 7999
#wget https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F"&"files=0000.am

cd datasets/prjs1359
mkdir -p AmiraSet 

for i in {0,999,1999,2999,3999,4999,5999,6999,7999}; do
    wget -O "AmiraSet/$(printf "%04d" $i).am" "https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=$(printf "%04d" $i).am"
done