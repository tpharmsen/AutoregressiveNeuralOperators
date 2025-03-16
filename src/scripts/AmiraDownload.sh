#wget https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=0000.am
#wget https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F"&"files=0000.am
for i in {0,10,100}; do
    wget "https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files=$(printf "%04d" $i).am"
done