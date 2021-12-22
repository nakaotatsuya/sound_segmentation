#!/bin/sh

cd $(dirname $0)
pwd

cd ../
pwd
cd house_audios/noise_processed_train/
pwd

for file in * ; do
    echo $file
    expect -c "
    set timeout 5
    spawn scp ./$file/noisereduce.wav nakao@dlbox6:/home/jsk/nakao/sound_segmentation/house_audios/noise_train2/$file/.
    expect \"password:\"
    send \"nakao\n\"
    expect \"$\"
    exit 0"
done
