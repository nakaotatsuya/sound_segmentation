#!/bin/sh

cd $(dirname $0)
pwd

cd ../
pwd
cd audios/wav_house/vacuum/
pwd

for file in `\find . -maxdepth 1 -type f`; do
    sox $file -b 16 -e signed-integer -r 16k "${file%.*}"_converted.wav
done

#sox input.wav -b 16 -e signed-integer hoge.wav
