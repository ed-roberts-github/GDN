#!/bin/sh
START=0
END=10
for (( i=$START; i < $END; i++))
do
echo "COUNTER is $COUNTER"
python /Users/edroberts/Desktop/im_gen/training_data/gen_im_csv.py --counter=$i --config_file="/Users/edroberts/Desktop/im_gen/training_data/config.yml" --output_dir="/Users/edroberts/Desktop/im_gen/training_data/train/train"
done
