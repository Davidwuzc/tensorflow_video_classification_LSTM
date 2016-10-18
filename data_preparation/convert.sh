#!/bin/bash

# convert all the video file to images
# the path/to/data structure will be something like /user/data/train_data/walk/video1.avi

# count = 0
# for file in /Volumes/passport/datasets/action_KTH/video/boxing/*.avi
# do
#     if [[ ! -d /Volumes/passport/datasets/action_KTH/video/boxing/$count ]]; then
#         mkdir -p /Volumes/passport/datasets/action_KTH/video/boxing/$count
#     fi
#     ffmpeg -i "$file" /Volumes/passport/datasets/action_KTH/video/boxing/$count/%05d.png
#     (( count++ ))
# done

for folder in /Volumes/passport/datasets/action_LCA/origin_video/*
do
    count = 0
    for file in "$folder"/*.avi
    do
        if [[ ! -d "$folder"/$count ]]; then
            mkdir -p "$folder"/$count
        fi
        ffmpeg -i "$file" -vf fps=5 "$folder"/$count/%05d.png
        rm "$file"
        (( count++ ))
    done
done