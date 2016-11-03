#!/bin/bash

# convert the avi video to images
#   Usage (sudo for the remove priviledge):
#       sudo ./conver.sh path/to/video/
#   Example Usage:
#       sudo ./conver.sh ~/document/videofile/
#   Example Output:
#       ~/document/videofile/walk/video1.avi 
#       #=>
#       ~/document/videofile/walk/video1/00001.jpg
#       ~/document/videofile/walk/video1/00002.jpg
#       ~/document/videofile/walk/video1/00003.jpg
#       ~/document/videofile/walk/video1/00004.jpg
#       ~/document/videofile/walk/video1/00005.jpg
#       ...

for folder in $1*
do
    for file in "$folder"/*.avi
    do
        if [[ ! -d "${file[@]%.avi}" ]]; then
            mkdir -p "${file[@]%.avi}"
        fi
        ffmpeg -i "$file" -vf fps=5 "${file[@]%.avi}"/%05d.jpg
        rm "$file"
    done
done