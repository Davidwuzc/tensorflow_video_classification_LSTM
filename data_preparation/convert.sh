#!/bin/bash

for folder in /path/to/video/folder/*
do
    count = 0
    for file in "$folder"/*.avi
    do
        if [[ ! -d "$folder"/$count ]]; then
            mkdir -p "$folder"/$count
        fi
        ffmpeg -i "$file" -vf fps=20 "$folder"/$count/%05d.png
        rm "$file"
        (( count++ ))
    done
done