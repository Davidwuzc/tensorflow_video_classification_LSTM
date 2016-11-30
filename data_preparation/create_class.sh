#!/bin/bash

# ex: 
#   sudo ./create_class.sh ../dataset/video_images
# output(../dataset/label):
#   walk
#   run
#   ...

for folder in $1/*
do
    if [[ ! -f "$1/../label" ]]; then
        touch "$1/../label"
    fi
    echo "${folder##*/}" >> "$1/../label"
done