# Data preprocessing

## Step 1 (conver.sh)
This will convert the video file to sequence of images. For example, if we 
set the fps to 30 and the video length is 2 seconds, then the video will be 
converted to 60 images.

1. config the path to your folder at line `for folder in /path/to/video/folder/*`
    > The folder structure is `.../video/classes_name/video_file.avi`
    > <br> Example: `.../video/arrive/video1.avi` & `.../video/run/video2.avi` etc.
    > <br> The code will be `for folder in .../video/*`

2. (option) change the fps setting at line `ffmpeg -i "$file" -vf fps=5 
    "$folder"/$count/%05d.png`

## Step 2 (convert_to_records.py)
This will convert sequence of images to [TFRecord](https://www.tensorflow.org/versions/r0.11/how_tos/reading_data/index.html#file-formats) 
which is the standard TensorFlow format.

1. (mandatory) set the **train_directory** parameter value to the path of your sequence of images.
2. (mandatory) set the **validation_directory** parameter value to the path of your sequence of images.
3. (mandatory) set the **output_directory** parameter value to the path where you want to store the data.
4. (mandatory) set the **labels_file** parameter value to the path of your label file.
    > the label file is just a txt file containing all your action classes name
    > <br> example file (label.txt)
    > ``` 
    > approach
    > run
    > play
    > walk
    > bury
    > exit
    > ...
    > ```
5. (option) set the **sequence_length** parameter to the length of one video clip. This value should be smaller than the origin length of the video.
 
### Example usage (only compatible with python 2.x)
```
python convert_to_records.py \
    --train_directory=.../training_data/run/1/images1.png \
    --validation_directory=.../validation_data/run/1/images1.png \
    --output_directory=.../result \
    --labels_file=.../label.txt \
    --sequence_length=16
```

---
**For my own reference**
```bash
python convert_to_records.py --train_directory=/Users/dgu/Documents/projects/machine_learning/lca_data/fps_5/origin_video --output_directory=/Users/dgu/Documents/projects/machine_learning/lca_data/fps_5/result_data --label_file=/Users/dgu/Documents/projects/machine_learning/lca_data/fps_5/label --sequence_length=5
```