# Data preprocessing

## Step 1 (convert.sh)

This will convert the video file to sequence of images. For example, if we 
set the fps to 30 and the video length is 2 seconds, then the video will be 
converted to 60 images.

The `convert.sh` script has two paramters:
* path to video folder
* fps number

### **Example Usage**:
* Folder structure is `.../video/walk/video_file.avi` 
* Fps setting is 20

Example command will be 
```
$ sudo ./convert.sh .../video/ 20
```

## Step 2 (convert_to_records.py)

This will convert sequence of images to [TFRecord](https://www.tensorflow.org/versions/r0.11/how_tos/reading_data/index.html#file-formats) 
which is the standard TensorFlow format.

1. (mandatory) set the **train_directory** parameter value to the path of your sequence of images.
2. (mandatory) set the **validation_directory** parameter value to the path of your sequence of images.
3. (mandatory) set the **output_directory** parameter value to the path where you want to store the data, must create the corresponding folder first.
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
5. (option) set the **sequence_length** parameter to the length of one video clip. This value should be smaller than the origin length of the video, the default value of **seuqnce_length**
is 16.
 
### Example usage (only compatible with python 2.x)

```
$ python convert_to_records.py 
    --train_directory=.../training_data 
    --validation_directory=.../validation_data 
    --output_directory=.../result 
    --labels_file=.../label.txt 
    --sequence_length=32
```

## Step 3
Sum up all the print out value and write this value to `{database}_data.py` file

Example:

```
Output:
$ [thread 3]: Wrote 354 videos in total
$ [thread 0]: Wrote 354 videos in total
$ [thread 1]: Wrote 354 videos in total
$ [thread 2]: Wrote 354 videos in total
```

Set the corresponding value inside `num_examples_per_epoch` function to 
`354+354+354+354=1416`

---
**For my own reference**
- kth
```bash
python convert_to_records.py --train_directory=/Users/dgu/Documents/projects/machine_learning/kth_data/origin_images --output_directory=/Users/dgu/Documents/projects/machine_learning/kth_data/sharded_data --label_file=/Users/dgu/Documents/projects/machine_learning/kth_data/label.txt
```

- lca
```bash
python convert_to_records.py --train_directory=/Users/dgu/Documents/projects/machine_learning/lca_data/origin_images --output_directory=/Users/dgu/Documents/projects/machine_learning/lca_data/sharded_data --label_file=/Users/dgu/Documents/projects/machine_learning/lca_data/label
```

remove the `.DS_Store` file in the video folder
```
find . -name '.DS_Store' -type f -delete
```

- ucf101
```bash
python convert_to_records.py --train_directory=/Volumes/passport/datasets/action_ucf101/origin_images --output_directory=/Volumes/passport/datasets/action_ucf101/sharded_data --label_file=/Volumes/passport/datasets/action_ucf101/label
```
