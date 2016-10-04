# Data preprocessing

How to use `convert_to_records.py` program

example usage (only compatible with python 2.x): 
```
python convert_to_records.py
    --train_directory=/path/to/folder/../data
    --validation_directory=/path/to/folder/../data
    --output_directory=/path/to/folder/../data
    --labels_file=/path/to/file/../data/label 
```

[**origin file link**](https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py)

[**reference link**](https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html#file-formats) for how to write data to tfrecord and read data from tfrecord:

# Run the LCA training program (only only compatible with python 3.x)

`python3 lca_train.py --data_dir=/path/to/folder/../data/sharded_data`

# Start the `TensorBoard` to monitor the status

`tensorboard --logdir=/path/to/folder/../summary --reload_interval=10`

---

### Remark for my own computer:

#### Train the LCA/KTH dataset

`python3 lca_train.py --data_dir=/Volumes/passport/datasets/action_LCA/sharded_data --train_dir=/Volumes/passport/datasets/action_LCA/train_result`
`python3 lca_train.py --data_dir=/Volumes/passport/datasets/action_KTH/sharded_data --train_dir=/Volumes/passport/datasets/action_KTH/train_result`

#### Start TensorBoard

`tensorboard --logdir=/Volumes/passport/datasets/action_LCA/train_result/summary --reload_interval=10`
`tensorboard --logdir=/Volumes/passport/datasets/action_KTH/train_result/summary --reload_interval=10`