[**origin file link**](https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py)

[**reference link**](https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html#file-formats) for how to write data to tfrecord and read data from tfrecord:

# Run the LCA training program (only only compatible with python 3.x)

`python3 lca_train.py --data_dir=/path/to/folder/../data/sharded_data`

# Start the `TensorBoard` to monitor the status

`tensorboard --logdir=/path/to/folder/../summary --reload_interval=10`

---

### Remark for my own computer:

#### Train the LCA/KTH dataset

`python3 lca_train.py --data_dir=/Users/dgu/Documents/projects/machine_learning/lca_data/fps_5/sharded_data_11 --train_dir=/Users/dgu/Documents/projects/machine_learning/lca_data/fps_5/train_result`
`python3 kth_train.py --data_dir=/Users/dgu/Documents/projects/machine_learning/kth_data/sharded_data --train_dir=/Users/dgu/Documents/projects/machine_learning/kth_data/train_result`

#### Start TensorBoard

`tensorboard --logdir=/Users/dgu/Documents/projects/machine_learning/lca_data/fps_5/train_result/summary --reload_interval=10`
`tensorboard --logdir=/Users/dgu/Documents/projects/machine_learning/kth_data/train_result/summary --reload_interval=10`