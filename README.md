# Dependencies

[tensorflow](https://www.tensorflow.org/api_docs/) >= 1.0

# Training 

## Step 1

preprocess the data accroding to the `README.md` in `data_preparation` folder 

## Step 2

1. Run `$ python kth_train.py --data_path=<path-to-data> --save_path=<path-to-folder>`

    - the `data_path` option is the path to the ptb data folder
    - the `save_path` option is the folder to store model/summary/checkpoint 
---

For my personal reference

`python kth_train.py --data_path=/Users/dgu/Documents/projects/machine_learning/kth_data/sharded_data --save_path=result`

`python lca_train.py --data_path=/Users/dgu/Documents/projects/machine_learning/lca_data/sharded_data --save_path=result`
