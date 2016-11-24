# Training 

## PTB

Steps:

1. The data required for this example is in the data/ dir of the PTB dataset from Tomas Mikolov's webpage:

    ```
    $ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    $ tar xvf simple-examples.tgz
    ```

2. Run `$ python ptb_train.py --data_path=simple-examples/data/ --save_path=result`

    - the `data_path` option is the path to the ptb data folder
    - the `save_path` option is the folder to store model/summary/checkpoint 

## KTH

1. Run `$ python kth_train.py --data_path=simple-examples/data/ --save_path=result`

    - the `data_path` option is the path to the ptb data folder
    - the `save_path` option is the folder to store model/summary/checkpoint 
---

For my personal reference

`python ptb_train.py --data_path=/Users/dgu/Documents/projects/machine_learning/ptb_data --save_path=result`

`python kth_train.py --data_path=/Users/dgu/Documents/projects/machine_learning/kth_data/sharded_data --save_path=result`

`python lca_train.py --data_path=/Users/dgu/Documents/projects/machine_learning/lca_data/sharded_data --save_path=result`
