# How to Train Your Agent to Read and Write
This repository contains the code for the AAAI paper: "How to Train Your Agent to Read and Write".

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

This project is implemented using the framework [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and the library [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric). Please, refer to their websites for further details on the installation and dependencies.

## Environments and Dependencies

- python 3.6
- PyTorch 1.1.0
- PyTorch Geometric 1.3.1
- subword-nmt 0.3.6

## Datasets

In our experiments, we use the following datasets:  [AGENDA](https://github.com/rikdz/GraphWriter/tree/master/data).

## Preprocess

First, convert the dataset into the format required for the model.

For the AGENDA dataset, run:
```
./preprocess_AGENDA.sh <dataset_folder>
```
For the WebNLG dataset, run:
```
./preprocess_WEBNLG.sh <dataset_folder>
```


## Training
For traning the model using the AGENDA dataset, execute:
```
./train_AGENDA.sh <graph_encoder> <gpu_id>
```

Options for `<graph_encoder>` is `cge-lw`. 

Examples:
```
./train_AGENDA.sh 0 cge-lw
```

## Decoding

For decoding, run:
```
./decode_AGENDA.sh <gpu_id> <model> <nodes_file> <graph_file> <output>
```

Example:
```
./decode_AGENDA.sh 0 model_agenda_cge_lw.pt test-nodes.txt test-graph.txt output-agenda-testset.txt
```


## More
For more details regading hyperparameters, please refer to [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).




