# Lightweight, Dynamic Graph Convolutional Networksfor AMR-to-Text Generation (EMNLP2020)



## Dependencies
The model requires:
- Python3
- [MXNet 1.5.0](https://github.com/apache/incubator-mxnet/tree/1.5.0)
- [Sockeye 1.18.56 (NMT framework based on MXNet)](https://github.com/awslabs/sockeye)
- CUDA 


## Installation
#### GPU

If you want to run sockeye on a GPU you need to make sure your version of Apache MXNet
Incubating contains the GPU bindings. Depending on your version of CUDA you can do this by running the following:

```bash
> pip install -r requirements/requirements.gpu-cu${CUDA_VERSION}.txt
> pip install .
```


## Training

To train the LDGCN model, run (e.g., for AMR2015):

```
./train_amr15gc.sh
```

## Decoding

When we finish the training, we can use the trained model to decode on the test set, run:

```
./decode_amr15.sh
```

This will use the last checkpoint by default. Use `--checkpoints` to specify a model checkpoint file.



## Postprocessing



We use BPE code. In the postprocessing stage, we need to merge them into natural language sequence for evaluation, run:

```
./merge_amr15.sh
```

## Evaluation

For BLEU score evaluation, run:

```
./eval_amr15_bleu.sh
```






