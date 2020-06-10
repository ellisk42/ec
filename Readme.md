#  Leveraging Natural Language for Program Search and Abstraction Learning

This repository is the official implementation of  Leveraging Natural Language for Program Search and Abstraction Learning (currently under review at NeurIPS 2020). This repository and branch is a static branch designed to reproduce the results in the paper. This README will be updated with a link to a deanonymized live branch after the review period.

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Getting Started -- Dependencies and Requirements
The following setup has been tested on an Ubuntu 16.04 machine. The codebase is implemented in both Python and OCaml.
##### Install Python and the Python requirements.
1. We test our implementation on Python 3.7.7. On Linux, you can create a fresh install (including pip) with:
```
sudo apt-get update; sudo apt-get install software-properties-common; sudo add-apt-repository ppa:deadsnakes/ppa; sudo apt-get update; sudo apt-get install python3.7; sudo apt install python3.7-dev; sudo apt install python3-pip; python3.7 -m pip install pip; pip install --upgrade setuptools;
```
2. 

```setup
pip install -r requirements.txt
```

> 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository.