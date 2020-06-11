#  Leveraging Natural Language for Program Search and Abstraction Learning

This repository is the official implementation of  Leveraging Natural Language for Program Search and Abstraction Learning (currently under review at NeurIPS 2020). This repository and branch is a static branch designed to reproduce the results in the paper. This README will be updated with a link to a deanonymized live branch after the review period.

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Getting Started -- Dependencies and Requirements
The following setup has been tested on an Ubuntu 16.04 machine. The codebase is implemented in both Python and OCaml.
##### Install Python 3.7.7 and the Python requirements.
1. We test our implementation on Python 3.7.7. On Linux, you can create a fresh install (including pip) with:
```
sudo apt-get update; sudo apt-get install software-properties-common; 
sudo add-apt-repository ppa:deadsnakes/ppa; sudo apt-get update; 
sudo apt-get install python3.7; sudo apt install python3.7-dev; 
sudo apt install python3-pip; python3.7 -m pip install pip;
pip install --upgrade setuptools;
```
2. Install the requirements.
```
pip install -r requirements.txt
```
3. Download the NLTK Tokenizer. At an interactive Python prompt, run:
```
> import nltk; nltk.download('punkt')
``` 

##### Build the OCaml binaries.
The repository contains prebuilt OCaml binaries that should run on most Linux-based machines. However, to build the OCaml binaries from scratch, you can run the following from the root of the repo.
1. Install OCaml.
```
sudo apt install ocaml ; 
sudo apt install opam ; 
opam init; opam update; 
opam switch 4.06.1+flambda ;
eval `opam config env`;
```
2. Install the OCaml requirements.
```
opam depext ppx_jane core re2 yojson vg cairo2 camlimages menhir ocaml-protoc zmq; 
opam install ppx_jane core re2 yojson vg cairo2 camlimages menhir ocaml-protoc zmq;
```
3. Run the following from the directory root to build the binaries.
```
make clean; make
```

##### Download Moses.
The codebase uses the [Moses](http://www.statmt.org/moses/?n=Moses.Releases) statistical machine translation system to implement the joint generative model for translating between program and natural language tokens.
On an Ubuntu 16.04 machine, you can download and use the prebuilt Moses binaries directly. (Binaries for [Ubuntu 17.04](http://www.statmt.org/moses/RELEASE-4.0/binaries/) are also available, as well as instructions for compiling from scratch on other architectures. As a caveat, we found this particularly hard to get working on a Mac machine.)

From the root of the directory, run:
```
wget http://www.statmt.org/moses/RELEASE-4.0/binaries/ubuntu-16.04.tgz;
tar -xvf ubuntu-16.04.tgz; mv ubuntu-16.04/ moses_compiled;
rm -rf  ubuntu-16.04.tgz;
```

##### Datasets
The datasets used in this paper are released in three versions (e.g. containing 200, 500, and 1000 training tasks) for evaluating data efficiency. We use the ```logo_unlimited_200``` and ```re2_1000``` datasets in the paper.
All datasets have the following directory structure.
```
|_ logo                           # Domain name.
  |__ language                    # Contains language datasets.
      |__logo_unlimited_200       # Dataset version (e.g. 200 tasks.)
          |_ synthetic            # Synthetic language data.            
              |_train
                |_ language.json  # Language annotations.
                |_ vocab.json     # Vocabulary.
              |_test ...
          |_ human   ...          # Where available, human language data.
      |...  
  |  
  |__ tasks                       # Contains task example specification.
      |__logo_unlimited_200       # Dataset version.
          |_ train                # Contains domain-specific tasks.
          |_ test  
```
Domain-specific loaders are provided for each dataset. See Training, below.
###### Graphics Programs
The graphics programs (LOGO) dataset is available on Zenodo [here](https://zenodo.org/record/3889088#.XuGEWp5KhTY).
The unzipped dataset should be stored in ```data/logo```. Our paper uses the ```logo_unlimited_200``` dataset (consisting of 200 training tasks and 111 testing tasks).

This dataset contains synthetic language for the 200, 500, and 1000 training task versions (and the 111 testing tasks); and human data for the 200 task version. We also provide rendered images for each task. Tasks are stored in the codebase-specific task format (which includes ground truth programs), and must be loaded through this repository (see Training).

The repository also contains code to generate the compositional graphics datasets from scratch (and synthetic language).
```
python bin/logo.py \
--generateTaskDataset [logo_unlimited_200 | logo_unlimited_500 | logo_unlimited_500] \
--generateLanguageDataset [logo_unlimited_200 | logo_unlimited_500 | logo_unlimited_500] \
```

###### Text Editing
The text editing (regex) dataset is available on Zenodo [here](https://zenodo.org/record/3889088#.XuGEWp5KhTY).
The unzipped dataset should be stored in ```data/re2```. Our paper uses the ```re2_1000``` dataset (consisting of 1000 training tasks and 500 testing tasks). 
The dataset tasks and human language comes from the regex domain in *Learning with Latent Language* (Andreas et. al, 2017) [[paper](https://arxiv.org/abs/1711.00482)] [[code](https://github.com/jacobandreas/l3)]

This dataset contains synthetic language and human language for all three training task versions. Tasks are stored in JSON containing the example inputs and outputs, but can be loaded with the domain-specific dataset loader.

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