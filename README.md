# Parsimonious Parser Transfer

This repository contains the code for the Parsimonious Parser Transfer (PPT) published
in EACL 2021 by Kemal Kurniawan, Lea Frermann, Philip Schulz, and Trevor Cohn.

## Fetching submodules

After cloning this repository, you need to also fetch the submodules with

    git submodule init
    git submodule update

## Installing requirements

We recommend you to use conda package manager. Then, create a virtual environment with
all the required dependencies with:

    conda env create -n [env] -f environment.yml

Replace `[env]` with your desired environment name. Once created, activate the environment.
The command above also installs the CPU version of PyTorch. If you need the GPU version,
follow the corresponding PyTorch installation docs afterwards. If you're using other package
manager (e.g., pip), you can look at the `environment.yml` file to see what the requirements are.

## Preparing dataset

Download UD treebanks v2.2 from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2837

## Preparing word embeddings

Next, download FastText's Wiki word embeddings from [this page](https://fasttext.cc/docs/en/pretrained-vectors.html).
You need to download the text format (`.vec`). Suppose you put the word embedding files
in `fasttext` directory. Then, perform the word embedding alignment to get the multilingual
embeddings:

    ./align_embedding.py with heetal

Lastly, minimise the word embedding files so they contain only words that actually occur in
the UD data. Assuming the UD data is stored in `ud-treebanks-v2.2`, then run

    ./minimize_vectors_file.py with vectors_path=aligned_fasttext/wiki.multi.id.vec output_path=aligned_fasttext/wiki.multi.min.id.vec corpus.lang=id

The command above minimises the word vector file for Indonesian (id). You can set `corpus.lang`
to other language codes mentioned in the paper, e.g., ar for Arabic, es for Spanish, etc.

## Training the source (English) parser

Assuming you have minimised the English word vectors file to `wiki.en.min.vec`,
run

    ./run_parser.py with ahmadetal word_emb_path=wiki.en.min.vec

The trained parser will be stored in `artifacts` directory.

## Performing direct transfer

Assuming the source parser parameters are saved in `artifacts/100_model.pth`, then run

    ./run_parser.py evaluate with ahmadetal heetal_eval_setup load_params=100_model.pth word_emb_path=aligned_fasttext/wiki.multi.min.id.vec corpus.lang=id

## Running the self-training baseline

    ./run_st.py with ahmadetal heetal_eval_setup distant word_emb_path=aligned_fasttext/wiki.multi.min.id.vec load_params=100_model.pth corpus.lang=id

Change `distant` to `nearby` to change the hyperparameters to ones optimised for nearby languages.

## Running PPT

    ./run_ppt.py with ahmadetal heetal_eval_setup distant word_emb_path=aligned_fasttext/wiki.multi.min.id.vec load_params=100_model.pth corpus.lang=id

Same as before, you can change `distant` to `nearby` for nearby languages.

## Running PPTX

Suppose you've trained the source parsers using `run_parser.py`, and the trained models are saved in `artifacts` and `de_artifacts` for English and German respectively (we use 2 languages in this example but you can have as many as you want). Also, the model parameters are `artifacts/100_model.pth` and `de_artifacts/150_model.pth` respectively. Then, you can run PPTX with:

    ./run_pptx.py with ahmadetal heetal_eval_setup distant load_src="{'en':('artifacts','100_model.pth'),'de':('de_artifacts','150_model.pth')}" \
         main_src=en word_emb_path=aligned_fasttext/wiki.multi.min.id.vec corpus.lang=id

Change `distant` to `nearby` for nearby languages.

## Sacred: an experiment manager

Almost all scripts in this repository use [Sacred](https://github.com/IDSIA/sacred/). The scripts are written so that you can store all about an experiment run in a MongoDB database. Simply set environment variables `SACRED_MONGO_URL` to point to a MongoDB instance and `SACRED_DB_NAME` to a database name to activate it. Also, invoke the `help` command of any such script to print its usage, e.g., `./run_parser.py help`.
