
# Synthetic Genomics D-SPACE Public Library

## Demo site
A public demo of the D-SPACE annotation and search capabilities is available here https://dspace.bio/.

## Install

* Clone this repo.
* Install your preferred version of [Tensorflow](https://www.tensorflow.org/) based on whether you're running on a GPU enabled machine or not. Either: `pip install tensorflow` or `pip install tensorflow-gpu`.
* Install this package:

    pip install -e .

## Training

1. Obtain .dat.gz files for Sprot and Trembl from Uniprot (We used 02_2018)
2. Obtain Uniref100 data from Uniprot and make a list of Uniref100 representative cluster proteins (1 per line) - these are the good ids that are non-redundant
3. Run parse_split_uniprot.py with data from steps 1 and 2
4. Run shuffle_uniprot.py on each subset of the data (train, test, val)
5. Move the resulting shuffled JSON files to 3 separate directories (named "train", "test", "val") within a common directory


6. Run

    train_dspace

