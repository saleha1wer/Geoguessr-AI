# Geo-Guess
This repository contains models that are used to classify the region and pinpoint the coordinates of an image, inspired by the game GeoGuessr.

make_grid.ipynb splits the map into 88 squares (method of splitting from https://nirvan66.github.io/geoguessr.html) 

gen_images.py generates 200 images per square in the grid, saves in 'images/'.

The generated data can be downloaded from:
https://www.dropbox.com/s/hmz8m8sd3d2u8xj/images.zip?dl=0

visualize_data.ipynb includes a visualisation of the gathered data

network.py initialises the network we will train.

load_data.py loads the train, test and validation data arrays from a directory containing the data.

main.py trains a model with Inception-ResNet-v2 for feature extraction for 100 epochs on the full dataset. To use a different feature extraction or to exclude augmented data, change the variables that are initialised in main.py <br />

evaluate.py evalautes the model, (using three different approaches).

training_plots.ipynb generated plots that demonstrate the training process.

