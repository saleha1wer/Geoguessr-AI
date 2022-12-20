# Geo-Guess
Model to classify image --> region, coordinates

make_grid.ipynb splits the map into 88 squares (method of splitting from https://nirvan66.github.io/geoguessr.html) 

gen_images.py generates 200 images per square in the grid, saves in 'images/'.

The generated data can be downloaded from:
https://www.dropbox.com/s/hmz8m8sd3d2u8xj/images.zip?dl=0


network.py initialises a network 

load_data.py loads the train, test and validation data arrays from a directory containing the data.

main.py does <br />
  training for 30 epochs <br />
  unfreeze the RSP <br />
  training for for 30 epochs <br />
  freeze everything but RSP <br />
  training for 20 epochs <br />
