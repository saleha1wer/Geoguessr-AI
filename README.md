# GeoGuessr AI
<!-- ABOUT THE PROJECT -->
## About The Project
This project aimed to train models that are used to classify the region and pinpoint the coordinates of an image, inspired by the game GeoGuessr. The models are trained on a dataset of google street view images and their corresponding location data. The goal of these models is to be able to accurately predict the location of an image based on its visual features.

More detailed info in [this medium post](https://medium.com/@salehalwer/geoguessr-inspired-exploration-of-cnns-predicting-street-view-image-locations-e7aaa2dc19f5)

### Built With

* Python
* Tensorflow
* Numpy
* Street View Static API 

## Usage

* main.py trains a model with Inception-ResNet-v2 for feature extraction for 125 epochs on the full dataset. To use a different feature extraction or to exclude augmented data, change the variables that are initialised in main.py <br />

The following scripts were used to split the map and gather data: 

* make_grid.ipynb splits the map into 88 squares (method of splitting from https://nirvan66.github.io/geoguessr.html) 

* gen_images.py generates 200 images per square in the grid, saves in 'images/'.

The generated data can be downloaded from:
https://www.dropbox.com/s/hmz8m8sd3d2u8xj/images.zip?dl=0

* visualize_data.ipynb includes a visualisation of the gathered data

The following scripts were used to implement the network and data loader: 

* network.py initialises the network we will train.

* load_data.py loads the train, test and validation data arrays from a directory containing the data.

training_plots.ipynb generated plots that demonstrate the training process.

<!-- ROADMAP -->
## Roadmap

- [x] Data Gathering
- [x] CNN that maps image --> region + coordinates 
- [ ] Maybe a network that maps image --> country
- [ ] LSTM with sequential input of the same area
- [ ] Expand Map and gather more data
- [ ] Chrome Extension that plays the game using a trained model

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- CONTACT -->
## Contact

Saleh Alwer - saleh.tarek.issa.alwer@umail.leidenuniv.nl


