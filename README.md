# Pokemon Classifier with Python
Simple Pokemon classifier with Python, OpenCV, and Keras

![1](https://imgur.com/VliJvzW.png)
![2](https://imgur.com/IDKfFOP.png)
![3](https://imgur.com/BZgoeMz.png)
![4](https://imgur.com/cGIHE2q.png)
![5](https://imgur.com/wGOxvFn.png)

## Installation 
- It is quite difficult to install **Tensorflow** and/or **Keras** and have it setup on a GPU, please check out how to do that [here](https://www.tensorflow.org/install)
- Install OpenCV for Python and `imutils`, also `matplotlib` if you want to graph, else it is not required.

## Limitation
- Since it was only trained on 5 types of Pokemon _(Bulbasaur, Charmander, Pikachu, Mewtwo, Squirtle)_ as of now so it cannot be used to classify all Pokemons
- The model is a very limited CNN model so the accuracy is not very high, please consider training on a bigger dataset (this model was trained on roughly **1000 images in total**) or loading a better pre-trained model.
