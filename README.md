# **Behavioral Cloning** 

**Project Goals**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[cnn_design]: ./9-layer-ConvNet-model.png "Nvidia Autopilot CNN Architecture"
[log]: ./output/driving_log.jpg "Driving Log CSV"
[flipped]: ./output/flipped.JPG "Flipped Image and Inverted Steering Angle"
[vis]: ./output/vis1.JPG "Training Data Visualization"
[run3-loss]: ./output/run3-loss.png "Ex. Overfitting on run 3"
[run5-loss]: ./output/run5-loss.png "Loss Graph on run 5"

### Model Architecture and Training Strategy

After testing a very basic model with just a single fully connected layer, I researched a model from  `End to End Learning for Self-Driving Cars` paper https://arxiv.org/abs/1604.07316. The Original CNN Design was developed by Nvidia for Autopilot training. Originally the network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers.
![alt text][cnn_design]

 
My model, based on the Nvidia model, consists of a convolution neural network with 2x2 filter sizes and depths between 24 and 64 (model.py lines 63-67) 

The model includes RELU layers to introduce nonlinearity (model.py line 63-73), the data is normalized in the model using a Keras lambda layer (model.y line 61) and a cropping layer. Cropping layer removes the upper and lower portion of the image eliminating the mountains & skies and the hud of the car this gives us a focused view of the road.

#### Attempts to reduce overfitting in the model

The model does not use dropout but rather we run a couple of epoch values between 2 & 7 stopped exactly at where the validation loss begins to peak. The ideal number of epochs was 5. visualizing the loss graph was helpful in finding the optimal epoch

![alt text][run3-loss]
The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 24). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 76).

#### Appropriate training data

Training data is from the sample data given for this exercise.
The Training data was split 80:20 to create a training, validation set
Sample data consist of 8037 images

![alt text][log]

![alt text][vis]

### Model Architecture and Training Strategy

#### Solution Design Approach

The overall strategy for deriving a model architecture was to start from a basic model and iteratively improve the 
accuracy of that model

My first step was to use a very basic convolution neural network model with 1 dense layer because I just wanted to set up a valid pipeline that I could iterate over. To do this I import the data samples and use just the center images and the corresponding angle measurement,
I normalize my data then run it through my basic model with 5 epochs

        model = Sequential()
        model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3)))
        model.add(Flatten())
        model.add(Dense(1))

The performance was not good. the vehicle just veered off the tracks almost going in circles
[![Watch the video](https://img.youtube.com/vi/-S8UjA13aQY/default.jpg)](https://youtu.be/-S8UjA13aQY)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Let's try epoch 2 ( since we noticed the data might have slightly overfitted on with 5 epochs)

[![Watch the video](https://img.youtube.com/vi/QGgGpJtCahY/default.jpg)](https://youtu.be/QGgGpJtCahY)

Epoch 2 a little bit better, but vehicle veers off to the left

Next step was to improve the CNN model. I skip LeNet all together since I had seen the results from the lesson so let's do Nvidia's  :)
Implemented a model based on this GitHub repository https://github.com/0bserver07/Nvidia-Autopilot-Keras
and trained for 2 epochs. The model was underfitting and vehicle runs into the bridge.

[![Watch the video](https://img.youtube.com/vi/9aulrl-H7ZM/default.jpg)](https://youtu.be/9aulrl-H7ZM)

Ok, let's train for 5 epochs. Training loss vs validation loss is not bad, let's try model on track.
and surprisingly model drives well But the vehicle dives into the river at the right turn after the bridge.

[![Watch the video](https://img.youtube.com/vi/sEglDtG8m6o/default.jpg)](https://youtu.be/sEglDtG8m6o)

At this point I observe the track has more left turns than right turns I am guessing that's
why our model based on the track data is able to navigate left turns well but not right turns
Ah! this is called the  left turn bias

Need more data? let's use multiple cameras! Train again with same CNN model for 5 epoch. Results? The vehicle again goes for a 
dive into the river.

Let's try more data with some augmented right turns. I removed multiple camera images
and used just the center camera. I flipped the images and the steering angle measurements

        image_flipped = np.fliplr(image)
        measurement_flipped = -angle
        images.append(image_flipped)
        measurements.append(measurement_flipped)

![alt text][flipped]
Train again with same CNN model for 5 epoch and Voila! 
The vehicle is able to drive autonomously around the track without leaving the road or going for a dive in the river. :)
[![Watch the video](https://img.youtube.com/vi/2ptZBSjly38/default.jpg)](https://youtu.be/2ptZBSjly38)
![alt text][run5-loss]

#### Final Model Architecture

The final model architecture (model.py lines 61-73) consisted of a convolution neural network with the following layers and layer sizes

        model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3)))
        model.add(Cropping2D(cropping=((70,25),(0,0)))) #Top 70 pixels and bottom 25 pixels cropped
        model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
        model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
        model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
        model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
        model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
        model.add(Flatten())
        model.add(Dense(1164, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='tanh'))

#### Project Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### How to Run
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
