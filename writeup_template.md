# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./TestImages/cnn-architecture-624x890png "Model Visualization"
[image2]: ./TestImages/Test0.jpg "Normal Image"
[image3]: ./TestImages/Test1.jpg "Flipped Image"
[image4]: ./TestImages/Test2.jpg "Normal Image"
[image5]: ./TestImages/Test3.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 32 and 64 (model.py lines 105-111) 

The model includes RELU layers to introduce nonlinearity (code line 105 ff), and the data is normalized and mean centered in the model using a Keras lambda layer (code line 104). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 117). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 124-126). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Tight curves and road sections with different curbs were driven several times. To generate a bigger variety of images I also drove into the "wrong" direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reduce the input size of the original camera image, increasing the channel depth and finally breaking it down to a single steering angle value.

My first step was to use a convolution neural network model similar to the Nvidia DAVE-2 approach. I thought this model might be appropriate because it showed very good results in the original paper "End-to-End Deep Learning for Self-Driving Cars" by Bojarksi, Firner, et al.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I reduced amount of epochs to only 10.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added more images of successfull ride of that particular section.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 103-125) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap on track one using center lane driving.
I then recorded the vehicle recovering from the left side and right side of the road back to center so that the vehicle would learn to return from the area near to the curbs back to the center of the road.
Then I repeated this process on track two in order to get more data.

To receive more data points the right and left camera images of the car were used too. The recorded steering angle was corrected by a tunable hyperparameter (correction = 0.21).
To augment the data set, I also flipped images and steering angles. So it is guaranteed that the training data is equally balanced between right curves and left curves. Examples of original and flipped image:

![alt text][image2]
![alt text][image3]
![alt text][image3]
![alt text][image4]

Images were augmended even more by randomly cropping the image and removing top and bottom pixels of the image. Since the upper part of the image (mostly sky) and the hood of the car imply no helpful information about the steering angle. By adding a random factor for each image I tried to improve the generalisation of the CNN and make predictions more robust. (model.py line 45-67)

After the collection process, I had about 57.000 data points.

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
