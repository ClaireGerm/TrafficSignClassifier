﻿﻿﻿﻿﻿# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./mdimg/histogram.JPG "Histogram"
[image4]: ./mdimg/signs.JPG "Traffic Sign 1"


### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! 

### Data Set Summary & Exploration

#### 1. Dataset summary

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread over the labels:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing
As a first step, I decided to convert the images to grayscale because it reduces the amount of features, which results in a shorter execution time. After that I normalized the images which converts the int values of each pixel [0,255] to float values with range [-1,1]. The goal of data normalization is to reduce and even eliminate data redundancy.

#### 2. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scaled image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU		|         									|
| Max pooling				| 2x2 stride, outputs 5x5x16        									|
| Flatten					| outputs 400												|
| Fully connected | 	outputs 84 |
| RELU | |
| Fully connected | outputs 43 |
| Softmax |										| 


#### 3. Training the model
To train the model, I used the following optimizer and values for the hyperparameters:
- Optimizer = AdamOptimizer
- Epochs = 25
- Batch_size = 125
- learning_rate = 0.0009
- mu = 0
- sigma = 0.1

#### 4. Finding a solution
My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.940
* test set accuracy of 0.905

An iterative approach was chosen:
* I started with the LeNet model and the conditions from the course. I changed the input depth to 3 because I started with RGB instead of gray scaled images. That didn't give the desired result at all.
* My second step was to add grayscaling. That already gave me a better validation accuracy.
* My third step also slightly increased the results. I normalized the training and validation data.
* After that I decided to fine tune the values of the epochs, batch_size and learning_rate.  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I extracted from Google street view:

![alt text][image4]

These pictures won't be too hard to classify due to none of them being dark or unclear. The speed limit is probably the hardest one because of the number 30 looking similar to 80 or 50.

#### 2. Performance on the images from the internet

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)   									| 
| Turn right ahead     			| Turn right ahead 										|
| No passing					| No passing											|
| Keep right	      		| Bumpy Road					 				|
| Yield			| Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This doesn't compares favorably to the accuracy on the test set of 0.905.

#### 3. Model certainty

For the first image, the model is relatively sure that this is a speed limit sign (probability of 100%), and the image does contain a speed limit. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100         			| Speed limit (30km/h)   									| 
| 0     				| Speed limit (50km/h) 										|
| 0					| Speed limit (20km/h)											|
| 0	      			| Speed limit (80km/h)					 				|
| 0				    | Bicycles crossing      							|


For the second image, the model is relatively sure that this is a 'Turn right ahead' sign (probability of 100%), and the image does contain a 'Turn right ahead' sign. It contains a 'Keep right' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|100        			| Turn right ahead  									| 
| 0    				| No passing 										|
| 0					| Ahead only											|
| 0	      			| Roundabout mandatory					 				|
| 0				    | Keep left      							|

For the third image, the model is relatively sure that this is a 'No passing' sign (probability of 100%), and the image does contain a 'No passing' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100         			| No passing   									| 
| 0     				| No passing for vehicles over 3.5 metric tons										|
| 0					| Roundabout mandatory										|
| 0	      			| Priority road					 				|
| 0				    | Speed limit (100km/h)      							|

For the fourth image, the model is relatively sure that this is a 'Bumpy road' sign (probability of 100%), and the image doesn't actually contain a 'Bumpy road' sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.98974        			| Bumpy road   									| 
| 0.01009     				| Keep right 										|
| 0.00010					| Turn left ahead											|
| 0.00004	      			| Go straight or right					 				|
| 0.00002				    | Bicycles crossing      							|

For the last image, the model is relatively sure that this is a 'Yield' sign (probability of 100%), and the image does contain a 'Yield' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.99973         			| Yield   									| 
| 0.00027     				| Ahead only										|
| 0					| Speed limit (60km/h)										|
| 0	      			| No passing for vehicles over 3.5 metric tons					 				|
| 0				    | Speed limit (80km/h)      							|









