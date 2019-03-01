# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/v1.png "Visualization"
[image2]: ./examples/v2.png "Visualization"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/bicyclecrossing.jpg "Traffic Sign 1"
[image5]: ./examples/childrencrossing.jpg "Traffic Sign 2"
[image6]: ./examples/nopassing.jpg "Traffic Sign 3"
[image7]: ./examples/roadwork.jpg "Traffic Sign 4"
[image8]: ./examples/straightorright.jpg "Traffic Sign 5"
[image9]: ./examples/t1.png "Test 1 "
[image10]: ./examples/t2.png "test 2"
[image11]: ./examples/t3.png "test 3"
[image12]: ./examples/t4.png "Test 4"
[image13]: ./examples/t5.png "Test 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to not to convert the images to grayscale because I tried with gray processing but didn't so much benefit. The color image  have more information such as RED color for stop sign etc. 

As next step, I normalized the image data because normalize the data to have same range of feature values. and Also shuffling data serves the purpose of reducing variance and making sure that models remain general and overfit less.

I decided to use original data without augmented data because the test accuracy is accpetable. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	   | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image    | 
| Convolution 5X5       | 1x1 stride, padding, outputs 28x28x6  	|
| RELU					|					|
| Max pooling	      	| 2x2 stride, outputs 14x14x16				 
| Convolution 5x5       | 14x14x6, outputs 10x10x16
| RELU		
| Dropout                | 						|
| Max pooling	      	| 2x2 stride, 5x5x16 	   		     
| Fully connected		| input 400, output 400.    |	
| Fully connected		| input 400, output 120. 
| RELU	
| Dropout                | 						|     
| Fully connected 		| input 120, output 84		                 
| RELU	
| Dropout 
| Fully connected       | input 84, output 43                       												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

* My final model results were:
    * training set accuracy of 99%
    * validation set accuracy of 96%
    * test set accuracy of 94.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
 
    * I choose LeNet as it is discussed in class the result is good.
* What were some problems with the initial architecture?
    *  LeNet needs to be modified for 43 output class. However there were pre-processing bugs at beginning for my initial part as well.

     
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

    *  For traffic sign classifier, it needed to output all 43 classes instead of the 10 for decimals. When running, I discovered it gave a low accuracy of without started tuning the hyperparameters, therefore it was obvious the model needed to be more complex to capture traffic signs.

* Which parameters were tuned? How were they adjusted and why?

    * Epocs change from 60 to 30 as no much improvement after 30. 
    * minibatch size changed 64 to 128 to improve speed
    * Dropout changed to multiple places and finally put 3 after FC layer to avoid overfitting. 
    *  tried with different the learning rate and lower to increase the model’s stability 
    * Adam optimizer for training, reducing the loss cross-validating a one-hot encoded label with the model’s output. The test accuracy is 96%.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

    *  Dropout to improve validation accuracy.  It provides a way to regularization in neural networks which helps reducing interdependent learning amongst the neurons.
    *  shuffle reduces variance and making sure that models remain general and overfit less.
    

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
    *  I think LeNet was chosen as it is balance between computing power and accuracy.It is proven to be good at 32X32 level of  images. Of course there are many other well known more deeper architecture can do better but for traffic sign level of image complexity it is good enough without much longer training. 
    * Without much effort of adjustment for LeNet the result can be quite good out of. the  training set accuracy of 99%, the validation set accuracy of 96% and test set accuracy of 94.5%. For 5 new images test if can achieve 80% accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Expected: 
       29 (bicycles crossing), 
       9 (no passing), 
       36 (go straight or right), 
       25 (road work), 
       28 (children crossing)

Actual:
[28  9 36 25 28]


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is not sure if  this is a bicycle sign and got wrong prediction. The top five soft max probabilities were


| Probability         	|     Prediction	        			| 
|:---------------------:|:---------------------------------------------:| 
| .22         			| 25  									| 
| .20     				| 2 										|
| .14					| 5										
| .11	      			| 1					 				|
| .08				    | 21      							|

For the second image is predict correctly with labe 9 which no pass. 

| Probability         	|     Prediction	        			| 
|:---------------------:|:---------------------------------------------:| 
| .78         			| 9  									| 
| .21     				| 10										|
| .0					| 41									
| .0	      			| 23					 				|
| .0				    | 26      							|


![alt text][image9]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


