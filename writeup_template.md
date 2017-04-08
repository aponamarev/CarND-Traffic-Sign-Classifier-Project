# **Traffic Sign Recognition** 


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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[train_img1]: ./reporting_results/train_img1.png "Train Img 1"
[train_img2]: ./reporting_results/train_img1.png "Train Img 2"
[train_img3]: ./reporting_results/train_img1.png "Train Img 3"
[data_dist]: ./reporting_results/data_distribution.png "Data Distribution"
[LeNet_accuracy]: ./reporting_results/LeNet_accuracy.png "Accuracy: LeNet"
[LeNet_Dro_accuracy]: ./reporting_results/LeNet_accuracy.png "Accuracy: LeNet"
[LeNet_dropout_bn_accuracy]: ./reporting_results/LeNet_dropout_bn.png "Accuracy: LeNet with dropout and bn"
[SmallFilters_Accuracy]: ./reporting_results/SmallFilters_Accuracy.png "Accuracy: SmallFilter Net"
[SmallFilters_Loss]: ./reporting_results/SmallFilters_Loss.png "Loss: SmallFilter Net"
[LeNet_graph]: ./reporting_results/LeNet_graph.png "Graph: LeNet"
[SmallFilters_graph]: ./reporting_results/smallfilters_graph.png "Graph: SmallFilters"



## Rubric Points  
---

The code in this repository presents two versions of the algorthm:
* PDF - The posterior probability of the classifier is adjusted for prior probability of the dataset. The adjustment provided as a tf.constant. Values for this constant calculated in PDF function (Probability Density Function) in the 3rd code cell of Traffic_Sign_Classifier-PDF.ipynb
* Rebalanced - rebalances the dataset through syntetic random oversampling to achieve equal probability of the dataset. The random is done by creating a new set of indeces of samples - trainset_balanced_ids. trainset_balanced_ids contains indeces for samples with an equal distribution of classes. Random oversampling is done in the 3rd cell of Traffic_Sign_Classifier_StoreSummaries.ipynb

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fifth code cell of the Traffic_Sign_Classifier-PDF.ipynb notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the samples across classes. As one can observer in the histogram below the dataset is very unbalanced. If the classificator trained on such dataset, it may result in unbalanced classifier that will more likely to classify images as belonging the class that has a large presence in the training data.

![alt text][data_dist]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth and sixth code cells of the IPython notebook.

As a first step, I normalized images with the dataset mean for each of the channels and scalled features by 255 (which represents a rough estimate of the dataset range). The reason for the step is to ensure consistency between the inputs (images) and the expected values for the acivaion layer and the convolution operations. Both expect the incoming featuremaps roughly normally distributed centered around 0.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the data provided with the original project. As provided data already was split into the train, test, and validation datasets, I decided to keep the data without any changes to save time.

As described above my training set had 34,799 number of images. My validation set and test set had 12,630 images.

Outside of the syntetic random oversampling, I did not use any data augmentation.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tested 3 model architectures in this assignment:
1. LeNet with Elu activation
2. LeNet with Elu activation, Dropout after fully connected layer 4, and Batch Normalization after each maxpool layer.
3. SmallFilterNet. Small filter net includes 3 modules that consist of two consecutive 3x3 convolutinons with increasing number of layers. The second layer of the module includes [1,2,2,1] stride (as an alternative to pooling function). The last layer of the module is a 1x1 bottleneck intended to compress the featuremaps to decrease computational load. 1x1 bottle neck structure was borrowed from network in the network and inception architectures. The modules are followed by batch normalization layers. SmallFilterNet also features a dropout layer (applied to the featuremaps of the last module). The modules are followed by two convolution layers decreasing to the featuremap size to 1x1. The last layer the network is 1x1 fully convolutional layer with the number of outputs equal to the number of classes.

Each of the nets architechtures features tf.nn.softmax_cross_entropy_with_logits loss. 

Please find tensorboard representation of the net architectures below:
LeNet:
![alt text][LeNet_graph]



SmallFilterNet:
![alt text][SmallFilters_graph]


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training evaluation functions is located in cells 8 and 9. The cell 7 contains a general code of initialization of net structures. The code for training the models is located in cells 10-12. 

The training includes the following:
Optimizer - Adam Optimizer
Batch size - 512 examples
Epochs - LeNet architectures are trained for 50 epochs. SmallFilter net was trained for 70 epochs.
Learning rate - Adam Optimizer initialized with 1e-3 learning rate.

Overall the training could have been done in a smaller number of iterations. The training for all of the listed above architectures plateaued after rouhgly 35-40 epochs:

SmallFilter Accuracy log:
![alt text][SmallFilters_Accuracy]

SmallFilter Loss log:
![alt text][SmallFilters_Loss]


Models were trained on ec2.P2 instance equiped with Nvidia Tesla K80 GPU (12Gb memory).

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
