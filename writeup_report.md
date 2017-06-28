#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[img_conv_net]:          ./plots/nvidia_conv_net.png                  "NVIDIA ConvNet architecture"
[img_angles_data]:       ./plots/angle_distribution_data.png          "Histogram of angles from downloaded data"
[img_angles_curves]:     ./plots/angle_distribution_data_curves.png   "Histogram of angles from collected curves data"
[img_raw]:               ./plots/input_image_initial.png              "Raw image input data"
[img_preprocessed]:      ./plots/input_image_preprocessed.png         "Preprocessed image data"
[img_mse_data_lr0050]:   ./plots/mse_training_data_lr-005.png         "MSE for downloaded data with angle correction +/-0.050"
[img_mse_data_lr0075]:   ./plots/mse_training_data_lr-0075.png        "MSE for downloaded data with angle correction +/-0.075"
[img_mse_data_lr0100]:   ./plots/mse_training_data_lr-010.png         "MSE for downloaded data with angle correction +/-0.100"
[img_mse_data_lr0125]:   ./plots/mse_training_data_lr-0125.png        "MSE for downloaded data with angle correction +/-0.125"
[img_mse_curves_lr0075]: ./plots/mse_training_data+curves_lr-0075.png "MSE for complete data with angle correction +/-0.075"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* config.py containing some constants shared by different python files
* data_processing.py containing the functions used for pre-processing and augmenting the image data
* plot_helper.py containing the functions used to visualize the data and training process
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I implemented the NVIDIA Convolutional Network model described in "End to End Learning for Self-Driving Cars" (M. Bojarski et al, 2016, arXiv:1604.07316). I thought that this model would be appropriate as it was defined for the same purpose and uses input images of comparable size. As expected, the model was already performing quite well with the driving data downloaded from the link provided in the Udacity Self-Driving-Car-Nanodegree lecture. These initial tests were performed on the re-scaled, cropped and normalized images (re-scaled and cropped in such a way that the image dimensions fit the input dimensions expected by the NVIDIA Convolutional Network model) of the center camera only. After training for 10 epochs I observed that the model had converged after approximately 5 epochs and the mean squared errors (MSE) of both training and validation data set had first decreased to and then oscillated around similar values.

I was first wondering whether it would be necessary to add some dropout or pooling layers to reduce overfitting of the model, but the training history didn't show any evidence of this necessity. Also, in the context of estimating an angle value, in particular dropout seemed counter-productive to me, as it would significantly change the magnitude of the estimated value when using it with a non-zero dropout probability during training and a zero dropout probability during validation. So I didn't further modify the network and continued working on data collection and augmentation.

After I had collected more training data (in particular curve driving data) and implemented the augmentation techniques suggested in the Udacity Self-Driving-Car-Nanodegree lecture (flipping and usage of left and right camera images), I trained the model again and tested it in the autonomous mode of the simulator. The car was able to successfully and smoothly drive around the complete track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 78-89) consisted of a convolution neural network with the following layers and layer sizes:
* a lambda layer to normalize the range of each color channel to [-0.5, 0.5]
* a convolutional layer with kernel size 5, stride size 2 and 24 feature dimensions
* a ReLU activation layer
* a convolutional layer with kernel size 5, stride size 2 and 36 feature dimensions
* a ReLU activation layer
* a convolutional layer with kernel size 5, stride size 2 and 48 feature dimensions
* a ReLU activation layer
* a convolutional layer with kernel size 3, stride size 1 and 64 feature dimensions
* a ReLU activation layer
* a convolutional layer with kernel size 3, stride size 1 and 64 feature dimensions
* a ReLU activation layer
* a flattening layer
* a fully connected layer with 100 feature dimensions
* a fully connected layer with 50 feature dimensions
* a fully connected layer with 10 feature dimensions
* a fully connected layer with 1 feature dimension (i.e., the output angle)

Here is a visualization of the architecture (taken from "End to End Learning for Self-Driving Cars" by M. Bojarski et al):

![NVIDIA ConvNet][img_conv_net]

I noticed only during this writeup that I forgot to specify the activation for the fully connected layers in my implementation, and therefore - as Keras default - no activation function was used. Although the original paper "End to End Learning for Self-Driving Cars" by M. Bojarski et al doesn't report where (and which) activation layers are used, I suppose that their fully connected layers use activation functions (except for the last one). So this probably explains why I didn't observe any overfitting: The 4 connected layers can be converted to one fully connected layer in a mathematically equivalent manner (the concatenation of 4 linear affine functions is a linear affine function), thereby significantly reducing the total effective size of the network. As evidenced by my training and verification results, the size of this reduced network well suits the problem at hand.

####3. Creation of the Training Set & Training Process

As a starting point I used the training data downloaded from the link provided in the Udacity Self-Driving-Car-Nanodegree lecture. In initial tests (performed on the re-scaled, cropped and normalized images of the center camera only) I observed that the car was getting off the track only in curves. The next images show the initial image data and the rescaled and cropped image data:

![Raw image data][img_raw]
![Preprocessed image data][img_preprocessed]

I then collected driving data from 2 rounds of forward, 2 rounds of reverse driving through track 1 and a number of recovering maneuvers, in which I drove the simulated vehicle back to the center of the road. Unfortunately the vehicle performed even worse (it was driving very close to the right boundary of the road and got off the road in an earlier curve than without the additional data). Most likely, this was due to the fact that the recovery maneuvers were started with a vehicle standing still at the side of the road and contained much data from a vehicle close to the side of the road without any steering interaction and little data from the recovery maneuver itself. So I collected data again, this time concentrating only on smooth curves driving and recording a few recovery maneuvers of a moving vehicle (such that only the recovery maneuver but not the driving at the side of the road would be recorded). Using augmented versions of this data together with augmented versions of the downloaded data worked fine and resulted in smooth driving of the vehicle. The next histograms show the distribution of the angle data of the downloaded and the newly collected driving data:

![Downloaded driving data][img_angles_data]
![Collected curve driving data][img_angles_curves]

Preprocessing of the images consisted of first rescaling it from (160, 320) to (100, 200) and then cutting off the top 34 pixels, resulting in an image size (66, 200) as expected by the NVIDIA Convolutional Network model. The related code is contained in lines 40 and 43 of model.py (resp. lines 229-257 and 300-324 of data_processing.py).

To augment the data set, I flipped each image (see line 42 of model.py and lines 265-276 of data_processing.py, respectively). I also used the images from the left and right camera together with slightly modified angles. To find out a good angle correction offset, I tried +/-0.050, +/-0.075, +/-0.100, +/-0.125 (i.e., a range of approximately +/-[3°, 7°]). In each case the model had converged after 5 epochs, the next table shows the results:

| Correction Value | Training Loss | Validation Loss | Autonomous Performance |
|------------------|---------------|-----------------|------------------------|
| +/-0.050         | 0.0086        | 0.0086          | smooth, got off the road in first curve before bridge |
| +/-0.075         | 0.0100        | 0.0092          | smooth, got off the road in first curve after bridge |
| +/-0.100         | 0.0110        | 0.0103          | slightly oscillating, got off the road in first curve after bridge |
| +/-0.125         | 0.0121        | 0.0120          | strongly oscillating, got off the road in first curve before bridge |

Here, values above +/-0.100 resulted in over-steering and a driving pattern, which oscillated between the left and right sides of the road. So I decided to use the value of +/-0.075, which had a reasonably slow loss value and showed a good perfomance in autonomous simulator mode.

After this collection and augmentation process, I had a number of @todo data points, which I used for the training of the model. To speed up the training process (which I executed on my CPU) I first saved chunks of preprocessed and shuffled data into pickle files, which were then accessed by generators during training and validation calculations (see lines 63-65 and 95 of model.py, 19-162 of data_processing.py, respectively).

In the training procedure, I finally randomly shuffled the data set and put 20% of the preprocessed data chunks into a validation set. I used the MSE as loss function and the Adam optimizer as optimization procedure. Because of the Keras built-in learning rate decay mechanism for the Adam optimizer, I didn't have to select a learning rate for the training procedure. I then trained the model for 5 epochs (which was evidenced by my initial tests with greater numbers of epochs), as can be seen in lines 97-100 of model.py. The result of the final training and validation procedure is shown below:

![MSE of final training and validation procedure][img_mse_curves_lr0075]

In this final run I observed a training loss of 0.0114 and a validation loss of 0.0114. Using the resulting model, the vehicle was able to drive smoothly in autonomous simulation mode throughout the complete first track without ever getting off the road.