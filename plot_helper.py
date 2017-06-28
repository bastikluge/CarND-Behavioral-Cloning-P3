from keras.models import Model
import matplotlib.pyplot as plt

###################################################################
# Function to plot the training and validation loss for each epoch
# param[in] history_object  history object output from network training
###################################################################
def plot_history(history_object):
	# print the keys contained in the history object
	# print(history_object.history.keys())

	# plot the training and validation loss for each epoch
	plt.figure()
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()

###################################################################
# Function to plot a histogram of the angle data
# param[in] x  numpy array containing the angle data
###################################################################	
def plot_angle_histogram(x):
	# define symmetric range
	x_min = min(x)
	x_max = max(x)
	if -x_max < x_min:
		x_min = -x_max
	if -x_min > x_max:
		x_max = -x_min
	
	plt.figure()
	plt.hist(x, range=(x_min, x_max), bins=25) # odd number of bins to have 0 in the middle of a bin
	plt.title('distribution of angles')
	plt.ylabel('frequency [1]')
	plt.xlabel('steering angle [deg]')
	plt.show()