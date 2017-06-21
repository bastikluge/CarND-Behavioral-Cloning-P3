import config
import data_processing as dp
import plot_helper as ph
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
import cv2
	
###################################################################
# Prepare the data
###################################################################
input_shape = None

do_preprocess_data = True
if do_preprocess_data:
	# Read the data
	file_paths = []
	file_paths.append('./data')
	# file_paths.append('./data_curves+correction')
	file_paths.append('./data_f+r_round')
	file_paths.append('./data_f+r_round2')
	x_data, y_data = dp.read_data(file_paths)
	print('x_data shape:', x_data.shape)
	cv2.imshow('Initial sample image', x_data[0])
	ph.plot_angle_histogram(y_data)

	# Preprocess the data
	x_data = dp.scale_images(x_data, config.IMAGE_RESCALE_FACTOR, verbose=True)
	print('x_data shape:', x_data.shape)
	x_data, y_data = dp.augment_flip(x_data, y_data)
	x_data = dp.crop_images(x_data, config.IMAGE_CROP_FACTOR, verbose=True)
	
	# Print some information about the data
	cv2.imshow('Final sample image', x_data[0])
	print('x_data shape:', x_data.shape)
	print('y_data shape:', y_data.shape)
	input_shape=x_data[0].shape
	print('Number of preprocessed samples:', len(x_data))
	
	# Save the data
	saved_size = int(len(x_data) / config.PREPROCESSED_CHUNK_SIZE) * config.PREPROCESSED_CHUNK_SIZE
	dp.save_preprocessed_chunks(x_data[0:saved_size, :, :, :], y_data[0:saved_size], 0, config.PREPROCESSED_CHUNK_SIZE)
else:
	x_data, y_data = dp.load_preprocessed_data()
	input_shape=x_data[0].shape

###################################################################
# Define the model
###################################################################
print('Defining the model...')
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
model.add(Convolution2D(24, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(36, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(48, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid', subsample=(1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

###################################################################
# Use the model
###################################################################
# Compile, run and save the model
train_gen, valid_gen, train_size, valid_size = dp.create_generators()
print('Compiling and running the model...')
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_gen, samples_per_epoch=train_size,
									 validation_data=valid_gen, nb_val_samples=valid_size,
									 nb_epoch=config.NB_EPOCHS)
print()

# Save the model
print('Saving the model to model.h5')
model.save('model.h5')
print()

###################################################################
# Plot some output
###################################################################
ph.plot_history(history_object)