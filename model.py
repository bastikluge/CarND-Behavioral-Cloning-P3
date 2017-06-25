import config
import data_processing as dp
import plot_helper as ph
import numpy as np
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
	file_paths.append('./data_curves')
	# file_paths.append('./data_curves+correction')
	# file_paths.append('./data_f+r_round')
	# file_paths.append('./data_f+r_round2')
	x_data = None
	y_data = None
	chunk_index = 0
	for file_path in file_paths:
		for column_index in range(config.NUMBER_COLUMN_INDICES):
			# Determine angle offset
			angle_offset = 0.0
			if column_index == config.COLUMN_INDEX_LEFT:
				angle_offset = config.ANGLE_LEFT_CAM_OFFSET
			if column_index == config.COLUMN_INDEX_RIGHT:
				angle_offset = config.ANGLE_RIGHT_CAM_OFFSET
			
			# Read the data
			x_tmp, y_tmp = dp.read_data(file_path, column_index=column_index, angle_offset=angle_offset)
			print('x_data shape:', x_tmp.shape)
			#cv2.imshow('Initial sample image', x_tmp[0])
			#cv2.imwrite('initial_image.png', x_tmp[0])
			ph.plot_angle_histogram(y_tmp)

			# Preprocess the data
			x_tmp = dp.scale_images(x_tmp, config.IMAGE_RESCALE_FACTOR, verbose=True)
			print('x_data shape:', x_tmp.shape)
			x_tmp, y_tmp = dp.augment_flip(x_tmp, y_tmp)
			x_tmp = dp.crop_images(x_tmp, config.IMAGE_CROP_FACTOR, verbose=True)
			#cv2.imshow('Final sample image', x_tmp[0])
			#cv2.imwrite('preprocessed_image.png', x_tmp[0])
			
			# Print some information about the data
			print('x_data shape:', x_tmp.shape)
			print('y_data shape:', y_tmp.shape)
			input_shape=x_tmp[0].shape
			print('Number of preprocessed samples:', len(x_tmp))
			
			# Append to data
			if chunk_index == 0:
				x_data = x_tmp
				y_data = y_tmp
			else:
				x_data = np.concatenate((x_data, x_tmp))
				y_data = np.concatenate((y_data, y_tmp))
			data_size = len(x_data)
			
			# Save the data
			saved_size = int(data_size / config.PREPROCESSED_CHUNK_SIZE) * config.PREPROCESSED_CHUNK_SIZE
			print('Saving', saved_size, 'samples, ', data_size - saved_size, 'remaining')
			chunk_index = dp.save_preprocessed_chunks(x_data[0:saved_size, :, :, :], y_data[0:saved_size], chunk_index=chunk_index, chunk_size=config.PREPROCESSED_CHUNK_SIZE)
			
			# Truncate data
			x_data = x_data[saved_size:data_size, :, :, :]
			y_data = y_data[saved_size:data_size]
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