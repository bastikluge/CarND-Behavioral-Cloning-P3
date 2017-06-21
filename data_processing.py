import csv
import cv2
import numpy as np
import pickle
import os
import glob
import sklearn
import random

###################################################################

PICKLE_FILE_NAME_BASE = 'pre_proc_'
PICKLE_SIZE_FILE_NAME = 'pre_proc_size.p'

###################################################################
# Function to construct a pickle file name for a given index
###################################################################
def get_pickle_file_name(index=0):
	pickle_file_name = PICKLE_FILE_NAME_BASE + str(index) + '.p'
	return pickle_file_name

###################################################################
# Function to check if preprocessed data from a pickle file exits
###################################################################
def has_preprocessed_data(index=0):
	return os.path.isfile(get_pickle_file_name(index))

###################################################################
# Function to load preprocessed data from a pickle file
# param[in] index	index of the pickle data file
# param[in] verbose	flag to activate console output
# return	image data, steering angles (as numpy arrays)
###################################################################
def load_preprocessed_data(index=0, verbose=False):
	if verbose:
		print('   Loading preprocessed data [', index, ']...')
	pickle_file = open(get_pickle_file_name(index), 'rb')
	preprocessed_data = pickle.load(pickle_file)
	x = preprocessed_data['features']
	y = preprocessed_data['labels']
	if verbose:
		print('   ...Done')
	return x, y

###################################################################
# Function to save preprocessed data to a pickle file
# param[in] x	image data (as numpy array)
# param[in] y	steering angle (as numpy array)
# param[in] index	index of the pickle data file
# param[in] verbose	flag to activate console output
###################################################################
def save_preprocessed_data(x, y, index=0, verbose=False):
	if verbose:
		print('   Saving preprocessed data [', index, ']...')
	preprocessed_data = {'features': x, 'labels': y}
	pickle_file = open(get_pickle_file_name(index), 'wb')
	pickle.dump(preprocessed_data, pickle_file)
	if verbose:
		print('   ...Done')

###################################################################
# Function to save preprocessed data in chunks to a pickle file
# param[in] x			image data (as numpy array)
# param[in] y			steering angle (as numpy array)
# param[in] chunk_index	index of first chunk to be written
#						(if 0, all saved chunks will be deleted)
# param[in] chunk_size	number of data entries per chunk
# return 	index of next chunk to be written
###################################################################
def save_preprocessed_chunks(x, y, chunk_index=0, chunk_size=1024):
	print()
	print('############## save_preprocessed_chunks ##############')
	pickle_sizes = {}
	if chunk_index == 0:
		# Remove old files
		files = glob.glob(PICKLE_FILE_NAME_BASE + '*')
		print('Removing old pickle files:', files)
		for f in files:
			os.remove(f)
	else:
		# update pickle sizes with existing files
		pickle_size_file = open(PICKLE_SIZE_FILE_NAME, 'rb')
		pickle_sizes = pickle.load(pickle_size_file)
	# Shuffle data
	print('Shuffling data...')
	x, y = sklearn.utils.shuffle(x, y)
	# Write new files
	data_size = len(x)
	for i in range(0, data_size, chunk_size):
		x_chunk = x[i:i+chunk_size]
		y_chunk = y[i:i+chunk_size]
		print('Processing chunk', chunk_index, ', size: ', len(x_chunk))
		pickle_sizes[chunk_index] = len(x_chunk)
		save_preprocessed_data(x_chunk, y_chunk, index=chunk_index, verbose=True)
		chunk_index += 1
	# Write pickle size file
	pickle_size_file = open(PICKLE_SIZE_FILE_NAME, 'wb')
	pickle.dump(pickle_sizes, pickle_size_file)
	print('######################################################')
	print()
	return chunk_index

###################################################################
# Function to generate data in batches
# param[in] chunk_indices	indices of chunks from which the data shall be generated
# param[in] batch_size		expected data batch size
# yield		image data, steering angles (as numpy arrays)
###################################################################
def generator(chunk_indices, batch_size=128):
	index = 0
	nbr_indices = len(chunk_indices)
	while 1:
		chunk_index = chunk_indices[index]
		# load the data for the current index
		x, y = load_preprocessed_data(chunk_index)
		# update the data index
		if (index + 1) < nbr_indices:
			index += 1
		else:
			index = 0
		# process and yield the data
		x, y = sklearn.utils.shuffle(x, y)
		num_samples = len(x)
		for offset in range(0, num_samples, batch_size):
			x_batch = x[offset:offset+batch_size]
			y_batch = y[offset:offset+batch_size]
			yield x, y

###################################################################
# Function to create training and validation generator
# param[in] batch_size	expected data batch size
# yield		training generator, validation generator,
#           number of training samples, number of validation samples
###################################################################
def create_generators(batch_size=128):
	print()
	print('################## create_generators #################')
	# split pickle files into training and validation files
	pickle_size_file = open(PICKLE_SIZE_FILE_NAME, 'rb')
	pickle_sizes = pickle.load(pickle_size_file)
	nbr_chunks = len(pickle_sizes)
	nbr_train  = int(0.8 * nbr_chunks)
	pickle_indices = list(range(nbr_chunks))
	random.shuffle(pickle_indices)
	train_indices = pickle_indices[0:nbr_train]
	valid_indices = pickle_indices[nbr_train:nbr_chunks]
	train_size = 0
	valid_size = 0
	for i in range(len(train_indices)):
		train_size += pickle_sizes[train_indices[i]]
	for i in range(len(valid_indices)):
		valid_size += pickle_sizes[valid_indices[i]]
	print('Training   generator uses', train_size, 'samples from', train_indices)
	print('Validation generator uses', valid_size, 'samples from', valid_indices)
	# create and return generators
	train_generator = generator(train_indices, batch_size=batch_size)
	valid_generator = generator(valid_indices, batch_size=batch_size)
	print('######################################################')
	print()
	return train_generator, valid_generator, train_size, valid_size
	
###################################################################
# Function to read the data in a list of file paths
# param[in] file_paths 	list of paths from which csv-files and image data shall be loaded
# return    numpy arrays of images and steering angles
###################################################################
def read_data(file_paths):
	print()
	print('###################### read_data #####################')
	images = []
	angles = []
	for file_path in file_paths:
		# read csv file
		print('Processing file path', file_path, ':')
		lines = []
		csv_file_name = file_path + '/driving_log.csv'
		print('   Opening', csv_file_name, '...')
		with open(csv_file_name) as csvfile:
			print('   ... Done')
			reader = csv.reader(csvfile)
			line_count = 0
			print('   Reading', csv_file_name, '...')
			for line in reader:
				line_count += 1
				lines.append(line)
			print('   ... Done (', line_count, 'lines )')
		# read image files
		is_header = True
		print('   Reading image data...')
		for line in lines:
			if is_header:
				is_header = False
			else:
				source_path = line[0]
				file_name = source_path.split('/')[-1]
				img_file_name = file_path + '/IMG/' + file_name
				image = cv2.imread(img_file_name)
				images.append(image)
				angle = float(line[3])
				angles.append(angle)
		print('   ... Done')
	print('######################################################')
	print()
	return np.array(images), np.array(angles)

###################################################################
# Test method for read_data
###################################################################
def test_read_data():
	file_paths = []
	file_paths.append('./data')
	images, angles = read_data(file_paths);
	print('images shape:', images.shape)
	print('angles shape:', angles.shape)
	
# test_read_data()

###################################################################
# Scales the images by the given factor
# param[in] images					image data
# param[in] image_resize_factor		factor, by which the images shall be scaled
# param[in] verbose					flag to activate textual output
# return scaled images
###################################################################
def scale_images(images, image_resize_factor, verbose=False):
	scaled_images = None
	if (len(images) != 0) & (image_resize_factor != 1.0):
		if verbose:
			print('#################### scale_images ####################')
			print('Scaling from (', images[0].shape[0], ',', images[0].shape[1],
			      ') to (', int(image_resize_factor * images[0].shape[0]),
				  ',',      int(image_resize_factor * images[0].shape[1]), ')...')
		scaled_images = []
		for image in images:
			scaled_image = cv2.resize(image, None,
									  fx=image_resize_factor,
									  fy=image_resize_factor,
									  interpolation = cv2.INTER_AREA)
			scaled_images.append(scaled_image)
		scaled_images = np.array(scaled_images)
		if verbose:
			print('Done...')
			print('######################################################')
	else:
		scaled_images = images
	return scaled_images

def scale_image(image, image_resize_factor):
	image = cv2.resize(image, None,
					   fx=image_resize_factor,
					   fy=image_resize_factor,
					   interpolation = cv2.INTER_AREA)
	return image

###################################################################
# Function to augment the data by flipping left-right
# param[in] images 	image data
# param[in] angles   steering angle data
# return    numpy arrays of augmented image and steering angle data
###################################################################
def augment_flip(images, angles):
	print()
	print('#################### augment_flip ####################')
	print('Processing...')
	images_flipped = np.fliplr(images)
	angles_flipped = -angles
	images_aug = np.concatenate((images, images_flipped))
	angles_aug = np.concatenate((angles, angles_flipped))
	print('...Done')
	print('######################################################')
	print()
	return images_aug, angles_aug

###################################################################
# Test method for augment_flip
###################################################################
def test_augment_flip():
	file_paths = []
	file_paths.append('./data')
	images, angles = read_data(file_paths);
	print('images shape:', images.shape)
	print('angles shape:', angles.shape)
	images, angles = augment_flip(images, angles)
	print('images shape:', images.shape)
	print('angles shape:', angles.shape)

# test_augment_flip()

###################################################################
# Function to crop part of the image
# param[in] x			image data
# param[in] crop_factor	factor, by which the top of the image shall be cropped
# return    numpy array of cropped image data
###################################################################
def crop_images(images, crop_factor, verbose=False):
	y_to   = images[0].shape[0]
	y_from = int(crop_factor * y_to)
	if verbose:
		print()
		print('##################### crop_data ######################')
		print('Processing: cropping y-axis to [', y_from, ',', y_to, ']...')
	images_cropped = images[:, y_from:y_to, :, :]
	if verbose:
		print('...Done')
		print('######################################################')
	return images_cropped

def crop_image(image, crop_factor, verbose=False):
	y_to   = image.shape[0]
	y_from = int(crop_factor * y_to)
	if verbose:
		print()
		print('##################### crop_data ######################')
		print('Processing: cropping y-axis to [', y_from, ',', y_to, ']...')
	image_cropped = image[y_from:y_to, :, :]
	if verbose:
		print('...Done')
		print('######################################################')
	return image_cropped