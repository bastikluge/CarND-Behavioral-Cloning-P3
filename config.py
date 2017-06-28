###################################################################
# Collection of parameters used in behavioral cloning processing
###################################################################
# Columns in csv file
COLUMN_INDEX_CENTER     = 0
COLUMN_INDEX_LEFT       = 1
COLUMN_INDEX_RIGHT      = 2
NUMBER_COLUMN_INDICES   = 3
# Angle corrections for left/right camera
ANGLE_LEFT_CAM_OFFSET   = 0.075
ANGLE_RIGHT_CAM_OFFSET  = -0.075
# Chunk size of preprocessed pickle files
PREPROCESSED_CHUNK_SIZE = 512
# Scaling factor applied to input images
IMAGE_RESCALE_FACTOR 	= 10/16 # (160, 320) -> (100, 200)
# Cropping factor (applied to crop the top of the image) applied to input images
IMAGE_CROP_FACTOR    	= 0.34  # (100, 200) -> (66, 200)
# Number of epochs for which the convolutional network is trained
NB_EPOCHS            	= 5