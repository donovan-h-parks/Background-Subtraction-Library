import numpy as np
import cv2
import pybgs

#params = { 
# 	'algorithm': 'eigenbackground', 
# 	'low': 15 * 15,
# 	'high': 15 * 15 * 2,
# 	'history_size': 100,
# 	'dims': 5 }

params = { 
 	'algorithm': 'adaptive_median', 
 	'low': 40,
 	'high': 40 * 2,
 	'sampling_rate': 10,
 	'learning_frames': 400 }

# params = { 
# 	'algorithm': 'grimson_gmm', 
# 	'low': 3.0 * 3.0,
# 	'high': 3.0 * 3.0 * 2,
# 	'alpha': 0.01,
# 	'max_modes': 3 }

# params = { 
# 	'algorithm': 'mean_bgs', 
# 	'low': 3 * 30 * 30,
# 	'high': 3 * 30 * 30 * 2,
# 	'alpha': 1e-6,
# 	'learning_frames': 30 }

# params = { 
# 	'algorithm': 'prati_mediod_bgs', 
# 	'low': 50,
# 	'high': 50 * 2,
# 	'weight': 1,
# 	'sampling_rate': 5,
# 	'history_size': 16 }

#params = { 
# 	'algorithm': 'wren_ga', 
# 	'low': 3.5 * 3.5,
# 	'high': 3.5 * 3.5 * 2,
# 	'alpha': 0.05,
# 	'learning_frames': 30 }

#params = { 
#	'algorithm': 'zivkovic_agmm', 
#	'low': 5 * 5,
#	'high': 5 * 5 * 2,
#	'alpha': 0.001,
#	'max_modes': 3 }

bg_sub = pybgs.BackgroundSubtraction()	
camera_source = cv2.VideoCapture()
camera_source.open(0)

i = 0
error, img = camera_source.read()
high_threshold_mask = np.zeros(shape=img.shape[0:2], dtype=np.uint8)
low_threshold_mask = np.zeros_like(high_threshold_mask)
bg_sub.init_model(img, params)

while cv2.waitKey(30) == -1:
    error, img = camera_source.read()
    bg_sub.subtract(i, img, low_threshold_mask, high_threshold_mask)
    bg_sub.update(i, img, high_threshold_mask)
    cv2.imshow('foreground', low_threshold_mask)
    # cv2.imshow('background', bg_sub.get_background())
    i += 1

 
