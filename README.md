# Overview
This library containing 7 popular background subtraction algorithms:
* adaptive median filtering
* eigenbackground
* single Gaussian
* Gaussian mixture models
* adaptive Gaussian mixture models
* running mean
*  mediod filtering

This library makes use of [OpenCV](http://opencv.org/).

# Building with CMake

	$ mkdir _build
	$ cd _build
	$ cmake ..
	$ make

# Building Python Interface

1. Install OpenCV with Python bindings enabled.
2. Edit **include_dirs** in the `setup.py`.
3. Build the cython module:

		$ pip install numpy cython
		$ python setup.py build_ext --inplace

4. Test the module using stream from the webcam:

		$ python pybgs_test.py

# Citation

If you find this software useful, please consider citing:

* Parks DH, Fels SS. 2008. Evaluation of background substraction algorithms with post-processing. IEEE Fifth International Conference on Advanced Video and Signal Based Surveillance (AVSS2008): 192-199.

# License

This software is released under GNU General Public License version 3. See LICENSE.txt for further details.

# Contact

Inquires regarding this software can be directed to: *donovan_parks_at_gmail_dot_com*
