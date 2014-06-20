#ifndef _CREATE_PARAMS_WRAPPER_HPP_
#define _CREATE_PARAMS_WRAPPER_HPP_

#include "AdaptiveMedianBGS.hpp"
#include "Eigenbackground.hpp"
#include "GrimsonGMM.hpp"
#include "MeanBGS.hpp"
#include "PratiMediodBGS.hpp"
#include "WrenGA.hpp"
#include "ZivkovicAGMM.hpp"
#include "Image.hpp"
#include <cv.h>
#include <cxcore.h>

/**
 * @brief Updates ImageBase and IplImage data pointers.
 * 
 * @param updated_image 
 * @param updated_ipl_image
 * @param data_ptr 				data pointer
 * @param step 					step between adjacent rows in bytes
 */	
void set_image_data(
	ImageBase* updated_image, 
	IplImage* updated_ipl_image, 
	unsigned char* data_ptr, int step)
{
	cvSetData(updated_ipl_image, data_ptr, step);
	(*updated_image) = updated_ipl_image;
}

using namespace Algorithms::BackgroundSubtraction;

AdaptiveMedianParams CreateAdaptiveMedianParams(int width, int height, 
	float low_threshold, float high_threshold, 
	int sampling_rate, int learning_frames)
{
	Algorithms::BackgroundSubtraction::AdaptiveMedianParams params;
	params.SetFrameSize(width, height);
	params.LowThreshold() = low_threshold;
	params.HighThreshold() = high_threshold;
	params.SamplingRate() = sampling_rate;
	params.LearningFrames() = learning_frames;
	return params;
}

EigenbackgroundParams CreateEigenbackgroundParams(int width, int height, 
	float low_threshold, float high_threshold, int history_size, int dims)
{
	Algorithms::BackgroundSubtraction::EigenbackgroundParams params;
	params.SetFrameSize(width, height);
	params.LowThreshold() = low_threshold;
	params.HighThreshold() = high_threshold;
	params.HistorySize() = history_size;
	params.EmbeddedDim() = dims;
	return params;
}

GrimsonParams CreateGrimsonGMMParams(int width, int height,
	float low_threshold, float high_threshold, 	
	float alpha, float max_modes)
{
	Algorithms::BackgroundSubtraction::GrimsonParams params;
	params.SetFrameSize(width, height);
	params.LowThreshold() = low_threshold;
	params.HighThreshold() = high_threshold;
	params.Alpha() = alpha;
	params.MaxModes() = max_modes;
	return params;
}

MeanParams CreateMeanBGSParams(int width, int height,
	unsigned int low_threshold, unsigned int high_threshold, 	
	float alpha, int learning_frames)
{
	Algorithms::BackgroundSubtraction::MeanParams params;
	params.SetFrameSize(width, height);
	params.LowThreshold() = low_threshold;
	params.HighThreshold() = high_threshold;
	params.Alpha() = alpha;
	params.LearningFrames() = learning_frames;
	return params;
}

PratiParams CreatePratiMediodBGSParams(int width, int height,
	unsigned int low_threshold, unsigned int high_threshold, 	
	int weight, int sampling_rate, int history_size)
{
	Algorithms::BackgroundSubtraction::PratiParams params;
	params.SetFrameSize(width, height);
	params.LowThreshold() = low_threshold;
	params.HighThreshold() = high_threshold;
	params.Weight() = weight;
	params.SamplingRate() = sampling_rate;
	params.HistorySize() = history_size;
	return params;
}

WrenParams CreateWrenGAParams(int width, int height,
	float low_threshold, float high_threshold, 	
	float alpha, int learning_frames)
{
	Algorithms::BackgroundSubtraction::WrenParams params;
	params.SetFrameSize(width, height);
	params.LowThreshold() = low_threshold;
	params.HighThreshold() = high_threshold;
	params.Alpha() = alpha;	
	params.LearningFrames() = learning_frames;	
	return params;
}

ZivkovicParams CreateZivkovicAGMMParams(int width, int height,
	float low_threshold, float high_threshold, 	
	float alpha, int max_modes)
{
	Algorithms::BackgroundSubtraction::ZivkovicParams params;
	params.SetFrameSize(width, height);
	params.LowThreshold() = low_threshold;
	params.HighThreshold() = high_threshold;
	params.Alpha() = alpha;	
	params.MaxModes() = max_modes;	
	return params;
}

#endif