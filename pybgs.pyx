# distutils: language = c++
# distutils: sources = Image.cpp Eigenbackground.cpp AdaptiveMedianBGS.cpp GrimsonGMM.cpp MeanBGS.cpp  PratiMediodBGS.cpp  WrenGA.cpp  ZivkovicAGMM.cpp
# distutils: libraries = opencv_core opencv_highgui

import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libcpp cimport bool

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef extern from "types_c.h":
    # C MACRO defined as an external int variable
    cdef int IPL_DEPTH_8U

    ctypedef struct IplImage:
        char *imageData
        int  widthStep
        int width
        int height

    cdef struct CvSize:
        int width
        int height

cdef extern from "core_c.h":
    IplImage *cvCreateImageHeader(CvSize size, int depth, int channels)
    void cvReleaseImageHeader(IplImage** image)
    void cvSetData(IplImage * arr, void* data, int step)

cdef extern from "Image.hpp":
    cdef cppclass ImageBase:
        ImageBase()
        ImageBase(IplImage* img)
        void ReleaseMemory(bool b)
        IplImage* Ptr()        

    cdef cppclass RgbImage(ImageBase):
        RgbImage()
        RgbImage(IplImage* img)      

    cdef cppclass BwImage(ImageBase):
        BwImage()
        BwImage(IplImage* img)
        

cdef extern from "Bgs.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass Bgs:
        void InitModel(const RgbImage& data)

        void Initalize(const BgsParams& param)

        void Subtract(int frame_num, const RgbImage& data,
                      BwImage& low_threshold_mask, BwImage& high_threshold_mask)
        void Update(int frame_num, const RgbImage& data,
                    const BwImage& update_mask)
        RgbImage* Background()

cdef extern from "BgsParams.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass BgsParams:
        pass

cdef extern from "Eigenbackground.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass EigenbackgroundParams(BgsParams):
        pass
    cdef cppclass Eigenbackground(Bgs):
        pass

cdef extern from "AdaptiveMedianBGS.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass AdaptiveMedianParams(BgsParams):
        pass
    cdef cppclass AdaptiveMedianBGS(Bgs):
        pass

cdef extern from "GrimsonGMM.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass GrimsonParams(BgsParams):
        pass
    cdef cppclass GrimsonGMM(Bgs):
        pass  

cdef extern from "MeanBGS.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass MeanParams(BgsParams):
        pass
    cdef cppclass MeanBGS(Bgs):
        pass  

cdef extern from "PratiMediodBGS.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass PratiParams(BgsParams):
        pass
    cdef cppclass PratiMediodBGS(Bgs):
        pass  

cdef extern from "WrenGA.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass WrenParams(BgsParams):
        pass
    cdef cppclass WrenGA(Bgs):
        pass  

cdef extern from "ZivkovicAGMM.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass ZivkovicParams(BgsParams):
        pass
    cdef cppclass ZivkovicAGMM(Bgs):
        pass  

cdef extern from "create_params_wrapper.hpp":
    AdaptiveMedianParams CreateAdaptiveMedianParams(int width, int height, 
    float low_threshold, float high_threshold, int sampling_rate, int learning_frames)

    EigenbackgroundParams CreateEigenbackgroundParams(int width, int height, 
    float low_threshold, float high_threshold, int history_size, int dims)

    GrimsonParams CreateGrimsonGMMParams(int width, int height,
    float low_threshold, float high_threshold, float alpha, float max_modes)

    MeanParams CreateMeanBGSParams(int width, int height,
    unsigned int low_threshold, unsigned int high_threshold,    
    float alpha, int learning_frames)

    BgsParams CreatePratiMediodBGSParams(int width, int height,
    unsigned int low_threshold, unsigned int high_threshold,    
    int weight, int sampling_rate, int history_size)

    WrenParams CreateWrenGAParams(int width, int height,
    float low_threshold, float high_threshold,  
    float alpha, int learning_frames)

    ZivkovicParams CreateZivkovicAGMMParams(int width, int height,
    float low_threshold, float high_threshold,  
    float alpha, int max_modes) 

    void set_image_data(ImageBase* updated_image, IplImage* updated_ipl_image, 
                        unsigned char* data_ptr, int step)


cdef class BackgroundSubtraction:
    cdef Bgs* bg
    cdef IplImage* rgb_iplimage
    cdef IplImage* low_mask_iplimage
    cdef IplImage* high_mask_iplimage
    cdef RgbImage rgb_image
    cdef BwImage low_mask_image
    cdef BwImage high_mask_image

    def __init__(self):
        self.rgb_image.ReleaseMemory(False)
        self.low_mask_image.ReleaseMemory(False)
        self.high_mask_image.ReleaseMemory(False)

    def __dealloc__(self):
        cvReleaseImageHeader(&self.rgb_iplimage)
        cvReleaseImageHeader(&self.low_mask_iplimage)
        cvReleaseImageHeader(&self.high_mask_iplimage)
        del self.bg

    def init_model(self, np.uint8_t[:, :, :] image, params):
        if params['algorithm'] == 'adaptive_median':
            self.bg = new AdaptiveMedianBGS()
            adaptive_median_params = CreateAdaptiveMedianParams(
                image.shape[1], image.shape[0], 
                params['low'], params['high'], 
                params['sampling_rate'], params['learning_frames'])
            self.bg.Initalize(adaptive_median_params)
        elif params['algorithm'] == 'eigenbackground':
            self.bg = new Eigenbackground()
            eigen_params = CreateEigenbackgroundParams(
                image.shape[1], image.shape[0], 
                params['low'], params['high'], 
                params['history_size'], params['dims'])
            self.bg.Initalize(eigen_params)
        elif params['algorithm'] == 'grimson_gmm':
            self.bg = new GrimsonGMM()
            grimson_gmm_params = CreateGrimsonGMMParams(
                image.shape[1], image.shape[0], 
                params['low'], params['high'], 
                params['alpha'], params['max_modes'])
            self.bg.Initalize(grimson_gmm_params)
        elif params['algorithm'] == 'mean_bgs':
            self.bg = new MeanBGS()
            mean_bgs_params = CreateMeanBGSParams(
                image.shape[1], image.shape[0], 
                params['low'], params['high'], 
                params['alpha'], params['learning_frames'])
            self.bg.Initalize(mean_bgs_params)      
        elif params['algorithm'] == 'prati_mediod_bgs':
            self.bg = new PratiMediodBGS()
            prati_mediod_bgs_params = CreatePratiMediodBGSParams(
                image.shape[1], image.shape[0], 
                params['low'], params['high'], 
                params['weight'], params['sampling_rate'], params['history_size'])
            self.bg.Initalize(prati_mediod_bgs_params)
        elif params['algorithm'] == 'wren_ga':
            self.bg = new WrenGA()
            wren_ga_params = CreateWrenGAParams(
                image.shape[1], image.shape[0], 
                params['low'], params['high'], 
                params['alpha'], params['learning_frames'])
            self.bg.Initalize(wren_ga_params)
        elif params['algorithm'] == 'zivkovic_agmm':
            self.bg = new ZivkovicAGMM()
            zivkovic_agmm_params = CreateZivkovicAGMMParams(
                image.shape[1], image.shape[0], 
                params['low'], params['high'], 
                params['alpha'], params['max_modes'])
            self.bg.Initalize(zivkovic_agmm_params)                       

        
        cdef CvSize size
        size.width = image.shape[1]
        size.height = image.shape[0]
        self.rgb_iplimage = cvCreateImageHeader(size, IPL_DEPTH_8U, 3)        
        self.low_mask_iplimage = cvCreateImageHeader(size, IPL_DEPTH_8U, 1)
        self.high_mask_iplimage = cvCreateImageHeader(size, IPL_DEPTH_8U, 1)

        set_image_data(&self.rgb_image, self.rgb_iplimage, 
                       &image[0, 0, 0], image.strides[0])
        assert image.strides[1] == 3
        assert image.strides[2] == 1        
        self.bg.InitModel(self.rgb_image)

    def subtract(self, frame_num, np.uint8_t[:, :, :] image, 
                 np.uint8_t[:, :] low_threshold_mask, 
                 np.uint8_t[:, :] high_threshold_mask):

        set_image_data(<ImageBase *>(&self.rgb_image), self.rgb_iplimage, 
                       &image[0, 0, 0], image.strides[0])
        set_image_data(<ImageBase *>(&self.low_mask_image), self.low_mask_iplimage, 
                       &low_threshold_mask[0, 0], low_threshold_mask.strides[0])
        set_image_data(<ImageBase *>(&self.high_mask_image), self.high_mask_iplimage, 
                       &high_threshold_mask[0, 0], high_threshold_mask.strides[0])
        self.bg.Subtract(frame_num, self.rgb_image, self.low_mask_image, self.high_mask_image)

    def update(self, frame_num, np.uint8_t[:, :, :] image, 
               np.uint8_t[:, :] low_threshold_mask):
        set_image_data(<ImageBase *>(&self.rgb_image), self.rgb_iplimage, 
                       &image[0, 0, 0], image.strides[0])
        set_image_data(<ImageBase *>(&self.low_mask_image), self.low_mask_iplimage, 
                       &low_threshold_mask[0, 0], low_threshold_mask.strides[0])
        self.bg.Update(frame_num, self.rgb_image, self.low_mask_image)

    def get_background(self):
        cdef RgbImage* background = self.bg.Background()
        h = background.Ptr().height
        w = background.Ptr().width
        cdef unsigned char [:,:,:] view = <unsigned char[:h, :w, :3]> <unsigned char*> background.Ptr().imageData
        return np.asarray(view)

