CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(BGS)

SET(BGS_SRCS AdaptiveMedianBGS.cpp
    AdaptiveMedianBGS.hpp
    Bgs.hpp
    BgsParams.hpp
    Eigenbackground.cpp
    Eigenbackground.hpp
    GrimsonGMM.cpp
    GrimsonGMM.hpp
    Image.cpp
    Image.hpp
    MeanBGS.cpp
    MeanBGS.hpp
    PratiMediodBGS.cpp
    PratiMediodBGS.hpp
    WrenGA.cpp
    WrenGA.hpp
    ZivkovicAGMM.cpp
    ZivkovicAGMM.hpp
)

FIND_PACKAGE(OpenCV REQUIRED)

ADD_LIBRARY(bgs ${BGS_SRCS})
ADD_EXECUTABLE(bgs_test main.cpp)
TARGET_LINK_LIBRARIES(bgs_test bgs ${OpenCV_LIBS})


