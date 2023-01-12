#ifndef __COMMON_INCLUDE__
#define __COMMON_INCLUDE__

/* Includes */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#endif


#ifndef __MAIN_H__
#define __MAIN_H__

/* Includes */
#include "eyeTracker.h"
#ifndef __GESTURE_DATA_H__
#define __GESTURE_DATA_H__
#include "_gesture.h"
#endif

/* Constants */
#define CAM_NUM     1 //Usually this is 0. In my case is 1.
#define KEY_ESC     27
#define INIT_SEC    2
#define CASCADE_FACE_PATH "./haarcascade_frontalface_alt2.xml"
#define CASCADE_EYE_PATH "./haarcascade_eye.xml"

/* Types */

/* Functions */
void initialSetUp(cv::VideoCapture& cap, cv::Mat& frame, const uint16_t FPS);
void paintMainWindow();
void showCameraFrame();
#endif