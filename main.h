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
#ifndef __INCLUDE_PAINTER__
#define __INCLUDE_PAINTER__
#include "painter.h"
#endif
#ifndef __GESTURE_DATA_H__
#define __GESTURE_DATA_H__
#include "_gesture.h"
#endif

/* Constants */
#define CAM_NUM     2 //Usually this is 0. In my case is 2.
#define KEY_ESC     27
#define INIT_SEC    2
#define CASCADE_FACE_PATH "./haarcascade_frontalface_alt2.xml"
#define CASCADE_EYE_PATH "./haarcascade_eye.xml"

/* Types */

/* Global Variables */
cv::Point CURSOR;
cv::Size DISPLAY_SIZE;
cv::Mat MAIN_WINDOW;
EyeTracker eyeTracker;

/* Functions */
void initialSetUp(cv::VideoCapture& cap, cv::Mat& frame, cv::Mat& mainWindow, const uint16_t FPS);
void adjustEyes2Face(cv::Rect& faceROI, cv::Rect& leftEyeROI, cv::Rect& rightEyeROI, cv::Point& leftEyeCenter, cv::Point& rightEyeCenter);

#endif