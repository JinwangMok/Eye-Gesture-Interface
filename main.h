#ifndef __MAIN_H__
#define __MAIN_H__

/* Includes */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/* Constants */
#define KEY_ESC 27
#define INIT_SEC 2
#define CASCADE_FACE_PATH "./haarcascade_frontalface_alt2.xml"
#define CASCADE_EYE_PATH "./haarcascade_eye.xml"
// > Display Resolution
//TODO: Required revise to getting the display resolution.
#ifdef _WIN32
#define DISPLAY_W 1280
#define DISPLAY_H 720

#elif __APPLE__ 
#define DISPLAY_W 2560
#define DISPLAY_H 1440

#else
#define DISPLAY_W 1920
#define DISPLAY_H 1080
#endif
/* Types */

/* Global Variables */
cv::Point CURSOR;
cv::Size DISPLAY_SIZE;
cv::Mat MAIN_WINDOW;

/* Functions */
cv::Point initialSetUp(cv::VideoCapture& cap, cv::Mat& frame, cv::Mat& mainWindow, const uint16_t FPS);


#endif