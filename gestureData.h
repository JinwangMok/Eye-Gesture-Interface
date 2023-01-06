#ifndef __COMMON_INCLUDE__
#define __COMMON_INCLUDE__

/* Includes */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#endif

#ifndef __GESTURE_DATA__
#define __GESTURE_DATA__

/* Includes */

/* Constants */

/* Types */


class GestureData{
    private:
        std::chrono::duration<double> frameTime;
        cv::Point leftEyeCenter;
        cv::Point rightEyeCenter;
        bool isLeftEyeOpen;
        bool isRightEyeOpen;

    public:
        GestureData(){}
        GestureData(std::chrono::duration<double> _duration, cv::Point lCenter, cv::Point rCenter, bool lToggle, bool rToggle){
            this->frameTime = _duration;
            this->leftEyeCenter = lCenter;
            this->rightEyeCenter = rCenter;
            this->isLeftEyeOpen = lToggle;
            this->isRightEyeOpen = rToggle;
        }
        void setFrameTime(std::chrono::duration<double> _duration){ this->frameTime = _duration; }
        std::chrono::duration<double> getFrameTime(){ return this->frameTime; }
        
        void setLeftEyeCenter(cv::Point center){ this->leftEyeCenter = center; }
        cv::Point getLeftEyeCenter(){ return this->leftEyeCenter; }

        void setRightEyeCenter(cv::Point center){ this->rightEyeCenter = center; }
        cv::Point getRightEyeCenter(){ return this->rightEyeCenter; }

        void setIsLeftEyeOpen(bool toggle){ this->isLeftEyeOpen = toggle; }
        bool getIsLeftEyeOpen(){ return this->isLeftEyeOpen; }

        void setIsRightEyeOpen(bool toggle){ this->isRightEyeOpen = toggle; }
        bool getIsRightEyeOpen(){ return this->isRightEyeOpen; }
};
/* Global Variables */

/* Fuctions */

#endif