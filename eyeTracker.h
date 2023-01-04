#ifndef __COMMON_LIB__
#define __COMMON_LIB__

/* Includes */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#endif

#ifndef __EYE_TRACKER_H__
#define __EYE_TRACKER_H__

/* Includes */

/* Constants */
// ET: Eye Tracker
#define ET__CASCADE_FACE_MIN_NEIGHBORS 9
#define ET__CASCADE_EYE_MIN_NEIGHBORS 15

/* Types */
//TODO: 안구 검출 부분 엔진 교체 필요(현재는 실험용으로 cascadeClassifier 사용)
class EyeTracker{
    private:
        cv::CascadeClassifier faceClassifier;
        cv::CascadeClassifier eyeClassifier;
        cv::Rect lastFaceROI;
        // 아래 4개 변수는 논문 알고리즘으로 추후 업데이트
        cv::Rect lastLeftEyeROI;
        cv::Point lastLeftEyeCenter;    //*중요
        cv::Rect lastRightEyeROI;
        cv::Point lastRightEyeCenter;   //*중요
    
    protected:
        void setLastFaceROI(cv::Rect faceROI){ this->lastFaceROI = faceROI; }
        void setLastLeftEyeROI(cv::Rect leftEyeROI){ this->lastLeftEyeROI = leftEyeROI; }
        void setLastLeftEyeCenter(cv::Point leftEyeCenter){ this->lastLeftEyeCenter = leftEyeCenter; }
        void setLastRightEyeROI(cv::Rect rightEyeROI){ this->lastRightEyeROI = rightEyeROI; }
        void setLastRightEyeCenter(cv::Point rightEyeCenter){ this->lastRightEyeCenter = rightEyeCenter; }

    public:
        cv::Rect getLastFaceROI(){ return this->lastFaceROI; }
        cv::Rect getLastLeftEyeROI(){ return this->lastLeftEyeROI; }
        cv::Point getLastLeftEyeCenter() { return this->lastLeftEyeCenter; }
        cv::Rect getLastRightEyeROI(){ return this->lastRightEyeROI; }
        cv::Point getLastRightEyeCenter() { return this->lastRightEyeCenter; }
        
    public:
        EyeTracker(cv::String casecadeFacePath, cv::String casecadeEyePath){
            this->faceClassifier.load(casecadeFacePath);
            this->eyeClassifier.load(casecadeEyePath);
            if(faceClassifier.empty()){
                std::cout << "ERROR: Invalid Cascade File Path. Please check CASCADE_FACE_PATH in main.h" << std::endl;
                return;
            }
            if(eyeClassifier.empty()){
                std::cout << "ERROR: Invalid Cascade File Path. Please check CASCADE_EYE_PATH in main.h" << std::endl;
                return;
            }
            std::cout << "Cascade Classifiers are successfully loaded!" << std::endl;
        };
        void detectFace(cv::Mat& cameraFrame);
        void detectEyes(cv::Mat& cameraFrame);
};

/* Global Variables */

/* Fuctions */

#endif