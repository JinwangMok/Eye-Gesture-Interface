#ifndef __COMMON_INCLUDE__
#define __COMMON_INCLUDE__

/* Includes */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#endif

#ifndef __EYE_TRACKER_H__
#define __EYE_TRACKER_H__

/* Includes */
#include <chrono>
#ifndef __GESTURE_DATA_H__
#define __GESTURE_DATA_H__
#include "gestureData.h"
#endif

/* Constants */
// ET: Eye Tracker
#define ET__CASCADE_FACE_MIN_NEIGHBORS  9
#define ET__CASCADE_EYE_MIN_NEIGHBORS   15
#define ET__MAX_BUFFER_LENGTH           100 // Aim to store data for 3sec.(approximately 30frame * 3 = 90 + 10(margin))
#define ET__MAX_ERROR_COUNT             30  // Aim to handle error with 1sec margin.(approximately 1sec = 30frame)
#define ET__MIN_SCROLL_MARGIN           5
/* Types */
//TODO: 안구 검출 부분 엔진 교체 필요(현재는 실험용으로 cascadeClassifier 사용)
enum Gesture{
    NONE,
    WAIT,
    INTERFACE_ENABLE,
    INTERFACE_DISABLE,
    LEFT_CLICK,
    RIGHT_CLICK,
    DOUBLE_CLICK,
    POINTER_MOVE,
    DRAG,
    DROP,
    SCROLL_UP,
    SCROLL_DOWN,
};
class EyeTracker{
    private:
        cv::CascadeClassifier faceClassifier;
        cv::CascadeClassifier eyeClassifier;
        cv::Rect lastFaceROI;
        // 아래 4개 변수는 논문 알고리즘으로 추후 업데이트
        cv::Rect lastLeftEyeROI;
        cv::Point lastLeftEyeCenter;
        cv::Rect lastRightEyeROI;
        cv::Point lastRightEyeCenter;
        // 아래 queue 자료형은 각 정보를 저장하는 버퍼의 역할
        std::queue<cv::Rect> faceROIBuffer;
        std::queue<cv::Rect> leftEyeROIBuffer;
        std::queue<cv::Rect> rightEyeROIBuffer;
        std::queue<cv::Point> leftEyeCenterBuffer;
        std::queue<cv::Point> rightEyeCenterBuffer;
        // 아래 멤버는 제스처 인식 관련
        GestureData lastGestureData;
        std::chrono::duration<double> frameDuration; // The duration for measuring gesture recognition in single frame.
        std::vector<GestureData> gestureDataBuffer;
        bool doubleClickFlag;
        bool rightClickFlag;
        bool dragFlag;
        bool interfaceEnableFlag;

    protected:
        void setLastFaceROI(cv::Rect faceROI){ this->lastFaceROI = faceROI; }
        void setLastLeftEyeROI(cv::Rect leftEyeROI){ this->lastLeftEyeROI = leftEyeROI; }
        void setLastLeftEyeCenter(cv::Point leftEyeCenter){ this->lastLeftEyeCenter = leftEyeCenter; }
        void setLastRightEyeROI(cv::Rect rightEyeROI){ this->lastRightEyeROI = rightEyeROI; }
        void setLastRightEyeCenter(cv::Point rightEyeCenter){ this->lastRightEyeCenter = rightEyeCenter; }
        
        void pushToFaceROIBuffer(cv::Rect faceROI){ this->faceROIBuffer.push(faceROI); }
        void pushToLeftEyeROIBuffer(cv::Rect leftEyeROI){ this->leftEyeROIBuffer.push(leftEyeROI); }
        void pushToLeftEyeCenterBuffer(cv::Point leftEyeCenter){ this->leftEyeCenterBuffer.push(leftEyeCenter); }
        void pushToRightEyeROIBuffer(cv::Rect rightEyeROI){ this->rightEyeROIBuffer.push(rightEyeROI); }
        void pushToRightEyeCenterBuffer(cv::Point rightEyeCenter){ this->rightEyeCenterBuffer.push(rightEyeCenter); }

        void popFromFaceROIBuffer(){ this->faceROIBuffer.pop(); }
        void popFromLeftEyeROIBuffer(){ this->leftEyeROIBuffer.pop(); }
        void popFromLeftEyeCenterBuffer(){ this->leftEyeCenterBuffer.pop(); }
        void popFromRightEyeROIBuffer(){ this->rightEyeROIBuffer.pop(); }
        void popFromRightEyeCenterBuffer(){ this->rightEyeCenterBuffer.pop(); }

        void resetFaceROIBuffer(){ std::queue<cv::Rect> emptyQ; std::swap(emptyQ, this->faceROIBuffer);}
        void resetLeftEyeROIBuffer(){ std::queue<cv::Rect> emptyQ; std::swap(emptyQ, this->leftEyeROIBuffer);}
        void resetLeftEyeCenterBuffer(){ std::queue<cv::Point> emptyQ; std::swap(emptyQ, this->leftEyeCenterBuffer);}
        void resetRightEyeROIBuffer(){ std::queue<cv::Rect> emptyQ; std::swap(emptyQ, this->rightEyeROIBuffer);}
        void resetRightEyeCenterBuffer(){ std::queue<cv::Point> emptyQ; std::swap(emptyQ, this->rightEyeCenterBuffer);}

        void setLastGestureData(GestureData gesture){ this->lastGestureData = gesture; }
        GestureData getLastGestureData(){ return this->lastGestureData; }

        void setGestureTime(std::chrono::duration<double> _duration){ this->frameDuration = _duration; }
        std::chrono::duration<double> getGestureTime(){ return this->frameDuration; }

        void pushToGestureDataBuffer(GestureData gesture){ this->gestureDataBuffer.push_back(gesture); }
        void popFromGestureDataBuffer(){ this->gestureDataBuffer.erase(this->gestureDataBuffer.begin()); }
        std::vector<GestureData> getGestureDataBuffer(){ return this->gestureDataBuffer; }
        void resetGestureDataBuffer(){ std::vector<GestureData> emptyBuf; std::swap(emptyBuf, this->gestureDataBuffer); }

        void setDoubleClickFlag(bool flag){ this->doubleClickFlag = flag; }
        bool getDoubleClickFlag(){ return this->doubleClickFlag; }

        void setRightClickFlag(bool flag){ this->rightClickFlag = flag; }
        bool getRightClickFlag(){ return this->rightClickFlag; }

        void setDragFlag(bool flag){ this->dragFlag = flag; }
        bool getDragFlag(){ return this->dragFlag; }

        void setInterfaceEnableFlag(bool flag){ this->interfaceEnableFlag = flag; }
        bool getInterfaceEnableFlag(){ return this->interfaceEnableFlag; }

    public:
        cv::Rect getLastFaceROI(){ return this->lastFaceROI; }
        cv::Rect getLastLeftEyeROI(){ return this->lastLeftEyeROI; }
        cv::Point getLastLeftEyeCenter() { return this->lastLeftEyeCenter; }
        cv::Rect getLastRightEyeROI(){ return this->lastRightEyeROI; }
        cv::Point getLastRightEyeCenter() { return this->lastRightEyeCenter; }
        
        std::queue<cv::Rect> getFaceROIBuffer(){ return this->faceROIBuffer; }
        std::queue<cv::Rect> getLeftEyeROIBuffer(){ return this->leftEyeROIBuffer; }
        std::queue<cv::Point> getLeftEyeCenterBuffer() { return this->leftEyeCenterBuffer; }
        std::queue<cv::Rect> getRightEyeROIBuffer(){ return this->rightEyeROIBuffer; }
        std::queue<cv::Point> getRightEyeCenterBuffer() { return this->rightEyeCenterBuffer; }
    
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
            this->frameDuration = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0));
            this->doubleClickFlag = false;
            this->rightClickFlag = false;
            this->dragFlag = false;
            this->interfaceEnableFlag = false;
        };
        void detectFace(cv::Mat& cameraFrame);
        void detectEyes(cv::Mat& cameraFrame);
        void adjustEyes2Face(cv::Rect& faceROI, cv::Rect& leftEyeROI, cv::Rect& rightEyeROI, cv::Point& leftEyeCenter, cv::Point& rightEyeCenter);
        Gesture traceAndTranslate2Gesture(cv::Mat& cameraFrame);
};
/* Global Variables */

/* Fuctions */

#endif