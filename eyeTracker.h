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

#ifndef __EYE_PICKER_H__
#define __EYE_PICKER_H__
#include "eyePicker.h"
#endif
/* Constants */
// ET: Eye Tracker
#define ET__CASCADE_FACE_MARGIN_PIXEL   200
#define ET__CASCADE_FACE_CHECK_NUM      30
#define ET__CASCADE_FACE_MIN_NEIGHBORS  9
#define ET__CASCADE_EYE_MIN_NEIGHBORS   15
#define ET__MAX_BUFFER_LENGTH           100 // Aim to store data for 3sec.(approximately 30frame * 3 = 90 + 10(margin))
#define ET__MAX_ERROR_COUNT             30  // Aim to handle error with 1sec margin.(approximately 1sec = 30frame)
#define ET__MIN_SCROLL_MARGIN           20
#define ET__BUFFER_ERROR_MARGIN_COUNT   4

/* Types */
//TODO: 안구 검출 부분 엔진 교체 필요(현재는 실험용으로 cascadeClassifier 사용)
enum EYE_STATE_TYPE{
    BOTH_OPEN_TO_BOTH_OPEN,
    SINGLE_OPEN_TO_BOTH_OPEN,
    BOTH_CLOSE_TO_BOTH_OPEN,
    BOTH_OPEN_TO_SINGLE_OPEN,
    SINGLE_OPEN_TO_SINGLE_OPEN,
    BOTH_CLOSE_TO_SINGLE_OPEN,
    BOTH_OPEN_TO_BOTH_CLOSE,
    SINGLE_OPEN_TO_BOTH_CLOSE,
    BOTH_CLOSE_TO_BOTH_CLOSE
};

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
class EyeTracker : public EyePicker{
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
        std::deque<cv::Rect> faceROIBuffer;
        std::deque<cv::Rect> leftEyeROIBuffer;
        std::deque<cv::Rect> rightEyeROIBuffer;
        std::deque<cv::Point> leftEyeCenterBuffer;
        std::deque<cv::Point> rightEyeCenterBuffer;
        
        // 아래 멤버는 제스처 인식 관련
        GestureData lastGestureData;
        std::vector<GestureData> gestureDataBuffer;

        cv::Point centerOfBothEyes;
        cv::Point lastCenterOfBothEyes;
        cv::Point* CURSOR_POINTER;

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
        
        void pushToFaceROIBuffer(cv::Rect faceROI){ this->faceROIBuffer.push_back(faceROI); }
        void pushToLeftEyeROIBuffer(cv::Rect leftEyeROI){ this->leftEyeROIBuffer.push_back(leftEyeROI); }
        void pushToLeftEyeCenterBuffer(cv::Point leftEyeCenter){ this->leftEyeCenterBuffer.push_back(leftEyeCenter); }
        void pushToRightEyeROIBuffer(cv::Rect rightEyeROI){ this->rightEyeROIBuffer.push_back(rightEyeROI); }
        void pushToRightEyeCenterBuffer(cv::Point rightEyeCenter){ this->rightEyeCenterBuffer.push_back(rightEyeCenter); }

        void popFromFaceROIBuffer(){ this->faceROIBuffer.pop_front(); }
        void popFromLeftEyeROIBuffer(){ this->leftEyeROIBuffer.pop_front(); }
        void popFromLeftEyeCenterBuffer(){ this->leftEyeCenterBuffer.pop_front(); }
        void popFromRightEyeROIBuffer(){ this->rightEyeROIBuffer.pop_front(); }
        void popFromRightEyeCenterBuffer(){ this->rightEyeCenterBuffer.pop_front(); }

        void resetFaceROIBuffer(){ std::deque<cv::Rect> emptyQ; std::swap(emptyQ, this->faceROIBuffer);}
        void resetLeftEyeROIBuffer(){ std::deque<cv::Rect> emptyQ; std::swap(emptyQ, this->leftEyeROIBuffer);}
        void resetLeftEyeCenterBuffer(){ std::deque<cv::Point> emptyQ; std::swap(emptyQ, this->leftEyeCenterBuffer);}
        void resetRightEyeROIBuffer(){ std::deque<cv::Rect> emptyQ; std::swap(emptyQ, this->rightEyeROIBuffer);}
        void resetRightEyeCenterBuffer(){ std::deque<cv::Point> emptyQ; std::swap(emptyQ, this->rightEyeCenterBuffer);}

        void setLastGestureData(GestureData gesture){ this->lastGestureData = gesture; }

        void pushToGestureDataBuffer(GestureData gesture){ this->gestureDataBuffer.push_back(gesture); }
        void popFromGestureDataBuffer(){ this->gestureDataBuffer.erase(this->gestureDataBuffer.begin()); }
        void resetGestureDataBuffer(){ std::vector<GestureData> emptyBuf; std::swap(emptyBuf, this->gestureDataBuffer); }

        void setCenterOfBothEyes(cv::Point pos){ this->centerOfBothEyes = pos; }
        void setLastCenterOfBothEyes(cv::Point pos){ this->lastCenterOfBothEyes = pos; }

        void setDoubleClickFlag(bool flag){ this->doubleClickFlag = flag; }

        void setRightClickFlag(bool flag){ this->rightClickFlag = flag; }

        void setDragFlag(bool flag){ this->dragFlag = flag; }

        void setInterfaceEnableFlag(bool flag){ this->interfaceEnableFlag = flag; }

    public:
        cv::Rect getLastFaceROI(){ return this->lastFaceROI; }
        cv::Rect getLastLeftEyeROI(){ return this->lastLeftEyeROI; }
        cv::Point getLastLeftEyeCenter() { return this->lastLeftEyeCenter; }
        cv::Rect getLastRightEyeROI(){ return this->lastRightEyeROI; }
        cv::Point getLastRightEyeCenter() { return this->lastRightEyeCenter; }
        
        std::deque<cv::Rect> getFaceROIBuffer(){ return this->faceROIBuffer; }
        std::deque<cv::Rect> getLeftEyeROIBuffer(){ return this->leftEyeROIBuffer; }
        std::deque<cv::Point> getLeftEyeCenterBuffer() { return this->leftEyeCenterBuffer; }
        std::deque<cv::Rect> getRightEyeROIBuffer(){ return this->rightEyeROIBuffer; }
        std::deque<cv::Point> getRightEyeCenterBuffer() { return this->rightEyeCenterBuffer; }
    
        GestureData getLastGestureData(){ return this->lastGestureData; }
        std::vector<GestureData> getGestureDataBuffer(){ return this->gestureDataBuffer; }

        cv::Point getCenterOfBothEyes(){  return this->centerOfBothEyes; }
        cv::Point getLastCenterOfBothEyes(){  return this->lastCenterOfBothEyes; }

        bool getDoubleClickFlag(){ return this->doubleClickFlag; }
        bool getRightClickFlag(){ return this->rightClickFlag; }
        bool getDragFlag(){ return this->dragFlag; }
        bool getInterfaceEnableFlag(){ return this->interfaceEnableFlag; }
        
        void attachCursor(cv::Point* pCursor){ this->CURSOR_POINTER = pCursor; std::cout << "Initial CURSOR Point : " << *pCursor << std::endl; }
        void resetFlags(){ this->setDoubleClickFlag(false); this->setDragFlag(false); this->setRightClickFlag(false); }

    public:
        EyeTracker(){}
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
            this->doubleClickFlag = false;
            this->rightClickFlag = false;
            this->dragFlag = false;
            this->interfaceEnableFlag = false;
        }
        void detectFace(cv::Mat& cameraFrame);
        void detectEyesUsingHaar(cv::Mat& cameraFrame);
        void detectEyesUsingEyePicker(cv::Mat& cameraFrame);
        void adjustEyes2Face(cv::Rect& faceROI, cv::Rect& leftEyeROI, cv::Rect& rightEyeROI, cv::Point& leftEyeCenter, cv::Point& rightEyeCenter);
        Gesture traceAndTranslate2Gesture(cv::Mat& cameraFrame);
        EYE_STATE_TYPE selectCaseFromGesture(bool isLeftEyeOpen, bool isRightEyeOpen, bool isLastLeftEyeOpen, bool isLastRightEyeOpen);
};
/* Global Variables */

/* Fuctions */

#endif