#include "eyeTracker.h"

void EyeTracker::detectFace(cv::Mat& cameraFrame){
    /* Local Variables */
    cv::Mat grayscale;
    std::vector<cv::Rect> faceLikes;
    cv::Rect faceROI;
    std::uint16_t faceROIErrorCount = 0;

    // 1. Preprocess the cameraFrame Mat.
    cv::flip(cameraFrame, cameraFrame, 1);
    cv::cvtColor(cameraFrame, grayscale, cv::COLOR_BGR2GRAY);

    // 2. Detect Face-like areas.
    this->faceClassifier.detectMultiScale(grayscale, faceLikes, 1.1, ET__CASCADE_FACE_MIN_NEIGHBORS);

    // 3. Select Most Face-like area to Face. -> 논문 내용으로 변경할 예정.
    //// TODO:프레임 간의 얼굴 영역 마진 처리
    if(faceLikes.size()==1){ 
        // Just one face has detected.
        faceROI = faceLikes.at(0);
        if(this->getFaceROIBuffer().size() > ET__MAX_BUFFER_LENGTH){
            this->popFromFaceROIBuffer();
        }
        this->pushToFaceROIBuffer(faceROI);
    }else{ 
        // Multiple face-like areas have detected.
        if(faceROIErrorCount < ET__MAX_ERROR_COUNT){ 
            // The time yet in error margin.
            if(this->getFaceROIBuffer().size() > 1){
                faceROI = this->getLastFaceROI();
                faceROIErrorCount++;
            }else{
                faceROI = cv::Rect();
            }
        }else{
            // The time is out of error margin.
            this->resetFaceROIBuffer();
            faceROI = cv::Rect();
            faceROIErrorCount = 0;
        }
    }

    // 4. Restore last face Region-Of-Interest.
    this->setLastFaceROI(faceROI);
}

void EyeTracker::detectEyes(cv::Mat& cameraFrame){
    /* Local Variables */
    std::uint16_t leftEyeROIErrorCount = 0;
    std::uint16_t rightEyeROIErrorCount = 0;
    cv::Rect faceROI = this->getLastFaceROI();
    cv::Point faceROILoc = cv::Point(faceROI.x, faceROI.y);
    std::uint16_t faceROIWidth = cvRound(faceROI.width);
    std::uint16_t faceROIHeight = cvRound(faceROI.height);
    std::vector<cv::Rect> leftEyeLikes, rightEyeLikes;
    cv::Rect leftEyeROI, rightEyeROI;
    cv::Point leftEyeLoc, rightEyeLoc;

    // 1. Select areas that each eye can exist.
    cv::Rect leftEyeArea(faceROILoc.x,                  faceROILoc.y, faceROIWidth/2, faceROIHeight);
    cv::Rect rightEyeArea(faceROILoc.x+faceROIWidth/2,  faceROILoc.y, faceROIWidth/2, faceROIHeight);

    // 2. Detect eye-like areas for each Rect.
    this->eyeClassifier.detectMultiScale(cameraFrame(leftEyeArea), leftEyeLikes, 1.1, ET__CASCADE_EYE_MIN_NEIGHBORS);
    this->eyeClassifier.detectMultiScale(cameraFrame(rightEyeArea), rightEyeLikes, 1.1, ET__CASCADE_EYE_MIN_NEIGHBORS);

    // 3. Select most eye-like area to each eye.
    //// TODO: 논문 내용(그레이디언트)으로 변경.
    //// (1)Left
    if(leftEyeLikes.size()==1){
        // Just one leftEye has detected.
        leftEyeROI = leftEyeLikes.at(0);
        if(this->getLeftEyeROIBuffer().size() > ET__MAX_BUFFER_LENGTH){
            this->popFromLeftEyeROIBuffer();
        }
        this->pushToLeftEyeROIBuffer(leftEyeROI);
    }else if(leftEyeLikes.empty()){
        // None leftEye has detected.(Probably this mainly meaning of blink.)
        std::cout << "Left Eye missing" << std::endl;
    }else{
        // Multiple leftEye-like areas have detected.
        if(leftEyeROIErrorCount < ET__MAX_ERROR_COUNT){ 
            // The time yet in error margin.
            if(this->getLeftEyeROIBuffer().size() > 1){
                leftEyeROI = cv::Rect(
                    cv::Point(this->getLastLeftEyeROI().tl() - this->getLastFaceROI().tl()),
                    cv::Size(this->getLastLeftEyeROI().width, this->getLastLeftEyeROI().height)
                );
                leftEyeROIErrorCount++;
            }else{
                leftEyeROI = cv::Rect();
            }
        }else{
            // The time is out of error margin.
            this->resetLeftEyeROIBuffer();
            leftEyeROI = cv::Rect();
            leftEyeROIErrorCount = 0;
        }
    }
    //// (2)Right
    if(rightEyeLikes.size()==1){
        // Just one rightEye has detected.
        rightEyeROI = rightEyeLikes.at(0);
        if(this->getRightEyeROIBuffer().size() > ET__MAX_BUFFER_LENGTH){
            this->popFromRightEyeROIBuffer();
        }
        this->pushToRightEyeROIBuffer(rightEyeROI);
    }else if(rightEyeLikes.empty()){
        // None rightEye has detected.(Probably this mainly meaning of blink.)
        std::cout << "Right Eye missing" << std::endl;
    }else{
        // Multiple rightEye-like areas have detected.
        if(rightEyeROIErrorCount < ET__MAX_ERROR_COUNT){ 
            // The time yet in error margin.
            if(this->getRightEyeROIBuffer().size() > 1){
                rightEyeROI = cv::Rect(
                    cv::Point(this->getLastRightEyeROI().tl() - this->getLastFaceROI().tl()),
                    cv::Size(this->getLastRightEyeROI().width, this->getLastRightEyeROI().height)
                );
                rightEyeROIErrorCount++;
            }else{
                rightEyeROI = cv::Rect();
            }
        }else{
            // The time is out of error margin.
            this->resetRightEyeROIBuffer();
            rightEyeROI = cv::Rect();
            rightEyeROIErrorCount = 0;
        }
    }

    // 4. Restore location, Region-Of-Interest area, center point of each eye.
    this->setLastLeftEyeROI(
        cv::Rect(
            (leftEyeArea.x + leftEyeROI.x),
            (leftEyeArea.y + leftEyeROI.y),
            leftEyeROI.width,
            leftEyeROI.height
        )
    );
    this->setLastRightEyeROI(
        cv::Rect(
            (rightEyeArea.x + rightEyeROI.x),
            (rightEyeArea.y + rightEyeROI.y),
            rightEyeROI.width,
            rightEyeROI.height
        )
    );
    this->setLastLeftEyeCenter(
        cv::Point(
            cvRound(leftEyeArea.x + leftEyeROI.x + leftEyeROI.width/2), 
            cvRound(leftEyeArea.y + leftEyeROI.y + leftEyeROI.height/2)
        )
    );
    this->setLastRightEyeCenter(
        cv::Point(
            cvRound(rightEyeArea.x + rightEyeROI.x + rightEyeROI.width/2),
            cvRound(rightEyeArea.y + rightEyeROI.y + rightEyeROI.height/2)
        )
    );
}