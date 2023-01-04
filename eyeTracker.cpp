#include "eyeTracker.h"

void EyeTracker::detectFace(cv::Mat& cameraFrame){
    /* Local Variables */
    cv::Mat grayscale;
    std::vector<cv::Rect> faceLikes;
    cv::Rect faceROI;

    // 1. Preprocess the cameraFrame Mat.
    cv::flip(cameraFrame, cameraFrame, 1);
    cv::cvtColor(cameraFrame, grayscale, cv::COLOR_BGR2GRAY);

    // 2. Detect Face-like areas.
    this->faceClassifier.detectMultiScale(grayscale, faceLikes, 1.1, ET__CASCADE_FACE_MIN_NEIGHBORS);

    // 3. Select Most Face-like area to Face. -> 논문 내용으로 변경할 예정.
    if(faceLikes.size()==1){
        faceROI = faceLikes.at(0);
        //TODO: 직전 검출 얼굴 버퍼 관리 코드 추가
    }else{
        //TODO: 직전 검출 얼굴 버퍼에서 최근 데이터 가져오기 등으로 수정
        faceROI = cv::Rect();
    }

    // 4. Restore last face Region-Of-Interest.
    this->setLastFaceROI(faceROI);
}

void EyeTracker::detectEyes(cv::Mat& cameraFrame){
    /* Local Variables */
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

    // 3. Select most eye-like area to each eye. -> 논문 내용으로 변경할 예정.
    // (1)Left
    if(leftEyeLikes.size()==1){
        leftEyeROI = leftEyeLikes.at(0);
        //TODO: 직전 검출 좌안 버퍼 관리 코드 추가
    }else{
        //TODO: 직전 검출 좌안 버퍼에서 최근 데이터 가져오기 등으로 수정
        leftEyeROI = cv::Rect();
    }
    // (2)Right
    if(rightEyeLikes.size()==1){
        rightEyeROI = rightEyeLikes.at(0);
        //TODO: 직전 검출 우안 버퍼 관리 코드 추가
    }else{
        //TODO: 직전 검출 우안 버퍼에서 최근 데이터 가져오기 등으로 수정
        rightEyeROI = cv::Rect();
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