#include "eyeTracker.h"

void EyeTracker::detectFace(cv::Mat& cameraFrame){
    /* Local Variables */
    cv::Mat grayscale;
    std::vector<cv::Rect> face_likes;
    cv::Rect faceROI;

    // 1. Preprocess the cameraFrame Mat.
    cv::flip(cameraFrame, cameraFrame, 1);
    cv::cvtColor(cameraFrame, grayscale, cv::COLOR_BGR2GRAY);

    // 2. Detect Face-like areas.
    this->faceClassifier.detectMultiScale(grayscale, face_likes, 1.1, ET__CASCADE_FACE_MIN_NEIGHBORS);

    // 3. Select Most Face-like area to Face.
    if(face_likes.size()==1){
        faceROI = face_likes.at(0);
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
}