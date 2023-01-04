#include "eyeTracker.h"

void EyeTracker::detectFace(cv::Mat& cameraFrame){
    /* Local Variables */
    cv::Mat grayscale;
    std::vector<cv::Rect> faceLikes;
    cv::Rect faceROI;
    uint16_t faceROIErrorCount = 0;

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

// [ NOTICE ]
// Left and Right Eye ROI Rect is independent from faceROI. 
// They have to add faceROI.tl() and faceROI.width/2(especially RightEyeROI) when they used.
void EyeTracker::detectEyes(cv::Mat& cameraFrame){
    /* Local Variables */
    uint16_t leftEyeROIErrorCount = 0;
    uint16_t rightEyeROIErrorCount = 0;
    cv::Rect faceROI = this->getLastFaceROI();
    cv::Point faceROILoc = cv::Point(faceROI.x, faceROI.y);
    uint16_t faceROIWidth = cvRound(faceROI.width);
    uint16_t faceROIHeight = cvRound(faceROI.height);
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
        
        // leftEyeROI = this->getLastLeftEyeROI();
        leftEyeROI = cv::Rect();
    }else{
        // Multiple leftEye-like areas have detected.
        if(leftEyeROIErrorCount < ET__MAX_ERROR_COUNT){ 
            // The time yet in error margin.
            if(this->getLeftEyeROIBuffer().size() > 1){
                leftEyeROI = this->getLastLeftEyeROI();
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

        // rightEyeROI = this->getLastRightEyeROI();
        rightEyeROI = cv::Rect();
    }else{
        // Multiple rightEye-like areas have detected.
        if(rightEyeROIErrorCount < ET__MAX_ERROR_COUNT){ 
            // The time yet in error margin.
            if(this->getRightEyeROIBuffer().size() > 1){
                rightEyeROI = this->getLastRightEyeROI();
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
    this->setLastLeftEyeROI(leftEyeROI);
    this->setLastRightEyeROI(rightEyeROI);
    this->setLastLeftEyeCenter(
        cv::Point(
            cvRound(leftEyeROI.x + leftEyeROI.width/2), 
            cvRound(leftEyeROI.y + leftEyeROI.height/2)
        )
    );
    this->setLastRightEyeCenter(
        cv::Point(
            cvRound(rightEyeROI.x + rightEyeROI.width/2),
            cvRound(rightEyeROI.y + rightEyeROI.height/2)
        )
    );
}

void EyeTracker::adjustEyes2Face(cv::Rect& faceROI, cv::Rect& leftEyeROI, cv::Rect& rightEyeROI, cv::Point& leftEyeCenter, cv::Point& rightEyeCenter){
    leftEyeROI = cv::Rect(
        cv::Point(faceROI.tl() + leftEyeROI.tl()),
        leftEyeROI.size()
    );
    rightEyeROI = cv::Rect(
        cv::Point(faceROI.tl() + rightEyeROI.tl() + cv::Point(faceROI.width/2, 0)),
        rightEyeROI.size()
    );
    leftEyeCenter = cv::Point(
        faceROI.x + leftEyeCenter.x,
        faceROI.y + leftEyeCenter.y
    );
    rightEyeCenter = cv::Point(
        faceROI.x + rightEyeCenter.x + faceROI.width/2,
        faceROI.y + rightEyeCenter.y
    );
}

Painter EyeTracker::traceAndTranslate(cv::Mat& cameraFrame){
    /* Variables */
    cv::Rect faceROI, leftEyeROI, rightEyeROI;
    cv::Point leftEyeCenter, rightEyeCenter;
    
    // Start Point Of Duration.
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    // 1. Detect face and eyes. (per single frame)
    this->detectFace(cameraFrame);
    this->detectEyes(cameraFrame);

    // 2. Adjusting both eyes' information.
    faceROI = this->getLastFaceROI();
    leftEyeROI = this->getLastLeftEyeROI();
    rightEyeROI = this->getLastRightEyeROI();
    leftEyeCenter = this->getLastLeftEyeCenter();
    rightEyeCenter = this->getLastRightEyeCenter();
    this->adjustEyes2Face(faceROI, leftEyeROI, rightEyeROI, leftEyeCenter, rightEyeCenter);

    // 3. Translate to Gesture.
    // TODO:
    //// (1) 멤버 변수로 프레임 당 시간 저장 및 누적 합 구하기
    //// (2) 시간과 함께 눈 뜬 정보(ex: leftEyeROI.empty()) 등을 이용해 제스처 구현
    //// (3) 이를 다른 함수로 구현?
    //// (4) 다 만들면 Painter::paint() 구현

    // End Point Of Duration(sec).
    std::chrono::duration<double> durationTime = std::chrono::system_clock::now() - start;
    // duration.count()
    // this->setGestureTime(durationTime);

    return Painter();
}