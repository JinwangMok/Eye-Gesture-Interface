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

    // 3. Select Most Face-like area to Face.
    if(faceLikes.size()==1){ 
        // Just one face has detected.
        faceROI = faceLikes.at(0);

        // Check face ROI is valid area.
        if(this->getFaceROIBuffer().empty()){
            this->pushToFaceROIBuffer(faceROI);
        }else{
            if(this->getFaceROIBuffer().size() >= ET__MAX_BUFFER_LENGTH){
                this->popFromFaceROIBuffer();
            }

            if( abs(this->getLastFaceROI().x - faceROI.x) > ET__CASCADE_FACE_MARGIN_PIXEL || 
                abs(this->getLastFaceROI().y - faceROI.y) > ET__CASCADE_FACE_MARGIN_PIXEL ){
                    // Replace Argorithm
                    std::vector<cv::Rect> faceROIAreas;
                    std::vector<int> faceROICounts;
                    int faceLikesNum = 0;

                    for(cv::Rect faceROIFromBuffer : this->getFaceROIBuffer()){
                        if(faceROIAreas.empty()){
                            faceROIAreas.push_back(faceROIFromBuffer);
                            faceROICounts.push_back(1);
                            faceLikesNum++;
                        }else{
                            int cnt = 0;
                            for(cv::Rect faceROIArea : faceROIAreas){
                                if( abs(faceROIArea.x - faceROIFromBuffer.x) > ET__CASCADE_FACE_MARGIN_PIXEL ||
                                    abs(faceROIArea.y - faceROIFromBuffer.y) > ET__CASCADE_FACE_MARGIN_PIXEL){

                                    faceROIAreas.push_back(faceROIFromBuffer);
                                    faceROICounts.push_back(1);
                                    faceLikesNum++;
                                }else{
                                    faceROIAreas.at(cnt) = faceROIFromBuffer; // Update for restore last position of faceROI. 
                                    faceROICounts.at(cnt)++;
                                }
                                cnt++;
                            }
                        }
                    }

                    int maxNum = 0;
                    for(int i = 0; i < faceROIAreas.size(); i++){
                        if(faceROICounts.at(i) > maxNum){
                            maxNum = faceROICounts.at(i);
                            faceROI = faceROIAreas.at(i);
                        }
                    }
            }

            this->pushToFaceROIBuffer(faceROI);
        }

    }else{ 
        // Multiple face-like areas have detected.
        if(faceROIErrorCount < ET__MAX_ERROR_COUNT){ 
            // The time yet in error margin.
            faceROI = this->getLastFaceROI();
            faceROIErrorCount++;
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
void EyeTracker::detectEyesUsingHaar(cv::Mat& cameraFrame){
    // Add
    this->detectFace(cameraFrame);
    
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

/* Detect Eyes Using EyePicker Algorithm. */
void EyeTracker::detectEyesUsingEyePicker(cv::Mat& cameraFrame){
    detectionData output;
    cv::Rect faceROI;
    cv::Mat grayFrame;

    this->detectFace(cameraFrame);
    faceROI = this->getLastFaceROI();
    cv::cvtColor(cameraFrame, grayFrame, cv::COLOR_BGR2GRAY);
    this->selectEyeArea(grayFrame, faceROI, output);

    // Restore adjusted data.
    int xDiff, yDiff;
    cv::Point lastCenter;
    switch(output.eyeState){
        case EP__EYE_STATE_OPEN:

            this->setLastLeftEyeROI(cv::Rect(cv::Point(output.leftEyeRegion.tl() + faceROI.tl()), cv::Size(output.leftEyeRegion.size())));
            this->setLastRightEyeROI(cv::Rect(cv::Point(output.rightEyeRegion.tl() + faceROI.tl()), cv::Size(output.rightEyeRegion.size())));
            this->setLastLeftEyeCenter(faceROI.tl() + output.leftEyePosition);
            this->setLastRightEyeCenter(faceROI.tl() + output.rightEyePosition);

            this->setCenterOfBothEyes(
                cv::Point(
                    cvRound(faceROI.x + output.leftEyePosition.x + (output.rightEyePosition.x - output.leftEyePosition.x)/2),
                    output.leftEyePosition.y < output.rightEyePosition.y ? 
                        cvRound(faceROI.y + output.leftEyePosition.y + (output.rightEyePosition.y - output.leftEyePosition.y)/2)
                        :
                        cvRound(faceROI.y + output.leftEyePosition.y + (output.leftEyePosition.y - output.rightEyePosition.y)/2)
                )
            );
            break;

        case EP__EYE_STATE_CLOSE:
            this->outputData.leftEyeRegion = cv::Rect();
            this->outputData.rightEyeRegion = cv::Rect();
            this->outputData.leftEyePosition = cv::Point();
            this->outputData.rightEyePosition = cv::Point();
            this->outputData.resultleftEyePosition = cv::Point();
            this->outputData.resultrightEyePosition = cv::Point();
            
            this->setLastLeftEyeROI(cv::Rect());
            this->setLastRightEyeROI(cv::Rect());
            this->setLastLeftEyeCenter(cv::Point());
            this->setLastRightEyeCenter(cv::Point());

            this->setCenterOfBothEyes(this->getLastCenterOfBothEyes());
            break;

        case EP__EYE_STATE_LEFT_CLOSED:
            this->outputData.leftEyeRegion = cv::Rect();
            this->outputData.leftEyePosition = cv::Point();
            this->outputData.resultleftEyePosition = cv::Point();

            xDiff = (this->getLastRightEyeCenter().x - faceROI.x) - output.rightEyePosition.x;
            yDiff = (this->getLastRightEyeCenter().y - faceROI.y) - output.rightEyePosition.y;
            lastCenter = this->getLastCenterOfBothEyes();
            this->setCenterOfBothEyes(
                cv::Point(
                    lastCenter.x - xDiff,
                    lastCenter.y - yDiff
                )
            );

            this->setLastLeftEyeROI(cv::Rect());
            this->setLastRightEyeROI(cv::Rect(cv::Point(output.rightEyeRegion.tl() + faceROI.tl()), cv::Size(output.rightEyeRegion.size())));
            this->setLastLeftEyeCenter(cv::Point());
            this->setLastRightEyeCenter(faceROI.tl() + output.rightEyePosition);
            
            break;

        case EP__EYE_STATE_RIGHT_CLOSED:
            this->outputData.rightEyeRegion = cv::Rect();
            this->outputData.rightEyePosition = cv::Point();
            this->outputData.resultrightEyePosition = cv::Point();

            xDiff = (this->getLastLeftEyeCenter().x - faceROI.x) - output.leftEyePosition.x;
            yDiff = (this->getLastLeftEyeCenter().y - faceROI.y) - output.leftEyePosition.y;
            lastCenter = this->getLastCenterOfBothEyes();
            this->setCenterOfBothEyes(
                cv::Point(
                    lastCenter.x - xDiff,
                    lastCenter.y - yDiff
                )
            );
        
            
            this->setLastLeftEyeROI(cv::Rect(cv::Point(output.leftEyeRegion.tl() + faceROI.tl()), cv::Size(output.leftEyeRegion.size())));
            this->setLastRightEyeROI(cv::Rect());
            this->setLastLeftEyeCenter(faceROI.tl() + output.leftEyePosition);
            this->setLastRightEyeCenter(cv::Point());
            
            break;
        default:
            break;
    }
}

/* MAIN ALGORITHM */
// TODO: 인터페이스 토글에 따라 활성 비활성 동작 추가해야함!! -> switch 문을 토글로 if-else 나눠서 걸어잠그기
Gesture EyeTracker::traceAndTranslate2Gesture(cv::Mat& cameraFrame){
    /* Variables */
    cv::Rect faceROI, leftEyeROI, rightEyeROI;
    cv::Point leftEyeCenter, rightEyeCenter, bothEyeCenter, lastBothEyeCenter;

    std::chrono::duration<double> totalTime, accumulatedTime, detectionTime, durationTime;
    std::chrono::microseconds totalTimeMS;
    std::chrono::system_clock::time_point start;

    bool isLeftEyeOpen = true; 
    bool isRightEyeOpen = true;

    GestureData lastGestureData, thisGestureData;

    Gesture result;


    // Start Point Of Duration.
    start = std::chrono::system_clock::now();

    // 1. Detect face and eyes. (per single frame)
    this->detectEyesUsingEyePicker(cameraFrame);

    // 2. Get both eyes' information.
    faceROI = this->getLastFaceROI();
    leftEyeROI = this->getLastLeftEyeROI();
    rightEyeROI = this->getLastRightEyeROI();
    leftEyeCenter = this->getLastLeftEyeCenter();
    rightEyeCenter = this->getLastRightEyeCenter();
    bothEyeCenter = this->getCenterOfBothEyes();
    lastBothEyeCenter = this->getLastCenterOfBothEyes();

    // 3. Translate to Gesture. 
    //// 1) Check eyes are opened.
    if(leftEyeROI.empty()){ isLeftEyeOpen = false; }
    if(rightEyeROI.empty()){ isRightEyeOpen = false; }  

    //// 2) Translate to gesture.
    lastGestureData = this->getLastGestureData();
    
    EYE_STATE_TYPE CASE = selectCaseFromGesture(isLeftEyeOpen, isRightEyeOpen, lastGestureData.getIsLeftEyeOpen(), lastGestureData.getIsRightEyeOpen());
    
    uint16_t bufErrorCnt;
    if(this->getInterfaceEnableFlag()){
        
        switch(CASE){

            /* 1) 양안을 지속적으로 뜬 경우 */
            case EYE_STATE_TYPE(BOTH_OPEN_TO_BOTH_OPEN):
                bufErrorCnt = 0;
                accumulatedTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0));
                for(std::vector<GestureData>::iterator it = this->getGestureDataBuffer().end()-1; it != this->getGestureDataBuffer().begin()-1; it--){
                    if(it->getIsLeftEyeOpen() && it->getIsRightEyeOpen()){
                        accumulatedTime += it->getFrameTime(); // sec
                    }else{
                        if(bufErrorCnt < ET__BUFFER_ERROR_MARGIN_COUNT){
                            bufErrorCnt++;
                        }else{
                            break;
                        }
                    }
                }

                if(accumulatedTime.count() <= 0){
                    // exception.
                    result =  Gesture(NONE);

                }else if(this->getDoubleClickFlag()){
                    // 누적시간이 1초 미만인가?
                    if(accumulatedTime.count() < ET__DOUBLE_CLICK_WAIT_THRESHOLD){
                        // 더블 클릭 대기
                        this->setDoubleClickFlag(true);
                        result = Gesture(WAIT);
                    }else{
                        // ET__DOUBLE_CLICK_WAIT_THRESHOLD초 이상이면서 더블클릭 플래그가 있으면 -> 더블 클릭 플래그 초기화
                        this->setDoubleClickFlag(false);
                        result = Gesture(NONE);
                    }
                }else{
                    // 더블클릭 플래그가 없는 경우
                    //좌우 센터 간 y 픽셀 차이 검사 -> 스크롤 || 포인터 이동 
                    if(leftEyeCenter.y > rightEyeCenter.y && (leftEyeCenter.y - rightEyeCenter.y) > ET__MIN_SCROLL_MARGIN){
                        // 고개 좌측 기울임. 상향 스크롤
                        result = Gesture(SCROLL_UP);
                    }else if(leftEyeCenter.y < rightEyeCenter.y && (rightEyeCenter.y - leftEyeCenter.y) > ET__MIN_SCROLL_MARGIN){
                        // 고개 우측 기울임. 하향 스크롤
                        result = Gesture(SCROLL_DOWN);
                    }else{
                        // 포인터 이동
                        if(lastBothEyeCenter == cv::Point()){
                            this->setLastCenterOfBothEyes(bothEyeCenter);
                            result = Gesture(POINTER_MOVE);
                        }else{
                            if( abs(lastBothEyeCenter.x - bothEyeCenter.x) > ET__MIN_POINTER_MOVE_THRESHOLD ||
                                abs(lastBothEyeCenter.y - bothEyeCenter.y) > ET__MIN_POINTER_MOVE_THRESHOLD){
                                this->moveCursor((lastBothEyeCenter.x - bothEyeCenter.x), (lastBothEyeCenter.y - bothEyeCenter.y));
                                this->setLastCenterOfBothEyes(bothEyeCenter);
                                result = Gesture(POINTER_MOVE);
                            }else{
                                result = Gesture(NONE);
                            }
                        }
                    }
                }

                break;

            /* 2) 지금은 양안을 떴지만 방금까지 한쪽 눈만 뜬 경우 */
            case EYE_STATE_TYPE(SINGLE_OPEN_TO_BOTH_OPEN):
                bufErrorCnt = 0;
                accumulatedTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0));
                for(std::vector<GestureData>::iterator it = this->getGestureDataBuffer().end()-1; it != this->getGestureDataBuffer().begin()-1; it--){
                    if((lastGestureData.getIsLeftEyeOpen() == it->getIsLeftEyeOpen()) && (lastGestureData.getIsRightEyeOpen() == it->getIsRightEyeOpen())){
                        accumulatedTime += it->getFrameTime(); // sec
                    }else{ 
                        if(bufErrorCnt < ET__BUFFER_ERROR_MARGIN_COUNT){
                            bufErrorCnt++;
                        }else{
                            break;
                        }
                    }
                }
                
                if(accumulatedTime.count() <= 0){
                    // exception.
                    result =  Gesture(NONE);
                }else if(this->getRightClickFlag()){
                    if(this->getDragFlag()){
                    // 우클릭 검사 플래그와 드래그 플래그가 있는 경우 -> 드롭 및 드래그/우클릭 플래그 초기화
                        result = Gesture(DROP);
                        this->setDragFlag(false);
                        this->setRightClickFlag(false);
                    }else{
                    // 우클릭 검사 플래그만 있는 경우 -> 우클릭 및 우클릭 플래그 초기화
                    result = Gesture(RIGHT_CLICK);
                        this->setRightClickFlag(false);
                    }
                }else{
                    // exception. 다 아니면 눈을 잘못 인식한 경우 등 예외
                    result = Gesture(NONE);
                }

                break;

            /* 3) 지금은 양안을 떳지만 방금까지 두 눈 모두 감은 경우 */
            case EYE_STATE_TYPE(BOTH_CLOSE_TO_BOTH_OPEN):
                bufErrorCnt = 0;
                accumulatedTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0));
                for(std::vector<GestureData>::iterator it = this->getGestureDataBuffer().end()-1; it != this->getGestureDataBuffer().begin()-1; it--){
                    if(!it->getIsLeftEyeOpen() && !it->getIsRightEyeOpen()){
                        accumulatedTime += it->getFrameTime(); // sec
                    }else{ 
                        if(bufErrorCnt < ET__BUFFER_ERROR_MARGIN_COUNT){
                            bufErrorCnt++;
                        }else{
                            break;
                        }
                    }
                }
                // std::cout << "안구 깜빡임에 걸린 누적 시간 : " << accumulatedTime.count() << "초" << std::endl;
                if(accumulatedTime.count() <= 0){
                    // exception. 일단 써놓음
                    result =  Gesture(NONE);
                }else if(accumulatedTime.count() >= ET__LEFT_CLICK_THRESHOLD){
                    // 0.6초 이상 3초 미만
                    if(accumulatedTime.count() < ET__INTERFACE_THRESHOLD){
                        if(this->getDoubleClickFlag()){
                            result = Gesture(DOUBLE_CLICK);
                            this->setDoubleClickFlag(false);
                        }else{
                            result = Gesture(LEFT_CLICK);
                            this->setDoubleClickFlag(true);
                        }
                    }else{
                        result = Gesture(INTERFACE_DISABLE);
                        this->setInterfaceEnableFlag(false);
                    }
                }else{
                    // 0.6초 미만 -> 그냥 눈 깜빡인 것이므로 무시
                    result = Gesture(NONE);
                }

                break;

            /* 4) 지금은 한 쪽 눈만 떴지만 방금까지 두 눈을 뜬 경우 */
            case EYE_STATE_TYPE(BOTH_OPEN_TO_SINGLE_OPEN):
                this->setDoubleClickFlag(false);
                result = Gesture(WAIT);

                break;

            /* 5) 지금도 한 쪽 눈만 떳고 방금도 한쪽눈만 뜬 경우  */
            case EYE_STATE_TYPE(SINGLE_OPEN_TO_SINGLE_OPEN):
                bufErrorCnt = 0;
                accumulatedTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0));
                for(std::vector<GestureData>::iterator it = this->getGestureDataBuffer().end()-1; it != this->getGestureDataBuffer().begin()-1; it--){
                    if((lastGestureData.getIsLeftEyeOpen() == it->getIsLeftEyeOpen()) && (lastGestureData.getIsRightEyeOpen() == it->getIsRightEyeOpen())){
                        accumulatedTime += it->getFrameTime(); // sec
                    }else{ 
                        if(bufErrorCnt < ET__BUFFER_ERROR_MARGIN_COUNT){
                            bufErrorCnt++;
                        }else{
                            break;
                        }
                    }
                }

                if(accumulatedTime.count() <= 0){
                    // exception
                    result =  Gesture(NONE);
                }else{
                    if(this->getRightClickFlag()){
                        // 우클릭 플래그인 경우 
                        if(this->getDragFlag()){
                            // 우클릭이면서 드래그인 경우 == 우클릭 후 움직임이 조금이라도 있었던 경우
                            // 아래와 달리, 조건 검사 없이, 바로 위치 차이 계산 후 초점 이동 
                            this->moveCursor((lastBothEyeCenter.x - bothEyeCenter.x), (lastBothEyeCenter.y - bothEyeCenter.y));
                            this->setLastCenterOfBothEyes(bothEyeCenter);
                            this->setDragFlag(true);
                            result = Gesture(DRAG);
                        }else{
                            // 우클릭이지만 아직 드래그 플래그는 없는 경우
                            if( abs(lastBothEyeCenter.x - bothEyeCenter.x) > ET__MIN_POINTER_MOVE_THRESHOLD ||
                                abs(lastBothEyeCenter.y - bothEyeCenter.y) > ET__MIN_POINTER_MOVE_THRESHOLD){
                                this->moveCursor((lastBothEyeCenter.x - bothEyeCenter.x), (lastBothEyeCenter.y - bothEyeCenter.y));
                                this->setRightClickFlag(true);
                                this->setDragFlag(true);
                                result = Gesture(DRAG);
                            }else{
                                this->setDragFlag(false);
                                result = Gesture(RIGHT_CLICK);
                            }
                            this->setLastCenterOfBothEyes(bothEyeCenter);
                        }
                    }else{
                        // 우클릭 플래그 없는 경우
                        if(accumulatedTime.count() < ET__RIGHT_CLICK_THRESHOLD){
                            result = Gesture(WAIT);
                        }else{
                            result = Gesture(WAIT);
                            this->setRightClickFlag(true);
                        }
                    }
                }

                break;

            /* 6) 지금은 한쪽 눈만 떴지만 방금까지는 두 눈을 감은 경우*/
            case EYE_STATE_TYPE(BOTH_CLOSE_TO_SINGLE_OPEN):
                // 계속 감고 있었는지 확인. 마진은 1-2개
                result = Gesture(NONE);

                break;

            /* 7) 지금은 두 눈 다 감았지만 방금까지는 두 눈 모두 뜬 경우 */
            case EYE_STATE_TYPE(BOTH_OPEN_TO_BOTH_CLOSE):
                result = Gesture(NONE);

                break;

            /* 8) 지금은 두 눈 다 감았지만 방금까지는 한 쪽 눈만 감은 경우 */
            case EYE_STATE_TYPE(SINGLE_OPEN_TO_BOTH_CLOSE):
                // 오인식 확인?
                result = Gesture(NONE);

                break;

            /* 9) 지금도 두 눈 다 감았고 방금까지도 두 눈 모두 감은 경우 */
            case EYE_STATE_TYPE(BOTH_CLOSE_TO_BOTH_CLOSE):
                result = Gesture(NONE);

                break;

            default:
                break;
        }
    }else{
        if(CASE == EYE_STATE_TYPE(BOTH_CLOSE_TO_BOTH_OPEN)){    
            if(this->getIsLastDisableEyeClosed()){
                if(this->getAccumlatedDuration4Enable().count() >= 3){
                    result = Gesture(INTERFACE_ENABLE);
                    this->setInterfaceEnableFlag(true);
                }else{
                    result = Gesture(NONE);
                    this->setInterfaceEnableFlag(false);
                    this->setIsLastDisableEyeClosed(false);
                    this->setAccumlatedDuration4Enable(std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0)));
                }
            }
        }else if(CASE == EYE_STATE_TYPE(BOTH_CLOSE_TO_BOTH_CLOSE)){
            this->setIsLastDisableEyeClosed(true);
            std::chrono::duration<double> duTime = this->getAccumlatedDuration4Enable() + lastGestureData.getFrameTime();
            this->setAccumlatedDuration4Enable(duTime);
        }else{
            this->setIsLastDisableEyeClosed(false);
            this->setAccumlatedDuration4Enable(std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0)));
        }
    }
    // End Point Of Duration(sec).
    durationTime = std::chrono::system_clock::now() - start;

    thisGestureData = GestureData(durationTime, leftEyeCenter, rightEyeCenter, isLeftEyeOpen, isRightEyeOpen);
    
    if(this->getGestureDataBuffer().size() >= ET__MAX_BUFFER_LENGTH){
        this->popFromGestureDataBuffer();
    }
    this->pushToGestureDataBuffer(thisGestureData);
    this->setLastGestureData(thisGestureData);

    return result;
}

EYE_STATE_TYPE EyeTracker::selectCaseFromGesture(bool isLeftEyeOpen, bool isRightEyeOpen, bool isLastLeftEyeOpen, bool isLastRightEyeOpen){
    if(isLeftEyeOpen && isRightEyeOpen){
        if(isLastLeftEyeOpen && isLastRightEyeOpen){
            return EYE_STATE_TYPE(BOTH_OPEN_TO_BOTH_OPEN);
        }else if(isLastLeftEyeOpen || isLastRightEyeOpen){
            return EYE_STATE_TYPE(SINGLE_OPEN_TO_BOTH_OPEN);
        }else{
            return EYE_STATE_TYPE(BOTH_CLOSE_TO_BOTH_OPEN);
        }
    }else if(isLeftEyeOpen || isRightEyeOpen){
        if(isLastLeftEyeOpen && isLastRightEyeOpen){
            return EYE_STATE_TYPE(BOTH_OPEN_TO_SINGLE_OPEN);
        }else if(isLastLeftEyeOpen || isLastRightEyeOpen){
            return EYE_STATE_TYPE(SINGLE_OPEN_TO_SINGLE_OPEN);
        }else{
            return EYE_STATE_TYPE(BOTH_CLOSE_TO_SINGLE_OPEN);
        }
    }else{
        if(isLastLeftEyeOpen && isLastRightEyeOpen){
            return EYE_STATE_TYPE(BOTH_OPEN_TO_BOTH_CLOSE);
        }else if(isLastLeftEyeOpen || isLastRightEyeOpen){
            return EYE_STATE_TYPE(SINGLE_OPEN_TO_BOTH_CLOSE);
        }else{
            return EYE_STATE_TYPE(BOTH_CLOSE_TO_BOTH_CLOSE);
        }
    }
}

void EyeTracker::moveCursor(int xDiff, int yDiff){
    int newX, newY;
    
    newX = this->CURSOR_POINTER->x - xDiff * ET__POINTER_X_MOVE_RATIO;
    newX = newX > MAIN_WINDOW_WIDTH ? MAIN_WINDOW_WIDTH : newX;
    newX = newX < 0 ? 0 : newX;
    this->CURSOR_POINTER->x = newX;

    newY = this->CURSOR_POINTER->y - yDiff * ET__POINTER_Y_MOVE_RATIO;
    newY = newY > MAIN_WINDOW_HEIGHT ? MAIN_WINDOW_HEIGHT : newY;
    newY = newY < 0 ? 0 : newY;
    this->CURSOR_POINTER->y = newY;

    // std::cout << "포인터 x : " << newX << " , y : " << newY << std::endl << std::endl;
}