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
    switch(output.eyeState){
        case EP__EYE_STATE_OPEN:
            this->setLastLeftEyeROI(cv::Rect(cv::Point(output.leftEyeRegion.tl() + faceROI.tl()), cv::Size(output.leftEyeRegion.size())));
            this->setLastRightEyeROI(cv::Rect(cv::Point(output.rightEyeRegion.tl() + faceROI.tl()), cv::Size(output.rightEyeRegion.size())));
            this->setLastLeftEyeCenter(faceROI.tl() + output.leftEyePosition);
            this->setLastRightEyeCenter(faceROI.tl() + output.rightEyePosition);
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
            break;

        case EP__EYE_STATE_LEFT_CLOSED:
            this->outputData.leftEyeRegion = cv::Rect();
            this->outputData.leftEyePosition = cv::Point();
            this->outputData.resultleftEyePosition = cv::Point();

            this->setLastLeftEyeROI(cv::Rect());
            this->setLastRightEyeROI(cv::Rect(cv::Point(output.rightEyeRegion.tl() + faceROI.tl()), cv::Size(output.rightEyeRegion.size())));
            this->setLastLeftEyeCenter(cv::Point());
            this->setLastRightEyeCenter(faceROI.tl() + output.rightEyePosition);
            break;

        case EP__EYE_STATE_RIGHT_CLOSED:
            this->outputData.rightEyeRegion = cv::Rect();
            this->outputData.rightEyePosition = cv::Point();
            this->outputData.resultrightEyePosition = cv::Point();

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
// TODO: 인터페이스 토글에 따라 활성 비활성 동작 추가해야함!!
// TODO: 인식률을 위해 버퍼 내의 값들에 대한 오류 마진을 두는 방향으로 수정해야함!!
// TODO: 커서 이동 표현
// TODO: 우클릭 안됨 수정
Gesture EyeTracker::traceAndTranslate2Gesture(cv::Mat& cameraFrame){
    /* Variables */
    cv::Rect faceROI, leftEyeROI, rightEyeROI;
    cv::Point leftEyeCenter, rightEyeCenter;

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
    // this->detectEyesUsingHaar(cameraFrame);
    this->detectEyesUsingEyePicker(cameraFrame);

    // 2. Adjusting both eyes' information.
    faceROI = this->getLastFaceROI();
    leftEyeROI = this->getLastLeftEyeROI();
    rightEyeROI = this->getLastRightEyeROI();
    leftEyeCenter = this->getLastLeftEyeCenter();
    rightEyeCenter = this->getLastRightEyeCenter();
    // this->adjustEyes2Face(faceROI, leftEyeROI, rightEyeROI, leftEyeCenter, rightEyeCenter); // Use only with detectEyesUsingHaar()

    // 3. Translate to Gesture. 
    //// 1) Check eyes are opened.
    if(leftEyeROI.empty()){ isLeftEyeOpen = false; }
    if(rightEyeROI.empty()){ isRightEyeOpen = false; }  

    //// 2) Translate to gesture.
    lastGestureData = this->getLastGestureData();
    if(isLeftEyeOpen && isRightEyeOpen){
        if(lastGestureData.getIsLeftEyeOpen() && lastGestureData.getIsLeftEyeOpen()){
            // (1.1) 양안을 지속적으로 뜬 경우
                // 제스처 버퍼에서 두 눈을 모두 뜨지 않은 경우를 찾을 때까지의 시간의 합
            accumulatedTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0));
            for(std::vector<GestureData>::iterator it = this->getGestureDataBuffer().end()-1; it != this->getGestureDataBuffer().begin()-1; it--){
                if(it->getIsLeftEyeOpen() && it->getIsRightEyeOpen()){
                    accumulatedTime += it->getFrameTime(); // sec
                }else{ 
                    break;
                }
            }

            if(accumulatedTime.count() <= 0){
                // exception. 일단 써놓음
                result =  Gesture(NONE);
            }else if(this->getDoubleClickFlag()){
                if(accumulatedTime.count() < 1){
                    // 1초 미만이면서 더블클릭 플래그가 있으면 -> 더블 클릭 대기
                    this->setDoubleClickFlag(true);
                    result = Gesture(WAIT);
                }else{
                    // 1초 이상이면서 더블클릭 플래그가 있으면 -> 더블 클릭 플래그 초기화
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
                    //// 🔧구현 예정
                }
            }
        }else if(lastGestureData.getIsLeftEyeOpen() || lastGestureData.getIsLeftEyeOpen()){
            // (1.2) 지금은 양안을 떴지만 방금까지 한쪽 눈만 뜬 경우
                // 제스처 버퍼에서 연속적으로 "동일한" 한쪽 눈만 뜬 경우들의 시간의 합
            accumulatedTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0));
            for(std::vector<GestureData>::iterator it = this->getGestureDataBuffer().end()-1; it != this->getGestureDataBuffer().begin()-1; it--){
                if((lastGestureData.getIsLeftEyeOpen() == it->getIsLeftEyeOpen()) && (lastGestureData.getIsLeftEyeOpen() == it->getIsRightEyeOpen())){
                    accumulatedTime += it->getFrameTime(); // sec
                }else{ 
                    break;
                }
            }
            
            if(accumulatedTime.count() <= 0){
                // exception. 일단 써놓음
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
        }else{
            // (1.3) 지금은 양안을 떳지만 방금까지 두 눈 모두 감은 경우
                // 제스처 버퍼에서 연속적으로 두 눈 모두 감은 경우들의 시간의 합
            accumulatedTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0));
            for(std::vector<GestureData>::iterator it = this->getGestureDataBuffer().end()-1; it != this->getGestureDataBuffer().begin()-1; it--){
                if(!it->getIsLeftEyeOpen() && !it->getIsRightEyeOpen()){
                    accumulatedTime += it->getFrameTime(); // sec
                }else{ 
                    break;
                }
            }
            // std::cout << "안구 깜빡임에 걸린 누적 시간 : " << accumulatedTime.count() << "초" << std::endl;
            if(accumulatedTime.count() <= 0){
                // exception. 일단 써놓음
                result =  Gesture(NONE);
            }else if(accumulatedTime.count() >= 0.3){
                // 0.6초 이상 3초 미만 -> !!수정 0.3초 이상. 0.2도 괜찮을듯
                if(accumulatedTime.count() < 3){
                    if(this->getDoubleClickFlag()){
                        result = Gesture(DOUBLE_CLICK);
                        this->setDoubleClickFlag(false);
                    }else{
                        result = Gesture(LEFT_CLICK);
                        this->setDoubleClickFlag(true);
                    }
                }else{
                // 3초 이상
                    if(this->getInterfaceEnableFlag()){
                        // 인터페이스 활성화였을 시
                        result = Gesture(INTERFACE_DISABLE);
                        this->setInterfaceEnableFlag(false);
                    }else{
                        // 인터페이스 비활성화였을 시
                        result = Gesture(INTERFACE_ENABLE);
                        this->setInterfaceEnableFlag(true);
                    }
                }
            }else{
                // 0.6초 미만 -> 그냥 눈 깜빡인 것이므로 무시
                result = Gesture(NONE);
            }
        }
        
    }else if(isLeftEyeOpen || isRightEyeOpen){
        if(lastGestureData.getIsLeftEyeOpen() && lastGestureData.getIsLeftEyeOpen()){
            // (2.1) 지금은 한 쪽 눈만 떳지만 방금까지 두눈을 뜬 경우
                // 우클릭 검사 대기(1초 이상 요구이므로).
                result = Gesture(WAIT);
        }else if(lastGestureData.getIsLeftEyeOpen() || lastGestureData.getIsLeftEyeOpen()){
            // (2.2) 지금도 한 쪽 눈만 떳고 방금도 한쪽눈만 뜬 경우 
                // 연속적으로 "동일한" 쪽의 눈을 감은 경우의 누적시간을 구함.
            accumulatedTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::nanoseconds(0));
            for(std::vector<GestureData>::iterator it = this->getGestureDataBuffer().end()-1; it != this->getGestureDataBuffer().begin()-1; it--){
                if((lastGestureData.getIsLeftEyeOpen() == it->getIsLeftEyeOpen()) && (lastGestureData.getIsLeftEyeOpen() == it->getIsRightEyeOpen())){
                    accumulatedTime += it->getFrameTime(); // sec
                }else{ 
                    break;
                }
            }

            if(accumulatedTime.count() <= 0){
                // exception. 일단 써놓음
                result =  Gesture(NONE);
            }else{
                 if(this->getRightClickFlag()){
                    // 우클릭 플래그인 경우 
                    if(this->getDragFlag()){
                        // 우클릭이면서 드래그인 경우 == 우클릭 후 움직임이 조금이라도 있었던 경우
                        // 아래와 달리, 조건 검사 없이, 바로 위치 차이 계산 후 초점 이동 
                        result = Gesture(DRAG);
                    }else{
                        // 우클릭이지만 아직 드래그 플래그는 없는 경우
                        if(isLeftEyeOpen && lastGestureData.getIsLeftEyeOpen()){
                            // 왼쪽 눈이 열린 경우
                            // 🔧 구현 필요: 왼쪽 눈의 직전 위치와 현재 위치 차이 계산
                                // 만약 위치 차이가 5픽셀 이상이면 -> this->setDragFlag(true); result = Gesture(DRAG); 위치 차이만큼 초점 이동
                        }else if(isRightEyeOpen && lastGestureData.getIsRightEyeOpen()){
                            // 오른쪽 눈이 열린 경우
                            // 🔧 구현 필요: 오른쪽 눈의 직전 위치와 현재 위치 차이 계산 후 저장 및 이동
                                // 만약 위치 차이가 5픽셀 이상이면 -> this->setDragFlag(true); result = Gesture(DRAG); 위치 차이만큼 초점 이동
                        }
                    }
                 }else{
                    // 우클릭 플래그 없는 경우
                    if(accumulatedTime.count() < 0.3){
                        // 누적 시간 1초 미만 -> 단순 대기
                        result = Gesture(WAIT);
                    }else{
                        // 누적 시간 1초 이상 -> 우클릭 플래그 및 대기(눈 뜰 때 적용이므로)
                        // 우클릭 검사 플래그가 없으면 1초 미만 -> 우클릭 대기 중 || 1초 이상 -> 우클릭만 누르고 드래그는 안한 경우 
                        // ⭐️ 이 부분 3초 미만 유지 내용 수정함. 논문에 표 1에 strike 그려놓음
                        result = Gesture(WAIT);
                        this->setRightClickFlag(true);
                    }
                 }
            }
        }else{
            // (2.3) 지금은 한쪽 눈만 떳지만 방금까지는 두눈을 감은 경우
            // 사용 안함 예외. (예외임.. 그냥 안쓰는 경우라고 가정할까함!)
            result = Gesture(NONE);
        }
    }else{
        if(lastGestureData.getIsLeftEyeOpen() && lastGestureData.getIsLeftEyeOpen()){
            // (3.1) 지금은 두 눈 다 감았지만 방금까지는 두 눈 모두 뜬 경우
                // (⭐️작성중!! 여기부터 하셈) 좌클릭 검사 시작 플래그 ON. 또는 더블클릭인지도 확인?
                // 사용 안해도 될 듯?
            result = Gesture(NONE);
        }else if(lastGestureData.getIsLeftEyeOpen() || lastGestureData.getIsLeftEyeOpen()){
            // (3.2) 지금은 두 눈 다 감았지만 방금까지는 한 쪽 눈만 감은 경우
            // 사용 안함 예외
            result = Gesture(NONE);
        }else{
            // (3.3) 지금은 두 눈 다 감았지만 방금까지도 두눈 모두 감은
                // 제스처 버퍼에서 연속적으로 두눈 모두 감은 경우의 누적시간 합을 구함
                // 3초 이상이면 인터페이스 플래그 토글 후 버퍼 등 초기화하고 다시 뜰때까지 계속 감고있는거 예외처리
                // 이것도 사용 안해도 될 듯?
            result = Gesture(NONE);
        }
    }

    // End Point Of Duration(sec).
    durationTime = std::chrono::system_clock::now() - start;

    thisGestureData = GestureData(durationTime, leftEyeCenter, rightEyeCenter, isLeftEyeOpen, isRightEyeOpen);
    
    // <will be deleted>
    // Accumulate total time.
    // this->setGestureTime(totalTime + durationTime);
    // </will be deleted>

    // Update gesture data and its buffer.
    //// 제스처 버퍼 사이즈가 최대 이상이면 pop 하는 연산 필요
    if(this->getGestureDataBuffer().size() >= ET__MAX_BUFFER_LENGTH){
        this->popFromGestureDataBuffer();
    }
    this->pushToGestureDataBuffer(thisGestureData);
    this->setLastGestureData(thisGestureData);

    return result;
}
// duration.count()
// this->setGestureTime(durationTime);

// 두눈을 다 뜬 경우
    // focusPoint = cv::Point(
    //     ((rightEyeCenter.x - leftEyeCenter.x)/2 + leftEyeCenter.x),
    //     (leftEyeCenter.y < rightEyeCenter.y ? ((leftEyeCenter.y - rightEyeCenter.y)/2 + leftEyeCenter.y) : ((rightEyeCenter.y - leftEyeCenter.y)/2 + rightEyeCenter.y))
    // );
// 한쪽 눈만 뜬 경우
    // 직전 눈의 정보와 감은 쪽이 일치하는지 확인 + 일치한다면 이동 거리 계산해서 focusPoint에 반영
// 두눈 다 감은 경우
    // 직전 눈의 focusPoint 정보 그대로 사용

// focusPoint = this->getLastGesture().getCursor();
