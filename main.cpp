#include "main.h"

//TODO:
//  1. 얼굴 인식, 동공 검출 -> 양안 좌표, perspective weight, 얼굴 rect 위치, CoE(Center of Eyes) 등 반환
//  2. 1번에서 반환된 값들을 토대로 커서 위치 계산 및 업데이트 -> 커서 위치 반환
//  3. 2번에서 반환된 커서 값과 1번에서 반환된 양안의 개폐 여부등을 토대로 제스처 판단 및 동작 수행
//  4. 평가를 위한 함수 -> 사용성을 평가하기 위해 정답률, 시간 등을 반환
int main(int argc, char** argv){
    cv::VideoCapture cap(CAM_NUM);

    if(!cap.isOpened()){
        std::cerr << "Camera load failed!" << std::endl;
        return -1;
    }
    
    Gesture result;
    cv::Mat cameraFrame;
    const uint16_t fps = cap.get(cv::CAP_PROP_FPS);

    initialSetUp(cap, cameraFrame, MAIN_WINDOW, fps);

    /* TEST for eyeTracker */
    cv::Rect faceROI, leftEyeROI, rightEyeROI;
    cv::Point leftEyeCenter, rightEyeCenter, bothEyeCenter;
    cv::String command;

    while(true){
        cap >> cameraFrame;

        result = eyeTracker.traceAndTranslate2Gesture(cameraFrame);
        if(result != Gesture(NONE) && result != Gesture(WAIT)){
            switch(result){
                case Gesture(INTERFACE_ENABLE):
                    command = "Interface enabled.";
                    break;
                case Gesture(INTERFACE_DISABLE):
                    command = "Interface disabled.";
                    break;
                case Gesture(LEFT_CLICK):
                    command = "Left click";
                    break;
                case Gesture(RIGHT_CLICK):
                    command = "Right click";
                    break;
                case Gesture(DOUBLE_CLICK):
                    command = "Double click";
                    break;
                case Gesture(POINTER_MOVE):
                    command = "Pointer move";
                    break;
                case Gesture(DRAG):
                    command = "Drag";
                    break;
                case Gesture(DROP):
                    command = "Drop";
                    break;
                case Gesture(SCROLL_UP):
                    command = "Scroll up";
                    break;
                case Gesture(SCROLL_DOWN):
                    command = "Scroll down";
                    break;
                default:
                    break;
            }
            std::cout << "Processing time : " <<  eyeTracker.getLastGestureData().getFrameTime().count() << " sec" << std::endl << std::endl;
        }
        // // Till now, two steps under this line are seperated for testing.
        // eyeTracker.detectFace(cameraFrame);
        // eyeTracker.detectEyes(cameraFrame);
        // read values and adjust for display.
        faceROI = eyeTracker.getLastFaceROI();
        leftEyeROI = eyeTracker.getLastLeftEyeROI();
        rightEyeROI = eyeTracker.getLastRightEyeROI();
        leftEyeCenter = eyeTracker.getLastLeftEyeCenter();
        rightEyeCenter = eyeTracker.getLastRightEyeCenter();
        bothEyeCenter = eyeTracker.getCenterOfBothEyes();

        cv::putText(cameraFrame, command, cv::Point(10, 30), 2, 1, cv::Scalar(0, 0, 255));
        cv::rectangle(cameraFrame, faceROI, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        cv::rectangle(cameraFrame, leftEyeROI, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        cv::rectangle(cameraFrame, rightEyeROI, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        cv::circle(cameraFrame, leftEyeCenter, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        cv::circle(cameraFrame, rightEyeCenter, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        cv::circle(cameraFrame, bothEyeCenter, 3, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
        cv::circle(cameraFrame, CURSOR, 10, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        cv::imshow("Camera", cameraFrame);
        // cv::circle(MAIN_WINDOW, CURSOR, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA); // Show CURSOR
        // cv::imshow("Main Window", MAIN_WINDOW);
        if(cv::waitKey(10)==27){ break; }
    }
    
    cv::destroyAllWindows();
}

void initialSetUp(cv::VideoCapture& cap, cv::Mat& frame, cv::Mat& mainWindow, const uint16_t FPS){
    /* VARIABLES */
    const int FRAME_NUM_FOR_INIT = FPS * INIT_SEC;

    //TODO: Add Initial display on the mainWindow like "loading..."

    /* Initialize for INIT_SEC */
    for(int i = 0; i < FRAME_NUM_FOR_INIT; i++){
        cap >> frame;
    }

    //TODO: Update the mainWindow for experiment.
    mainWindow = cv::Mat(cv::Size(DISPLAY_W, DISPLAY_H), CV_8UC3, cv::Scalar::all(255));

    eyeTracker = EyeTracker(CASCADE_FACE_PATH, CASCADE_EYE_PATH);

    CURSOR = cv::Point(cvRound(DISPLAY_W/2), cvRound(DISPLAY_H/2));

    eyeTracker.attachCursor(&CURSOR);
}