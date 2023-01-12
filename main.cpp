#include "main.h"

// TODO:
// 1. 얼굴 안정화 코드 찾기 & 수정하기
// 2. 마우스 클릭, 드래그 등에 따른 포인터 흔적 남기기.
// 3. 실험을 위한 부수적인 설정

EyeTracker eyeTracker;
Gesture result;
cv::Mat cameraFrame, MAIN_WINDOW;
cv::Point CURSOR;

int main(int argc, char** argv){
    cv::VideoCapture cap(CAM_NUM);

    if(!cap.isOpened()){
        std::cerr << "Camera load failed!" << std::endl;
        return -1;
    }

    const uint16_t fps = cap.get(cv::CAP_PROP_FPS);

    initialSetUp(cap, cameraFrame, fps);

    while(true){
        cap >> cameraFrame;
        
        result = eyeTracker.traceAndTranslate2Gesture(cameraFrame);

        // std::cout <<  eyeTracker.getLastGestureData().getFrameTime().count() << " sec per frame." << std::endl;

        paintMainWindow();
        showCameraFrame();
        
        if(cv::waitKey(10)==27){ break; }
    }
    
    cv::destroyAllWindows();
}

void initialSetUp(cv::VideoCapture& cap, cv::Mat& frame, const uint16_t FPS){
    /* VARIABLES */
    const int FRAME_NUM_FOR_INIT = FPS * INIT_SEC;

    //TODO: Add Initial display on the mainWindow like "loading..."

    /* Initialize for INIT_SEC */
    for(int i = 0; i < FRAME_NUM_FOR_INIT; i++){
        cap >> frame;
    }

    //TODO: Update the mainWindow for experiment.
    MAIN_WINDOW = cv::Mat(cv::Size(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT), CV_8UC3, cv::Scalar::all(255));

    eyeTracker = EyeTracker(CASCADE_FACE_PATH, CASCADE_EYE_PATH);

    CURSOR = cv::Point(cvRound(MAIN_WINDOW_WIDTH/2), cvRound(MAIN_WINDOW_HEIGHT/2));

    eyeTracker.attachCursor(&CURSOR);
}

void paintMainWindow(){
    static cv::String command;
    MAIN_WINDOW = cv::Mat(cv::Size(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT), CV_8UC3, cv::Scalar::all(255));

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
    
    cv::putText(MAIN_WINDOW, command, cv::Point(0, 30), 1, 1, cv::Scalar(0, 0, 255));

    cv::circle(MAIN_WINDOW, CURSOR, 5, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
    cv::imshow("MAIN", MAIN_WINDOW);
}

void showCameraFrame(){
    cv::Rect faceROI, leftEyeROI, rightEyeROI;
    cv::Point leftEyeCenter, rightEyeCenter, bothEyeCenter;
    

    faceROI = eyeTracker.getLastFaceROI();
    leftEyeROI = eyeTracker.getLastLeftEyeROI();
    rightEyeROI = eyeTracker.getLastRightEyeROI();
    leftEyeCenter = eyeTracker.getLastLeftEyeCenter();
    rightEyeCenter = eyeTracker.getLastRightEyeCenter();
    bothEyeCenter = eyeTracker.getCenterOfBothEyes();
    
    cv::rectangle(cameraFrame, faceROI, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    cv::rectangle(cameraFrame, leftEyeROI, cv::Scalar(0, 128, 255), 3, cv::LINE_AA);
    cv::rectangle(cameraFrame, rightEyeROI, cv::Scalar(0, 128, 255), 3, cv::LINE_AA);
    cv::circle(cameraFrame, leftEyeCenter, 5, cv::Scalar(0, 128, 255), -1, cv::LINE_AA);
    cv::circle(cameraFrame, rightEyeCenter, 5, cv::Scalar(0, 128, 255), -1, cv::LINE_AA);
    cv::circle(cameraFrame, bothEyeCenter, 3, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);

    cv::imshow("WebCam", cameraFrame);
}