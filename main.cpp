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
    EyeTracker eyeTracker(CASCADE_FACE_PATH, CASCADE_EYE_PATH);

    cv::Mat cameraFrame;
    const uint16_t fps = cap.get(cv::CAP_PROP_FPS);

    CURSOR = initialSetUp(cap, cameraFrame, MAIN_WINDOW, fps);
    
    std::cout << CURSOR << std::endl;

    cap >> cameraFrame;
    // cv::circle(MAIN_WINDOW, CURSOR, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA); // Show CURSOR
    cv::imshow("Main Window", MAIN_WINDOW);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

cv::Point initialSetUp(cv::VideoCapture& cap, cv::Mat& frame, cv::Mat& mainWindow, const uint16_t FPS){
    /* VARIABLES */
    const int FRAME_NUM_FOR_INIT = FPS * INIT_SEC;

    //TODO: Add Initial display on the mainWindow like "loading..."

    /* Initialize for INIT_SEC */
    for(int i = 0; i < FRAME_NUM_FOR_INIT; i++){
        cap >> frame;
    }

    //TODO: Update the mainWindow for experiment.
    mainWindow = cv::Mat(cv::Size(DISPLAY_W, DISPLAY_H), CV_8UC3, cv::Scalar::all(255));

    return cv::Point(cvRound(DISPLAY_W/2), cvRound(DISPLAY_H/2));
}