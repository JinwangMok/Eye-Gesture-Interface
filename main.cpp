#include "main.h"

// CASCADE CLASSIFIER
cv::CascadeClassifier faceClassifier;
cv::CascadeClassifier eyeClassifier;

int main(int argc, char** argv){
    cv::VideoCapture cap(0);

    if(!cap.isOpened()){
        std::cerr << "Camera load failed!" << std::endl;
        return -1;
    }
    
    cv::Mat cameraFrame;
    const uint16_t fps = cap.get(cv::CAP_PROP_FPS);

    CURSOR = initialSetUp(cap, cameraFrame, mainWindow, fps);
    
    std::cout << CURSOR << std::endl;
    
    cap >> cameraFrame;
    
    cv::imshow("Main Window", mainWindow);
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
    MAIN_WINDOW(cv::Size(DISPLAY_W, DISPLAY_H), CV_8UC3, cv::Scalar::all(255));
    
    return cv::Point(0, 0);
}