#include "main.h"

// CASCADE CLASSIFIER
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eye_cascade;

int main(int argc, char** argv){
    cv::VideoCapture cap(0);

    if(!cap.isOpened()){
        std::cerr << "Camera load failed!" << std::endl;
        return -1;
    }

    cv::Mat cameraFrame;
    
    while(true){

        cap >> cameraFrame;
        
        cv::imshow("User", cameraFrame);
        
        if(cv::waitKey(5)==27){
            break;
        }
    }
    cv::destroyAllWindows();
}