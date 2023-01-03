#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(){
    cv::VideoCapture cap(0);

    if(!cap.isOpened()){
        std::cerr << "Camera load failed!" << std::endl;
        return -1;
    }
    
    cv::Mat cameraFrame;

    while(true){
        cap >> cameraFrame;
        flip(cameraFrame, cameraFrame, 1);
        imshow("Frame", cameraFrame);

        if(waitKey(10)==27){ break; }
    }
}