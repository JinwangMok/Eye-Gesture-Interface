#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(){
    Rect a(10, 20, 5, 5), b(20, 30, 5, 5);
    cout << b - a << endl;
}