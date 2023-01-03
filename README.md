# Look-Click

## Environment
> ***OS***: MacOS Ventura 13.1
>
> ***Compiler***: g++
>
> ***OpenCV***: OpenCV 4.7.0(stable, installed using homebrew)
>
> ***OpenCV Path***: /opt/homebrew/opt/opencv@4
>
> ***Camera Number***: 1 (this is important when you create `VideoCapture` instance. Please Change CAM_NUM in main.h if your camera doesn't work.)

To compile and execute, follow steps below ⬇️

1. Download current repository to your `$PATH`
> Click the "code" button on top of this page and select "Download ZIP".
>
> or
> 
> ```shell
> cd $PATH
> git clone https://github.com/JinwangMok/Look-Click.git
> cd Look-Click
> ```

2. 
> Compile the project. (using g++)
> 
> ```shell
> g++ main.cpp eyeTracker.cpp gestureTranslator.cpp -o main `pkg-config --libs --cflags opencv4` -std=c++11
> ```

3. Execute the binary.
> 
> ```shell
> main
> ```

🚨This project is on the way. Thus, these steps are now in modifying.
If something better than this guide-line, PLEASE telling me what you know🙏🏻