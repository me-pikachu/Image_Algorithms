Installing the dependencies :
    Linux : 
        sudo apt update
        sudo apt install libopencv-dev
    Windows :
        Download the OpenCV library from the official OpenCV website.
    Mac OS:
        brew install opencv
    Make sure to keep the installation files in the environment variables

Compiling : g++ image.cpp -o image `pkg-config --cflags --libs opencv4`
Executing : ./image
    
