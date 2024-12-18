// --- Basic face detection --- //

#include <iostream>
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

void detect(Mat frame);

// face cascade classifier
CascadeClassifier face_cascade;

// Function main
int main(void)
{
    // load the initial cascade
    if (!face_cascade.load("HAAR FILE LOCATION")){
        cout << "Error: Couldn't find the Haar file." << endl;
        return -1;
    }

    // load in the image to analyse
    Mat frame = imread("IMAGE TO ANALYSE LOCATION");

    // apply the classifier
    if (!frame.empty()){
        detect(frame);
    }else{
        cout << "Error: No image submitted." << endl;
    }
    
    return 0;
}

// detect the faces in the image
void detect(Mat frame)
{
    // declare vector of faces variable
    vector<Rect> faces;
    Mat frame_gray;

    // convert image to grayscale
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // detect the faces in the image -- adjust 4th parameter for detail
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 10, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // loop through all the faces found
    cout << faces.size() << " face(s) were/was found." << endl;
    for (int i = 0; i < faces.size(); i++)
    {                 
        cout << "Face geometry for face " << i << " -- x:" << faces[i].x << ", y:" << faces[i].y << ", width:" << faces[i].width << ", height:" << faces[i].height << ";" << endl;
    }
}