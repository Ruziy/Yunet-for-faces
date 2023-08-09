
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"

//define the buffer size. Do not change the size!
//0x9000 = 1024 * (16 * 2 + 4), detect 1024 face at most
#define DETECT_BUFFER_SIZE 0x9000
using namespace cv;
using namespace std;

int main()
{

    Mat image = imread("images/work-16.jpg");
    Mat resize_image;
    

    int* pResults = NULL;
    //pBuffer is used in the detection functions.
    //If you call functions in multiple threads, please create one buffer for each thread!
    unsigned char* pBuffer = (unsigned char*)malloc(DETECT_BUFFER_SIZE);
    if (!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }


    ///////////////////////////////////////////
    // CNN face detection 
    // Best detection rate
    //////////////////////////////////////////
    //!!! The input image must be a BGR one (three-channel) instead of RGB
    //!!! DO NOT RELEASE pResults !!!
    TickMeter cvtm;
    cvtm.start();

    pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);

    cvtm.stop();
    printf("time = %gms\n", cvtm.getTimeMilli());

    printf("%d faces detected.\n", (pResults ? *pResults : 0));
    Mat result_image = image.clone();
    //print the detection results
    for (int i = 0; i < (pResults ? *pResults : 0); i++)
    {
        short* p = ((short*)(pResults + 1)) + 16 * i;
        int confidence = p[0];
        int x = p[1];
        int y = p[2];
        int w = p[3];
        int h = p[4];

        //show the score of the face. Its range is [0-100]
        char sScore[256];
        snprintf(sScore, 256, "%d", confidence);
        cv::putText(result_image, sScore, cv::Point(x, y - 3), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        //draw face rectangle
        if (confidence > 65) {
            rectangle(result_image, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
            //draw five face landmarks in different colors
            cv::circle(result_image, cv::Point(p[5], p[5 + 1]), 1, cv::Scalar(255, 0, 0), 2);
            cv::circle(result_image, cv::Point(p[5 + 2], p[5 + 3]), 1, cv::Scalar(0, 0, 255), 2);
            cv::circle(result_image, cv::Point(p[5 + 4], p[5 + 5]), 1, cv::Scalar(0, 255, 0), 2);
            cv::circle(result_image, cv::Point(p[5 + 6], p[5 + 7]), 1, cv::Scalar(255, 0, 255), 2);
            cv::circle(result_image, cv::Point(p[5 + 8], p[5 + 9]), 1, cv::Scalar(0, 255, 255), 2);

            //print the result
            printf("face %d: confidence=%d, [%d, %d, %d, %d] (%d,%d) (%d,%d) (%d,%d) (%d,%d) (%d,%d)\n",
                i, confidence, x, y, w, h,
                p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14]);
        }
    }
    resize(result_image, resize_image, Size(600, 400));
    imshow("result", resize_image);

    waitKey();

    //release the buffer
    free(pBuffer);

    return 0;
}

