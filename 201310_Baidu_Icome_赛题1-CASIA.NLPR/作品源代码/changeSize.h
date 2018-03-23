#ifndef __CHANGESIZE_H__
#define __CHANGESIZE_H__

#include "cv.h"
#include <math.h>
#include "highgui.h"

IplImage* rotateImage(IplImage* src, int angle, bool clockwise);
IplImage* changeSize(IplImage* img, int fixSize_M, int fixSize_N);
void changeSize_hBig(IplImage* img, IplImage* newImg, int fixSize_M, int fixSize_N);
void changeSize_wBig(IplImage* img, IplImage* newImg, int fixSize_M, int fixSize_N);

#endif
