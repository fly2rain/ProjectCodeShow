#ifndef __HOG_H__
#define __HOG_H__

#include <stdlib.h>
#include <math.h>
#include <vector>
#include "cv.h"
#include "highgui.h"
#include<fstream>

using namespace std;  

CvMat* getHogFeature(IplImage* img);
void img2Matrix(IplImage* img, double * pixelMat);
void newHoG(IplImage *img, double *params, int *img_size, double *dth_des, unsigned int grayscale);
void saveHogFeatureTxt(CvMat** hogFeature, int i);
void readHogFeatureTxt(CvMat* hogFeature);

#endif