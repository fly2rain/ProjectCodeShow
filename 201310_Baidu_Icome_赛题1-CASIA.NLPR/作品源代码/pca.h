#ifndef __PCA_H__
#define __PCA_H__

#include <fstream>
#include <string>
#include "cv.h"
#include <iostream>

using namespace std;
using namespace cv;

void readPCAcoef(CvMat* coef,string fileName);
void rd_PCA(CvMat* hogFeature, CvMat* coef, CvMat* rdFeature);


#endif