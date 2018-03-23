#ifndef __READIMG_H__
#define __READIMG_H__

#include "cv.h"
#include <afx.h>
#include "highgui.h"
#include <fstream>
#include <iostream>     
#include <io.h>     
#include <direct.h>     
#include <string>     
#include <vector>     
#include <iomanip>     
#include <ctime>  

using namespace std;

IplImage* readImg(long imgNum);
CvMat* readTrainsetNum(char* fileName);
void getFiles(string path, vector<string>& files, vector<string>& filesName);


#endif