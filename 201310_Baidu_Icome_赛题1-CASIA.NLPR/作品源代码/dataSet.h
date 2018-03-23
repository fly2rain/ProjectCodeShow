#ifndef __DATASET_H
#define __DATASET_H


#include <stdio.h>
#include <list>
#include <fstream>
#include <iostream>
#include <string>
#include "cv.h"

#define TRUE 1
#define FALSE 0

using namespace std;

class dataSet
{
private:
	double* data;   //样本矩阵
    double* label;  //样本标签
public:
	int l;        //样本集的大小
	int dim;      //特征的维数
public:
	dataSet(int length, int featureDim);
	int readFeatureTxt(string fPath);
	int readLabelTxt(string lPath);
	int initFeature(CvMat* feature);
	double getData(int sampleNum, int fNum);
	double getLabel(int sampleNum);
	double* getLabel();
	~dataSet();
};

#endif