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
	double* data;   //��������
    double* label;  //������ǩ
public:
	int l;        //�������Ĵ�С
	int dim;      //������ά��
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