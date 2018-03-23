#include "dataSet.h"
/* 
 dataSet is used to initial data to svm format.
*/

dataSet::dataSet(int length, int featureDim)
{
	l = length;
	dim = featureDim;

	data = new double [l*dim];
	label = new double [l];
}

int dataSet::initFeature(CvMat* feature)
{
	for(int i=0; i<feature->rows; i++)
	{
		for(int j=0; j<feature->cols; j++)
		{
			data[i*dim+j] = (double)*(feature->data.fl + feature->cols * i + j);
		}
	}

	return 1;
}

int dataSet::readFeatureTxt(string fPath)
{
    ifstream inFile; 

	//read train set
	inFile.open(fPath);
	if(!inFile.is_open())
	{
		cout<<"can not open trainSet.txt"<<endl;
		//system("pause");
		return FALSE;
	}

	for(int i=0; i<l; i++)
	{
		for(int j=0; j<dim; j++)
		{
			inFile >> data[i*dim+j];
		}
		inFile.get();
	}
	inFile.close();

	return TRUE;
}

int dataSet::readLabelTxt(string lPath)
{
	//read label
	ifstream inFile; 
	inFile.open(lPath);
	if(!inFile.is_open())
	{
		cout<<"can not open label.txt"<<endl;
		//system("pause");
		return FALSE;
	}
	for(int i=0; i<l; i++)
	{
		inFile >> label[i];
		inFile.get();
	}
	inFile.close();
	return TRUE;
}

double dataSet::getData(int sampleNum, int fNum)
{
	if(sampleNum > l)
	{
		return FALSE;
	}

	if(fNum > dim)
	{
		return FALSE;
	}
	//读第sampleNum样本的第fNum维特征；
	return data[(sampleNum-1)*dim+(fNum-1)]; 
}

double dataSet::getLabel(int sampleNum)
{
	return label[sampleNum-1];
}
double* dataSet::getLabel()
{
	return label;
}

dataSet::~dataSet()
{
	delete [] data;
	delete [] label;
}