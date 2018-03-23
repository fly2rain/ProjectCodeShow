#include "pca.h"

void readPCAcoef(CvMat* coef , string fileName)
{
	ifstream inFile; 

	inFile.open(fileName);
	int rows = coef->rows;
	int cols = coef->cols;
	for(int i=0; i<rows; i++)
	{
		double* ptr = coef->data.db + i*coef->cols;
		for(int j=0; j<cols; j++)
		{
			inFile >> ptr[j];
		}
		inFile.get();
	}

	inFile.close();
}

void rd_PCA(CvMat* hogFeature, CvMat* coef, CvMat* rdFeature)
{

	cvMatMulAdd(hogFeature, coef, 0, rdFeature);

}