// toC++.cpp : 定义控制台应用程序的入口点。
//
#include "readImg.h"
#include "changeSize.h"
#include "HoG.h"
#include "pca.h"
#include "dataSet.h"
#include "svm.h"
#include "Gist_.h"

int main(int argc , char* argv[])
{

	string imagePath ;
	string resultFile ;
	// command line parameter parsing
	if ( argc != 3 )
	{
		cout<<"Input arguments are not enough! Executable terminated."<<endl;
		system("pause");
		return -1;
	} 
	else
	{
		imagePath = argv[1];
		resultFile = argv[2];
	}



	cout<<"Preparing..."<<endl;
	//  all images resized to 150*150
	int fixSize_M = 150; 
	int fixSize_N = 150;
	// Load PCA coefficients for HOG and Gist

	CvMat* coeff_HOG_64 = cvCreateMat(11664, 64, CV_64FC1);
	string coeff_HOG_64_file="coeff_HOG_64.txt";
	readPCAcoef(coeff_HOG_64 , coeff_HOG_64_file);

	CvMat* coeff_Gist_64= cvCreateMat(512, 64, CV_64FC1);
	string coeff_Gist_64_file="coeff_Gist_64.txt";
	readPCAcoef(coeff_Gist_64 , coeff_Gist_64_file);

	CvMat* coeff_Gist_128= cvCreateMat(512, 128, CV_64FC1);
	string coeff_Gist_128_file="coeff_Gist_128.txt";
	readPCAcoef(coeff_Gist_128 , coeff_Gist_128_file);

	// Load SVM model

	svm_model *model_Gist = svm_load_model("model_gist_128_sigma@1.5_c@5.3.txt");
	svm_model *model_HOG_Gist = svm_load_model("model_HOG&Gist(sep_PCA)_128_sigma@0.012_c@6.8.txt");

	// predict images label.
    vector<string> files;    
	vector<string> filesName; 
    getFiles(imagePath, files, filesName);
	int numerImages = files.size();
	int* predictLabel = new int [numerImages];  



	cout<<"Start to process."<<endl;

	ofstream fp;
	fp.open(resultFile,ios::out);
	fp.close();

	for(long i=0; i<numerImages; i++)
	{
		
		IplImage* img = cvLoadImage(files[i].c_str(), -1);

		

		if(!img)
		{
			predictLabel[i] = 1;
			cout<<"The "<<i<<" th image processed."<<endl;
			fp.open(resultFile,ios::app);
			fp << filesName[i]<<" "<< predictLabel[i]<<endl;
			fp.close();
			cvReleaseImage(&img);
			continue;
		}

		if (img->depth != 8 || img->nChannels !=3)
		{
			predictLabel[i] = 1;
			cout<<"The "<<i<<" th image processed."<<endl;
			fp.open(resultFile,ios::app);
			fp << filesName[i]<<" "<< predictLabel[i]<<endl;
			fp.close();
			cvReleaseImage(&img);
			continue;
		}

		else
		{
		    IplImage* resizedImg = changeSize(img, fixSize_M, fixSize_N);
			cvReleaseImage(&img);

		    // Extract hog and gist feature.
		    CvMat* hogFeature = getHogFeature(resizedImg);
		    CvMat* gistFeature = getGistFeature(resizedImg);
		
		    // hog feature reduction dimension.
		    CvMat* rdFeature_HOG_64 = cvCreateMat(1, 64, CV_64FC1);
			CvMat* rdFeature_Gist_64 = cvCreateMat(1,64, CV_64FC1);
			CvMat* rdFeature_Gist_128=cvCreateMat(1, 128,CV_64FC1);
		    rd_PCA(hogFeature, coeff_HOG_64, rdFeature_HOG_64);
			rd_PCA(gistFeature, coeff_Gist_64 , rdFeature_Gist_64);
			rd_PCA(gistFeature, coeff_Gist_128 , rdFeature_Gist_128);

		    cvReleaseImage(&resizedImg);
			cvReleaseMat(&hogFeature);
			cvReleaseMat(&gistFeature);
		    
			// predict probs
	        svm_node* sample_gist = new svm_node [rdFeature_Gist_128->cols+1];
			svm_node* sample_hog_gist = new svm_node [rdFeature_HOG_64->cols+rdFeature_Gist_64->cols+1];

		    for(int j=0; j<rdFeature_Gist_128->cols; j++)
		    {
			    sample_gist[j].index = j+1;
		        sample_gist[j].value = rdFeature_Gist_128->data.db[j];
		     }
		     sample_gist[rdFeature_Gist_128->cols].index = -1;

			 int temp=0;
			 for (int j=0; j<rdFeature_HOG_64->cols+rdFeature_Gist_64->cols; j++)
			 {
				 // 0-63 hog
				 if (j<=rdFeature_HOG_64->cols-1)
				 {
					 sample_hog_gist[j].index = j+1;
					 sample_hog_gist[j].value = rdFeature_HOG_64->data.db[j];
				 } 
				 else
				 {
					 sample_hog_gist[j].index = j+1;
					 sample_hog_gist[j].value = rdFeature_Gist_64->data.db[temp];
					 temp++;
				 }
			 }
			 sample_hog_gist[rdFeature_HOG_64->cols+rdFeature_Gist_64->cols].index = -1;

			 double probUnit_Gist[2];
		     double predict_Gist = svm_predict_probability(model_Gist , sample_gist , probUnit_Gist);

			 double probUnit_HOG_Gist[2];
			 double predict_HOG_Gist = svm_predict_probability(model_HOG_Gist , sample_hog_gist , probUnit_HOG_Gist);

			 if ( (probUnit_Gist[0]+probUnit_HOG_Gist[0]) < (probUnit_Gist[1]+probUnit_HOG_Gist[1]) )
			 {
				 predictLabel[i]=0;
			 } 
			 else
			 {
				 predictLabel[i]=1;
			 }
			 cvReleaseMat(&rdFeature_Gist_128);
			 cvReleaseMat(&rdFeature_HOG_64);
			 cvReleaseMat(&rdFeature_Gist_64);
			 cout<<"The "<<i<<" th image processed."<<endl;

			 fp.open(resultFile,ios::app);
			 fp << filesName[i]<<" "<< predictLabel[i]<<endl;
			 fp.close();
		}
	}

	
	cout<<"All image have been processed successfully !"<<endl;


	delete predictLabel;

	cvReleaseMat(&coeff_Gist_128);
	cvReleaseMat(&coeff_Gist_64);
	cvReleaseMat(&coeff_HOG_64);

}

