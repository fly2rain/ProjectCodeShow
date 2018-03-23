#include "Gist_.h"

CvMat* getGistFeature(IplImage* img)
{
	Mat input =img;
	
	Mat im_gray( input.rows , input.cols, CV_32FC1);
	cv::cvtColor(input , im_gray , CV_RGB2GRAY);

	Mat im_resize;
	resize(im_gray , im_resize , Size(150,150) , 0 , 0 , CV_INTER_LINEAR);

	Mat im_resize_float;
	im_resize.convertTo(im_resize_float, CV_32FC1);
	//////////////////////////////////////////////////////////////////////////
	int nblocks=4;
	int n_scale=4;
	int orientations_per_scale[4]={8,8,8,8};
	int descsize=512;

	//printf("width: %d height:%d \n",im_resize_float.cols,im_resize_float.rows);
	//for(int i=0;i<5;i++)
	//{
	//	for (int j=0;j<5;j++)
	//	{
	//		printf("%f ",im_resize_float.at<float>(i,j));
	//	}
	//	printf("\n");
	//}


	float *desc=cvGist(im_resize_float, nblocks, n_scale, orientations_per_scale, descsize);
	//////////////////////////////////////////////////////////////////////////

	//for(int i=0;i<descsize;i++)
	//	printf("%.4f \n",desc[i]);
	//printf("\n");


	CvMat* gist =  cvCreateMat(1 , 512 , CV_64FC1);

	//cvInitMatHeader(gist , 1 , 512 , CV_32FC1 , desc);

	for (int i=0; i<512; i++)
	{
		gist->data.db[i]=(float)desc[i];
	}
	


	delete desc;

	return gist;

}