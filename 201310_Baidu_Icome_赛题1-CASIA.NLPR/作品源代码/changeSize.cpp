/***** changeSize *****/
// use this function to resize img to size(fixSize_M, fixSize_N)

#include "changeSize.h"

IplImage* changeSize(IplImage* img, int fixSize_M, int fixSize_N)
{
	int row = img->height;
	int col = img->width;
	int channel = img->nChannels;

	if(row >= col)
	{
	    IplImage* newImg = cvCreateImage(cvSize(fixSize_M, fixSize_N), IPL_DEPTH_8U, channel);
	    cvSetZero(newImg);
        changeSize_hBig(img, newImg, fixSize_M, fixSize_N);
		return newImg;
	}
	else
	{
	    IplImage* rotateFillImg = cvCreateImage(cvSize(fixSize_N, fixSize_M), IPL_DEPTH_8U, channel);
	    cvSetZero(rotateFillImg);
		IplImage* rotateImg = rotateImage(img, 90, 1);
		changeSize_hBig(rotateImg, rotateFillImg, fixSize_N, fixSize_M);
		IplImage* newImg = rotateImage(rotateFillImg, 90, 0);

		cvReleaseImage(&rotateFillImg);
		cvReleaseImage(&rotateImg);
		return newImg;
	}
    return NULL;	
}
/* change size in the situation of height bigger than width */
void changeSize_hBig(IplImage* img, IplImage* newImg, int fixSize_M, int fixSize_N)
{
    int row = img->height;
	int col = img->width;
	int channel = img->nChannels;
	
	if(row >= col)
	{
		int rowNew = fixSize_M;
		int colNew = (col*fixSize_M)/row;

		if(labs(rowNew-colNew)<=1)
		{
			rowNew = fixSize_M;
			colNew = fixSize_N;
		}

		IplImage* resizeImg=cvCreateImage(cvSize(colNew,rowNew),IPL_DEPTH_8U,channel);  
		cvResize(img,resizeImg,CV_INTER_LINEAR);   

		if(rowNew == colNew)
		{
			cvCopy(resizeImg, newImg);
			cvReleaseImage(&resizeImg);
			return;
		}
		// put the resized img to the center of newImg.
		CvRect imgROI = cvRect((fixSize_N - colNew)/2, 0, colNew, rowNew);
		cvSetImageROI(newImg, imgROI);
		cvCopy(resizeImg, newImg);
		cvResetImageROI(newImg);

		/* fill the left and right partital. */
		IplImage* leftFillunit = cvCreateImage(cvSize(10, rowNew),IPL_DEPTH_8U,channel);   
		imgROI = cvRect(0, 0, 10, rowNew);
		cvSetImageROI(resizeImg, imgROI);
		cvCopy(resizeImg, leftFillunit);
		cvResetImageROI(resizeImg);
	    IplImage* rightFillunit = cvCreateImage(cvSize(10, rowNew),IPL_DEPTH_8U,channel); 
		imgROI = cvRect((colNew-10), 0, 10, rowNew);
		cvSetImageROI(resizeImg, imgROI);
        cvCopy(resizeImg, rightFillunit);
		cvResetImageROI(resizeImg);

		IplImage* leftFillimg = cvCreateImage(cvSize((fixSize_N - colNew)/2, fixSize_M), IPL_DEPTH_8U, channel);
		IplImage* rightFillimg = cvCreateImage(cvSize((fixSize_N - colNew)/2, fixSize_M), IPL_DEPTH_8U, channel);
   
		int fillSize = 0; 
		while(fillSize<(fixSize_N - colNew)/2)
		{
			if((((fixSize_N - colNew)/2) - fillSize) >= 10)
			{
				// left fill
			    imgROI = cvRect((((fixSize_N - colNew)/2)-fillSize-10), 0, 10, rowNew);
		        cvSetImageROI(leftFillimg, imgROI);
                cvCopy(leftFillunit, leftFillimg);
		        cvResetImageROI(leftFillimg);
				// right fill
				imgROI = cvRect(fillSize, 0, 10, rowNew);
		        cvSetImageROI(rightFillimg, imgROI);
                cvCopy(rightFillunit, rightFillimg);
		        cvResetImageROI(rightFillimg);
			}
			else
			{
			    // left fill partital.
			    imgROI = cvRect(0,0,((fixSize_N - colNew)/2 - fillSize), rowNew);
		        cvSetImageROI(leftFillimg, imgROI);
		        cvSetImageROI(leftFillunit, imgROI);
                cvCopy(leftFillunit, leftFillimg);
		        cvResetImageROI(leftFillimg);
				cvResetImageROI(leftFillunit);
				// right fill partitial.
				imgROI = cvRect(fillSize, 0, ((fixSize_N - colNew)/2-fillSize), rowNew);
		        cvSetImageROI(rightFillimg, imgROI);
				CvRect imgROI1 = cvRect(10-((fixSize_N - colNew)/2-fillSize), 0, ((fixSize_N - colNew)/2-fillSize), rowNew);
				cvSetImageROI(rightFillunit, imgROI1);
                cvCopy(rightFillunit, rightFillimg);
		        cvResetImageROI(rightFillimg);
				cvResetImageROI(rightFillunit);
			}
			fillSize = fillSize + 10;
		}

     imgROI = cvRect(0, 0, ((fixSize_N - colNew)/2), rowNew);
     cvSetImageROI(newImg, imgROI);
     cvCopy(leftFillimg, newImg);
     cvResetImageROI(newImg);

	 imgROI = cvRect((fixSize_N-(fixSize_N - colNew)/2), 0, fixSize_N, rowNew);
     cvSetImageROI(newImg, imgROI);
     cvCopy(rightFillimg, newImg);
     cvResetImageROI(newImg);

	 cvReleaseImage(&resizeImg);
	 cvReleaseImage(&leftFillunit);
	 cvReleaseImage(&rightFillunit);
	 cvReleaseImage(&leftFillimg);
     cvReleaseImage(&rightFillimg);
	}
}
/*
void changeSize_wBig(IplImage* img, IplImage* newImg, int fixSize_M, int fixSize_N)
{
	
    IplImage* rotateImg = rotateImage(img, 90, 1);
}*/
/* rotateImage */
IplImage* rotateImage(IplImage* src, int angle, bool clockwise)     
{  
    angle = abs(angle) % 180;  
    if (angle > 90)  
    {  
        angle = 90 - (angle % 90);  
    }  
    IplImage* dst = NULL;  
    int width =  
        (int)((double)(src->height * sin(angle * CV_PI / 180.0)) +  
        (double)(src->width * cos(angle * CV_PI / 180.0 )) + 1);  
    int height =  
        (int)((double)(src->height * cos(angle * CV_PI / 180.0)) +  
        (double)(src->width * sin(angle * CV_PI / 180.0 )) + 1);  
    int tempLength = (int)(sqrt((double)src->width * src->width + src->height * src->height) + 10);  
    int tempX = (tempLength + 1) / 2 - src->width / 2;  
    int tempY = (tempLength + 1) / 2 - src->height / 2;  
    int flag = -1;  
  
    dst = cvCreateImage(cvSize(width, height), src->depth, src->nChannels);  
    cvZero(dst);  
    IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), src->depth, src->nChannels);  
    cvZero(temp);  
  
    cvSetImageROI(temp, cvRect(tempX, tempY, src->width, src->height));  
    cvCopy(src, temp, NULL);  
    cvResetImageROI(temp);  
  
    if (clockwise)  
        flag = 1;  
  
    float m[6];  
    int w = temp->width;  
    int h = temp->height;  
    m[0] = (float) cos(flag * angle * CV_PI / 180.);  
    m[1] = (float) sin(flag * angle * CV_PI / 180.);  
    m[3] = -m[1];  
    m[4] = m[0];  
    // 将旋转中心移至图像中间  
    m[2] = w * 0.5f;  
    m[5] = h * 0.5f;  
    //  
    CvMat M = cvMat(2, 3, CV_32F, m);  
    cvGetQuadrangleSubPix(temp, dst, &M);  
    cvReleaseImage(&temp);  
    return dst;  
}  