#include "HOG.h" 

CvMat* getHogFeature(IplImage* img)
{

	/* input parameters */
	double* params = new double[5];
    params[0]=9;               //number of orientation bins. 
    params[1]=8;               //cell size.
    params[2]=2;               //block size. 
    params[3]=0;               //1 for oriented gradients and 0 otherwise. 
    params[4]=0.2;             //value for clipping of the L2-norm. 
	/* input img size */
	int img_size[2];
	img_size[0] = img->height;
	img_size[1] = img->width;

	/* input grayscale */
	unsigned int grayscale = 0;

	/* output HoG feature */
	int nb_bins       = (int) params[0];    // 有多少个bins
    int block_size    = (int) params[2];    // block size
	int hist1= 2+ceil(-0.5 + img_size[0]/params[1]);
    int hist2= 2+ceil(-0.5 + img_size[1]/params[1]);
    CvMat* HoGfeature = cvCreateMat((hist1-2-(block_size-1))*(hist2-2-(block_size-1))*nb_bins*block_size*block_size, 1, CV_64FC1);
	double* dth_des = (double *) HoGfeature->data.db;

	newHoG(img, params, img_size, dth_des, grayscale);

	// transposition HoGfeature
	CvMat* hogFeature = cvCreateMat(1, HoGfeature->rows, CV_64FC1);
	for(int i=0; i<HoGfeature->rows; i++)
	{
		hogFeature->data.db[i] = HoGfeature->data.db[i];
	}
	cvReleaseMat(&HoGfeature);

	delete[] params;
	return hogFeature;
}

void saveHogFeatureTxt(CvMat** hogFeature, int i)
{
    ofstream fp;
    int txtNum = (i/100);
	string sTxtName = "I:\\Baidu\\\hogFeatureAll\\";
    string s1 = ".txt\0";
	char s[5];
	itoa(txtNum, s, 10);
	string sTxtNum = s;
	string name = sTxtName+sTxtNum+s1;

	fp.open(name.c_str());

	int line;
	if(i>29900)
	{
		line = 68;
	}
	else
	{
		line = 100;
	}
	for(int j=0; j<line; j++)
	{
		double * ptr = hogFeature[j]->data.db;
		for(int i=0; i<hogFeature[j]->cols; i++)
	    {
            fp<<* ptr;
			fp<<" ";
			ptr ++;
		}
		fp <<"\n";
	}
	fp.close();
}

void readHogFeatureTxt(CvMat* hogFeature)
{
	//ifstream fp;
	unsigned int cnt = 0;
	int lineCnt = 0;
	for(int i=0; i<300; i++)
	{
		cout<< "get hog feature from "<< i <<".txt"<<endl;
	    string sTxtName = "I:\\Baidu\\hogFeatureAll\\";
		string s1 = ".txt\0";
		char s[5];
		itoa(i, s, 10);
		string sTxtNum = s;
		string name = sTxtName+sTxtNum+s1;

		ifstream fp;
		fp.open(name.c_str());
		if(i<299)
		{
			lineCnt = 100;
		}
		else
		{
			lineCnt = 68;
		}
		for(int k=0; k<lineCnt; k++)
		{
			//CvMat* hogTemp = cvCreateMat(1, 11664, CV_64FC1);
			float * ptr = hogFeature->data.fl + cnt*hogFeature->cols;
			float p;
			for(int j=0; j<11664; j++)
			{
		        fp >> p;
				/*if(j<100)
				{
					*ptr = p;
				}*/
                *ptr = p;
				ptr ++;
			}
			fp.get();
			cnt ++;

		/*	for(int m=0; m<11664; m++)
			{
				cout<<m<<"   "<<hogFeature[0]->data.db[m]<<"\n"<<endl;
			}*/
		}
		fp.close();
	}
}

void img2Matrix(IplImage* img, double * pixelMat)
{
	for(int y=0; y<img->height; y++)
	{
		unsigned char* ptr = (unsigned char*)(img->imageData + y*img->widthStep);
		for(int x=0; x<img->width; x++)
		{
			pixelMat[x*img->height+y] = (double)ptr[3*x];
			pixelMat[x*img->height+y + img->height*img->width] = (double)ptr[3*x+1];
			pixelMat[x*img->height+y + 2*img->height*img->width] = (double)ptr[3*x+2];
		}
	}
}

/********  newHOG   **********/
//pixels: input image matrix. saved by channel.
//params: input HOG parameters. 
//         params[0]=9;               //number of orientation bins. 
//         params[1]=8;               //cell size.
//         params[2]=2;               //block size. 
//         params[3]=0;               //1 for oriented gradients and 0 otherwise. 
//         params[4]=0.2;             //value for clipping of the L2-norm. 
//img_size:the size of input image.
//dth_des: output HOG feature.
//grayscale: 0 for color image,otherwise,1 for grayscale.

void newHoG(IplImage *img, double *params, int *img_size, double *dth_des, unsigned int grayscale)
{
    
    const float pi = 3.1415926536;
    
    int nb_bins       = (int) params[0];
    double cwidth     =  params[1];
    int block_size    = (int) params[2];
    int orient        = (int) params[3];
    double clip_val   = params[4];
    
    int img_width  = img_size[1];
    int img_height = img_size[0];
    
    int hist1= 2+ceil(-0.5 + img_height/cwidth);
    int hist2= 2+ceil(-0.5 + img_width/cwidth);
    
    double bin_size = (1+(orient==1))*pi/nb_bins;
    
    float dx[3], dy[3], grad_or, grad_mag, temp_mag;
    float Xc, Yc, Oc, block_norm;
    int x1, x2, y1, y2, bin1, bin2;
    int des_indx = 0;
    
    vector<vector<vector<double> > > h(hist1, vector<vector<double> > (hist2, vector<double> (nb_bins, 0.0) ) );    
    vector<vector<vector<double> > > block(block_size, vector<vector<double> > (block_size, vector<double> (nb_bins, 0.0) ) );
    
    //Calculate gradients (zero padding)

	double* pixels = new double[3*img->width * img->height];
	img2Matrix(img, pixels);
    
    for(unsigned int y=0; y<img_height; y++) 
	{
        for(unsigned int x=0; x<img->width; x++) 
		{
            if (grayscale == 1)
			{
                if(x==0)  /* X gradient */
				{
					dx[0] = pixels[y +(x+1)*img_height];
				}
                else
				{
                    if (x==img_width-1) 
				    {
					     dx[0] = -pixels[y + (x-1)*img_height];
					}
                    else
					{
						dx[0] = pixels[y+(x+1)*img_height] - pixels[y + (x-1)*img_height];
					}
                }
                if(y==0) 
				{
					dy[0] = -pixels[y+1+x*img_height];
				}
                else
				{
                    if (y==img_height-1) 
					{
						dy[0] = pixels[y-1+x*img_height];
					}
                    else 
					{
						dy[0] = -pixels[y+1+x*img_height] + pixels[y-1+x*img_height];
					}
                }
            }
            else // grayscale = 0
			{
                if(x==0)
				{
                    dx[0] = pixels[y +(x+1)*img_height];
                    dx[1] = pixels[y +(x+1)*img_height + img_height*img_width];
                    dx[2] = pixels[y +(x+1)*img_height + 2*img_height*img_width];   
                }
                else
				{
                    if (x==img->width-1)
					{
                        dx[0] = -pixels[y + (x-1)*img_height];                        
                        dx[1] = -pixels[y + (x-1)*img_height + img_height*img_width];
                        dx[2] = -pixels[y + (x-1)*img_height + 2*img_height*img_width];
                    }
                    else
					{
                        dx[0] = pixels[y+(x+1)*img_height] - pixels[y + (x-1)*img_height];
                        dx[1] = pixels[y+(x+1)*img_height + img_height*img_width] - pixels[y + (x-1)*img_height + img_height*img_width];
                        dx[2] = pixels[y+(x+1)*img_height + 2*img_height*img_width] - pixels[y + (x-1)*img_height + 2*img_height*img_width];
                    }
                }
                if(y==0)
				{
                    dy[0] = -pixels[y+1+x*img_height];
                    dy[1] = -pixels[y+1+x*img_height + img_height*img_width];
                    dy[2] = -pixels[y+1+x*img_height + 2*img_height*img_width];
                }
                else
				{
                    if (y==img_height-1)
					{
                        dy[0] = pixels[y-1+x*img_height];
                        dy[1] = pixels[y-1+x*img_height + img_height*img_width];
                        dy[2] = pixels[y-1+x*img_height + 2*img_height*img_width];
                    }
                    else
					{
                        dy[0] = -pixels[y+1+x*img_height] + pixels[y-1+x*img_height];
                        dy[1] = -pixels[y+1+x*img_height + img_height*img_width] + pixels[y-1+x*img_height + img_height*img_width];
                        dy[2] = -pixels[y+1+x*img_height + 2*img_height*img_width] + pixels[y-1+x*img_height + 2*img_height*img_width];
                    }
                }
            }
            
            grad_mag = sqrt(dx[0]*dx[0] + dy[0]*dy[0]);
            grad_or= atan2(dy[0], dx[0]);
            
            if (grayscale == 0)
			{
                temp_mag = grad_mag;
                for (unsigned int cli=1;cli<3;++cli)
				{
                    temp_mag= sqrt(dx[cli]*dx[cli] + dy[cli]*dy[cli]);
                    if (temp_mag>grad_mag)
					{
                        grad_mag=temp_mag;
                        grad_or= atan2(dy[cli], dx[cli]);
                    }
                }
            }
            
            if (grad_or<0) grad_or+=pi + (orient==1) * pi;

            // trilinear interpolation
            
            bin1 = (int)floor(0.5 + grad_or/bin_size) - 1;
            bin2 = bin1 + 1;
            x1   = (int)floor(0.5+ x/cwidth);
            x2   = x1+1;
            y1   = (int)floor(0.5+ y/cwidth);
            y2   = y1 + 1;
            
            Xc = (x1+1-1.5)*cwidth + 0.5;
            Yc = (y1+1-1.5)*cwidth + 0.5;
            
            Oc = (bin1+1+1-1.5)*bin_size;
            
            if (bin2==nb_bins)
			{
                bin2=0;
            }
            if (bin1<0)
			{
                bin1=nb_bins-1;
            }            
           
            h[y1][x1][bin1]= h[y1][x1][bin1]+grad_mag*(1-((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
            h[y1][x1][bin2]= h[y1][x1][bin2]+grad_mag*(1-((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
            h[y2][x1][bin1]= h[y2][x1][bin1]+grad_mag*(1-((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
            h[y2][x1][bin2]= h[y2][x1][bin2]+grad_mag*(1-((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
            h[y1][x2][bin1]= h[y1][x2][bin1]+grad_mag*(((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
            h[y1][x2][bin2]= h[y1][x2][bin2]+grad_mag*(((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
            h[y2][x2][bin1]= h[y2][x2][bin1]+grad_mag*(((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
            h[y2][x2][bin2]= h[y2][x2][bin2]+grad_mag*(((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
        }
    }
    
    
    
    //Block normalization
    
    for(unsigned int x=1; x<hist2-block_size; x++){
        for (unsigned int y=1; y<hist1-block_size; y++){
            
            block_norm=0;
            for (unsigned int i=0; i<block_size; i++){
                for(unsigned int j=0; j<block_size; j++){
                    for(unsigned int k=0; k<nb_bins; k++){
                        block_norm+=h[y+i][x+j][k]*h[y+i][x+j][k];
                    }
                }
            }
            
            block_norm=sqrt(block_norm);
            for (unsigned int i=0; i<block_size; i++){
                for(unsigned int j=0; j<block_size; j++){
                    for(unsigned int k=0; k<nb_bins; k++){
                        if (block_norm>0){
                            block[i][j][k]=h[y+i][x+j][k]/block_norm;
                            if (block[i][j][k]>clip_val) block[i][j][k]=clip_val;
                        }
                    }
                }
            }
            
            block_norm=0;
            for (unsigned int i=0; i<block_size; i++){
                for(unsigned int j=0; j<block_size; j++){
                    for(unsigned int k=0; k<nb_bins; k++){
                        block_norm+=block[i][j][k]*block[i][j][k];
                    }
                }
            }
            
            block_norm=sqrt(block_norm);
            for (unsigned int i=0; i<block_size; i++){
                for(unsigned int j=0; j<block_size; j++){
                    for(unsigned int k=0; k<nb_bins; k++){
                        if (block_norm>0) dth_des[des_indx]=block[i][j][k]/block_norm;
                        else dth_des[des_indx]=0.0;
                        des_indx++;
                    }
                }
            }
        }
    }
	delete []pixels;
}