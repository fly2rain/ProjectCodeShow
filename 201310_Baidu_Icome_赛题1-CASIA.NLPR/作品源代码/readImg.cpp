#include "readImg.h"

void getFiles(string path, vector<string>& files, vector<string>& filesName) 
{  
    // handle of file   
    long hFile = 0;    
    // info of file     
    struct _finddata_t fileinfo;    
  
    string p;  
  
    if((hFile = _findfirst(p.assign(path).append("/*").c_str(),&fileinfo)) != -1)  
	{    
        do  
		{    
            // if it's catalog, iteration. else add to the list.
            if((fileinfo.attrib & _A_SUBDIR)) 
			{    
                if((strcmp(fileinfo.name,".") != 0)&&(strcmp(fileinfo.name,"..")!=0))    
                    getFiles(p.assign(path).append("/").append(fileinfo.name), files, filesName);    
            }  else  {    
                files.push_back(p.assign(path).append("/").append(fileinfo.name));  
				filesName.push_back(fileinfo.name);
            }    
        }while(_findnext(hFile,&fileinfo) == 0);    
  
        _findclose(hFile);    
    }  
}



IplImage* readImg(long imgNum)
{
	IplImage* img; 
	CFileFind fileFinder;
	static char isLastImg = 0;
	CString strPicDir;

	if(imgNum<=4999)
	{
		strPicDir = "E:\\Baidu\\Data\\train-origin-pics-part1";
	}
	else if(imgNum<=9999)
	{
		strPicDir = "E:\\Baidu\\Data\\train-origin-pics-part2";
		imgNum -= 4999;
	}
	else if(imgNum<=14999)
	{
		strPicDir = "E:\\Baidu\\Data\\train-origin-pics-part3";
		imgNum -= 9999;
	}
	else if(imgNum<=19999)
	{
		strPicDir = "E:\\Baidu\\Data\\train-origin-pics-part4";
		imgNum -= 14999;
	}
	else if(imgNum<=24999)
	{
		strPicDir = "E:\\Baidu\\Data\\train-origin-pics-part5";
		imgNum -= 19999;
	}
	else
	{
		strPicDir = "E:\\Baidu\\Data\\train-origin-pics-part6";
		imgNum -= 24999;
	}

	/*if(isLastImg)
	{
		return NULL;
	}*/

	if (strPicDir.Right(1) == TEXT('\\'))
	{

		int nPos  = strPicDir.ReverseFind(TEXT('\\'));
		strPicDir = strPicDir.Left(nPos);
	}

	CString strPicFile = TEXT("");
	strPicFile.Format(TEXT("%s\\%s"),strPicDir,TEXT("*.jpg"));
	CString strFilePath;
	bool bWorking = fileFinder.FindFile(strPicFile);
	long imgNumCnt = 0;
	while(imgNumCnt < imgNum)
	{
		imgNumCnt ++;
		if(fileFinder.FindNextFile() == NULL)
		{
			cout<< "There is the last in this folder!(function readImg)" << endl;
			strFilePath = fileFinder.GetFilePath();
			isLastImg = 1;
		    //return NULL;
		}
		else
		{
			strFilePath = fileFinder.GetFilePath();
		}
	}

    char* cstr = new char[strFilePath.GetLength()+1]; 
    WideCharToMultiByte(CP_OEMCP, 0, strFilePath, -1, cstr, strFilePath.GetLength(), NULL, NULL);  
	cstr[strFilePath.GetLength()] = '\0';
	img = cvLoadImage(cstr,1);
	if(img==NULL)
	{
		cout<<"read image wrong in function readImg!"<<endl;
	}
	fileFinder.Close();
	//cvNamedWindow("1", 1);
	//cvShowImage("1", img);
    delete [] cstr;
    return img;
}  

CvMat* readTrainsetNum(char* fileName)
{
	ifstream inFile; 
	char line[128];

	//read train set
	inFile.open(fileName);
	if(!inFile.is_open())
	{
		cout<<"can not open trainsetNum.txt"<<endl;
		//system("pause");
		return NULL;
	}

	// count the num of train samples.
	float trainSampleCnt = 0;
	while(!inFile.eof())
	{
		inFile.getline(line, sizeof(line));
		trainSampleCnt++;
	}
	trainSampleCnt --;

	// read the sample number.
	CvMat* trainsetNum = cvCreateMat(trainSampleCnt, 1, CV_32FC1);
	inFile.clear(); // clear eof.
	inFile.seekg(0, ios_base::beg);// relocate the pointer to the begining of the file. 
	for(long i=0; i<trainSampleCnt; i++)
	{
		inFile >> trainsetNum->data.fl[i] ;
	}
	inFile.close();

	return trainsetNum;
}

