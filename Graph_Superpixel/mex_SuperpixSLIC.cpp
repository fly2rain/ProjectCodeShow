/**
 * ==================================================================
 *  \brief: calculate the superpixel (over-segmentation) for a given
 *          image by the SLIC method. This algorithm could tackle with
 *          image with multiple channels. For example, it could handle
 *          hyperspectral spectral images of hundreds of channels.
 *
 *  \attention: this algorithm is based on the paper: <SLIC Superpixels Compared
 *              to State-of-the-Art Superpixel Methods>
 *
 *  \file: mex_SuperpixelSLIC.cpp
 *
 *  \author:  Feiyun Zhu (fyZhu), fyzhu@nlpr.ia.ac.cn
 *  \date:    2014 - 01 - 10
 *  \version: 1.0
 *  Company:  Institute of Automation, Chinese Academy of Sciences
 * ===================================================================
*/

//#include "D:\ProgramFiles_D\MATLAB2013\extern\include\matrix.h"
//#include "D:\ProgramFiles_D\MATLAB2013\extern\include\mex.h"
//#include "/home/zfy/Matlab2013/extern/include/mex.h"
//#include "/home/zfy/Matlab2013/extern/include/matrix.h"
#include "mex.h"
#include "matrix.h"
#include <vector>
#include <cmath>
using namespace  std;
typedef unsigned int uint32; // 32 bit == 8 byte

struct Sizes {
    Sizes (const uint32& row, const uint32& col, const uint32& npix, \
           const uint32 &band, const double& compact=20, const uint32& mIter=3, \
           const bool& perturb=false) \
        : nRow (row), nCol(col), nPix(npix), nBand(band), m(compact),  \
          maxIter(mIter), perturbSeedsOrNot (perturb) {}

    uint32 nRow, nCol, nPix, nBand, maxIter;
    double m; // m is the compactness balance parameter.
    bool perturbSeedsOrNot;
};

//! ==============================================================
//! ------------- Declarations for sub Functions ---------------
//! ==============================================================
void DoSuperpix_givenNumber ( //! do superpixel given the superpixel number
                              const double*       pdat,
                              const Sizes&        sz,
                              const uint32&       K,
                              uint32*             labelForPix );

void DoSuperpix_givenStep ( //! do superpixel given the seeds' size
                            const double*       pdat,
                            const Sizes&        sz,
                            const uint32&       STEP,
                            uint32*             labelForPix );

void GetFeatureXY_forSeeds (
        const double*&      pdat,
        mxArray*&           array_kSeedFea,
        vector<double>&     kSeedr,
        vector<double>&     kSeedc,
        const uint32&       STEP,
        const Sizes&        sz );

void PerformSlicSuperpix (
        const double*       pdat,
        mxArray*&           array_kSeedFea,
        vector<double>&     kSeedr,
        vector<double>&     kSeedc,
        const Sizes&        sz,
        const uint32&       STEP,
        uint32 *labelForPix );

void EnforceLabelConnectivity (const uint32 *labels, //input labels that need to be corrected to remove stray labels
                               const int			width,
                               const int			height,
                               int*&               nlabels, //new labels
                               uint32&             numlabels, //the number of labels changes in the end if segments are removed
                               const int&			K ); //the number of superpixels desired by the user;


void GetContourForSuperpixels (
        const uint32*       pLabels,
        const Sizes&        sz,
        bool*               pContour );
//! ===================================================


/**
 * \brief In the following, the input and output parameters are reported
 * \param -input parameters: ------------------------------------------------
 *    1. pdat       2-dim matrix. E.g., Images. Each column is a sample vector.
 *                  The value should be rescaled into double format, which is in
 *                  the range of [0, 1].
 *    2. pdims      [nRow, nCol, nBand].
 *    ------------- first optional ---------------
 *    3. K          the number of the superpixels.
 *    4. m          the strength of compactness.
 *    ------------ second optional ---------------
 *    5. maxIter    the maximum number for iteration.
 *    6. edgeMap    the edge map of the given image.
 *    7. perturbSeedsOrNot      the bool paramter that determine perturb the
 *                              initial seeds or not
 * \param -output parameters: -----------------------------------------------
 *    1. labelForPix   2-dim matrix, the label for each pixel. Each label describes the belonging.
 *    2. pContours     2-dim matrix, the boundary of the superpixels
*/


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    //! check the conditions
    if ( 2 > nrhs )
        mexErrMsgTxt ("Error input: at least 2 input.\n");

    const int ndim = (int) mxGetNumberOfDimensions (prhs[0]);
    if (2 != ndim)
        mexErrMsgTxt ("The input data should be a 2-dimension matrix.\n");

    //! Get the input parameters
    double*         pdat    = mxGetPr (prhs[0]);
    const double*   pdims   = mxGetPr (prhs[1]);
    uint32          K       = mxGetScalar (prhs[2]);
    double          m       = mxGetScalar (prhs[3]);

    K = std::max (double(K), 1.0);

    if (!pdat) // if not data is read in.
        mexErrMsgTxt ("There is no image data input.\n");

    if(3 != mxGetNumberOfElements (prhs[1]))
        mexErrMsgTxt ("must input three parameters: [nRow, nCol, nBand], for the second variable.\n");

    // the sizes of the image data
    const int& nRow = pdims[0];
    const int& nCol = pdims[1];
    const int& nBand = pdims[2];
    const int  nPixel = nRow * nCol;
    const Sizes sz (nRow, nCol, nPixel, nBand, m);

    //! Create the output parameters
    plhs[0]             = mxCreateNumericMatrix (nRow, nCol, mxUINT32_CLASS, 0);
    uint32* labelForPix = (uint32*) mxGetPr (plhs[0]); // uint32 is a self defined format, i.e., unsigned int.

    plhs[1]             = mxCreateLogicalMatrix (nRow, nCol);
    bool* pContours     = (bool*) mxGetPr (plhs[1]);

    DoSuperpix_givenNumber (pdat, sz, K, labelForPix);

    GetContourForSuperpixels (labelForPix, sz, pContours);
    //    mexPrintf ( "no problem.\n");
}

//! -------------------------------------------------------
//! --- the temporary function 1 that do SLIC superpixel.--
//! -------------------------------------------------------
void DoSuperpix_givenNumber ( //! do superpixel given the superpixel number
                              const double*       pdat,
                              const Sizes&        sz,
                              const uint32&       K,
                              uint32*             labelForPix )
{

    float xySpacing = std::sqrt (float(sz.nPix)/float(K));
    const uint32 STEP = xySpacing + 0.5;
    DoSuperpix_givenStep (pdat, sz, STEP, labelForPix);
}


//! -------------------------------------------------------
//! - the most important function that do SLIC superpixel.-
//! -------------------------------------------------------
void DoSuperpix_givenStep ( //! do superpixel given the seeds' size
                            const double*       pdat,
                            const Sizes&        sz,
                            const uint32&       STEP,
                            uint32*             labelForPix )
{
    mxArray* array_kSeedFea = NULL;
    vector<double> kSeedr (0);
    vector<double> kSeedc (0);

    // this version not perturbseeds and not get the edgemag
    GetFeatureXY_forSeeds (pdat, array_kSeedFea, kSeedr, kSeedc, STEP, sz);

    // the main function to do the SLIC superpixel process.
    PerformSlicSuperpix (pdat, array_kSeedFea, kSeedr, kSeedc, sz, STEP, labelForPix);

}

//! This function is used to get the feature and spatial location for each seed.
//! Note that the seeds are equally distributed in the spatial plane.
void GetFeatureXY_forSeeds (
        const double*&      pdat,
        mxArray*&           array_kSeedFea,
        vector<double>&     kSeedr,
        vector<double>&     kSeedc,
        const uint32&       STEP,
        const Sizes&        sz )
{
    // number of point in the row dimesnioin and in the column dimension
    int rStrip = (0.5+double(sz.nRow)/double(STEP));
    int cStrip = (0.5+double(sz.nCol)/double(STEP));

    double rErr = double(sz.nRow) - double(rStrip*STEP);
    while (rErr<0)  rErr = sz.nRow - (--rStrip)*STEP;

    double cErr = double(sz.nCol) - double(cStrip*STEP);
    while (cErr<0)  cErr = sz.nCol - (--cStrip)*STEP;



    const double rErrPerStrip = rErr / rStrip;
    const double cErrPerStrip = cErr / cStrip;

    const uint32 rOff = STEP / 2;
    const uint32 cOff = STEP / 2;

    const uint32 nSeeds = rStrip * cStrip;

    //    mexPrintf ("rStrip=%d,\t cStrip=%d\n", rStrip, cStrip);

    array_kSeedFea  = mxCreateDoubleMatrix (sz.nBand, nSeeds, mxREAL);
    double* p_kSeeds =mxGetPr (array_kSeedFea);

    kSeedr.resize (nSeeds);
    kSeedc.resize (nSeeds);

    uint32 cAjusted, rAjusted, cAdd, rAdd, n(0);
    uint32 imgIdx;
    for (uint32 c=0; c != cStrip; c++) {
        cAdd = c * cErrPerStrip;
        cAjusted = c*STEP + cAdd + cOff;
        for (uint32 r=0; r != rStrip; r++) {
            rAdd = r * rErrPerStrip;
            rAjusted = r*STEP + rAdd + rOff;

            imgIdx = cAjusted * sz.nRow + rAjusted;

            //! copy features from the original data to the seed matrix, each column is a sample.
            for (uint32 ch=0; ch != sz.nBand; ch++)
                p_kSeeds[n*sz.nBand + ch] = pdat[imgIdx*sz.nBand + ch];

            // record the spatial point in the image.
            kSeedr[n] = rAjusted;
            kSeedc[n] = cAjusted;
            n++;
        }
    }
}


//! given the initial seeds feature and spatial index, this function perform the SLIC
//! superpxiel process. It returns the label matrix, whose element describes the belonging
//! of the corresponding pixel to the seeds (i.e., centers).
void PerformSlicSuperpix (
        const double*       pdat,
        mxArray*&           array_kSeedFea,
        vector<double>&     kSeedr,
        vector<double>&     kSeedc,
        const Sizes&        sz,
        const uint32&       STEP,
        uint32*             labelForPix )
{
    //! get the seeds feature vectors.
    double* p_kSeedFea = mxGetPr(array_kSeedFea);

      const uint32 Nseeds = kSeedr.size(); // the number of seeds.

    //! the local kmeans centers, which are the new seed features and spatial point (r, c).
    vector<double> tmpArrayMu (sz.nBand * Nseeds); // 2-dim matrix, each column is a sample.
    vector<double> tmpArrayRC (2 * Nseeds);

    //! the previous distance of each pixel to the centers.
    vector<double> preDistVec (sz.nPix, DBL_MAX);
    double weightFea2Spatial = (sz.m/STEP) * (sz.m/STEP); // the  (m/S)^2


    vector<uint32>  nPixForSeeds (Nseeds); // number of pixels belonged each seeds.

    uint32 rMin(0), rMax(0), cMin(0), cMax(0), correntPix(0), correntLabel(0);
    double distFea, distRC, dist, distmp;
    uint32 correntSeed = 0; double tmpNum = 0;
    uint32 tmpCount (0);

    mexPrintf ("----------maxIter=%d\n", sz.maxIter);
    for (uint32 iter=0; iter!=sz.maxIter; iter++)
    {
        preDistVec.assign(sz.nPix, DBL_MAX);
        for(uint32 n=0; n != Nseeds; n++ )
        {
            rMin = std::max (0.0,               kSeedr[n]-STEP);
            rMax = std::min (double(sz.nRow),   kSeedr[n]+STEP);

            cMin = std::max (0.0,               kSeedc[n]-STEP);
            cMax = std::min (double(sz.nCol),   kSeedc[n]+STEP);

            for (uint32 c=cMin; c!=cMax; c++)
            {
                for (uint32 r=rMin; r != rMax; r++)
                {
                    correntPix = c*(sz.nRow) + r;
                    // correntPix = c*395 + r;
                    distFea = 0;
                    for (uint32 ch=0; ch != sz.nBand; ch++)
                    {
                        distmp = p_kSeedFea[n*sz.nBand+ch] - pdat[correntPix*sz.nBand+ch];
                        distFea += distmp * distmp;
                    }

                    distRC = (r-kSeedr[n]) * (r-kSeedr[n]) + \
                            (c-kSeedc[n]) * (c-kSeedc[n]);

                    dist = distFea*3/sz.nBand + weightFea2Spatial*distRC;

                    if ( tmpCount < 10 ) {
                        mexPrintf ("dist = %f\n", dist);
                        ++tmpCount;
                    }

                    if (dist < preDistVec[correntPix]) {
                        preDistVec[correntPix] = dist;
                        labelForPix[correntPix] = n;
                    }
                }
            }
        }

        tmpArrayMu.assign (sz.nBand*Nseeds, 0);
        tmpArrayRC.assign (2*Nseeds, 0);
        nPixForSeeds.assign (Nseeds, 0);

        for (uint32 c=0; c != sz.nCol; c++) // column
        {
            for (uint32 r=0; r != sz.nRow; r++) // row
            {
                correntPix      = c*sz.nRow + r;
                correntLabel    = labelForPix[correntPix];

                for (uint32 ch=0; ch != sz.nBand; ch++)
                    tmpArrayMu[correntLabel*sz.nBand + ch] += pdat[correntPix*sz.nBand + ch];

                tmpArrayRC[correntLabel*2 + 0] += r; // first "row"
                tmpArrayRC[correntLabel*2 + 1] += c; // second "column"

                nPixForSeeds[correntLabel]++;
            }
        }

        for (uint32 n=0; n != Nseeds; n++)
        {
            tmpNum = std::max (uint32(1), nPixForSeeds[n]);
            for (uint32 ch=0; ch != sz.nBand; ch++)
            {
                correntSeed = n*sz.nBand + ch;
                p_kSeedFea[correntSeed] = tmpArrayMu[correntSeed] / tmpNum;
            }

            kSeedr[n] = tmpArrayRC[n*2 + 0] / tmpNum; // first "row"
            kSeedc[n] = tmpArrayRC[n*2 + 1] / tmpNum; // second "column"
        }
    } // end of iteration
}

//! -------------------------------------------------------
//! ------ enforce the connectivity of superpixels -------
//! 这个程序非常巧妙, 值得多研究几次.
//! -------------------------------------------------------
void EnforceLabelConnectivity (
        const uint32*       labels, //input labels that need to be corrected to remove stray labels
        const int			width,
        const int			height,
        int*&               nlabels, //new labels
        uint32&             numlabels,//the number of labels changes in the end if segments are removed
        const int&			K ) //the number of superpixels desired by the user
{

    const int dx4[4] = {-1,  0,  1,  0};
    const int dy4[4] = { 0, -1,  0,  1};

    const int sz = width*height;
    const int SUPSZ = sz/K;
    //nlabels.resize(sz, -1);
    for(int i = 0; i < sz; i++ ) nlabels[i] = -1;
    int label(0);
    int* xvec = new int[sz];
    int* yvec = new int[sz];
    int oindex(0);
    int adjlabel(0);//adjacent label
    for( int j = 0; j < height; j++ )
    {
        for( int k = 0; k < width; k++ )
        {
            if( 0 > nlabels[oindex] )
            {
                nlabels[oindex] = label;

                // Start a new segment
                xvec[0] = k;
                yvec[0] = j;
                //! Quickly find an adjacent label to use later if needed
                //! 找到相superpixel的标签 (左边的或者上边, 尤其上边的可能性更大).
                {
                    for( int n = 0; n < 4; n++ )
                    {
                        int x = xvec[0] + dx4[n];
                        int y = yvec[0] + dy4[n];
                        if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                        {
                            int nindex = y*width + x;
                            if(nlabels[nindex] >= 0)
                                adjlabel = nlabels[nindex]; //break;
                        }
                    }
                }

                //! Here are some Chinese Comments which display abnormally 
                //! 这是本程序的核心部分, 避免了递归运算的繁琐, 而是采用了不断 for 结束条件也在增长的方式达
                //! 到递归地效果. 对于每个像素, 一旦条件成立, 这种四邻域判断就一直会向左下角方向延伸.
                //! 同时计数, 这个计数便于将来判断保留该superpixel与否.
                int count (1);
                for( int c = 0; c < count; c++ )
                {
                    for( int n = 0; n < 4; n++ )
                    {
                        int x = xvec[c] + dx4[n];
                        int y = yvec[c] + dy4[n];

                        if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                        {
                            int nindex = y*width + x;

                            if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
                            {
                                xvec[count] = x;
                                yvec[count] = y;
                                nlabels[nindex] = label;
                                count++;
                            }
                        }

                    }
                }

                //-------------------------------------------------------
                // If segment size is less then a limit, assign an
                // adjacent label found before, and decrement label count.
                //-------------------------------------------------------
                if(count <= 8)
                {
                    for( int c = 0; c < count; c++ )
                    {
                        int ind = yvec[c]*width+xvec[c];
                        nlabels[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }
    numlabels = label;

    if(xvec) delete [] xvec;
    if(yvec) delete [] yvec;
}

//! -------------------------------------------------------
//! ------------ get contours for superpixels ------------
//! -------------------------------------------------------
void GetContourForSuperpixels (
        const uint32*       pLabels,
        const Sizes&        sz,
        bool*               pContour )
{
    const int dr8[8] = {-1, -1,  0,  1, 1, 1, 0, -1}; // row
    const int dc8[8] = { 0, -1, -1, -1, 0, 1, 1,  1}; // columnLL

    uint32 centerPix (0), neiborPix (0), count(0);
    for (uint32 c=0; c != sz.nCol; c++)
    {
        for (uint32 r=0; r != sz.nRow; r++)
        {
            count = 0;
            for (int ch=0; ch !=8; ch++)
            {
                int row = r + dr8[ch];
                int col = c + dc8[ch];
                if ( (row >= 0 && row < sz.nRow) && (col>=0 && col < sz.nCol) )
                {
                    neiborPix = row + col*sz.nRow;
                    if (pLabels[centerPix] != pLabels[neiborPix]) count++;
                }
            }

            if ( count >1 )
                pContour[centerPix] = true;
            centerPix++;
        }
    }
}
