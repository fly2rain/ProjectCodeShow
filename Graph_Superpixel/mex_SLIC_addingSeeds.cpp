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
 *  \date:    2014 - 05 - 17
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
#include <algorithm>
typedef float  inputType;
#include "QuickSet_KthElement.h"
using namespace  std;
typedef unsigned int uint32; // 32 bit == 8 byte

struct Sizes {
    Sizes (const uint32& row, const uint32& col, const uint32& npix, \
           const uint32 &band, const inputType& compact=20, const uint32 newSeeds=50, const uint32& mIter=10, \
           const bool& perturb=false) \
        : nRow (row), nCol(col), nPix(npix), nBand(band), m(compact), NNEWSEEDS(newSeeds),  \
          maxIter(mIter), perturbSeedsOrNot (perturb) {}

    uint32 nRow, nCol, nPix, nBand, maxIter, NNEWSEEDS;
    inputType m; // m is the compactness balance parameter.
    bool perturbSeedsOrNot;
};

inline void ind2sub ( const uint32 nRow, const uint32& idx, uint32& rstRow, uint32& rstCol )
{
    rstRow = idx % nRow;
    rstCol = idx / nRow;
}

//! ==============================================================
//! ------------- Declarations for sub Functions ---------------
//! ==============================================================
//! --- the temporary function 1 that do SLIC superpixel.--
void DoSuperpix_givenNumber ( //! do superpixel given the superpixel number
                              const inputType*    pDat,
                              const Sizes&        sz,
                              const uint32&       K,
                              uint32*             pLbl,
                              vector<uint32>&     vecAddPos );

//! This function is used to get the feature and spatial location for each seed.
//! Note that the seeds are equally distributed in the spatial plane.
void GetFeatureXY_forSeeds (
        const inputType*&   pDat,
        vector<inputType>&  kSeedFea,
        vector<inputType>&  kSeedRow,
        vector<inputType>&  kSeedCol,
        const uint32&       STEP,
        const Sizes&        sz );

//! given the initial seeds feature and spatial index, this function perform the SLIC
//! superpxiel process. It returns the label matrix, whose element describes the belonging
//! of the corresponding pixel to the seeds (i.e., centers).
void PerformSlicSuperpix (
        const inputType*    pDat,
        vector<inputType>&  kSeedFea,
        vector<inputType>&  kSeedRow,
        vector<inputType>&  kSeedCol,
        const Sizes&        sz,
        const uint32&       STEP,
        uint32*             pLbl,
        vector<uint32>&     vecAddPos );
//! ===================================================

/**
 * \brief In the following, the input and output parameters are reported
 * \param -input parameters: ------------------------------------------------
 *    1. pDat       2-dim matrix. E.g., Images. Each column is a sample vector.
 *                  The value should be rescaled into inputType format, which is in
 *                  the range of [0, 1].
 *    2. pDims      [nRow, nCol, nBand].
 *    ------------- first optional ---------------
 *    3. K          the number of the superpixels.
 *    4. m          the strength of compactness.
 *    ------------ second optional ---------------
 *    5. maxIter    the maximum number for iteration.
 * \param -output parameters: -----------------------------------------------
 *    1. pLbl       2-dim matrix, the label for each pixel. Each label describes the belonging.
*/
#define     IN_DATA         prhs[0]
#define     IN_DIMS         prhs[1]
#define     IN_N_SUPIX      prhs[2]
#define     IN_COMPACT_FACT prhs[3]
#define     IN_N_NEWSEEDS   prhs[4]
#define     OUT_LABEL       plhs[0]
#define     OUT_NEWSEEDS    plhs[1]
using std::vector;

//void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    //! check the conditions
    if ( 2 > nrhs )
        mexErrMsgTxt ("Error input: at least 2 input.\n");

    const int ndim = (int) mxGetNumberOfDimensions (prhs[0]);
    if (2 != ndim)
        mexErrMsgTxt ("The input data should be a 2-dimension matrix.\n");

    //! Get the input parameters
    inputType*     pDat    = (inputType*) mxGetPr (IN_DATA);
    const double*  pDims   = mxGetPr (IN_DIMS);
    uint32         K       = mxGetScalar (IN_N_SUPIX);
    inputType      m       = mxGetScalar (IN_COMPACT_FACT);
    uint32         NNEWSEEDS = mxGetScalar (IN_N_NEWSEEDS);

    K = std::max (inputType(K), inputType(1.0));

    if ( !pDat ) // if not data is read in.
        mexErrMsgTxt ("There is no image data input.\n");

    if( 3 != mxGetNumberOfElements (IN_DIMS) )
        mexErrMsgTxt ("must input three parameters: [nRow, nCol, nBand], for the second variable.\n");

    // the sizes of the image data
    const uint32& nRow  = pDims[0];
    const uint32& nCol  = pDims[1];
    const uint32& nBand = pDims[2];
    const uint32  nPixel = nRow * nCol;
    const Sizes   sz (nRow, nCol, nPixel, nBand, m, NNEWSEEDS);


    //! create the ouptut matrixes
    OUT_LABEL      = mxCreateNumericMatrix (nRow, nCol, mxUINT32_CLASS, mxREAL);
    uint32* pLbl   = (uint32*) mxGetPr (OUT_LABEL); // uint32 is a self defined format, i.e., unsigned int.

    //! do the superpixel processures
    vector<uint32> vecAddPos;
    DoSuperpix_givenNumber (pDat, sz, K, pLbl, vecAddPos);

    if ( vecAddPos.size () % 2 )
        mexWarnMsgTxt ( "There is something wrong about how to add seeds.\n" );

    const uint32 numNewAddSeeds = vecAddPos.size () / 2;
    OUT_NEWSEEDS  = mxCreateNumericMatrix (2, numNewAddSeeds, mxUINT32_CLASS, mxREAL);
    uint32* pnewSeed = (uint32*) mxGetPr ( OUT_NEWSEEDS );
    for ( uint32 n=0; n != vecAddPos.size (); ++n )
        pnewSeed[n] = vecAddPos[n];
}

//! -------------------------------------------------------
//! --- the temporary function 1 that do SLIC superpixel.--
//! -------------------------------------------------------
void DoSuperpix_givenNumber ( //! do superpixel given the superpixel number
                              const inputType*    pDat,
                              const Sizes&        sz,
                              const uint32&       K,
                              uint32*             pLbl,
                              vector<uint32>&     vecAddPos )
{
    inputType xySpacing = std::sqrt (inputType(sz.nPix)/inputType(K));
    const uint32 STEP   = xySpacing + 0.5;

    std::vector<inputType> kSeedFea;
    std::vector<inputType> kSeedr;
    std::vector<inputType> kSeedc;

    // this version not perturbseeds and not get the edgemag
    GetFeatureXY_forSeeds (pDat, kSeedFea, kSeedr, kSeedc, STEP, sz);

    // the main function to do the SLIC superpixel process.
    PerformSlicSuperpix (pDat, kSeedFea, kSeedr, kSeedc, sz, STEP, pLbl, vecAddPos);
}

//! This function is used to get the feature and spatial location for each seed.
//! Note that the seeds are equally distributed in the spatial plane.
void GetFeatureXY_forSeeds (
        const inputType*&  pDat,
        vector<inputType>& kSeedFea,
        vector<inputType>& kSeedRow,
        vector<inputType>& kSeedCol,
        const uint32&      STEP,
        const Sizes&       sz )
{
    // number of point in the row dimesnioin and in the column dimension
    int rStrip = (0.5+inputType(sz.nRow)/inputType(STEP));
    int cStrip = (0.5+inputType(sz.nCol)/inputType(STEP));

    inputType rErr = inputType(sz.nRow) - inputType(rStrip*STEP);
    while (rErr<0)  rErr = sz.nRow - (--rStrip)*STEP;

    inputType cErr = inputType(sz.nCol) - inputType(cStrip*STEP);
    while (cErr<0)  cErr = sz.nCol - (--cStrip)*STEP;

    const inputType rErrPerStrip = rErr / rStrip;
    const inputType cErrPerStrip = cErr / cStrip;

    const uint32 rOff = STEP / 2;
    const uint32 cOff = STEP / 2;

    uint32 cAjusted, rAjusted, cAdd, rAdd, n(0);
    uint32 imgIdx;
    for (uint32 c=0; c != cStrip; ++c) {
        cAdd = c * cErrPerStrip;
        cAjusted = c*STEP + cAdd + cOff;
        for (uint32 r=0; r != rStrip; ++r) {
            rAdd = r * rErrPerStrip;
            rAjusted = r*STEP + rAdd + rOff;

            imgIdx = cAjusted * sz.nRow + rAjusted;

            //! copy features from the original data to the seed matrix, each column is a sample.
            for (uint32 ch=0; ch != sz.nBand; ++ch)
                kSeedFea.push_back ( pDat[imgIdx*sz.nBand + ch] );

            // record the spatial point in the image.
            kSeedRow.push_back (rAjusted);
            kSeedCol.push_back (cAjusted);
            ++n;
        }
    }
}

//! regarding the previous feature distance, adding "NNEWSEEDS" new seeds
void AddNewSeeds (
        const inputType*    pDat,
        vector<inputType>&  kSeedFea,
        vector<inputType>&  kSeedRow,
        vector<inputType>&  kSeedCol,
        vector<inputType>&  preFeaDists,
        const Sizes&        sz,
        uint32*             pLbl,
        vector<uint32>&     vecAddPos )
{
    uint32  NNEWSEEDS = sz.NNEWSEEDS;
    const uint32  CAND_NNEWSEEDS = std::min ( 50*NNEWSEEDS, sz.nPix/2 );

    // get the set of all K biggest idxs
    vector<uint32>     candSeedLbl; // the label of candidate added pixels as new seeds.
    vector<inputType>  candFeaDist; // the idx of the candidate added pixels as new seeds
    vector<uint32>     candRelIdx;  // the candidate relative idx.

    // get the K-th biggest value
    inputType  tmpDist = Get_KthElement_copy ( preFeaDists, CAND_NNEWSEEDS );
    for ( uint32 n=0; n != preFeaDists.size (); ++n)
    {
        if ( preFeaDists[n] >= tmpDist )
        {
            candSeedLbl.push_back ( pLbl [n] );
            candFeaDist.push_back ( preFeaDists [n] );
            candRelIdx.push_back ( n );
        }
    }

    // find the unique label set of the candidate label set in candSeedLbl
    vector<uint32> uniqueLbl ( candSeedLbl );
    vector<uint32>::iterator it = std::unique ( uniqueLbl.begin(), uniqueLbl.end () );
    uniqueLbl.resize ( std::distance (uniqueLbl.begin (), it) );

    std::sort ( uniqueLbl.begin (), uniqueLbl.end () );
    it = std::unique ( uniqueLbl.begin(), uniqueLbl.end () );
    uniqueLbl.resize ( std::distance (uniqueLbl.begin (), it) );

    // calculate the distance of each unique label ----------------------
    vector<inputType>  uniqueDist ( uniqueLbl.size (), 0 );
    vector<uint32>     uniqueIdx ( uniqueLbl.size (), 0 );
    for ( uint32 n=0; n != uniqueLbl.size (); ++n )
    {   tmpDist = 0;
        for ( uint32 ch=0; ch != candSeedLbl.size (); ++ch )
        {
            if ( uniqueLbl[n] == candSeedLbl[ch] )
            {
                uniqueDist[n] += candFeaDist[ch];
                if ( tmpDist < candFeaDist[ch] )
                {
                    tmpDist = candFeaDist[ch];
                    uniqueIdx[n] = candRelIdx[ch];
                }
            }
        }
    }

    if ( NNEWSEEDS > uniqueLbl.size () )
        tmpDist = 0;
    else
        tmpDist = Get_KthElement_copy ( uniqueDist, NNEWSEEDS );

    //    mexPrintf ("NNEWSEEDS=%d \t uniSize=%d \t tmpDist=%f\n", NNEWSEEDS,\
    //               uniqueLbl.size (), tmpDist);

    bool isInImageOrNot ( false );
    uint32  seedLbl (kSeedRow.size ()), r, c, eleIdx;
    for ( uint32 n=0; n != uniqueLbl.size (); ++n )
    {
        if ( uniqueDist[n] >= tmpDist )
        {
            uint32 relIdx = uniqueIdx [n];
            ind2sub ( sz.nRow, relIdx, r, c );
            isInImageOrNot = ( r >= 0 && r < sz.nRow) && (c >= 0 && c < sz.nCol);
            if ( !isInImageOrNot ) { // if not in image, there is wrong with ind2sub
                mexErrMsgTxt ("Out of the image: something wrong with ind2sub.\n");
                return; }

            kSeedRow.push_back ( r );
            kSeedCol.push_back ( c );

            vecAddPos.push_back (r);
            vecAddPos.push_back (c);

            eleIdx = relIdx * sz.nBand;
            for (uint32 ch=0; ch != sz.nBand; ++ch)
                kSeedFea.push_back ( pDat[eleIdx + ch] );

            pLbl [relIdx] = seedLbl;
            ++ seedLbl;
        }
    }
}

//! given the initial seeds feature and spatial index, this function perform the SLIC
//! superpxiel process. It returns the label matrix, whose element describes the belonging
//! of the corresponding pixel to the seeds (i.e., centers).
void PerformSlicSuperpix (
        const inputType*    pDat,
        vector<inputType>&  kSeedFea,
        vector<inputType>&  kSeedRow,
        vector<inputType>&  kSeedCol,
        const Sizes&        sz,
        const uint32&       STEP,
        uint32*             pLbl,
        vector<uint32>&     vecAddPos )
{
    uint32 NSEEDS = kSeedRow.size(); // the number of seeds.
    const inputType stepTimes = 1;

    //! the local kmeans centers, which are the new seed features and spatial point (r, c).
    vector<inputType> tmpArrayMu (sz.nBand * NSEEDS); // 2-dim matrix, each column is a mean sample.
    vector<inputType> tmpArrayRC (2 * NSEEDS);

    //! the previous distance of each pixel to the centers.
    vector<inputType> preDists (sz.nPix, inputType(DBL_MAX));
    vector<inputType> preFeaDists (sz.nPix, inputType(0));

    //    const uint32       NNEWSEEDS = sz.NNEWSEEDS; //! the number of new seed

    inputType weightFea2Spatial = (sz.m/STEP);
    weightFea2Spatial *= weightFea2Spatial; // the  (m/S)^2


    vector<uint32>  NPixInSeeds; // number of pixels belonged each seeds.

    uint32 rMin(0), rMax(0), cMin(0), cMax(0), opix(0), oLabel(0);
    inputType distFea, distRC, dist, distmp;
    uint32 correntSeed = 0; inputType tmpNum = 0;

    for (uint32 iter=0; iter!=sz.maxIter; ++iter)
    {
        for(uint32 n=0; n != NSEEDS; ++n )
        {
            rMin = std::max (inputType(0.0),       kSeedRow[n]-stepTimes*STEP);
            rMax = std::min (inputType(sz.nRow),   kSeedRow[n]+stepTimes*STEP);

            cMin = std::max (inputType(0.0),       kSeedCol[n]-stepTimes*STEP);
            cMax = std::min (inputType(sz.nCol),   kSeedCol[n]+stepTimes*STEP);

            for (uint32 c=cMin; c!=cMax; ++c)
            {
                for (uint32 r=rMin; r != rMax; ++r)
                {
                    opix = c*(sz.nRow) + r;
                    // opix = c*395 + r;
                    distFea = 0;
                    for (uint32 ch=0; ch != sz.nBand; ++ch)
                    {
                        distmp = kSeedFea[n*sz.nBand+ch] - pDat[opix*sz.nBand+ch];
                        distFea += distmp * distmp;
                    }

                    distRC = (r-kSeedRow[n]) * (r-kSeedRow[n]) + \
                            (c-kSeedCol[n]) * (c-kSeedCol[n]);

//                    dist = distFea*3/sz.nBand + weightFea2Spatial*distRC;
                    dist = distFea + weightFea2Spatial*distRC;

                    if (dist < preDists[opix]) {
                        preFeaDists[opix]   = distFea;
                        preDists[opix]      = dist;
                        pLbl[opix]          = n;
                    }
                }
            }
        }

        //        mexPrintf ( "NNEWSEEDS=%d\n", sz.NNEWSEEDS );
        if ( sz.NNEWSEEDS > 0 ) {
            //! add 20 new seed per iter, after the 3-rd iteration.
            if ( 2 < iter  &&  iter%2 == 1 && iter < sz.maxIter - 2 )
            {
                AddNewSeeds ( pDat, kSeedFea, kSeedRow, kSeedCol, \
                              preFeaDists, sz, pLbl, vecAddPos );
//                mexPrintf ( "iter=%d\n", iter );
            }
        }

        //        mexPrintf ("[%d]\tNSeeds=%d\n", iter, kSeedRow.size () );
        NSEEDS = kSeedRow.size ();
        tmpArrayMu.assign (sz.nBand*NSEEDS, 0);
        tmpArrayRC.assign (2*NSEEDS, 0);
        NPixInSeeds.assign (NSEEDS, 0);

        preDists.assign (sz.nPix, inputType(DBL_MAX));
        preFeaDists.assign (sz.nPix, inputType(0));

        for (uint32 c=0; c != sz.nCol; ++c) // column
        {
            for (uint32 r=0; r != sz.nRow; ++r) // row
            {
                opix      = c*sz.nRow + r;
                oLabel    = pLbl[opix];

                for (uint32 ch=0; ch != sz.nBand; ++ch)
                    tmpArrayMu[oLabel*sz.nBand + ch] += pDat[opix*sz.nBand + ch];

                tmpArrayRC[oLabel*2 + 0] += r; // first "row"
                tmpArrayRC[oLabel*2 + 1] += c; // second "column"

                ++(NPixInSeeds[oLabel]);
            }
        }

        for (uint32 n=0; n != NSEEDS; ++n)
        {
            tmpNum = std::max (uint32(1), NPixInSeeds[n]);
            for (uint32 ch=0; ch != sz.nBand; ++ch)
            {
                correntSeed = n*sz.nBand + ch;
                kSeedFea[correntSeed] = tmpArrayMu[correntSeed] / tmpNum;
            }

            kSeedRow[n] = tmpArrayRC[n*2 + 0] / tmpNum; // first "row"
            kSeedCol[n] = tmpArrayRC[n*2 + 1] / tmpNum; // second "column"
        }
    } // end of iteration
}
