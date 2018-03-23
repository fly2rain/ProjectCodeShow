/* Lear's GIST implementation, version 1.1, (c) INRIA 2009, Licence: PSFL */


#ifndef GIST_H_INCLUDED
#define GIST_H_INCLUDED


//#include "gist.h"
#include "standalone_image.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

image_list_t *create_gabor(int nscales, const int *orr, int width, int height);



/*! Graylevel GIST for various scales. Based on Torralba's Matlab
 * implementation. http://people.csail.mit.edu/torralba/code/spatialenvelope/
 *
 * Descriptor size is  w*w*sum(n_orientations[i],i=0..n_scale-1)
 *
 *    @param src Source image
 *    @param w Number of bins in x and y axis
 *	   w : number of blocks
*     fc : 4 (default)
*     arientationsPerScale=[4 4 4 4]
 *    
 */

float *bw_gist_scaletab(image_t *src, int nblocks, int n_scale, const int *n_orientations);

/*! @brief implementation of grayscale GIST descriptor.
 * Descriptor size is w*w*(a+b+c)
 *
 *    @param src Source image
 *    @param w Number of bins in x and y axis
 */
float *bw_gist(image_t *scr, int nblocks, int a, int b, int c);

/*! @brief implementation of color GIST descriptor.
 *
 *    @param src Source image
 *    @param w Number of bins in x and y axis
 */
float *color_gist(color_image_t *src, int nblocks, int a, int b, int c);

/*! Color GIST for various scales. Based on Torralba's Matlab
 * implementation. http://people.csail.mit.edu/torralba/code/spatialenvelope/  */

float *color_gist_scaletab(color_image_t *src, int nblocks, int n_scale, const int *n_orientations);
float *color_gist_scaletab_gabor(color_image_t *src, int nblocks, int n_scale, const int *n_orientations, image_list_t *G);
//src :CV_32FC3
//int nblocks=4;
//int n_scale=3;
//int orientations_per_scale[3]={8,8,4};
//int descsize=0;
float *cvGist(cv::Mat &src, int nblocks, int n_scale, int *orientations_per_scale, int &descsize);

#endif
