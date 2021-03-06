#ifndef _siddon_h
#define _siddon_h

#ifdef WIN32
#define DLL __declspec(dllexport)
#else
#define DLL
#endif

/* @brief Calculates the indices of the pixels and rays which intersect along
          along with their lengths of intersection.

@param ozmin The minimum coordinate of grid in the z dimension.
@param zsize The size of the grid in the z dimension.
@param oz The number of pixels along the grid in the z dimension.
@param theta[i] The angle of the ith ray.
@param h[i] The horizontal position of the ith ray.
@param v[i] The vertical position of the ith ray.
@param dsize The size of theta, h, v. aka the number of rays.
@param gridx The locations of grid lines along x dimension.
@return pixels The linear index in the grid of pixels that intersect rays.
@return rays The linear index of the rays which intersect the pixels.
@return lengths The intersections lengths at each pixel.
@return psize The size of pixels and lengths.
*/
void DLL
get_pixel_indexes_and_lengths(
    const float ozmin, const float oxmin, const float oymin,
    const float zsize, const float xsize, const float ysize,
    const int oz, const int ox, const int oy,
    const float * const theta, const float * const h, const float * const v,
    const int dsize,
    const float *gridx, const float *gridy,
    int **pixels, int **rays, float **lengths, int *psize);

/* @brief Returns 1 for first and third quadrants, 0 otherwise.
*/
int
calc_quadrant(
    const float theta_p);

/* @brief Computes the list of intersections of the line (xi, yi) and the grid.

The intersections are then located in two lists: (gridx, coordy) and
(coordx, gridy). The length of gridx is ngridx+1.
*/
void DLL
calc_coords(
    const int ngridx, const int ngridy,
    const float xi, const float yi,
    const float sin_p, const float cos_p,
    const float *gridx, const float *gridy,
    float * const coordx, float * const coordy);

/* @brief Remove points from these sets that lay outside the boundaries of the
          grid.

(coordx, gridy) and (gridx, coordy) are sets of points along a line.
*/
void DLL
trim_coords(
    const int ox, const int oy,
    const float *coordx, const float *coordy,
    const float *gridx, const float *gridy,
    int *asize, float *ax, float *ay,
    int *bsize, float *bx, float *by);

/* @brief Combine the two sets of points into (coorx, coory).

(ax, ay) and (bx, by) are two sets of ordered points along a line. The total
number of points is asize + bsize = csize.
*/
void DLL
sort_intersections(
    const int ind_condition,
    const int asize, const float *ax, const float *ay,
    const int bsize, const float *bx, const float *by,
    int *csize, float *coorx, float *coory);

/* @brief Finds the distances between adjacent points and return the midpoints
          of these line segments.

(coorx, coory) describe the ordered points where the line intersects the
grid.
*/
void DLL
calc_dist(
    int const csize, const float *coorx, const float *coory,
    float *midx, float *midy, float *dist);

/* @brief Finds the linear index of the pixels containing the points
          (midx, midy) on the grid.
*/
void DLL
calc_index(
    int const ox, int const oy, int const oz,
    float const oxmin, float const oymin, float const ozmin,
    float const xstep, float const, float const,
    int const msize, const float *midx, const float *midy,
    int const indz, int *indi);
#endif
