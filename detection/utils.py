import cv2
import math
import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
try:
    import uncertainties.unumpy as unp 
    nom = unp.nominal_values
except ImportError:
    nom = lambda x: x
from scipy.interpolate import UnivariateSpline, RectBivariateSpline


#This class provides a way to easily access the basic paramenters of the video
#Also it adds a utility function to display progress bar on the video when
#displayed with OpenCV imshow function
class utilsVideo:
    #Ctor that takes following arguments
    #capture - video capture class object from OpenCV
    def __init__(self, capture):
        self.capture = capture
        
    # function used access and return basic details of the video
    def getStats(self):
        #get frames per seconds (fps) from the video
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        #total frame counts
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        #codec used to capture the video. This is useful when you are saving the
        #video to disc, currently not used anywhere
        #full parameter list at
        #https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
        codec = self.capture.get(cv2.CAP_PROP_FOURCC)
        #total duraiton of video in milli seconds
        durationSec = frame_count/fps * 1000
        return (fps,frame_count,durationSec)
    
    
    #This function implements display of progress bar on video frame.
    #Parametes:
    #currentframe: the current frame of video to be saved. This frame
    #        will be modified by adding progress bar on top of it and
    #        returned to the caller
    
    def displayProgressBar(self, currentframe):
        #get next frame number out of all the frames for video
        nextFrameNo = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        #get total number of frames in the video
        totalFrames =self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        #calculate the percent complete based on the frame currently
        #playing. OpenCV does provide a variable to access this
        #property directly (CAP_PROP_POS_AVI_RATIO), however
        #it seems to not work all the time, hence we calculate internally
        complete = nextFrameNo/totalFrames
        
        #progress bar thickness
        lineThickness = 2
        #progress bar will be displayed 4% from the bottom of the frame
        y = math.ceil(currentframe.shape[1] - currentframe.shape[1]/2)
        #display progress bar across the width of the video
        x = 0
        w = currentframe.shape[0]
        #white line as background for progressbar
        cv2.line(currentframe, (x, y), (w, y), (255,255,255), lineThickness)
        #red line as progress on top of that
        cv2.line(currentframe, (x, y), (math.ceil(w*complete), y), (0,0,255), lineThickness)
        return currentframe


"""
Rebin 1D histograms.
"""


class BoundedUnivariateSpline(UnivariateSpline):
    """
    1D spline that returns a constant for x outside the specified domain.
    """
    def __init__(self, x, y, fill_value=0.0, **kwargs):
        self.bnds = [x[0], x[-1]]
        self.fill_value = fill_value
        UnivariateSpline.__init__(self, x, y, **kwargs)

    def is_outside_domain(self, x):
        x = np.asarray(x)
        return np.logical_or(x<self.bnds[0], x>self.bnds[1])

    def __call__(self, x):
        outside = self.is_outside_domain(x)

        return np.where(outside, self.fill_value, 
                                 UnivariateSpline.__call__(self, x))
        
    def integral(self, a, b):
        # capturing contributions outside domain of interpolation
        below_dx = np.max([0., self.bnds[0]-a])
        above_dx = np.max([0., b-self.bnds[1]])

        outside_contribution = (below_dx + above_dx) * self.fill_value

        # adjusting interval to spline domain
        a_f = np.max([a, self.bnds[0]])
        b_f = np.min([b, self.bnds[1]])

        if a_f >= b_f:
            return outside_contribution
        else:
            return (outside_contribution +
                      UnivariateSpline.integral(self, a_f, b_f) )


class BoundedRectBivariateSpline(RectBivariateSpline):
    """
    2D spline that returns a constant for x outside the specified domain.
    Input
    -----
      x : array_like, length m+1, bin edges in x direction
      y : array_like, length n+1, bin edges in y direction
      z : array_like, m by n, values of function to fit spline
    """
    def __init__(self, x, y, z, fill_value=0.0, **kwargs):
        self.xbnds = [x[0], x[-1]]
        self.ybnds = [y[0], y[-1]]
        self.fill_value = fill_value
        RectBivariateSpline.__init__(self, x, y, z, **kwargs)

    def is_outside_domain(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.logical_or( np.logical_or(x<self.xbnds[0], x>self.xbnds[1]),
                              np.logical_or(y<self.ybnds[0], y>self.xbnds[1]) )

    def __call__(self, x, y):
        outside = self.is_outside_domain(x, y)

        return np.where(outside, self.fill_value, 
                                 RectBivariateSpline.__call__(self, x, y))
        
    def integral(self, xa, xb, ya, yb):
        assert xa <= xb
        assert ya <= yb

        total_area = (xb - xa) * (yb - ya)

        # adjusting interval to spline domain
        xa_f = np.max([xa, self.xbnds[0]])
        xb_f = np.min([xb, self.xbnds[1]])
        ya_f = np.max([ya, self.ybnds[0]])
        yb_f = np.min([yb, self.ybnds[1]])

        # Rectangle does not overlap with spline domain
        if xa_f >= xb_f or ya_f >= yb_f:
            return total_area * self.fill_value


        # Rectangle overlaps with spline domain
        else:
            spline_area = (xb_f - xa_f) * (yb_f - ya_f)
            outside_contribution = (total_area - spline_area) * self.fill_value
            return (outside_contribution +
                      RectBivariateSpline.integral(self, xa_f, xb_f, ya_f, yb_f) )


def midpoints(xx):
    """Return midpoints of edges in xx."""
    return xx[:-1] + 0.5*np.ediff1d(xx)


def edge_step(x, y, **kwargs):
    """
    Plot a histogram with edges and bin values precomputed. The normal
    matplotlib hist function computes the bin values internally.
    Input
    -----
     * x : n+1 array of bin edges.
     * y : n array of histogram values.
    """
    return plt.plot(x, np.hstack([y, y[-1]]), drawstyle='steps-post', **kwargs)


def rebin_along_axis(y1, x1, x2, axis=0, interp_kind=3):
    """
    Rebins an N-dimensional array along a given axis, in a piecewise-constant
    fashion.
    Parameters
    ----------
    y1 : array_like
        The input image
    x1 : array_like
        The monotonically increasing/decreasing original bin edges along
        `axis`, must be 1 greater than `np.size(y1, axis)`.
    y2 : array_like
        The final bin_edges along `axis`.
    axis : int
        The axis to be rebinned, it must exist in the original image.
    interp_kind : how is the underlying unknown continuous distribution
                  assumed to look: {3, 'piecewise_constant'}
                  3 is cubic splines
                  piecewise_constant is constant in each histogram bin
    Returns
    -------
    output : np.ndarray
        The rebinned image.
    """

    orig_shape = np.array(y1.shape)
    num_axes = np.size(orig_shape)

    # Output is going to need reshaping
    new_shape = np.copy(orig_shape)
    new_shape[axis] = np.size(x2) - 1

    if axis > num_axes - 1:
        raise ValueError("That axis is not in y1")

    if np.size(y1, axis) != np.size(x1) - 1:
        raise ValueError("The original number of xbins does not match the axis"
                         "size")

    odtype = np.dtype('float')
    if y1.dtype is np.dtype('O'):
        odtype = np.dtype('O')

    output = np.empty(new_shape, dtype=odtype)

    it = np.nditer(y1, flags=['multi_index', 'refs_ok'])
    it.remove_axis(axis)

    while not it.finished:
        a = list(it.multi_index)
        a.insert(axis, slice(None))

        rebinned = rebin(x1, y1[a], x2, interp_kind=interp_kind)

        output[a] = rebinned[:]
        it.iternext()

    return output


def rebin(x1, y1, x2, interp_kind=3):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.
    Input
    -----
     * x1 : m+1 array of old bin edges.
     * y1 : m array of old histogram values. This is the total number in 
              each bin, not an average.
     * x2 : n+1 array of new bin edges.
     * interp_kind : how is the underlying unknown continuous distribution
                      assumed to look: {3, 'piecewise_constant'}
                      3 is cubic splines
                      piecewise_constant is constant in each histogram bin
    Returns
    -------
     * y2 : n array of rebinned histogram values.
    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """

    if interp_kind == 'piecewise_constant':
        return rebin_piecewise_constant(x1, y1, x2)
    else:
        return rebin_spline(x1, y1, x2, interp_kind=interp_kind)
     

def rebin_spline(x1, y1, x2, interp_kind):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.
    Input
    -----
     * x1 : m+1 array of old bin edges.
     * y1 : m array of old histogram values. This is the total number in 
              each bin, not an average.
     * x2 : n+1 array of new bin edges.
     * interp_kind : how is the underlying unknown continuous distribution
                      assumed to look: {'cubic'}
    Returns
    -------
     * y2 : n array of rebinned histogram values.
    The cubic spline fit (which is the only interp_kind tested) 
    uses the UnivariateSpline class from Scipy, which uses FITPACK.
    The boundary condition used is not-a-knot, where the second and 
    second-to-last nodes are not included as knots (but they are still
    interpolated).
    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """
    m = y1.size
    n = x2.size - 1

    # midpoints of x1
    x1_mid = midpoints(x1)

    # constructing data for spline
    #  To get the spline to flatten out at the edges, duplicate bin mid values
    #   as value on the two boundaries.
    xx = np.hstack([x1[0], x1_mid, x1[-1]])
    yy = np.hstack([y1[0], y1, y1[-1]])

    # strip uncertainties from data
    yy = nom(yy)

    # instantiate spline, s=0 gives interpolating spline
    spline = BoundedUnivariateSpline(xx, yy, s=0., k=interp_kind)

    # area under spline for each old bin
    areas1 = np.array([spline.integral(x1[i], x1[i+1]) for i in range(m)])


    # insert old bin edges into new edges
    x1_in_x2 = x1[ np.logical_and(x1 > x2[0], x1 < x2[-1]) ]
    indices  = np.searchsorted(x2, x1_in_x2)
    subbin_edges = np.insert(x2, indices, x1_in_x2)

    # integrate over each subbin
    subbin_areas = np.array([spline.integral(subbin_edges[i], 
                                             subbin_edges[i+1]) 
                              for i in range(subbin_edges.size-1)])

    # make subbin-to-old bin map
    subbin_mid = midpoints(subbin_edges)
    sub2old = np.searchsorted(x1, subbin_mid) - 1

    # make subbin-to-new bin map
    sub2new = np.searchsorted(x2, subbin_mid) - 1

    # loop over subbins
    y2 = [0. for i in range(n)]
    for i in range(subbin_mid.size):
        # skip subcells which don't lie in range of x1
        if sub2old[i] == -1 or sub2old[i] == x1.size-1:
            continue
        else:
            y2[sub2new[i]] += ( y1[sub2old[i]] * subbin_areas[i] 
                                               / areas1[sub2old[i]] )

    return np.array(y2)


def rebin_piecewise_constant(x1, y1, x2):
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)

    # the fractional bin locations of the new bins in the old bins
    i_place = np.interp(x2, x1, np.arange(len(x1)))

    cum_sum = np.r_[[0], np.cumsum(y1)]

    # calculate bins where lower and upper bin edges span
    # greater than or equal to one original bin.
    # This is the contribution from the 'intact' bins (not including the
    # fractional start and end parts.
    whole_bins = np.floor(i_place[1:]) - np.ceil(i_place[:-1]) >= 1.
    start = cum_sum[np.ceil(i_place[:-1]).astype(int)]
    finish = cum_sum[np.floor(i_place[1:]).astype(int)]

    y2 = np.where(whole_bins, finish - start, 0.)

    bin_loc = np.clip(np.floor(i_place).astype(int), 0, len(y1) - 1)

    # fractional contribution for bins where the new bin edges are in the same
    # original bin.
    same_cell = np.floor(i_place[1:]) == np.floor(i_place[:-1])
    frac = i_place[1:] - i_place[:-1]
    contrib = (frac * y1[bin_loc[:-1]])
    y2 += np.where(same_cell, contrib, 0.)

    # fractional contribution for bins where the left and right bin edges are in
    # different original bins.
    different_cell = np.floor(i_place[1:]) > np.floor(i_place[:-1])
    frac_left = np.ceil(i_place[:-1]) - i_place[:-1]
    contrib = (frac_left * y1[bin_loc[:-1]])

    frac_right = i_place[1:] - np.floor(i_place[1:])
    contrib += (frac_right * y1[bin_loc[1:]])

    y2 += np.where(different_cell, contrib, 0.)

    return y2
