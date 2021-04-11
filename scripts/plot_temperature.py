"Script to plot the histogram of the detected object position (along x) over time."
import matplotlib.pyplot as plot
import numpy as np
import cv2
import matplotlib
import argparse
import os
from detection.tracking import tracking
from scipy.interpolate import interp1d
from detection.utils import rebin
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable


font = {'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

seconds_x_frame = 0.033846048376665376


def plot_temperature(value_file, min_temp, max_temp):

    plot_name = value_file.split('.txt')[0]+'_histogram_temperature.png'

    values = np.loadtxt(value_file)
    n_bins = int(max_temp - min_temp)
    bins = np.linspace(min_temp, max_temp, n_bins+1)
    median_xpos = np.median(values)


    hist = plot.hist(values, bins)
    plot.close()
    #hist, edges = np.histogram(values, bins)
    _, ax = plot.subplots(figsize=(30, 15))
    hist1 = hist[0][np.newaxis, :]
    extent = [bins.min(), bins.max(), 0, 1]
    ax.axes.yaxis.set_ticklabels([])
    
#     ax = plot.gca()
#     im = ax.imshow(hist1, extent=extent, aspect=2, cmap='plasma')
#     
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     plot.colorbar(im, cax=cax)
    plot.imshow(hist1, extent=extent, aspect=2, cmap='plasma')
    plot.colorbar()

    #yticks = np.linspace(0, max(hist[0]), 10)
    #ytick_labels = (yticks*seconds_x_frame).astype(int)
    #ax.set_yticks(yticks)
    #ax.set_yticklabels(ytick_labels)

#     text_ypos = np.percentile(hist[0], 99)
#     text_xpos = bins[-9]
    
    hist_counts_name = (value_file.split('.txt')[0]+
                        '_histogram_counts_temperature.txt')
    hist_bins_name = (value_file.split('.txt')[0]+
                          '_histogram_bins_temperature.txt')
    np.savetxt(hist_counts_name, hist[0]*seconds_x_frame)
    np.savetxt(hist_bins_name, bins)
    
#     cm = plot.cm.get_cmap('RdYlBu_r')
#     bin_centers = 0.5 * (hist[1][:-1] + hist[1][1:])
#     col = bin_centers - min(bin_centers)
#     col /= max(col)
#     
#     for c, p in zip(col, hist[2]):
#         plot.setp(p, 'facecolor', cm(c))
#     
#     plot.text(text_xpos, text_ypos, r'median value: {:.2f}'.format(median_xpos))
    plot.xlabel("Temperature [C°]")
    plot.ylabel("Time spent [seconds]")
    
    plot.savefig(plot_name)
    plot.show()
    plot.close()


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--coordinates_file", type=str,
        help="path to the text file with the coordinates of the detected box from a previuos "
        "tracking.")
    ap.add_argument("--min_temp", type=float,
        help="Minimum temperature recorded by the infrared camera.")
    ap.add_argument("--max_temp", type=float,
        help="Maximum temperature recorded by the infrared camera.")

    args = vars(ap.parse_args())

    plot_temperature(args["coordinates_file"], args["min_temp"], args["max_temp"])

    print('Done')


if __name__ == "__main__":
    main()
