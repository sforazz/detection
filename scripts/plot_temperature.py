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
from matplotlib.backends.backend_pdf import PdfPages


font = {'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

seconds_x_frame = 0.033846048376665376
MAX_COUNTS = 464.4354758246023/seconds_x_frame
MIN_COUNTS = 0.0


def plot_temperature(value_file, min_temp, max_temp):

    n_bins = int(max_temp - min_temp)
    bins = np.linspace(min_temp, max_temp, n_bins+1)
    plot_name = value_file.split('.txt')[0]+'_histogram_temperature.png'

    values = np.loadtxt(value_file)
    
    median_xpos = np.median(values)
    hist = plot.hist(values, bins)

    text_ypos = np.percentile(hist[0], 99)
    text_xpos = bins[-9]
    
    hist_counts_name = (value_file.split('.txt')[0]+
                        '_histogram_counts_temperature.txt')
    hist_bins_name = (value_file.split('.txt')[0]+
                          '_histogram_bins_temperature.txt')
    np.savetxt(hist_counts_name, hist[0]*seconds_x_frame)
    np.savetxt(hist_bins_name, bins)
    
    cm = plot.cm.get_cmap('RdYlBu_r')
    bin_centers = 0.5 * (hist[1][:-1] + hist[1][1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
     
    for c, p in zip(col, hist[2]):
        plot.setp(p, 'facecolor', cm(c))
     
    plot.text(text_xpos, text_ypos, r'median value: {:.2f}'.format(median_xpos))
    plot.xlabel("Temperature [C°]")
    plot.ylabel("Time spent [seconds]")
    
    plot.savefig(plot_name)
    plot.show()
    plot.close()


def plot_heatmap(temperature_files, min_temp, max_temp):

    n_bins = int(max_temp - min_temp)
    bins = np.linspace(min_temp, max_temp, n_bins+1)
    max_counts = []
    min_counts = []
    for temperature_file in temperature_files:
        temp = np.loadtxt(temperature_file)
        hist = plot.hist(temp, bins)
        max_counts.append(hist[0].max())
        min_counts.append(hist[0].min())
        plot.close()
        hist_counts_name = (temperature_file.split('.txt')[0]+
                            '_histogram_counts_temperature.txt')
        hist_bins_name = (temperature_file.split('.txt')[0]+
                              '_histogram_bins_temperature.txt')
        np.savetxt(hist_counts_name, hist[0]*seconds_x_frame)
        np.savetxt(hist_bins_name, bins)

    for temperature_file in temperature_files:
        plot_name = temperature_file.split('.txt')[0]+'_heatmap_position.png'
        pdf_name = temperature_file.split('.txt')[0]+'_heatmap_position.pdf'
        temp = np.loadtxt(temperature_file)
        hist = plot.hist(temp, bins)
        plot.close()

        _, ax = plot.subplots(figsize=(30, 15))
        hist1 = hist[0][np.newaxis, :]
        hist1 = ((hist1 - np.min(min_counts))/
                 (np.max(max_counts) - np.min(min_counts)))
        extent = [bins.min(), bins.max(), 0, 1]
        plot.gca().set_yticks([])

        i = ax.imshow(hist1, extent=extent, aspect=3, cmap='plasma',
                      vmin=0, vmax=1)
        cbar = plot.colorbar(i)
#         cbar.ax.tick_params(size=0)
        cbar.set_ticks([0, 0.33, 0.66, 1])
        cbar.set_ticklabels([int(np.min(min_counts)),
                              int(0.33*np.max(max_counts)*seconds_x_frame),
                              int(0.66*np.max(max_counts)*seconds_x_frame),
                              int(np.max(max_counts)*seconds_x_frame)])
        cbar.ax.set_ylabel('Time spent [seconds]', rotation=90, labelpad=40)

        plot.xlabel("Temperature [C°]")
    #     plot.ylabel("Time spent [seconds]")
        
        plot.savefig(plot_name)
        plot.savefig(pdf_name)
#         pp.savefig(ax)
        plot.show()
        plot.close()


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--coordinates_file", type=str, nargs='+',
        help="path to the text file with the coordinates of the detected box from a previuos "
        "tracking.")
    ap.add_argument("--min_temp", type=float,
        help="Minimum temperature recorded by the infrared camera.")
    ap.add_argument("--max_temp", type=float,
        help="Maximum temperature recorded by the infrared camera.")

    args = vars(ap.parse_args())

    plot_heatmap(args["coordinates_file"], args["min_temp"], args["max_temp"])

    print('Done')


if __name__ == "__main__":
    main()
