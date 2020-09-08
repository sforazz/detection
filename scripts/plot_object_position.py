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


font = {'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

FRAME_LEN_CM = 110 #lenght of the metal bar seen in the video
XLABELS_REF = [22.5, 24.6, 26.3, 27.7, 29.5, 31, 35.4, 38.5, 44, 50, 52, 56]
# XLABELS_REF = [19, 20, 21, 23, 24.5, 26.5, 29, 31, 37, 43, 52, 55]


def get_length(cap):

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    return duration


def plot_box_coordinates(video, coordinates_file, xlabels=[],
                         seconds_to_plot=None, rebinning=True):
    
    with open(coordinates_file, 'r') as f: 
        xpos = [int(x.strip()) for x in f]
    
    if xpos:
        plot_name = coordinates_file.split('.txt')[0]+'.png'
        captured_video = cv2.VideoCapture(video)
    
        video_len = get_length(captured_video)
        print('[INFO] Input video has a duration of {} seconds'.format(video_len))
    
        n_frames = int(captured_video.get(cv2.CAP_PROP_FRAME_COUNT))
        print('[INFO] The total number of frames is {}'.format(n_frames))
        
        seconds_x_frame = video_len/n_frames
        print('[INFO] Each frame last for {} seconds'.format(seconds_x_frame))
        
        frame_len_pixel = captured_video.read()[1].shape[1]
        print('[INFO] Acquired video has a frame width of {} pixels'.format(frame_len_pixel))
        pixels_x_cm = frame_len_pixel/FRAME_LEN_CM
        n_bins = int(frame_len_pixel/(2*pixels_x_cm))
        print('[INFO] The number of bins is {}'.format(n_bins))
        
        if seconds_to_plot is not None:
            print('[INFO] Only the last {} seconds of the recording will be plotted'
                  .format(seconds_to_plot))
            frames = int(seconds_to_plot/seconds_x_frame)
            xpos = xpos[-frames:]
        
        #need this to revert the temperature along x to go from Tmin to Tmax
        xpos = [np.abs(x-frame_len_pixel) for x in xpos]
#         if xlabels:
#             xtick_labels = xlabels
#             xticks = np.linspace(0, frame_len_pixel, len(xlabels))
#             mapping_function = interp1d(xticks, xtick_labels)
#             xpos = [float(mapping_function(x)) for x in xpos]
#             bins = np.linspace(np.min(xlabels), np.max(xlabels), n_bins)
#         else:
        bins = np.linspace(0, frame_len_pixel, n_bins)
        median_xpos = np.median(xpos)
        
        
        _, ax = plot.subplots(figsize=(30, 15))
        hist = ax.hist(xpos, bins)
        
        yticks = np.linspace(0, max(hist[0]), 10)
        ytick_labels = (yticks*seconds_x_frame).astype(int)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)

        text_ypos = np.percentile(hist[0], 99)
        text_xpos = np.percentile(xpos, 100)
        
        # chage the labels each time based on infrared camera reading!!!
    #     xmax = np.abs(102-frame_len_pixel)
    #     xmin = np.abs(5-frame_len_pixel)
        if xlabels:
    #         xtick_labels = [22.5, 24.6, 26.3, 27.7, 29.5, 31, 35.4, 38.5, 44, 50, 52, 56]
            xtick_labels = xlabels
            xticks = np.linspace(0, frame_len_pixel, len(xlabels))
            mapping_function = interp1d(xticks, xtick_labels)
            # xtick_labels.reverse()
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)
            median_xpos = float(mapping_function(median_xpos))
            print('[INFO] The median value along x is {}.'.format(median_xpos))
            if rebinning:
                plot_name_ref = coordinates_file.split('.txt')[0]+'_rebinned2ref.png'
                hist_counts_ref_name = (coordinates_file.split('.txt')[0]+
                                        '_histogram_counts_rebinned2ref.txt')
                hist_bins_ref_name = (coordinates_file.split('.txt')[0]+
                                      '_histogram_bins_rebinned2ref.txt')
                bins_label = np.asarray([float(mapping_function(x)) for x in bins])
                xtick_labels_ref = XLABELS_REF
                mapping_function_ref = interp1d(xticks, xtick_labels_ref)
                bins_label_ref = np.asarray([float(mapping_function_ref(x)) for x in bins])
                hist_ref = rebin(bins_label, hist[0], bins_label_ref,
                                 'piecewise_constant')
                np.savetxt(hist_counts_ref_name, hist_ref*seconds_x_frame)
                np.savetxt(hist_bins_ref_name, bins_label_ref)
            else:
                hist_counts_name = (coordinates_file.split('.txt')[0]+
                                        '_histogram_counts.txt')
                hist_bins_name = (coordinates_file.split('.txt')[0]+
                                      '_histogram_bins.txt')
                bins_label = np.asarray([float(mapping_function(x)) for x in bins])
                np.savetxt(hist_counts_name, hist[0]*seconds_x_frame)
                np.savetxt(hist_bins_name, bins_label)

            
        
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
        if rebinning:
            _, ax = plot.subplots(figsize=(30, 15))
            bar = ax.hist(bins_label_ref[:-1], bins=bins_label_ref,
                          weights=hist_ref)
            yticks = np.linspace(0, max(bar[0]), 10)
            ytick_labels = (yticks*seconds_x_frame).astype(int)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels)
            cm = plot.cm.get_cmap('RdYlBu_r')
            bin_centers = 0.5 * np.asarray(
                (bar[1][:-1] + bar[1][1:]))
            col = bin_centers - min(bin_centers)
            col /= max(col)
            for c, p in zip(col, bar[2]):
                plot.setp(p, 'facecolor', cm(c))
            plot.xlabel("Temperature [C°]")
            plot.ylabel("Time spent [seconds]")
            plot.savefig(plot_name_ref)
            plot.show()

    else:
        print('[INFO] the coordinates file is empty. Nothing to plot.')
        os.remove(coordinates_file)


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="csrt",
        help="OpenCV object tracker type")
    ap.add_argument("-r", "--resample", type=int,
        help=("Width, in pixels, that will be used to resample the input video."
              " By default is not resampled."))
    ap.add_argument("--rebinning", action='store_true',
        help=("Whether or not to rebin the histogram to a reference binning scheme"
              "N.B. In this case it assumes the --xlabel is provided and contains "
              "temperature values. Default is False."))
    ap.add_argument("-cf", "--coordinates_file", type=str,
        help="path to the text file with the coordinates of the detected box from a previuos "
        "tracking.")
    ap.add_argument("--seconds_to_plot", type=int,
        help=("Set if you want to plot only a fraction of the video. "
              "For example, if you set this to 1000, then only the "
              "last 1000 seconds will be plotted."))
    ap.add_argument("--xlabel", nargs='+', type=float,
        help="List of floats that will be used as labels for the x axis.")

    args = vars(ap.parse_args())
    
    if not args.get("coordinates_file", False):

        if not args.get("resample", False):
            resample = None
        else:
            resample = args["resample"]

        coordinates = tracking(args["video"], args["tracker"], resample=resample,
                               save_box_center=True)
    else:
        coordinates = args["coordinates_file"]
        if not args.get("video", False):
            basename = args["coordinates_file"].split('_box')[0]
            try:
                args["video"] = glob.glob(basename+'.*')[0]
                print(args["video"])
            except IndexError:
                raise Exception(
                    'Video file was not provided and could not be found in the'
                    ' folder of the coordinates file. You must provied it!')
    
    if not args.get("seconds_to_plot", False):
        seconds_to_plot = None
    else:
        seconds_to_plot = args["seconds_to_plot"]

    plot_box_coordinates(args["video"], coordinates, xlabels=args['xlabel'],
                         seconds_to_plot=seconds_to_plot,
                         rebinning=args["rebinning"])

    print('Done')


if __name__ == "__main__":
    main()
