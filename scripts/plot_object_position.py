"Script to plot the histogram of the detected object position (along x) over time."
import matplotlib.pyplot as plot
import numpy as np
import cv2
import matplotlib
import argparse
import os
from detection.tracking import tracking
from scipy.interpolate import interp1d


font = {'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

FRAME_LEN_CM = 110 #lenght of the metal bar seen in the video


def get_length(cap):

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    return duration


def plot_box_coordinates(video, coordinates_file, xlabels=[],
                         seconds_to_plot=None):
    
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
        median_xpos = np.median(xpos)
        
        bins = np.linspace(0, frame_len_pixel, n_bins)
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
    ap.add_argument("-s", "--save_box_coordinates", action='store_true',
        help=("Whether or not to save the x coordinates of each detected "
              "box to file. Default is False."))
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
    
    if not args.get("seconds_to_plot", False):
        seconds_to_plot = None
    else:
        seconds_to_plot = args["seconds_to_plot"]

    plot_box_coordinates(args["video"], coordinates, xlabels=args['xlabel'],
                         seconds_to_plot=seconds_to_plot)

    print('Done')


if __name__ == "__main__":
    main()
