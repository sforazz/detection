"Script to convert position to temperature."
import numpy as np
import cv2
import argparse
import os
from scipy.interpolate import interp1d
import glob


FRAME_LEN_CM = 110 #lenght of the metal bar seen in the video


def position2temperature(video, coordinates_file, xlabels):
    
    with open(coordinates_file, 'r') as f: 
        xpos = [int(x.strip()) for x in f]
    
    if xpos:
        file_name = coordinates_file.split('.txt')[0]+'_position2temperature.txt'
        captured_video = cv2.VideoCapture(video)
      
        frame_len_pixel = captured_video.read()[1].shape[1]
        print('[INFO] Acquired video has a frame width of {} pixels'.format(frame_len_pixel))
        
        #need this to revert the temperature along x to go from Tmin to Tmax
        xpos = [np.abs(x-frame_len_pixel) for x in xpos]

        xtick_labels = xlabels
        xticks = np.linspace(0, frame_len_pixel, len(xlabels))
        mapping_function = interp1d(xticks, xtick_labels)
        xpos = [float(mapping_function(x)) for x in xpos]

        median_xpos = np.median(xpos)
        print('[INFO] The median value along x is {}.'.format(median_xpos))
        np.savetxt(file_name, xpos)

    else:
        print('[INFO] the coordinates file is empty. Nothing to plot.')
        os.remove(coordinates_file)


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
    ap.add_argument("-cf", "--coordinates_file", type=str,
        help="path to the text file with the coordinates of the detected box from a previuos "
        "tracking.")
    ap.add_argument("--xlabel", nargs='+', type=float,
        help="List of floats that will be used as labels for the x axis.")

    args = vars(ap.parse_args())

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

    position2temperature(args["video"], coordinates, args['xlabel'])

    print('Done')


if __name__ == "__main__":
    main()
