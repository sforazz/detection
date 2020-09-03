# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt

# import the necessary packages
from imutils.video import FPS
import argparse
import imutils
import datetime
import cv2
from .utils import utilsVideo


def tracking(video, tracker_name='csrt', resample=None, save_box_center=True):

    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
    
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
        tracker = cv2.Tracker_create(tracker_name.upper())
    
    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # approrpiate object tracker constructor:
    else:
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
    
        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
    
    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None

    vs = cv2.VideoCapture(video)
    uv = utilsVideo(vs)
    (fps,frame_count,durationSec) = uv.getStats()
    print ("Total time: {}sec FrameRate: {} FrameCount: {}"
           .format(durationSec, fps, frame_count))

    # initialize the FPS throughput estimator
    fps = None

    # loop over frames from the video stream
    xpos = []
    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    file_name = video.split('.m4v')[0]+'_box_x_coordinates_{}.txt'.format(
        current_time)
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1]
    
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # resize the frame (so we can process it faster) and grab the
        # frame dimensions.
        if resample is not None:
            frame = imutils.resize(frame, width=resample)
        (H, W) = frame.shape[:2]
    
        # check to see if we are currently tracking an object
        if initBB is not None:
            t = 1
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
    
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)
                cv2.line(frame, (x+int(w/2), 0), (x+int(w/2), 50),(0, 0, 255),2)
            xpos.append(x+int(w/2))
            # update the FPS counter
            fps.update()
            fps.stop()
    
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", tracker_name),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            t = 10000
        # show the output frame
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', 2000, 1000)
        frame = uv.displayProgressBar(frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(t) & 0xFF
    
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
    
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()
        if key == ord("c"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = None
            tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break

    if save_box_center:
        with open(file_name, 'w') as f:
            for el in xpos:
                f.write(str(el)+'\n')

    vs.release()
    # close all windows
    cv2.destroyAllWindows()

    return file_name


if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="kcf",
        help="OpenCV object tracker type")
    ap.add_argument("-r", "--resample", type=int, default=0,
        help=("Width, in pixels, that will be used to resample the input video."
              " By default is not resampled."))
    ap.add_argument("-s", "--save_box_coordinates", action='store_true',
        help=("Whether or not to save the x coordinates of each detected "
              "box to file. Default is False."))
    args = vars(ap.parse_args())
    
    _ = tracking(args["video"], args["tracker"], resample=args["resample"],
                 save_box_center=args["save_box_coordinates"])

    print('Done')
