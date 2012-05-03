#!/usr/bin/env python

# David Cain
# Justin Sperry
# 2012-04-25
# CS365, Brian Eastwood

import os
 
import glob
from scipy import ndimage
import cv
import cv2
import numpy

import avgimage
import pipeline
import source

class Display(pipeline.ProcessObject):
    """
        Pipeline object to display the numpy image in a CV window
    """
    
    def __init__(self, input=None, name="pipeline"):
        pipeline.ProcessObject.__init__(self, input)
        cv2.namedWindow(name, cv.CV_WINDOW_NORMAL)
        self.name = name
        
    def generateData(self):
        input = self.getInput(0).getData()
        # output here so channels don't get flipped
        self.getOutput(0).setData(input)

        # Convert back to OpenCV BGR from RGB
        if input.ndim == 3 and input.shape[2] == 3:
            input = input[..., ::-1]
        
        cv2.imshow(self.name, input.astype(numpy.uint8))        


class BackgroundSubtraction(pipeline.ProcessObject):
    """
        Segments the bacteria colonies in the images.
    """
    def __init__(self, input=None, bgImg=None, threshold=2.0):
        pipeline.ProcessObject.__init__(self, input)
        self.bgImg = bgImg
        self.threshold = threshold
    
    def generateData(self):
        """
            Perform background subtraction on the image, segment the
            bacteria colonies (foreground) from the background data.
        """

        input = self.getInput(0).getData()
        #background subtraction
        output = (input.astype(numpy.float) - self.bgImg.astype(numpy.float))
        fish_present = (output.mean() > self.threshold)
        
        self.getOutput(0).setData(fish_present)
        
if __name__ == "__main__":
    #cam = cv2.VideoCapture(0)
    avi_filename = "83.avi"
    blank_frames_dir = "blanks"

    if not os.path.isdir(blank_frames_dir):
        os.makedirs(blank_frames_dir)
    num = 11 # We already have 10 frames

    assert os.path.isfile(avi_filename)
    key = None

    video_stream = source.VideoCV(avi_filename)
    display = Display(video_stream.getOutput(), "Testosterone-laden fish")
    fish_presence = BackgroundSubtraction(video_stream.getOutput())


    while key != 27:
        video_stream.update()
        display.update()
        fish_presence.update()

        key = cv2.waitKey(10)
        key &= 255


    '''
    cam = cv2.VideoCapture(avi_filename)
    while key != 27:
        # grab a frame from the camera
        flag, frame = cam.read()

        # If space is pressed, save new background image
        if key == 32:
            fn = os.path.join(blank_frames_dir, "%.3i.npy" % num)
            numpy.save(fn, frame)
            num += 1
        
        # display the frame in a window
        cv2.imshow("camera", frame)
        key = cv2.waitKey(20) & 255
    '''


    '''
    # Read in the bg frames, average them, and save to a numpy image
    bg_frames = avgimage.AvgImage(buffer_size = 10)
    for image_fn in glob.glob( os.path.join(blank_frames_dir, "*.npy") ):
        image = numpy.load(image_fn)
        bg_frames.add_image(image)

    bg = bg_frames.get_avg_image()
    numpy.save("bg.npy", bg)
    '''
