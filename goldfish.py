#!/usr/bin/env python

# David Cain
# Justin Sperry
# 2012-05-02
# CS365, Brian Eastwood

import csv
import os
 
import glob
#from scipy import ndimage
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
        pipeline.ProcessObject.__init__(self, input, outputCount=2)
        self.bgImg = bgImg
        self.threshold = threshold
    
    def generateData(self):
        """
            Perform background subtraction on the image, segment the
            bacteria colonies (foreground) from the background data.
        """

        input = self.getInput(0).getData()
        #background subtraction
        diff = abs(input.astype(numpy.float) - self.bgImg.astype(numpy.float))
        fish_present = diff.mean() > self.threshold

        self.setOutput(fish_present, 0)
        self.setOutput(diff.mean(), 1)
        
if __name__ == "__main__":

    # All frames in the dataset
    all_frame_fns = sorted(glob.glob("fish-74.2/*.tif"))

    # A list of all frames where the goldfish and its shadow are absent
    bg_frame_fns = sorted(glob.glob("fish-74.2/blanks/*.tif"))

    # Use pipeline object to read background frames, an object to average them
    bg_images = source.FileStackReader(bg_frame_fns)
    bg_frames = avgimage.AvgImage(buffer_size = len(bg_frame_fns))

    # Add all the background frames to the average image object
    for i in range(len(bg_frame_fns)):
        bg_images.update()
        image = (bg_images.getOutput()).getData()
        bg_frames.add_image( image )
        bg_images.increment()

    # Create a numpy image that is the average of all background frames
    avg_bg = bg_frames.get_avg_image()



    #all_frame_fns = ["fish-74.2/fish-74.2-000002.tif"] # fish frame - 1.76

    all_images = source.FileStackReader(all_frame_fns)
    display = Display(all_images.getOutput(), "Testosterone-laden fish")
    fish_presence = BackgroundSubtraction(all_images.getOutput(), avg_bg)

    # Display video, gather data about fish's presence, abs mean value
    intensity_data = []
    prev_frame = None
    key = None
    while (key != 27) and (all_images.getFrameName() != prev_frame):
        all_images.update()
        display.update()
        fish_presence.update()

        fish_present = fish_presence.getOutput(0)
        avg_val = fish_presence.getOutput(1)
        intensity_data.append( (avg_val, all_images.getFrameName(), fish_present) )

        # TODO: add a delay that's either consistent with the FPS Brian
        #       obtained, or sped up but still reasonably visible
        all_images.increment()
        key = cv2.waitKey(20)
        key &= 255

    intensity_out = csv.writer(open("image-bg_values.csv", "wb"))
    for (value, frame_name, fish_present) in sorted(intensity_data):
        intensity_out.writerow( [frame_name, value, fish_present] )
