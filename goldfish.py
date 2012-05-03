#!/usr/bin/env python

# David Cain
# Justin Sperry
# 2012-05-02
# CS365, Brian Eastwood

import csv
 
import glob
#from scipy import ndimage
import cv
import cv2
import numpy

import avgimage
import pipeline
import source
import color
import particle_filter

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

class ShowFeatures(pipeline.ProcessObject):
    """
        Draws boxes around the features in an image
    """
    def __init__(self, input = None, features = None, n = None):
        pipeline.ProcessObject.__init__(self, input, 2)
        self.setInput(features, 1)
        self.r = n/2.
        
    def generateData(self):
        input = self.getInput(0).getData()
        feature = self.getInput(1).getData()
        x = feature[1]
        y = feature[0]
        r = self.r
        cv2.rectangle(input, (int(x-r), int(y-r)), (int(x+r), int(y+r)), (255,0,0), thickness=2)
        self.getOutput(0).setData(input)

def average_images(filenames):
    """
        Return a numpy image of the averaged images from the filenames.
    """
    num_images = len(filenames)

    # Use a pipeline object to read all the images into a buffer
    images = source.FileStackReader(filenames)
    avg_buffer = avgimage.AvgImage(buffer_size = num_images)
    for i in range(num_images):
        images.update()
        image = (images.getOutput()).getData()
        avg_buffer.add_image( image )
        images.increment()

    # Return the average of all frames in the buffer (a numpy image)
    return avg_buffer.get_avg_image()
    

def fish_identification():
    """
        Identify whether or not the fish is in the image by using background
        subtraction
        
        If the mean absolute intensity exceeds a given threshold, we conclude
        that the fish is present (and if does not exceed the threshould,
        absent).
    """

    # All frames in the data set
    all_frame_fns = sorted(glob.glob("fish-74.2/*.tif"))

    # A list of all frames where the goldfish and its shadow are absent
    bg_frame_fns = sorted(glob.glob("fish-74.2/blanks/*.tif"))
    avg_bg = average_images(bg_frame_fns)

    all_images = source.FileStackReader(all_frame_fns)
    display = Display(all_images.getOutput(), "Testosterone-laden fish")
    fish_presence = BackgroundSubtraction(all_images.getOutput(), avg_bg, 2.0)

    # Display video, gather data about fish's presence, abs mean value
    intensity_data = []
    prev_frame = None
    key = None
    while (key != 27) and (all_images.getFrameName() != prev_frame):
        all_images.update()
        display.update()
        fish_presence.update()

        # Get data about the fish's presence, append to list
        fish_present = fish_presence.getOutput(0)
        avg_val = fish_presence.getOutput(1)
        intensity_data.append( (avg_val, all_images.getFrameName(), fish_present) )

        # Read the key, get ready for the next image
        all_images.increment()
        key = cv2.waitKey(20)
        key &= 255

    # Create a CSV of sorted images and their intensity
    # (useful in determining if the threshold was reasonable)
    intensity_out = csv.writer(open("image-bg_values.csv", "wb"))
    for (value, frame_name, fish_present) in sorted(intensity_data):
        intensity_out.writerow( [frame_name, value, fish_present] )

def particle_filter_test():
    """
        Test the particle filter, tracking the eyeball of the fish.
    """
    patch_n = 20
    
    frames = sorted(glob.glob("fish-83.2/*.tif"))
    raw = source.FileStackReader(frames)
    src = color.Grayscale(raw.getOutput())
    display = Display(src.getOutput(), "Testosterone Laden Goldfish")
    p_filter = particle_filter.Particle_Filter(src.getOutput(), numpy.array([102,123]), patch_n, 100)
    features = ShowFeatures(src.getOutput(), p_filter.getOutput(), patch_n)
    display2 = Display(features.getOutput(), "Eye_Tracking?")
    
    key = None
    frame = 0
    while key != 27:
        raw.update()
        raw.increment()
        src.update()
        display.update()
        p_filter.update()
        features.update()
        display2.update()
        
        frame += 1
        print "Frame: %d" % (frame)
        

        key = cv2.waitKey(10)
        key &= 255
        
if __name__ == "__main__":
    """
        Test the particle filter
    """
    particle_filter_test()

