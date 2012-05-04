#!/usr/bin/env python

# David Cain
# Justin Sperry
# 2012-05-04
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
    
    def __init__(self, inpt=None, name="pipeline"):
        pipeline.ProcessObject.__init__(self, inpt)
        cv2.namedWindow(name, cv.CV_WINDOW_NORMAL)
        self.name = name
        
    def generateData(self):
        """
            Get the next input, adjust its color channels accordingly,
            and display the results in an openCV window
        """
        inpt = self.getInput(0).getData()
        assert inpt is not None, "Can't display! Image is null"

        # output here so channels don't get flipped
        self.getOutput(0).setData(inpt)

        # Convert back to OpenCV BGR from RGB
        if inpt.ndim == 3 and inpt.shape[2] == 3:
            inpt = inpt[..., ::-1]
        
        cv2.imshow(self.name, inpt.astype(numpy.uint8))        


class BackgroundSubtraction(pipeline.ProcessObject):
    """
        Segments the bacteria colonies in the images.

        inpt: a pipeline.Image object
        threshold: the minimum absolute mean difference to consider the
            fish as part of the image (2.0 is a good value, can be
            tweaked to any threshold, though)
    """
    def __init__(self, inpt=None, bgImg=None, threshold=2.0):
        pipeline.ProcessObject.__init__(self, inpt, outputCount=3)
        self.bgImg = bgImg
        self.threshold = threshold
    
    def generateData(self):
        """
            Perform background subtraction on the image, segment the
            bacteria colonies (foreground) from the background data.

            Outputs:
                [0] = <boolean> true if fish present in frame
                [1] = <float> absolute mean value of pixel differences
                      of the frame and background image.
                [2] = Image object containing mask image of difference
        """
        inpt = self.getInput(0).getData()

        # Perform a pixel-by-pixel absolute background subtraction
        diff = abs(inpt.astype(numpy.float) - self.bgImg.astype(numpy.float))
        fish_present = diff.mean() > self.threshold

        self.setOutput(fish_present, 0)
        self.setOutput(diff.mean(), 1)
        self.getOutput(2).setData(diff)

class ShowFeatures(pipeline.ProcessObject):
    """
        Draws boxes around the features in an image

        inpt: The input image to draw on
        features: an (x,y) tuple of the feature center
            TODO: list of features?
    """
    def __init__(self, inpt=None, features=None, n=None):
        pipeline.ProcessObject.__init__(self, inpt, 2)
        self.setInput(features, 1)
        self.r = n/2.
        
    def generateData(self):
        """
            Draw a rectangle at the feature location
        """
        inpt = self.getInput(0).getData()
        feature = self.getInput(1).getData()
        x = feature[1]
        y = feature[0]
        r = self.r
        cv2.rectangle(inpt, (int(x-r), int(y-r)), (int(x+r), int(y+r)),
                (255,0,0), thickness=2)
        self.getOutput(0).setData(inpt)

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
    blobs = particle_filter.DifferenceOfGaussian(src.getOutput())
    p_filter = particle_filter.Particle_Filter(blobs.getOutput(),
            numpy.array([102,123]), patch_n, 100)
    #p_filter3 = particle_filter.Particle_Filter(src.getOutput(),
    #       numpy.array([102,123]), patch_n, 100, True)
    features = ShowFeatures(src.getOutput(), p_filter.getOutput(), patch_n)
    #features3 = ShowFeatures(src.getOutput(), p_filter3.getOutput(), patch_n)
    display2 = Display(features.getOutput(), "Eye_Tracking")
    display3 = Display(blobs.getOutput(), "DoG")


    # Get averaged background image
    bg_frame_fns = sorted(glob.glob("fish-83.2/blanks/*.tif"))
    avg_bg = average_images(bg_frame_fns)

    print raw.getOutput()
    fish_presence = BackgroundSubtraction(raw.getOutput(), avg_bg, 2.0)

    display4 = Display(fish_presence.getOutput(2), "Fish background subtraction")
    
    key = None
    frame = 0
    while key != 27:
        raw.update()
        src.update()
        display.update()
        p_filter.update()
        #p_filter3.update()
        #features3.update()
        features.update()
        display2.update()
        display3.update()

        fish_presence.update()
        display4.update()
        
        frame += 1
        print "Frame: %d" % (frame)

        key = cv2.waitKey(10)
        key &= 255
        raw.increment()

def subtract_and_track():

    patch_n = 20
    
    frames = sorted(glob.glob("fish-83.2/*.tif"))
    raw = source.FileStackReader(frames)
    src = color.Grayscale(raw.getOutput())
    fish = BackgroundSubtraction(src.getOutput())
    display = Display(fish.getOutput(1), "Testosterone-laden Goldfish")
    blobs = particle_filter.DifferenceOfGaussian(src.getOutput(1))
    p_filter = particle_filter.Particle_Filter(blobs.getOutput(),
            numpy.array([102,123]), patch_n, 100)
    features = ShowFeatures(src.getOutput(), p_filter.getOutput(), patch_n)
    display2 = Display(features.getOutput(), "Eye Tracking")
    display3 = Display(blobs.getOutput(), "DoG")
    
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
        display3.update()
        
        frame += 1
        print "Frame: %d" % (frame)
        

        key = cv2.waitKey(10)
        key &= 255


if __name__ == "__main__":
    """
        Test the particle filter
    """
    particle_filter_test()

