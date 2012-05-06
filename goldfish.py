#!/usr/bin/env python

# David Cain
# Justin Sperry
# 2012-05-06
# CS365, Brian Eastwood

import csv
import math
import random
import time
 
import glob
from scipy import ndimage
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


#NOTE: settings currently tweaked to best suit 83.2 dataset
class LocateFish(pipeline.ProcessObject):
    """
        Segment the goldfish from the rest of the tank and other objects.

        This class does a good job of separating the fish from its shadow and
        other particles floating through the tank, except when the fish is
        towards the bottom of the tank. 

        inpt: a grayscale pipeline.Image object
        threshold: the minimum absolute mean difference to consider the
            fish as part of the image (2.0 is a good value, can be
            tweaked to any threshold, though)
        isolate_fish: use morphological operations to return a mask
            where we've attempted to isolate the fish, open its mask to
            cover its entire body, and dilate slightly past its edges.
    """
    def __init__(self, inpt, bgImg, threshold=2.0, isolate_fish=True):
        pipeline.ProcessObject.__init__(self, inpt, outputCount=3)
        assert bgImg.ndim == 2 # make sure image is grayscale
        self.bgImg = bgImg
        self.threshold = threshold
        self.binary = numpy.zeros(bgImg.shape)
        self.isolate_fish = isolate_fish
    
    def generateData(self):
        """
            Perform background subtraction on the image, segment the
            fish from the rest of the objects in the tank as best as
            possible. Manually exclude the left side of the frame too.

            Outputs:
                [0] = Image object containing mask image of difference
                [1] = <boolean> True if fish present in frame
                [2] = <float> absolute mean value of pixel differences
                      of the frame and background image.
        """
        inpt = self.getInput(0).getData()

        # Perform a pixel-by-pixel absolute background subtraction
        diff = abs(inpt.astype(numpy.float) - self.bgImg.astype(numpy.float))
        fish_present = diff.mean() > self.threshold

        # We use the frame and background difference to conclude if the
        # fish is present, saving this value as a metric of confidence
        self.setOutput(fish_present, 1)
        self.setOutput(diff.mean(), 2)

        # Initialize arrays that tell us where the fish is (1 == foreground)
        binary_img = numpy.zeros(self.bgImg.shape)
        fishmask = binary_img

        # If the fish is not present, we don't want to identify anything
        if not fish_present:
            self.getOutput(0).setData(binary_img)
            return

        # Create a binary image where 1's indicate a significant
        # difference from the background image
        binary_img[diff > 10] = 1
        binary_img[:,:103] = 0 # Manually exclude the wall

        # Use morphological operations to isolate the fish from other data
        if self.isolate_fish:

            # Open the image to fill the fish's body, eliminate small noise
            binary_img = ndimage.morphology.binary_opening(binary_img,
                    numpy.ones( (3,3) ), iterations=1)
            # Extra dilation to ensure we extend a bit past fish's body
            binary_img = ndimage.morphology.binary_dilation(binary_img,
                    numpy.ones( (3,3) ), iterations=2)
            
            # Divide the image into connected components (e.g. the fish and two
            # shadows below it)
            (labels, num_components) = ndimage.measurements.label(binary_img)
            labels = labels.reshape(binary_img.shape)

            # Exploit the fact that the fish is always above its shadows to
            # restrict the image to just the top connected component
            fishmask = numpy.zeros(binary_img.shape)
            fishmask[labels==1] = 1 # Highest group always indexed to 1

        assert fishmask.ndim == 2 # make sure it's an x,y grid of booleans
        self.getOutput(0).setData(fishmask*255)

class FishOrientation(pipeline.ProcessObject):
    """
        Determines the fish's left/right orientation, and its horizontality.

        A fish is deemed to be horizontal if its x-axis distance exceeds
        the threshold supplied in min_dist

        Inputs:
            [0] = Image object to draw over
            [1] = Image object containing coordinates to eye feature
            [2] = Image object containing coordinates to fin feature
        Outputs:
            [0] = Input 0 with features drawn over it
            [1] = Tuple of distance data (TODO)
    """
    # TODO: fin_feature is currently allowed to be None, so this works
    #       for just one feature. we should eventually disable this, though
    def __init__(self, inpt, eye_feature, fin_feature=None, min_dist=90,
            shape_size=10, debug=False):
        """
            Initialize a pipeline object with the appropriate count of
            inputs and outputs, adding values to specifying feature
            shape sizes.
        """
        pipeline.ProcessObject.__init__(self, inpt, inputCount=3,
                outputCount=2)
        self.setInput(eye_feature, 1)
        self.setInput(fin_feature, 2)
        self.min_dist = min_dist
        self.r = shape_size /2
        self.debug = debug

    def draw_features(self, coords):
        """
            Draw shapes around each (y,x) coordinate in coords.

            The first coordinate will be drawn as a circle, and any
            remaining coordinates as rectangles. self.r gives shape size.
        """
        inpt = numpy.copy(self.getInput(0).getData())
        box_color = (255, 0, 0) # red
        r = self.r

        # Draw the first feature as a circle
        (y,x) = coords[0]
        cv2.circle(inpt, (x,y), self.r, box_color, thickness=2)

        # Draw remaining features as rectangles
        for (y, x) in coords[1:]:
            top_left = ( int(x-r), int(y-r) )
            bottom_right = ( int(x+r), int(y+r) )
            cv2.rectangle(inpt, top_left, bottom_right, box_color, thickness=2)

        self.getOutput(0).setData(inpt)

    def generateData(self):
        """
            Draw features on the image, calculate orientation, horizontality.

            If no fin feature has been supplied, it will simply draw the eye
            feature, and skip all calculations

        """
        eye_coords = (y1,x1) = self.getInput(1).getData()

        # TODO: this block addresses when only one feature is supplied
        #       it should ultimately be deleted
        fin_object = self.getInput(2)
        if fin_object == None:
            self.draw_features( [eye_coords] )
            return

        fin_coords = (y2,x2) = self.getInput(2).getData()
        self.draw_features( [eye_coords, fin_coords] )

        euclidean_dist = math.hypot( x2-x1, y2-y1 )
        x_dist = (x2-x1) # fin - eye (positive is fish facing left

        fish_horizontal = abs(x_dist) > self.min_dist
        facing_left = (x_dist > 0)
        self.setOutput((fish_horizontal, facing_left), 1)

        if self.debug:
            output = [ ("Fish's x distance", x_dist),
                       ("Euclidean distance", euclidean_dist),
                       ("Min. horizontal dist", self.min_dist),
                       ("Fish is horizontal", fish_horizontal),
                       ("Fish is facing left", facing_left) ]
            print 10 * "-"
            for (msg, value) in output:
                print ("{0:20} : {1}".format(msg, value))


def average_images(filenames):
    """
        Return a numpy image of the averaged grayscale images from the filenames.
    """
    # Use a pipeline object to read all the images into a buffer
    image_stack = source.FileStackReader(filenames)
    image_reader = color.Grayscale(image_stack.getOutput())

    avg_buffer = avgimage.AvgImage(buffer_size = image_stack.getLength())
    for i in range(avg_buffer.get_buffer_size()):
        image_stack.update()
        image_reader.update()

        image = (image_reader.getOutput()).getData()
        avg_buffer.add_image( image )

        image_stack.increment()

    # Return the average of all frames in the buffer (a numpy image)
    return avg_buffer.get_avg_image()
    

def test_orientation():
    """
        Test the ability of determining fish orientation.

        Generates fake coordinates, and activates debug mode so that we
        can see how the FishOrientation class handles these datapoints.
        Includes a time delay and visual output so we can confirm its
        accuracy with visual data.
    """

    # All frames in the data set
    all_frame_fns = sorted(glob.glob("fish-83.2/*.tif"))

    # Read frames, convert to grayscale for segmenting
    raw = source.FileStackReader(all_frame_fns)
    src = color.Grayscale(raw.getOutput())

    def get_random_coord():
        """
            Generate a random coordinate within the tank space
        """
        x = random.randint(110,240)
        y = random.randint(0,240)
        return (y,x)

    eye_coord = pipeline.Image(data = get_random_coord())
    fin_coord = pipeline.Image(data = get_random_coord())

    orientation = FishOrientation(src.getOutput(), eye_coord, fin_coord,
            debug=True)
    feature_display = Display(orientation.getOutput(0),
            "Dummy features")

    # Display video with random features, print results of calculations
    prev_frame = None
    key = None
    while (key != 27) and (raw.getFrameName() != prev_frame):
        raw.update()
        orientation.update()
        feature_display.update()

        # Assign random coordinates for the next iteration
        eye_coord.setData(get_random_coord())
        fin_coord.setData(get_random_coord())

        # Read the key, increment the reader to the next image
        raw.increment()
        key = cv2.waitKey(20)
        key &= 255
        time.sleep(1)

def test_identification():
    """
        Identify whether or not the fish is in the image by using background
        subtraction and some morphological operations. Demonstrates the
        improvements over pure background subtraction and thresholding
        done by the LocateFish class.

        Saves the indicator variable, and floating point metric used to
        estimate whether or not the fish is in the screen to a sorted
        .csv file.
    """

    # All frames in the data set
    all_frame_fns = sorted(glob.glob("fish-83.2/*.tif"))

    # A list of all frames where the goldfish and its shadow are absent
    bg_frame_fns = sorted(glob.glob("fish-83.2/blanks/*.tif"))
    avg_bg = average_images(bg_frame_fns)

    # Read frames, convert to grayscale for segmenting
    raw = source.FileStackReader(all_frame_fns)
    src = color.Grayscale(raw.getOutput())
    display = Display(src.getOutput(), "Testosterone-laden fish")

    # Create two windows - one with fish isolation on, one without
    fish_presence = LocateFish(src.getOutput(), avg_bg, 2.0)
    fish_isolated = Display(fish_presence.getOutput(0), "Fish isolation")
    simple_fish_presence = LocateFish(src.getOutput(), avg_bg, 2.0, False)
    fish_unisolated = Display(simple_fish_presence.getOutput(0),
            "Fish isolation (no morphological operations)")

    # Display video, gather data about fish's presence, abs mean value
    intensity_data = []
    prev_frame = None
    key = None
    while (key != 27) and (raw.getFrameName() != prev_frame):
        raw.update()
        display.update()
        fish_presence.update()

        # Update the comparison windows side-by-side
        fish_isolated.update()
        fish_unisolated.update()

        # Get data about the fish's presence, append to list
        fish_present = fish_presence.getOutput(1)
        avg_val = fish_presence.getOutput(2)
        intensity_data.append( (avg_val, raw.getFrameName(), fish_present) )

        # Read the key, get ready for the next image
        raw.increment()
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
    patch_n = 8
    
     # A list of all frames where the goldfish and its shadow are absent
    bg_frame_fns = sorted(glob.glob("fish-83.2/blanks/*.tif"))
    avg_bg = average_images(bg_frame_fns)
    
    frames = sorted(glob.glob("fish-83.2/*.tif"))
    raw = source.FileStackReader(frames)
    src = color.Grayscale(raw.getOutput())
    fish_presence = LocateFish(src.getOutput(), avg_bg, 2.0)
    display = Display(src.getOutput(), "Testosterone Laden Goldfish")
    
    blobs = particle_filter.DifferenceOfGaussian(src.getOutput())
    p_filter = particle_filter.Particle_Filter(blobs.getOutput(), 
            fish_presence.getOutput(0), numpy.array([102,123]), patch_n, 100,True)
    #p_filter3 = particle_filter.Particle_Filter(src.getOutput(),
    #       numpy.array([102,123]), patch_n, 100, True)
    features = FishOrientation(src.getOutput(), p_filter.getOutput(),
            shape_size=patch_n)
    #features3 = FishOrientation(src.getOutput(), p_filter3.getOutput(),
    #       shape_size=patch_n)
    display2 = Display(features.getOutput(), "Eye_Tracking")
    #display3 = Display(blobs.getOutput(), "DoG")

    display4 = Display(fish_presence.getOutput(0), "Fish background subtraction")
    
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
        #display3.update()

        fish_presence.update()
        display4.update()
        
        frame += 1
        # TODO: frame numbers increment indefinitely
        print "Frame: %d" % (frame)

        key = cv2.waitKey(10)
        key &= 255
        raw.increment()

if __name__ == "__main__":
    """
        Test some aspect of the program in a specialized environment,
        unique of other program components (i.e. unit testing)
    """
    test_orientation()
    #test_identification()
    #particle_filter_test()

