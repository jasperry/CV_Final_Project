#!/usr/bin/env python

# David Cain
# Hieu Phan
# Justin Sperry
# 2012-03-28
# CS365, Brian Eastwood

"""
    A file dedicated to the average image buffer class, and a small function
    that uses this class to generate a flat-field image.
"""

import time

import numpy

def get_flat_field(np_images, buffer_size=None):
    """
        Create a flat field image from the input numpy images
    """
    if buffer_size is None:
        buffer_size = len(np_images)
    avg = AvgImage(buffer_size)

    for image in np_images:
        avg.add_image(image)

    avg_image = avg.get_avg_image()
    avg_mean = avg_image.mean() 
    avg_image[avg_image==0] = avg_mean # Avoid divide-by-zero error

    return (avg_mean / avg_image)


class AvgImage:
    """
        A class to hold image frames, and average them into one frame.
        Useful for flat field correction.
    """
    def __init__(self, buffer_size=30, image_shape=None):
        """
            Create space for the frames, current frame pointer.

            Creates a list to hold <buffer_size> images- every time an
            image is added, it goes to position <self.i> in this list-
            cycling back to the beginning if the end of the list has
            been reached.

            buffer_size: <int> The maximum number of frames to hold
            image_shape: <tuple> numpy shape for all images to be read
                         if None, shape is not set until first image read in
        """
        self.buffer_size = buffer_size
        self.image_shape = image_shape
        self.frame_buffer = None
        self.i = 0 # NOTE: Need to set this in case frame buffer not init'd

        # Allow user to specify size of images to come
        # Note: reason for manually specifying shape here is to avoid delay
        #       while numpy allocates memory after reading first image
        if self.image_shape != None:
            self._init_frames_buffer(image_shape)

        self.avg_frame = None
        self.all_scanned = False # True after <buffer_size> images read in

    def set_buffer_size(self, buffer_size):
        """
            Set the number of frames.

            If the numpy image space is not null, recreate it. Otherwise, do
            nothing.
        """
        self.buffer_size = buffer_size
        self.clear_frames()

    def get_buffer_size(self):
        """
            Returns the number of frames the buffer can hold
        """
        return self.buffer_size

    def get_num_frames_obtained(self):
        """
            Returns the number of frames stored in the buffer
        """
        return self.buffer_size if (self.all_scanned) else self.i

    def _init_frames_buffer(self, image_shape):
        """
            Initialize the buffer for frames to have <buffer_size> images
            of the size specified by image_shape

            image_shape: <tuple> numpy shape for all images to be read
        """
        self.i = 0
        self.all_scanned = False
        self.image_shape  = image_shape
        self.frame_buffer = numpy.zeros( [self.buffer_size] + list(image_shape))

    def add_image(self, image, recalc_avg=False):
        """
            Add an image to the next available spot in the image space.
            
            image: numpy.ndarray image (shape must match self.image_shape)
            recalc_avg: Compute the average each time an image is added
        """
        if self.image_shape == None:
            print "Setting frame buffer image sizes to: %s" % str(image.shape)
            self._init_frames_buffer(image.shape)

        assert image.shape == self.image_shape
        self.frame_buffer[self.i] = image
        self.i += 1

        if self.i >= self.buffer_size:
            self.all_scanned = True
            self.i -= self.buffer_size

        if recalc_avg == True:
            self.get_avg_image()

    def get_avg_image(self):
        """
            Compute the average image with the available frames.

            include_empty_frames: Average in empty frames, if any exist
        """

        frames_obtained = self.get_num_frames_obtained()
        if frames_obtained == 0:
            print "No frames have been read in, cannot compute average"
            return
        print "Averaging %i obtained frames." % frames_obtained
        start = time.time()
        self.avg_frame = sum(self.frame_buffer) / frames_obtained
        print "Took %.2f seconds to make average image" % (time.time()-start)
        return self.avg_frame

    def clear_frames(self):
        """
            Clear all the frames read in, reset next image pointer.
            Does nothing if no frames have been read in.
        """
        if self.frame_buffer != None:
            self._init_frames_buffer(self.image_shape)

    def get_frames(self):
        frames_obtained = self.get_num_frames_obtained()
        return self.frame_buffer[:frames_obtained]

if __name__ == "__main__":
    print "To see this code tested, refer to Project 2's 'calibration' stage"
