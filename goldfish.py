#!/usr/bin/env python

# David Cain
# Justin Sperry
# 2012-05-02
# CS365, Brian Eastwood

import os
 
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



def background_subtraction():
	# A list of all the goldfish-free frames
	bg_frame_fns = glob.glob("fish-74.2/blanks/*.tif")

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
	average_background = bg_frames.get_avg_image()
	

class ShowFeatures(pipeline.ProcessObject):
	'''
	Draws boxes around the features in an image
	'''
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


def particle_filter_test():
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
	particle_filter_test()
	'''
	display = Display(video_stream.getOutput(), "Testosterone-laden fish")
	fish_presence = BackgroundSubtraction(video_stream.getOutput())

	while key != 27:
		video_stream.update()
		display.update()
		fish_presence.update()

		key = cv2.waitKey(10)
		key &= 255
	'''


