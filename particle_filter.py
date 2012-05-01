import cv
import cv2
import numpy
import pipeline
from scipy import ndimage 



'''
Implements particle filtering after Nummiaro, Koller-Meier, Van Gool


Grayscale images only


Input(0) = next Frame
Input(1) = Last Position of objects to be tracked

'''
class Particle_Filter(Pipeline.ProcessObject):


	def __init__(self, input = None, pos = None, stepsize = None, n = 10)
		pipeline.ProcessObject.__init__(self, input, inputCount = 2)
		
		self.start_position = pos
		self.n = n
		self.x = numpy.ones((n,2), int) * pos[0]
		self.hist = None
		
		
		
	def generateData(self):
		
		input = self.getInput(0).getData()
		
		#if there is no histogram for the initial object to be tracked, grab one
		if self.hist = None:
			self.hist = self.makeHist(input, self.start_position, self.n)
			
		else:
			self.x += numpy.random.uniform(-self.stepsize, self.stepsize, self.x.shape)
			
			
	
	
	
	#returns a histogram of intensity values over an nxn patch centered over
	#the specified position
	def make_histogram(self, img, pos, n):
		x,y = pos
		r = n/2
		patch = ndimage.map_coordinates(img, [[y-r, y+r],[x-r,x+r]])
		hist, bin_edges = numpy.histogram(patch, bins = 10) #10 bins
		return hist
	
	#bhattacharyya distance
	def bhattacharyya(v1, v2)
		return -numpy.log((numpy.sqrt(v1*v2)).sum())
	