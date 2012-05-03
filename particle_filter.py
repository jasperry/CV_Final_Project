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
		self.stepsize = stepsize
		self.n = n
		self.x = numpy.ones((n,2), int) * pos[0]
		self.hist = None
		
		
		
	def generateData(self):
		
		input = self.getInput(0).getData()
		
		#if there is no histogram for the initial object to be tracked, grab one
		if self.hist = None:
			self.hist = self.makeHist(input, self.start_position, self.n)
			
		else:
		
			#perturb particles, clip for size, make histograms for each particle
			self.x += numpy.random.uniform(-self.stepsize, self.stepsize, self.x.shape)
			self.x = self.x.clip(numpy.zeros(2), array(input.shape)-1).astype(int))
			new_hist = self.make_histogram(input, x, self.n)
			
			#calculate weights (as battacharyya distances)
			w = self.get_weights(new_hist, self.hist)
			w /= numpy.sum(w)
			self.getOutput(0).setData(numpy.sum(self.x.T*w))
			
			
			if 1./sum(w**2) < n/2.:
				self.x = self.x[self.resample(w),:]
			
			
			
	
	def resample(self, weights):
		n = len(weights)
		indices = []
		c = [0.] + sum(weights[:i+1]) for i in range(n)]
		u0 = numpy.random()
		for u in [(u0+i)/n for i in range(n)]:
			while u > C[j]:
				j+=1
			indices.append(j-1)
		return indices	
	
	#returns a list of histograms of intensity values over an nxn patch
	#centered over the specified positions
	def make_histogram(self, img, pos, n):
		hists = []
		for each in pos:
			x = each[0]
			y = each[1]
			r = n/2
			patch = ndimage.map_coordinates(img, [[y-r, y+r],[x-r,x+r]])
			hist, bin_edges = numpy.histogram(patch, bins = 10) #10 bins
			hists.append(hist)
		return hists
	
	
	#returns an array of 1/(the bhattacharyya distance from the original
	#histogram to the current histogram)
	def get_weights(self,f, fo):
		weights = []
		compare = zip(f,fo)
		for each in compare:
			weights.append(1./self.bhattacharyya(each[0],each[1]))
		
		return numpy.array(weights)
		
	#bhattacharyya distance metric
	def bhattacharyya(self,v1, v2):
		return -numpy.log((numpy.sqrt(v1*v2)).sum())
	
	


def test_particle_filter()
	import glob
	import os.path
	
	files = glob.glob("path")



if __name__ == "__main__":
	test_particle_filter()




