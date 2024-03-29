#import cv
#import cv2
import numpy
import random
import pipeline
import imgutil
from scipy import ndimage
from scipy.ndimage import filters


'''
Implements particle filtering after Nummiaro, Koller-Meier, Van Gool


Grayscale images only


Input(0) = next Frame
Input(1) = Last Position of objects to be tracked

'''
class Particle_Filter(pipeline.ProcessObject):

    def __init__(self, input=None, mask=None, pos=None, stepsize=None, n=None, r = 8, best=False):
        pipeline.ProcessObject.__init__(self, input, 2)
        
        self.pos = pos
        self.stepsize = stepsize
        self.n = n
        self.r = r
        self.x = numpy.ones((n,2), int) * pos
        self.hist = None
        self.best = best
        self.threshold = 0
        self.setInput(mask, 1)
        
        
    def generateData(self):
        
        input = self.getInput(0).getData()
        mask = self.getInput(1).getData()
        r = self.r
        
        #if there is no histogram for the initial object to be tracked, grab one
        if self.hist == None:
            self.hist = self.make_histogram(input, self.x, self.r)
            x,y = self.x[0][:]
            self.feature_patch = input[x-r:x+r,y-r:y+r].flatten()
            self.getOutput().setData(self.pos)
            
            
        else:
        
            #perturb particles, clip for size, make histograms for each particle
            self.x += numpy.random.uniform(-self.stepsize, self.stepsize, self.x.shape)

            #clip to exclude the wall
            self.x = self.x.clip(numpy.array([0,103]), numpy.array(input.shape)-1).astype(int)

            # List of (x,y) positions fish is present (already excludes wall)
            nonzero_positions = numpy.argwhere(mask != 0)

            # Ensure that all particles are on the fish
            # If a particle is off the fish, randomly choose one on its body
            for i in range(self.x.shape[0]):
                y,x = self.x[i,:]
                if mask[y,x] == 0.0:
                    self.x[i,:] = random.choice(nonzero_positions)

                    #offset = numpy.random.randint(-self.stepsize, self.stepsize, (2))
                    #temp = numpy.array([y,x]).clip(numpy.array([0,103]), numpy.array(input.shape)-1)
                    #y,x = temp
            
            new_hist = self.make_histogram(input, self.x, self.stepsize)
            
            #calculate weights (as battacharyya distances)
            w = self.get_weights(new_hist, self.hist)
            w /= numpy.sum(w) #normalize
            
            if self.best:
                #picks the location of the best particle
                new_pos = self.x[numpy.argmax(w),:]
            
            else:
                #sums the weighted particle positions
                new_pos = numpy.sum(self.x.T*w, axis = 1)
            
            
            
            self.getOutput(0).setData(new_pos)
            #self.setOutput(True,1)
            x, y = new_pos
            self.new_patch = input[x-r:x+r,y-r:y+r].flatten()
            ncc = imgutil.ncc(self.feature_patch, self.new_patch)
            print "NCC = %d" % (ncc)
            #if ncc value < threshold, eye is not in frame
            if ncc < self.threshold:
            	#self.setOutput(1, False)
            	pass
            	
            self.x = numpy.ones((self.n,2), int) * new_pos
            self.pos = new_pos
            
            
            if 1./sum(w**2) < self.n/2.:
                self.x = self.x[self.resample(w),:]
            
    
    def resample(self, weights):
        n = len(weights)
        indices = []
        c = [0.] + [sum(weights[:i+1]) for i in range(n)]
        u0, j = numpy.random.random(), 0
        for u in [(u0+i)/n for i in range(n)]:
            while u > c[j]:
                j+=1
            indices.append(j-1)
        return indices    
    
    #returns a list of histograms of intensity values over an nxn patch
    #centered over the specified positions
    def make_histogram(self, img, pos, step_size):
        hists = []
        for each in pos:
            x = each[0]
            y = each[1]
            r = step_size/2
            patch = ndimage.map_coordinates(img, numpy.array([[y-r, y+r],[x-r,x+r]]))
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


'''
Implements a difference of two Gaussians with different sigma to accentuate
The blob-like eye to track
'''
class DifferenceOfGaussian(pipeline.ProcessObject):
    def __init__(self, inpt=None):
        super(DifferenceOfGaussian, self).__init__(inpt)

    def generateData(self):
        inpt = self.getInput().getData().astype(numpy.float32)

        gI1 = filters.gaussian_filter(inpt, 1.2, 0)
        gI2 = filters.gaussian_filter(inpt, 2.0, 0)
        output = gI2 - gI1
        self.getOutput().setData(output)


def test_particle_filter():
    import glob

    files = glob.glob("path")


if __name__ == "__main__":
    test_particle_filter()




