'''
Created on Feb 27, 2012

Color manipulation pipeline objects.

@author: bseastwo
'''

import cv2
import numpy

import imgutil
import pipeline

class Convert(pipeline.ProcessObject):
    '''
    Conversion to and from RGB and XYZ color spaces. This uses a numpy dot
    product operation (3x3 color twist), not the OpenCV color conversion
    functions. OpenCV seems to want only uint8 data for RGB to XYZ conversions.
    '''
    rgb2xyzmat = numpy.array([[0.412453, 0.357580, 0.180423],
                              [0.212671, 0.715160, 0.072169],
                              [0.019334, 0.119193, 0.950227]])
    xyz2rgbmat = numpy.linalg.inv(rgb2xyzmat)
    
    @staticmethod
    def rgb2xyz(input):
        input = input.copy()
        (h, w, c) = input.shape
        input.shape = (h * w, c)
        output = numpy.dot(input, Convert.rgb2xyzmat.transpose())
        output.shape = (h, w, c)
        return output

    @staticmethod
    def xyz2rgb(input):
        input = input.copy()
        (h, w, c) = input.shape
        input.shape = (h * w, c)
        output = numpy.dot(input, Convert.xyz2rgbmat.transpose())
        output.shape = (h, w, c)
        return output
        
    def __init__(self, input=None, mode=cv2.COLOR_RGB2XYZ):
        super(Convert, self).__init__(input)
        self.mode = mode
    
    def generateData(self):
        input = self.getInput().getData().copy()
        (h, w, c) = input.shape
        
        if c != 3:
            print "Convert:generateData(): wrong number of color channels:", c
            self.getOutput().setData(input)
            return
        
        if self.mode == cv2.COLOR_RGB2XYZ:
            output = Convert.rgb2xyz(input)
        elif self.mode == cv2.COLOR_XYZ2RGB:
            output = Convert.xyz2rgb(input)
        else:
            print "Convert:generateData(): conversion not supported:", self.mode
            output = input.copy()
        
        self.getOutput().setData(output)
    
    def getMode(self): return self.mode
    
    def setMode(self, mode):
        if mode in [cv2.COLOR_RGB2XYZ, cv2.COLOR_XYZ2RGB]:
            self.mode = mode
            self.modified()

class Grayscale(pipeline.ProcessObject):
    '''
    Converts a color image to grayscale using standard luminance weights
    [0.29, 0.59, 0.11].
    '''
    def __init__(self, input=None):
        super(Grayscale, self).__init__(input)
    
    def generateData(self):
        input = self.getInput().getData()
        
        if input.ndim == 3 and input.shape[2] == 3:
            output = 0.29 * input[..., 0] + 0.59 * input[..., 1] + 0.11 * input[..., 2]
            output = numpy.require(output, input.dtype)
        else:
            output = input.copy()
        
        self.getOutput().setData(output)

class Split(pipeline.ProcessObject):
    '''
    Splits an image into separate color channels.
    '''
    def __init__(self, input=None):
        super(Split, self).__init__(input, outputCount=3)
    
    def generateData(self):
        input = self.getInput().getData()
        
        # if the input is 2D, replicate it on all output channels
        if input.ndim < 3:
            for d in range(self.getOutputCount()):
                self.getOutput(d).setData(input.copy())
            return

        # if the input is at least 3D, make sure we have enough outputs
        elif input.shape[-1] != self.getOutputCount():
            self.setOutputCount(input.shape[-1])
            self.ensureOutput()
        
        # separate input into channels
        for d in range(input.shape[-1]):
            self.getOutput(d).setData(input[..., d])

def testColor():
    import source
    import sink
    
    video = source.CameraCV()
    colorXYZ = Convert(video.getOutput())
    colorRGB = Convert(colorXYZ.getOutput())
    colorRGB.setMode(cv2.COLOR_XYZ2RGB)
    splitRGB = Split(colorRGB.getOutput())
    splitXYZ = Split(colorXYZ.getOutput())
    grayscale = Grayscale(video.getOutput())
    
    displays = []
    displays.append(sink.DisplayCV(video.getOutput(), "video"))
    displays.append(sink.DisplayCV(splitRGB.getOutput(0), "R"))
    displays.append(sink.DisplayCV(splitRGB.getOutput(1), "G"))
    displays.append(sink.DisplayCV(splitRGB.getOutput(2), "B"))
    displays.append(sink.DisplayCV(splitXYZ.getOutput(0), "X"))
    displays.append(sink.DisplayCV(splitXYZ.getOutput(1), "Y"))
    displays.append(sink.DisplayCV(splitXYZ.getOutput(2), "Z"))
    displays.append(sink.DisplayCV(colorRGB.getOutput(), "RGB"))
    displays.append(sink.DisplayCV(grayscale.getOutput(), "gray"))
    
    while cv2.waitKey(10) != 27:
        video.updatePlayMode()
        for disp in displays:
            disp.update()

if __name__ == "__main__":
    testColor()