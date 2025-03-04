import numpy as np
import scipy as sp
import vtk
from vtkmodules.util import numpy_support

imageMath = vtk.vtkImageMathematics()

class Vertebra:
    def __init__(self,mask,volume):
        self.mask = mask
        self.volume = volume
        imageMath.SetOperationToMultiply()
        imageMath.SetInputData(0, volume)
        imageMath.SetInputData(1, mask)
        imageMath.Update()
        self.maskedVolume = imageMath.GetOutput()