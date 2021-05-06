#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:51:04 2021

@author: jerome
"""

import numpy as np

from fenics import UserExpression


class CylindricalPotential(UserExpression):
    """
    Zero everywhere except inside a circle of radius R around a designated
    point
    """
    
    def __init__(self):
        super().__init__()
        
        self.center = np.zeros(2)
        self.radius = 1
        
    def set_center(self, center):
        self.center = center
        
    def set_radius(self, radius):
        self.radius = radius
        
    def eval(self, value, x):
        """
        Evaluate function and assign its value to 'value' variable
        Evaluates to -1 inside the circle and 0 elsewhere

        Parameters
        ----------
        value : basically a pointer
            output value.
        x : array of length 2
            x = [x, y] are coordinates for point evaluation.

        Returns
        -------
        None.

        """
        
        shifted_x = x - self.center
        r = np.linalg.norm(shifted_x)
        
        if r < self.radius:
            value[0] = -1.0
            
        else:
            value[0] = 0.0
        
        
    def value_shape(self):
        """
        Shape of the value output by this function

        Returns
        -------
        () : empty tuple
            indicates a scalar output.

        """
        return ()
        
if __name__ == "__main__":
    print("I am a test")


