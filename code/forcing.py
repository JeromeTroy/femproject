#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:14:04 2021

@author: jerome

Build the forcing function on the input to a tube.
Requires build_domain.
The forcing function takes value zero at the sides of the tube,
and has the lowest allowable mode in x.
It oscillates in time at a designated frequency.
It only occurs at the input of one tube, hence a constant
to determine if the position given is valid for the forcing
"""

from fenics import UserExpression
import numpy as np
import matplotlib.pyplot as plt

from build_domain import domain_params

# [[xmin, xmax], [ymin, ymax]]
ACCEPTABLE_INPUT = [
    [domain_params["tube_distance_from_boundary"] - domain_params["tol"], 
     domain_params["tube_distance_from_boundary"] + 
         domain_params["tube_diameter"] + domain_params["tol"]], 
    [-domain_params["tol"], domain_params["tol"]]]

SIN_X_FREQ = np.pi / domain_params["tube_diameter"]

# how quickly f oscillates in time
oscillating_frequency = 20

# TODO: make a 2 x 1 vector
def forcing(x, t, momentum):
    """
    Forcing function for the problem

    Parameters
    ----------
    x : array of length 2 of floats
        x = [x0, x1] = [x, y] is the position.
    t : scalar float
        time.
    momentum : scalar float
        value for momentum of wave

    Returns
    -------
    result : scalar float
        f(x, t; p) value.

    """
    # default to zero
    result = 0
    
    # check x coordinate
    if (ACCEPTABLE_INPUT[0][0] <= x[0]) and (x[0] <= ACCEPTABLE_INPUT[0][1]):
        # check y coordinate
        if (ACCEPTABLE_INPUT[1][0] <= x[1]) and (
                x[1] <= ACCEPTABLE_INPUT[1][1]):
            # both in acceptable range, compute result
            
            # shift x to left side of input tube
            shifted_x = x[0] - ACCEPTABLE_INPUT[0][0]
            
            # sinusoid curve
            result = np.sin(SIN_X_FREQ * shifted_x)
            
    # apply time scaling, sinusoid curve
    result *= np.sin(2 * np.pi * oscillating_frequency * t)
    
    # apply momentum
    result *= np.exp(1j * momentum)
    if t > 1 / oscillating_frequency:
        result = 0
    
    return result
            
            
class Forcer(UserExpression):
    
    def set_time(self, time):
        self.time = time
        
    def set_function(self, f):
        self.f = f
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
        
        value[0] = self.f(x, self.time)[0]
        value[1] = self.f(x, self.time)[1]
        
        
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
    
    print("Acceptable inputs")
    print(ACCEPTABLE_INPUT)
    x_fine = np.linspace(
        ACCEPTABLE_INPUT[0][0], ACCEPTABLE_INPUT[0][1], 100)
    
    print(forcing([1.5, 0], 0.25))   
    
    coords = np.zeros([len(x_fine), 2])
    coords[:, 0] = x_fine
    
    plt.plot(x_fine, 
             np.array(list(map(lambda x: forcing(x, 0.25), list(coords)))))
    plt.show()
    
    plt.plot(x_fine, 
             np.array(list(map(lambda x: forcing(x, 0.5), list(coords)))))
    plt.show()
    plt.plot(x_fine, 
             np.array(list(map(lambda x: forcing(x, 0.75), list(coords)))))
    plt.show()
    
    