#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:51:04 2021

@author: jerome

Construction for the potential of the problem
Requires build_domain.
Constructs a potential which takes on value -1 inside the designated 
subdomains from build_domain, and 0 everywhere else
"""

import numpy as np
import matplotlib.pyplot as plt

from fenics import UserExpression, plot, FunctionSpace, interpolate

from build_domain import build_mesh, mark_tubes


class Potential(UserExpression):
    """
    Zero everywhere except inside a designated region
    point
    """

    def set_tubes(self, tubes):
        self.tubes = tubes        
        
    def eval_cell(self, value, x, cell):
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
        
        if self.tubes[cell.index] == 1:
            value[0] = 0
            
        else:
            value[0] = 1
        
        
    def value_shape(self):
        """
        Shape of the value output by this function

        Returns
        -------
        () : empty tuple
            indicates a scalar output.

        """
        return ()
        
class OutboundExtractor(Potential):
    """
    Extracts boundary term which is only on outbound sides
    Implements same methods as Potential, overrides evel_cell
    """
    
    def eval_cell(self, value, x, cell):
        """
        Evaluates function and assigns value
        1 on outbound boundary
        0 elsewhere

        Parameters
        ----------
        value : basically a pointer
            value to be assigned.
        x : array of length 2
            coordinate to be evaluated.
        cell : mesh cell
            cell in question.

        Returns
        -------
        None.

        """
        if self.tubes[cell.index] == 2:
            value[0] = 1
        
        else:
            value[0] = 0
            
     
if __name__ == "__main__":
    
    print("Building domain")
    
    nx = 20
    mesh = build_mesh(nx, nx)
    tubes = mark_tubes(mesh)
    
    print("Building potential")
    v = Potential(degree=1)
    v.set_tubes(tubes)

    
    plot(v, mesh=mesh)
    plt.show()
    
    L2 = FunctionSpace(mesh, "CG", 1)
    v_proj = interpolate(v, L2)
    
    plot(v_proj)
    plt.show()
    
    print(v_proj.vector()[:])
    


