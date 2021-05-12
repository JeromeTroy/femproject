#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:55:47 2021

@author: jerome

Construct the domain for the problem
This is a set of two tubes, on either side of a central ring.
The domain itself is a rectangle.
The tubes indicate where a potential is applied, and are stored
as subdomains
"""

from fenics import SubDomain, Point, near, MeshFunction, plot, RectangleMesh
#from mshr import Rectangle, generate_mesh
import numpy as np
import matplotlib.pyplot as plt

# tolerances
TOL = 1e-14

# paramters
tube_diameter = 1
ring_inner_radius = 2
tube_distance_from_boundary = 1

domain_width = 10
domain_length = 10

domain_params = {"tol": TOL, 
                 "tube_diameter": tube_diameter,
                 "ring_inner_radius": ring_inner_radius, 
                 "tube_distance_from_boundary": tube_distance_from_boundary,
                 "domain_width": domain_width,
                 "domain_length": domain_length}

# build domain
origin = Point(0, 0)
far_corner = Point(domain_length, domain_width)

def build_mesh(nx, ny):
    
    if np.mean([nx, ny]) < 20:
        print("WARNING! Mesh resolutions below 20 do not resolve geometry!")
    
    return RectangleMesh(origin, far_corner, nx, ny, "crossed")

class InputTube(SubDomain):
    
    def inside(self, x, on_boundary):
        
        return (tube_distance_from_boundary <= x[0]) and (
            x[0] <= tube_distance_from_boundary + tube_diameter)

input_tube = InputTube()
    
class OutputTube(SubDomain):
    
    def inside(self, x, on_boundary):
        
        return (domain_length - tube_distance_from_boundary - tube_diameter <= 
                x[0]) and (x[0] <= domain_length - tube_distance_from_boundary)
    
output_tube = OutputTube()

class RingResonator(SubDomain):
    
    def inside(self, x, on_boundary):
        
        x_from_center = x[0] - 0.5 * domain_length
        y_from_center = x[1] - 0.5 * domain_width 
        radius = np.sqrt(np.power(x_from_center, 2) + 
                         np.power(y_from_center, 2))
        
        return (ring_inner_radius <= radius) and \
            (radius <= ring_inner_radius + tube_diameter)
            
ring_resonator = RingResonator()

class InputBoundary(SubDomain):
    
    def inside(self, x, on_boundary):
        
        return near(x[1], 0, TOL) and input_tube.inside(x, on_boundary) and \
            on_boundary
    
class OutputBoundary(SubDomain):
    
    def inside(self, x, on_boundary):
        
        on_input_outbound = near(x[1], domain_width, TOL) and \
            input_tube.inside(x, on_boundary) and on_boundary
            
        on_output = output_tube.inside(x, on_boundary) and on_boundary
        
        return on_input_outbound or on_output
   
outbound = OutputBoundary()

class DirichletBoundary(SubDomain):
    
    def inside(self, x, on_boundary):
        
        return on_boundary and (not input_tube.inside(x, on_boundary)) and \
            (not output_tube.inside(x, on_boundary))
            
  
def mark_tubes(mesh):
    
    tubes = MeshFunction("size_t", mesh, 2)
    input_tube.mark(tubes, 1)
    output_tube.mark(tubes, 2)
    ring_resonator.mark(tubes, 3)
    outbound.mark(tubes, 4)
        
    return tubes


if __name__ == "__main__":
    
    print("Building test mesh")
    
    nx = 100
    mesh = build_mesh(nx, nx)
    
    print("Marking tubes")
    tubes = mark_tubes(mesh)    

    print("Plotting")
    plot(mesh)
    plt.savefig("mesh.pdf")
    plt.show()
    
    plot(tubes)
    plt.show()    
