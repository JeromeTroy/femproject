#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:55:47 2021

@author: jerome
"""

from fenics import *
from mshr import *

class Domain():
    
    def __init__(self, width=0, length=0, resolution=1):
        
        self.width = width
        self.length = length
        self.resolution = resolution
        
        self.x = self.width / 2
        self.y = self.length / 2

        self.build_domain()
        
    def build_domain(self):
        
        corners = [Point(-self.x, -self.y), 
                   Point(self.x, self.y)]
        self.domain = Rectangle(corners[0], corners[1])
        
        self.mesh = generate_mesh(self.domain, self.resolution)
        
    def get_mesh(self):
        return self.mesh
    
    def get_domain(self):
        return self.domain
        

parameters = {}
mesh_params = {"width": 5, 
               "length": 5, 
               "resolution": 20}
parameters["domain"] = mesh_params

original_potential = {"radius": 0.95,
                      "center": [-1, 0], 
                      "nu": 50}

parameters["original_potential"] = original_potential

second_potential = {"radius": 0.95,
                    "center": [1, 0],
                    "nu": 50}

parameters["second_potential"] = second_potential

parameters["initial_savefile"] = "initial_condition.txt"
parameters["time_evolution_savefile"] = "time_evolution.txt"

time_stepping = {"tmax": 40,
                 "nt": 400}
parameters["time_stepping"] = time_stepping