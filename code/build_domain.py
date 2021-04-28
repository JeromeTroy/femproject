#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:55:47 2021

@author: jerome
"""

from fenics import *
from mshr import *

width = 6
length = 6

resolution = 20

x = width / 2
y = length / 2

corners = [Point(-x, -y), 
           Point(x, y)]

main_stage = Rectangle(corners[0], corners[1])

mesh = generate_mesh(main_stage, resolution)

