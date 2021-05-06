#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:16:11 2021

@author: jerome
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eigh

from fenics import FunctionSpace, interpolate, dot, grad, dx
from fenics import TrialFunction, TestFunction, Function, plot

from potential import CylindricalPotential
from build_domain import Domain, parameters
from fenics_utils import convert_fenics_form_to_csr

from time_utils import crank_nicolson

loadname = parameters["initial_savefile"]
savename = parameters["time_evolution_savefile"]

domain_params = parameters["domain"]
domain = Domain(width=domain_params["width"], 
                length=domain_params["length"], 
                resolution=domain_params["resolution"])

mesh = domain.get_mesh()

# setup potentials
zero_potential_params = parameters["original_potential"]
nu_zero = zero_potential_params["nu"]

zero_potential_expression = CylindricalPotential()
zero_potential_expression.set_radius(zero_potential_params["radius"])
zero_potential_expression.set_center(zero_potential_params["center"])

new_potential_params = parameters["second_potential"]
nu_new = new_potential_params["nu"]

new_potential_expression = CylindricalPotential()
new_potential_expression.set_radius(new_potential_params["radius"])
new_potential_expression.set_center(new_potential_params["center"])

# V = H^2
H2 = FunctionSpace(mesh, "CG", 3)
L2 = FunctionSpace(mesh, "DG", 0)
# interpolate potential to function space
zero_potential_interp = interpolate(zero_potential_expression, L2)
new_potential_interp = interpolate(new_potential_expression, L2)

# variational  formulation
u = TrialFunction(H2)
v = TestFunction(H2)
stiffness = dot(grad(v), grad(u)) * dx
mass = v * u * dx
potential_form = (zero_potential_interp + new_potential_interp) * v * u * dx

# convert to scipy matrices
stiffness_matrix = convert_fenics_form_to_csr(stiffness)
mass_matrix = convert_fenics_form_to_csr(mass)
potential_matrix = convert_fenics_form_to_csr(potential_form)
potential_matrix.eliminate_zeros()

# get initial condition
loaded = np.loadtxt(loadname)
u_init = loaded[:-1] + 0j
eigval = loaded[-1]

time_stepping_params = parameters["time_stepping"]
tmax = time_stepping_params["tmax"]
nt = time_stepping_params["nt"]

time = np.linspace(0, tmax, nt)

u_sols = crank_nicolson(tmax, u_init, nt, 
                        mass_matrix, stiffness_matrix, potential_matrix, 
                        nu_zero)

prob_final = Function(H2)

prob_final.vector()[:] = np.abs(np.power(u_sols[:, -1], 2))

fig = plot(prob_final)
plt.colorbar(fig)


