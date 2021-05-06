#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:14:04 2021

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

savename = parameters["initial_savefile"]

domain_params = parameters["domain"]
domain = Domain(width=domain_params["width"], 
                length=domain_params["length"], 
                resolution=domain_params["resolution"])

mesh = domain.get_mesh()

# setup potential
potential_params = parameters["original_potential"]
nu = potential_params["nu"]

potential_expression = CylindricalPotential()
potential_expression.set_radius(potential_params["radius"])
potential_expression.set_center(potential_params["center"])


# V = H^2
H2 = FunctionSpace(mesh, "CG", 3)
L2 = FunctionSpace(mesh, "DG", 0)
# interpolate potential to function space
potential_interp = interpolate(potential_expression, L2)

# variational  formulation
u = TrialFunction(H2)
v = TestFunction(H2)
stiffness = dot(grad(v), grad(u)) * dx
mass = v * u * dx
potential_form = potential_interp * v * u * dx

# convert to scipy matrices
stiffness_matrix = convert_fenics_form_to_csr(stiffness)
mass_matrix = convert_fenics_form_to_csr(mass)
potential_matrix = convert_fenics_form_to_csr(potential_form)
potential_matrix.eliminate_zeros()

# solve eigenvalue problem
upper_index = 10
selected_index = 0
A = stiffness_matrix + nu * potential_matrix
M = mass_matrix
eigvals, eigvecs = eigh(A.toarray(), b=M.toarray(), 
                        subset_by_index=[0, upper_index])

initial_condition = Function(H2)
initial_condition.vector()[:] = np.power(eigvecs[:, selected_index], 2)

save_array = np.zeros(len(eigvecs[:, selected_index]) + 1)
save_array[:-1] = eigvecs[:, selected_index]
save_array[-1] = eigvals[selected_index]
np.savetxt(savename, save_array)

fig = plot(initial_condition)
plt.colorbar(fig)