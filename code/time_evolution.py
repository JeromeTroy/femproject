#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:16:11 2021

@author: jerome
"""

import numpy as np
import matplotlib.pyplot as plt
from fenics import FunctionSpace, FacetNormal, interpolate, \
    TrialFunction, TestFunction, dx, ds, \
        Function, DirichletBC, dot, grad, Constant, plot

from scipy.sparse.linalg import spsolve
        
from fenics_utils import convert_fenics_form_to_csr, get_boundary_indices, \
    get_coordinates_on_boundary, integrate_on_domain, \
        build_xdmf_file, write_soln_to_file
from build_domain import build_mesh, mark_tubes, OutputBoundary, \
    InputBoundary, DirichletBoundary
from potential import Potential, OutputExtractor, InputExtractor, \
    OutboundExtractor
from forcing import forcing


# time stepping parameters
tmax = 5
nt = 1000
dt = tmax / nt

# potential strength
nu = 250

# momentum at input
momentum = 500

output_filename = "pdf_x_t.xdmf"

# construct domains and mesh
res = 100
mesh = build_mesh(res, res)
tubes = mark_tubes(mesh)

# construct potential and mark values
V_expr = Potential(degree=1)
V_expr.set_tubes(tubes)

# outbound extracter
# WARNING!
# Output boundary condition is not a "Boundary Condition"
# but is incorperated into the PDE using a conservation law.
Out_expr = OutboundExtractor()
Out_expr.set_tubes(tubes)

Output_expr = OutputExtractor()
Output_expr.set_tubes(tubes)
Input_expr = InputExtractor()
Input_expr.set_tubes(tubes)


input_boundary = InputBoundary()
dirichlet_boundary = DirichletBoundary()
output_boundary = OutputBoundary()

# construct function spaces
L2 = FunctionSpace(mesh, "DG", 0)
H1 = FunctionSpace(mesh, "CG", 1)

input_indicator = interpolate(Input_expr, L2)
output_indicator = interpolate(Output_expr, L2)

# template boundary condition
# constant is irrelavant, only used for indices
forcing_bc = DirichletBC(H1, Constant(0.0), input_boundary)

# zero dirichlet bc
zero_bc = DirichletBC(H1, Constant(0.0), dirichlet_boundary)

# outbound_boundary
out_bc = DirichletBC(H1, Constant(0.0), output_boundary)

# unit normal 
normal = FacetNormal(mesh)

# place outbound extractor and potential in L2
V = interpolate(V_expr, L2)
Out = interpolate(Out_expr, L2)

# trial and test functions
Phi = TrialFunction(H1)
Psi = TestFunction(H1)

# variational formulation

# term by term
# these will be used to solve for the next timestep
mass = Phi * Psi * dx
stiffness = dot(grad(Phi), grad(Psi)) * dx
potential = Phi * V * Psi * dx
dir_boundary = Phi * Psi * Out * ds

# convert each form into a sparse csr matrix
mass_mat = convert_fenics_form_to_csr(mass)
stiffness_mat = convert_fenics_form_to_csr(stiffness)
potential_mat = convert_fenics_form_to_csr(potential)
dir_boundary_mat = convert_fenics_form_to_csr(dir_boundary)

# construct left and right hand side operators
LHS_Operator = 1j * (mass_mat - dir_boundary_mat) - \
    0.5 * dt * (stiffness_mat + nu * potential_mat)
RHS_Operator = 1j * (mass_mat - dir_boundary_mat) + \
    0.5 * dt * (stiffness_mat + nu * potential_mat)

# indices for boundaries
not_zero_bc_indices, zero_bc_indices = get_boundary_indices(zero_bc, H1)
not_input_bc_indices, input_bc_indices = get_boundary_indices(forcing_bc, H1)

# all the free indices
free_indices = list(set(not_zero_bc_indices).intersection(
    set(not_input_bc_indices)))


LHS = LHS_Operator[free_indices, :][:, free_indices]

# extract RHS operators indices corresponding to input in free indices
# this comes from the free indices of the last timestep,
# and the input boundary indices
RHS_mat = RHS_Operator[free_indices, :][:, not_zero_bc_indices]


input_boundary_coords = get_coordinates_on_boundary(H1, input_bc_indices)

# initial condition, complex vector, zero
nx = LHS_Operator.shape[1]
Psi_init = 1j * np.zeros(nx)

Psi_solns = 1j * np.zeros([nx, nt])
Psi_solns[:, 0] = Psi_init

# input forcing, time = 0
force = np.array(list(map(lambda x: forcing(x, 0 * dt, momentum),
                          list(input_boundary_coords))))
    
# storage for transmission coefficient
transmission_coefficients = np.zeros(nt)

plotting_freq = 10
# time stepping
for time_index in range(nt - 1):
    
    # rhs vector, Psi_solns includes previous force
    rhs = RHS_mat @ Psi_solns[not_zero_bc_indices, time_index]
     
    # solve problem
    Psi_next_free = spsolve(LHS, rhs)
    
    # store solution
    Psi_solns[free_indices, time_index + 1] = Psi_next_free
    
    # next force (update)
    # kept as update for next iteration
    force = np.array(list(map(
        lambda x: forcing(x, (time_index + 1) * dt, momentum), 
        list(input_boundary_coords))))
    
    if time_index % plotting_freq == 0 and time_index > 0:
        # plot
        pdf = Function(H1)
        pdf.vector()[:] = np.abs(Psi_solns[:, time_index + 1])
        #pdf.vector()[:] /= assemble(pdf * dx)
        fig = plot(pdf, vmin=0, vmax=1)
        plt.colorbar(fig)
        plt.show()
        
    Psi_solns[input_bc_indices, time_index + 1] = force
    
    Psi_transmitted = integrate_on_domain(
        np.power(np.abs(Psi_solns[:, time_index + 1]), 2), H1, 
        cutoff=output_indicator)
    Psi_not_transmitted = integrate_on_domain(
        np.power(np.abs(Psi_solns[:, time_index + 1]), 2), H1, 
        cutoff=input_indicator)
    
    transmission_coefficients[time_index + 1] = \
        Psi_transmitted / Psi_not_transmitted
  
# write data to file
xdmf = build_xdmf_file(output_filename)
# make sure data written is probability density function (pdf)
# regular wave function is messy and is difficult to interpret
# build empty function for saving
empty_function = Function(H1)
write_soln_to_file(xdmf, Psi_solns, 
                   np.linspace(0, tmax, nt), empty_function)
        
np.savetxt("transmission_coefficients.txt", transmission_coefficients)
        
