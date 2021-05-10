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
        Function, DirichletBC, dot, grad, Constant, plot, assemble

from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat
        
from fenics_utils import convert_fenics_form_to_csr, get_boundary_indices, \
    get_coordinates_on_boundary
from build_domain import build_mesh, mark_tubes, OutputBoundary, \
    InputBoundary, DirichletBoundary
from potential import Potential, OutboundExtractor
from forcing import forcing


# time stepping parameters
tmax = 0.75
nt = 100
dt = tmax / nt

# potential strength
nu = 500

# momentum at input
momentum = 2000

# construct domains and mesh
res = 40
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

input_boundary = InputBoundary()
dirichlet_boundary = DirichletBoundary()
output_boundary = OutputBoundary()

# construct function spaces
L2 = FunctionSpace(mesh, "DG", 0)
H4 = FunctionSpace(mesh, "CG", 1)

# template boundary condition
# constant is irrelavant, only used for indices
forcing_bc = DirichletBC(H4, Constant(0.0), input_boundary)

# zero dirichlet bc
zero_bc = DirichletBC(H4, Constant(0.0), dirichlet_boundary)

# outbound_boundary
out_bc = DirichletBC(H4, Constant(0.0), output_boundary)

# unit normal 
normal = FacetNormal(mesh)

# place outbound extractor and potential in L2
V = interpolate(V_expr, L2)
Out = interpolate(Out_expr, L2)

# trial and test functions
Phi = TrialFunction(H4)
Psi = TestFunction(H4)

# variational formulation

# term by term
# these will be used to solve for the next timestep
mass = Phi * Psi * dx
stiffness = dot(grad(Phi), grad(Psi)) * dx
potential = Phi * V * Psi * dx
dir_boundary = Phi * dot(grad(Psi), normal) * Out * ds
neu_boundary = Phi * dot(grad(Psi), normal) * Out * ds

# convert each form into a sparse csr matrix
mass_mat = convert_fenics_form_to_csr(mass)
stiffness_mat = convert_fenics_form_to_csr(stiffness)
potential_mat = convert_fenics_form_to_csr(potential)
dir_boundary_mat = convert_fenics_form_to_csr(dir_boundary)
neu_boundary_mat = convert_fenics_form_to_csr(neu_boundary)

# construct left and right hand side operators
LHS_Operator = 1j * mass_mat - 0.5 * dt * stiffness_mat - \
    0.5 * nu * dt * potential_mat
        #+ 0.5 * dt * dir_boundary_mat
RHS_Operator = 1j * mass_mat + 0.5 * dt * stiffness_mat + \
    0.5 * nu * dt * potential_mat
        #- 0.5 * dt * dir_boundary_mat

# indices for boundaries
not_zero_bc_indices, zero_bc_indices = get_boundary_indices(zero_bc, H4)
not_input_bc_indices, input_bc_indices = get_boundary_indices(forcing_bc, H4)
not_out_bc_indices, output_bc_indices = get_boundary_indices(out_bc, H4)

# technically all the free indices
tech_free_indices = list(set(not_zero_bc_indices).intersection(
    set(not_input_bc_indices)))

# schrodinger equation indices
free_indices = list(set(not_zero_bc_indices).intersection(
    set(not_input_bc_indices).intersection(set(not_out_bc_indices))))

# two coupled PDEs
# one is the interior (schrodinger equation)
# other is the boundary condition (advection equation)
LHS_free = LHS_Operator[free_indices, :][:, free_indices]
LHS_out = 0.5 * dt * dir_boundary_mat[output_bc_indices, :][:, output_bc_indices]
LHS_free_to_out = LHS_Operator[output_bc_indices, :][:, free_indices]
LHS = bmat([[LHS_free, None], [LHS_free_to_out, LHS_out]])

# extract RHS operators indices corresponding to input in free indices
# this comes from the free indices of the last timestep,
# and the input boundary indices
RHS_free_to_free = RHS_Operator[free_indices, :][:, free_indices]
RHS_input_to_free = RHS_Operator[free_indices, :][:, input_bc_indices]
RHS_out_to_free = RHS_Operator[free_indices, :][:, output_bc_indices]
RHS_out_to_out = RHS_Operator[output_bc_indices, :][:, output_bc_indices]


input_boundary_coords = get_coordinates_on_boundary(H4, input_bc_indices)

# initial condition, complex vector, zero
nx = LHS_Operator.shape[1]
Psi_init = 1j * np.zeros(nx)

Psi_solns = 1j * np.zeros([nx, nt])
Psi_solns[:, 0] = Psi_init

# input forcing, time = 0
force = np.array(list(map(lambda x: forcing(x, 0 * dt, momentum),
                          list(input_boundary_coords))))
    

plotting_freq = 10
# time stepping
for time_index in range(nt - 1):
    
    # rhs vector
    rhs = RHS_free_to_free @ Psi_solns[free_indices, time_index] + \
        RHS_input_to_free @ force
    
    # solve problem
    Psi_next_free = spsolve(LHS_free, rhs)
    
    # store solution
    Psi_solns[free_indices, time_index + 1] = Psi_next_free
    
    # next force (update)
    # kept as update for next iteration
    force = np.array(list(map(
        lambda x: forcing(x, (time_index + 1) * dt, momentum), 
        list(input_boundary_coords))))
    
    if time_index % plotting_freq == 0 and time_index > 0:
        # plot
        pdf = Function(H4)
        pdf.vector()[:] = np.abs(Psi_solns[:, time_index + 1])
        #pdf.vector()[:] /= assemble(pdf * dx)
        fig = plot(pdf, vmin=0, vmax=1)
        plt.colorbar(fig)
        plt.show()
        
    Psi_solns[input_bc_indices, time_index + 1] = force
    
    
        
        