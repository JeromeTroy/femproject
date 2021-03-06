#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:30:22 2021

@author: jerome

Helper functions for navigating FEniCS
"""

from fenics import as_backend_type, assemble, Function, dx, XDMFFile
from scipy.sparse import csr_matrix
import numpy as np

from potential import Potential

def convert_fenics_form_to_csr(form):
    """
    Convert a fenics variational form to a sparse matrix (csr)

    Parameters
    ----------
    form : fenics variational form
        input form to convert to a matrix.

    Returns
    -------
    sparse_matrix : scipy csr_matrix
        matrix form of variational form for FEM.

    """
    
    # assemble the ufl form and convert it to a petsc sparse matrix
    petsc_matrix = as_backend_type(assemble(form)).mat()
    
    # convert the petsc sparse matrix to a scipy.sparse matrix
    sparse_matrix = csr_matrix(petsc_matrix.getValuesCSR()[::-1], 
                               shape=petsc_matrix.size)
    
    return sparse_matrix
    
def get_boundary_indices(boundary_condition, fun_space):
    """
    Gather the indices for the boundary

    Parameters
    ----------
    boundary_condition : Fenics DirichletBC object
        dirichlet boundary condtion.
    fun_space : Fenics FunctionSpace object
        function space on which the problem is defined.

    Returns
    -------
    interior_indices : array of ints >= 0
        matrix indices corresponding to interior nodes (not on boundary).
    boundary_indices : array of ints >= 0
        matrix indices corresponding to boundary nodes.

    """
    # get the boundary indices
    boundary_degs_of_freedom = boundary_condition.get_boundary_values().keys()
    boundary_indices = list(boundary_degs_of_freedom)
    
    # get all possible indices
    first, last = fun_space.dofmap().ownership_range()
    all_degs_of_freedom = range(last - first)
    
    # chop out boundary indices to get the interior indices
    interior_indices = list(set(all_degs_of_freedom) - 
                            set(boundary_degs_of_freedom))
    
    return interior_indices, boundary_indices
    
def get_coordinates_on_boundary(fun_space, boundary_indices):
    """
    Extract coordinates for the boundary

    Parameters
    ----------
    fun_space : Fenics FunctionSpace object
        function space on which the problem is defined.
    boundary_indices : array of ints >= 0
        indices cooresponding to matrix entries for boundary.

    Returns
    -------
    boundary_coordinates : n x 2 array of floats
        boundary_coordinates[j, :] = [x_j, y_j] on the boundary.

    """
    
    # all coordinates for the problem
    all_coordinates = fun_space.tabulate_dof_coordinates()
    
    # index on the boundary
    boundary_coordinates = all_coordinates[boundary_indices, :]
    
    return boundary_coordinates


def integrate_on_domain(solution, fun_space, cutoff=None):
    """
    Integrate a solution (on a subdomain)

    Parameters
    ----------
    solution : vector of floats
        solution vector.
    fun_space : FEniCS FunctionSpace
        function space on which the solution is defined.
    cutoff : FEniCS Expression, optional
        Indicator function for subdomain on which integration 
        should take place. 
        The default is None, in which case the integration takes place
        on the whole domain.

    Returns
    -------
    integral : float
        integral value.

    """
    # construct a fenics function and assign values
    u = Function(fun_space)
    u.vector()[:] = solution
    
    # initialize
    integral = None
    
    # integration
    if cutoff is None:
        # no domain specified - default to entire domain
        integral = assemble(u * dx)
        
    else:
        # build a cutoff for that domain
        
        # integrate
        integral = assemble(u * cutoff * dx)
        
    return integral
    
    
def build_xdmf_file(fname):
    """
    Build an XDMF File

    Parameters
    ----------
    fname : string
        file name.

    Returns
    -------
    xdmf : XDMFFIle
        output file.

    """
    xdmf = XDMFFile(fname)
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.parameters["rewrite_function_mesh"] = False

    return xdmf

def write_soln_to_file(xdmf, solns, times, empty_function, save_freq=1):
    """
    Write solution to xdmf

    Parameters
    ----------
    xdmf : XDMFFile
        output file.
    solns : array of floats (nx  x nt)
        solution at each node in x and time.
    times : array of floats
        time nodes.
    empty_function : FEniCS Function
        function template to write data to
    save_freq : int, optional
        frequency to save. The default is 1 (save each time step).

    Returns
    -------
    xdmf : XDMFFile
        same XDMFFile, just modified.

    """
    
    for j in range(0, len(times), save_freq):
        # assign to function which is used in XDMF
        empty_function.vector()[:] = np.power(np.abs(solns[:, j]), 2)
        xdmf.write(empty_function, times[j])

    return xdmf

if __name__ == "__main__":
    
    print("Testing")
    
    from fenics import UnitSquareMesh, FunctionSpace, DirichletBC, Constant, \
        TrialFunction, TestFunction, dx, plot
    mesh = UnitSquareMesh(4, 4)

    import matplotlib.pyplot as plt
    import numpy as np
    
    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, Constant(0), 'on_boundary')
    indices, bound_indices = get_boundary_indices(bc, V)
    
    v = TestFunction(V)
    u = TrialFunction(V)
    M = convert_fenics_form_to_csr(u * v * dx)
    
    int_submat = M[indices, :][:, indices]
    
    print("interior submatrix")
    print(int_submat)
    
    print("Tabulating coordinates on boundary")
    coords_from_V = V.tabulate_dof_coordinates()
    print(coords_from_V)
    
    boundary_coords = coords_from_V[bound_indices, :]
    print(boundary_coords)
    
    plot(mesh)
    plt.plot(boundary_coords[:, 0], boundary_coords[:, 1], "*")
    plt.show()