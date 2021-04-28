#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:30:22 2021

@author: jerome
"""

from fenics import as_backend_type, assemble
from scipy.sparse import csr_matrix

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
    
    petsc_matrix = as_backend_type(assemble(form)).mat()
    
    sparse_matrix = csr_matrix(petsc_matrix.getValuesCSR()[::-1], 
                               shape=petsc_matrix.size)
    
    return sparse_matrix
    