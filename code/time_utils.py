#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:22:55 2021

@author: jerome
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve

def ode(t, y, mass, stiffness, potential, nu):
    """
    ODE function for time evolution

    Parameters
    ----------
    t : float > 0
        time.
    y : array of length n of floats
        solution values y(t).
    mass : sparse matrix
        mass matrix for FEM.
    stiffness : sparse matrix
        stiffness matrix for FEM.
    potential : sparse matrix
        potential term matrix for FEM.
    nu : float > 0
        scaling for potential.

    Returns
    -------
    res : array of length n of floats
        res = dy / dt.

    """
    rhs = (stiffness + nu * potential) @ y
    res = -1j * spsolve(mass, rhs)
    
    return res

def crank_nicolson(tmax, y_init, nt, mass, stiffness, potential, nu):
    """
    Crank Nicolson method for first order ODE systems

    Parameters
    ----------
    tmax : float > 0
        maximum time.
    y_init : array of length n of floats
        initial condition.
    nt : int > 0
        number of time nodes.
    mass : sparse matrix
        mass matrix for FEM.
    stiffness : sparse matrix
        stiffness matrix for FEM.
    potential : sparse matrix
        potential matrix for FEM.
    nu : float > 0
        scaling for potential.

    Returns
    -------
    y_vals : len(y_init) x nt array
        y[:, j] = y(t^j).

    """
    dt = tmax / nt
    y_vals = np.zeros([len(y_init), nt])
    y_vals[:, 0] = y_init
    
    A = 1j * mass - 0.5 * dt * (stiffness + nu * potential)
    B = 1j * mass + 0.5 * dt * (stiffness + nu * potential)
    
    
    for j in range(nt - 1):
        
        y_vals[:, j + 1] = spsolve(A, B @ y_vals[:, j])
        
    return y_vals
    