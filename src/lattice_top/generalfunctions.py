import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from matplotlib import cm
import time
import math as m


"""
This file contains general use functions
"""


def round_sig(x, sig=2):
    return round(x, sig - int(m.floor(m.log10(abs(x)))) - 1)


def list_repeated_energies(energies, threshold=1e-8, shift=0):
    """
    This code takes a list of energies (has to be in ascending order or it won't work) and returns
    the list of each index of degenerate states!
    :param shift:
    :param energies: real list of energies
    :param threshold: threshold difference at which two energies are considered equal
    :return: a list of tuples of every repeated set of eigenvalues
    """
    if (all(energies[i] <= energies[i + 1] for i in range(len(energies) - 1))) is False:
        raise Exception(
            'You need to pass a list of energies in ascending order')

    energies_last = -np.inf
    degen_list = []
    temp_degen_list = [0]
    degen_last = False

    for i, energy_current in enumerate(energies):

        if abs(energy_current - energies_last) <= threshold:
            degen_current = True
        else:
            degen_current = False

        if degen_current is True and i != energies.__len__() - 1:
            temp_degen_list.append(i + shift)
        elif degen_current is True and i == energies.__len__() - 1:
            temp_degen_list.append(i + shift)
            degen_list.append(tuple(temp_degen_list))
        elif degen_current is False and degen_last is True:
            degen_list.append(tuple(temp_degen_list))
            temp_degen_list = [i + shift]
        else:
            temp_degen_list = [i + shift]

        degen_last = degen_current
        energies_last = energy_current

    return degen_list


def pick_right_states_extended(H: np.ndarray, perturbations: list, threshold=1e-8, return_list=False):
    """
    This code takes the eigenstates of H. Then it looks at all the degenerate
    states of H and uses each matrix in the list 'perturbations' to lift
    the degeneracy.
    :param H: Initial Hamiltonian to diagonalise
    :param perturbations: A list of perturbing Hamiltonians in the order that you want them applied to H
    :param threshold: Threshold at which two energies are considered degenerate
    :param return_list: do you want to be given a list of the remaining degeneracies?!
    :return: energies, states, and a list of remaining degeneracies!
    """

    num_perturbations = perturbations.__len__()

    # solve the states of the hamiltonian
    energies, states = la.eigh(H)
    degen_list = list_repeated_energies(energies, threshold=threshold)

    list_out = [degen_list]

    if degen_list == []:
        # the states are perfect - there is already no degeneracy :)
        if return_list == True:
            return energies, states, list_out
        else:
            return energies, states

    for i, pert in enumerate(perturbations):
        # for each successive perturbation this keeps track of the degeneracy that remains!
        remaining_degeneracy = []

        for j, deg_set in enumerate(degen_list):
            # for each degenerate set of states

            subspace = states[:, deg_set]
            hdot_on_subspace = subspace.conj().T @ pert @ subspace
            hdot_on_subspace = 0.5 * \
                (hdot_on_subspace + hdot_on_subspace.conj().T)

            # find the eigenvectors the perturbation on the degenerate subspace
            e, v = la.eigh(hdot_on_subspace)

            subdegen_shifted = list_repeated_energies(e, shift=deg_set[0])
            subdegen = list_repeated_energies(e)

            if subdegen_shifted == [deg_set]:
                # all the states are still degenerate so don't do anything
                remaining_degeneracy.extend(subdegen_shifted)
            else:
                # now if there is some degeneracy that can be lifted
                states[:, deg_set] = subspace @ v
                remaining_degeneracy.extend(subdegen_shifted)

        degen_list = remaining_degeneracy

        list_out.append(remaining_degeneracy)

    if return_list == True:
        return energies, states, list_out
    else:
        return energies, states


def fermi_occupation(energy: np.ndarray, beta: float, fermi_energy: float):
    # check if the value is too big for exp - if it is we jst use a step function!

    if beta == np.inf:
        return (1. * (energy <= fermi_energy))
    else:
        non_overload_array = abs(beta * (energy-fermi_energy)) <= 700

        a = 1 / (np.exp(beta * (energy-fermi_energy), where=non_overload_array) +
                 1) * non_overload_array
        b = (beta * (energy-fermi_energy) <= 0) * (1 - non_overload_array)

        return a + b
