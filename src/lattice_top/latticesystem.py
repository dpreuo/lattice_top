from .generalfunctions import *
from .plotting_functions import *

###############################################################
############ functions for solving the hamiltonian ############
###############################################################


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


##############################################################
########## parent class for lattice quantum systems ##########
##############################################################


class Lattice_System():
    # root stuff
    def __init__(self, lengths):
        """
        this initialises the system - we are just gonna set up the system size and the boundary conditions here
        :param lengths: length in the x and y direction
        """
        self._lengths = lengths

        # These must be set by you initialiser - the positions of each cell
        self._x_list = None
        self._y_list = None

        # These after you initialise the Hamiltonian
        self._edges = None
        self._hamiltonian = None
        self._x_dif_hamiltonian = None
        self._y_dif_hamiltonian = None
        self._n_sites = None

        # then once you solve the Hamiltonian - these are used in a lot of scripts for different quantities.
        self._fermi_energy = None
        self._energies = None
        self._states = None
        self._degenerate_list = None
        self._projector = None

        # note this is a matrix of the difference of energy eigenvalues
        self._E_dif = None

        # derivative of hamiltonian wrt magnetic vector potential in energy basis
        self._jx_energy_basis = None
        self._jy_energy_basis = None

        # finally the local markers
        self._bott_index = None
        self._chern_marker = None
        self._adiabatic_bott_index = None
        self._local_kubo = None

    def __str__(self):

        # this tells you what bits of the object have been initialised and what the system size is

        str_out = f"System size: {self._lengths} \nEdges: {self._edges}\nInitialised:\n"

        for item in vars(self):
            # str_out += item + ' : ' + str(vars(self)[item] is not None) + '\n'
            str_out += str(vars(self)[item] is not None) + ' : ' + item + '\n'

        return str_out

    #########################################################
    ################# initialise the system #################
    #########################################################

    def solve_Hamiltonian(self, fermi_energy=0):
        """
        This code solves the Hamiltonian and sets the states projectors and projected derivatives
        :return: none
        Internally sets:

        _energies
        _states
        _degenerate_list
        _projector
        _jx_energy_basis
        _jy_energy_basis
        _E_dif
        """
        t1 = time.time()

        self._energies, self._states, self._degenerate_list = \
            pick_right_states_extended(self._hamiltonian,
                                       [self._x_dif_hamiltonian, self._y_dif_hamiltonian], return_list=True)

        self._projector = self._states @ np.diag(
            self._energies <= fermi_energy) @ np.conj(self._states).T

        self._jx_energy_basis = self._states.conj().T \
            @ self._x_dif_hamiltonian @ self._states
        self._jy_energy_basis = self._states.conj().T \
            @ self._y_dif_hamiltonian @ self._states

        self._fermi_energy = fermi_energy
        self._E_dif = self._energies[:, np.newaxis] - self._energies
        self._E_dif = -self._E_dif + 1e20 * (self._E_dif.__abs__() <= 1e-8)
        dt = time.time() - t1
        print(f'Hamiltonian solved - This took {round_sig(dt)} seconds')

    def reset_fermi_level(self, fermi_energy):
        self._projector = self._states @ np.diag(
            self._energies <= fermi_energy) @ np.conj(self._states).T

    ##########################################################
    ############## calculate various indicators ##############
    ##########################################################

    def find_bott_index(self):
        t1 = time.time()
        Lx = self._lengths[0]
        Ly = self._lengths[1]

        delta_x = 2 * np.pi / Lx
        delta_y = 2 * np.pi / Ly

        X_exp = np.exp(1j * self._x_list * delta_x)
        Y_exp = np.exp(1j * self._y_list * delta_y)
        X_exp_star = np.exp(-1j * self._x_list * delta_x)
        Y_exp_star = np.exp(-1j * self._y_list * delta_y)

        M = self._projector * X_exp @ self._projector * Y_exp @ \
            self._projector * X_exp_star @ self._projector * \
            Y_exp_star @ self._projector + \
            (np.eye(self._n_sites) - self._projector)

        M = la.logm(M)

        out = np.diag(M) * Lx * Ly

        bott_index_tr = np.imag(out[::2] + out[1::2]) / (2 * np.pi)
        self._bott_index = bott_index_tr

        dt = time.time() - t1
        print(f'Bott Index found - This took {round_sig(dt)} seconds')

    def find_adiabatic_bott_index(self):

        t1 = time.time()
        Lx = self._lengths[0]
        Ly = self._lengths[1]

        P_vector = (self._energies <= 0)
        Q_vector = (self._energies > 0)

        Mx = P_vector[:, np.newaxis] * self._jx_energy_basis * \
            Q_vector / self._E_dif
        My = Q_vector[:, np.newaxis] * self._jy_energy_basis * \
            P_vector / self._E_dif

        MxMy = self._states @ Mx @ My @ self._states.conj().T

        bott_inside = np.diag(MxMy).imag
        bott_inside = bott_inside[::2] + bott_inside[1::2]
        self._adiabatic_bott_index = -bott_inside * 4 * np.pi

        dt = time.time() - t1
        print(
            f'Adiabatic Bott index found - This took {round_sig(dt)} seconds')

    def find_local_kubo(self, beta=np.inf):
        t1 = time.time()
        Lx = self._lengths[0]
        Ly = self._lengths[1]

        efactor = 1 / (self._E_dif * self._E_dif)
        fi = fermi_occupation(self._energies, beta, self._fermi_energy)

        fij = fi[:, np.newaxis] - fi
        factor = fij * (efactor)

        matrix = self._states @ (factor * self._jx_energy_basis) @ \
            self._jy_energy_basis @ self._states.conj().T - \
            self._states @ self._jx_energy_basis @ \
            (factor * self._jy_energy_basis) @ self._states.conj().T

        bott_inside = np.diag(matrix).imag
        bott_inside = bott_inside[::2] + bott_inside[1::2]

        self._local_kubo = bott_inside * np.pi

        dt = time.time() - t1
        print(f'Local Kubo found - This took {round_sig(dt)} seconds')

        return self._local_kubo

    def find_chern_marker(self):
        t1 = time.time()

        lx = self._lengths[0]
        ly = self._lengths[1]

        X = self._x_list
        Y = self._y_list

        M = self._projector * X @ self._projector * Y @ self._projector

        out = np.diag(M).imag
        out = out[::2] + out[1::2]
        out = -out * np.pi * 4
        self._chern_marker = out

        dt = time.time() - t1
        print(f'Chern Marker found - This took {round_sig(dt)} seconds')

        return self._chern_marker

    @property
    def adiabatic_bott_index(self):
        if self._adiabatic_bott_index is not None:
            return (self._adiabatic_bott_index)
        else:
            raise Exception('Need to create the adiabatic bott index first')

    @property
    def local_kubo(self):
        if self._local_kubo is not None:
            return (self._local_kubo)
        else:
            raise Exception('Need to create the local kubo first')

    @property
    def bott_index(self):
        if self._bott_index is not None:
            return (self._bott_index)
        else:
            raise Exception('Need to create the bott index first')

    @property
    def chern_marker(self):
        if self._chern_marker is not None:
            return (self._chern_marker)
        else:
            raise Exception('Need to create the chern marker first')

    ########################################################
    #################### plotting stuff ####################
    ########################################################

    def plot_dos(self, spacing=0.1):

        bin = (self._energies[-1] - self._energies[0])/spacing
        bin = int(bin)
        plt.hist(self._energies, bins=bin)
        plt.title('Density Of States')
        plt.xlabel('Energy')
        plt.ylabel('n')
        plt.show()
