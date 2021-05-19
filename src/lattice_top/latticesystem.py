from .generalfunctions import *
from .plotting_functions import *


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
        self._n_occupied = None

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

        # crosshair marker
        self._crosshair_position = None
        self._crosshair_list = None
        self._crosshair_sums = None
        self._crosshair_value = None

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

        self._n_occupied = sum(self._energies <= fermi_energy)

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
        self._fermi_energy = fermi_energy
        self._n_occupied = sum(self._energies <= fermi_energy)
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

        out = np.diag(M) * self._n_occupied

        bott_index_tr = np.imag(out[::2] + out[1::2]) / (2 * np.pi)
        self._bott_index = bott_index_tr

        dt = time.time() - t1
        print(f'Bott Index found - This took {round_sig(dt)} seconds')

    def find_adiabatic_bott_index(self):

        t1 = time.time()

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

    #########################################################
    #################### Crosshair stuff ####################
    #########################################################

    def find_crosshair_marker(self, position):

        Q = np.eye(len(self._x_list)) - self._projector

        step_x = self._x_list >= position[0]
        step_y = self._y_list >= position[1]

        crosshair = self._projector * step_x @ Q * step_y @ self._projector
        self._crosshair_list = 4*np.pi*np.diag(crosshair).imag

        s = []
        limit = max(self._lengths)

        for size in np.linspace(0, limit, 100):
            circle_in = np.sqrt(
                (self._x_list-position[0])**2 + (self._y_list-position[1])**2) <= size
            s.append(sum(circle_in * self._crosshair_list *
                     circle_in[:np.newaxis]))

        max_index = np.argmax(np.abs(s))

        self._crosshair_value = s[max_index]
        self._crosshair_sums = s
        self._crosshair_radii = np.linspace(0, limit, 100)
        self._crosshair_position = position

    @property
    def crosshair(self):

        if self._crosshair_value is not None:
            return (self._crosshair_value)
        else:
            raise Exception('Need to create the crosshair first')

    def plot_crosshair_graph(self):
        limit = max(self._lengths)
        plt.plot(self._crosshair_radii, self._crosshair_sums)
        plt.show()

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
