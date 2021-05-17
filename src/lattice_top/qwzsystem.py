from .latticesystem import *


class QWZ_System(Lattice_System):

    """
    The process for this type of system is:

    init: just writes down the system size and edge boundary conditions
    setup_QWZ_system: sets up the lattice and the hoppings on all sites, makes the hamitonian
    solve_hamiltonian: part of parent class - solves the hammy

    ~~ then you can do whatever you want with the system ~~
    """

    def __init__(self, lengths):
        super().__init__(lengths)

        # These qwz specific values will be created after initialising u_array
        self._u_vals = None

    def setup_QWZ_system(self, edges, u1, u2, u_type, u_noise_sigma=0, e_noise_sigma=0, circle_width=None, strip_width=None,
                         angle=(0., 0.)):
        """
        This sets up the system as a Qi-Wu-Zhang system, we initialise the Hamiltonian as well as the x and y
        derivatives. This has two parts:

        1) We setup the u_values - these are the local values of on site energy in our material.

        2) We use these values to set up the QWZ Hamiltonian as well as its derivatives

        :param u1:u in first region
        :param u2: u in second region
        :param u_type: either 'strip', 'circle' or 'uniform'
        :param u_noise_sigma: adds some gaussian noise to the u values with sigma, default is 0 (no noise)
        :param e_noise_sigma: adds some energy cost noise - ie overall energy for being in a unit cell
        :param circle_width: if you want to put a custom width for the circle, default is a third of the width
        :param strip_width: if you want to put a custom width for the strip, default is half the width
        :param angle: this adds a static magnetic field to the system - this is to stop you from hitting degenerate points
        in your discretisation of the states
        :return: nothing

        Internally sets:
        _system_type
        _u_vals
        _hamiltonian
        _x_dif_hamiltonian
        _y_dif_hamiltonian
        _x_list
        _y_list

        """

        # PART 1: set up the u_values

        t1 = time.time()

        self._edges = edges

        Lx = self._lengths[0]
        Ly = self._lengths[1]

        xs = np.arange(Lx)
        ys = np.arange(Ly)

        if u_type == 'strip':
            if strip_width is None:
                distance = Lx / 4
            else:
                distance = strip_width / 2

            x_mask = abs(xs - Lx / 2) <= distance
            x_values = u1 * x_mask + u2 * (1 - x_mask)

            u_vals = np.array([x_values] * Ly)
        elif u_type == 'uniform':
            u_vals = np.array([[u1] * Lx] * Ly)

        elif u_type == 'circle':
            minL = min(Lx, Ly)

            if circle_width is None:
                radius = minL / 3
            else:
                radius = circle_width / 2

            # note there is a slight shift here because symmetry causes degeneracy and degeneracy is a pain in the ass
            center = (Lx / 2 + 0.3, Ly / 2 - 0.2)

            dis = (xs - center[0]) ** 2 + (ys[:, np.newaxis] - center[1]) ** 2
            dis = np.sqrt(dis)
            u_vals = u1 * (dis <= radius) + u2 * (dis > radius)
        else:
            raise Exception('u_values type invalid')

        # add noise to the u_values
        if u_noise_sigma != 0:
            noise_matrix = np.random.normal(
                loc=0, scale=u_noise_sigma, size=self._lengths)
            u_vals = u_vals + noise_matrix.T

        self._u_vals = u_vals

        # PART 2: Set up the Hamiltonian

        xedge = self._edges[0]
        yedge = self._edges[1]

        x_list = np.tile(np.arange(Lx), Ly)
        y_list = np.kron(np.arange(Ly), np.ones(Lx, dtype='int'))

        self._x_list = np.kron(x_list, [1, 1])
        self._y_list = np.kron(y_list, [1, 1])

        A = np.array([[1, 0], [0, -1]])
        Bx = np.exp(1j * angle[0]) * (1 / 2) * np.array([[1, 1j], [1j, -1]])
        By = np.exp(1j * angle[1]) * (1 / 2) * np.array([[1, 1], [-1, -1]])

        lx_limit = Lx if xedge is False else 2 * Lx
        ly_limit = Ly if yedge is False else 2 * Ly

        x_mask = np.array((x_list[:, np.newaxis] - x_list) % lx_limit == 1)
        x_mask *= np.array(y_list[:, np.newaxis] - y_list == 0)

        y_mask = np.array((y_list[:, np.newaxis] - y_list) % ly_limit == 1)
        y_mask *= np.array(x_list[:, np.newaxis] - x_list == 0)

        A_part = np.kron(np.diag(np.ndarray.flatten(self._u_vals)), A)
        Bx_part = np.kron(x_mask, Bx) + np.kron(x_mask, Bx).conj().T
        By_part = np.kron(y_mask, By) + np.kron(y_mask, By).conj().T

        Bx_part_deriv = np.kron(x_mask, 1j * Bx) + \
            np.kron(x_mask, 1j * Bx).conj().T
        By_part_deriv = np.kron(y_mask, 1j * By) + \
            np.kron(y_mask, 1j * By).conj().T

        # add energy potential noise
        if e_noise_sigma != 0:
            e_noise = np.random.normal(
                loc=0, scale=e_noise_sigma, size=Lx * Ly)
            self._hamiltonian = A_part + Bx_part + \
                By_part + np.diag(np.kron(e_noise, [1, 1]))
        else:
            self._hamiltonian = A_part + Bx_part + By_part

        self._x_dif_hamiltonian = Bx_part_deriv
        self._y_dif_hamiltonian = By_part_deriv
        self._n_sites = 2*Lx*Ly

        dt = time.time() - t1
        print(f'Hamiltonian initialised - This took {round_sig(dt)} seconds')


########################################################
#################### plotting stuff ####################
########################################################

    def plot_u_values(self):
        """
        this plots the u_values of the system in a 3d plot!
        """
        try:
            self._u_vals
        except:
            raise Exception(
                'You need to create the u_values before you plot them')

        plot_surface_3d(self._u_vals, self._lengths,
                        title='QWZ u Values', xy_labels=('x', 'y'))

    def plot_state(self, state_id):
        """
        this plots an arbitrary state in our system
        :param state_id: the number of the state you want to plot
        :return: nowt
        """

        Z = self._states[:, state_id]
        Z = abs(Z)[::2] + abs(Z)[1::2]

        Z = Z.reshape(self._lengths[1], self._lengths[0])
        plot_surface_3d(Z, self._lengths, title='State Number: ' +
                        str(state_id), xy_labels=('x', 'y'))

    def cmap_state(self, state_id):
        Z = self._states[:, state_id]
        Z = abs(Z)[::2] + abs(Z)[1::2]
        Z = Z.reshape((self._lengths[1], self._lengths[0]))
        plt.pcolor(Z)
        plt.colorbar()
        plt.show()

    def cmap_index(self, index, name=None):
        Z = index.reshape((self._lengths[1], self._lengths[0]))
        plt.pcolor(Z)
        if name is not None:
            plt.title(name)
        plt.colorbar()
        plt.show()

    def plot_index(self, index, name=None):
        Z = index.reshape((self._lengths[1], self._lengths[0]))
        plot_surface_3d(Z, self._lengths, title=name, xy_labels=('x', 'y'))
