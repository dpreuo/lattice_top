# TODO Get this thing working!

'''
    def setup_Haldane_system(self, M, t1, t2, phi, u_type):
        """
        This gives you a Haldane hamiltonian based on the parameters t1,t2,phi and M
        Note that the hopping terms always affect the hopping from a given cell to the cells to the right and above it

        :param M:
        :param t1:
        :param t2:
        :param phi:
        :param u_type:
        :return:

        Internally sets:
        _system_type
        _u_vals
        _hamiltonian
        _x_dif_hamiltonian
        _y_dif_hamiltonian
        _x_list
        _y_list
        """
        t = time.time()

        self._system_type = 'Haldane'

        xedge = self._edges[0]
        yedge = self._edges[1]

        Lx = self._lengths[0]
        Ly = self._lengths[1]

        m_x = np.arange(Lx)
        m_y = np.arange(Ly)

        if u_type == 'uniform':
            t1_vals = np.array([[t1] * Lx] * Ly)
            t2_vals = np.array([[t2] * Lx] * Ly)
            phi_vals = np.array([[phi] * Lx] * Ly)
            M_vals = np.array([[M] * Lx] * Ly)

        self._u_vals = {'t1': t1_vals, 't2': t2_vals, 'phi': phi_vals, 'M': M_vals}

        # now we make the hamiltonian!

        # list of mx and my positions
        mx_list = np.tile(m_x, Ly)
        my_list = np.kron(m_y, np.ones(Lx, dtype='int'))

        t1_list = np.ndarray.flatten(t1_vals)
        phi_list = np.ndarray.flatten(phi_vals)
        t2_list = np.ndarray.flatten(t2_vals)
        M_list = np.ndarray.flatten(M_vals)

        mx_pos_list = np.kron(mx_list, [1, 1])
        my_pos_list = np.kron(my_list, [1, 1])

        # define our internal matrices
        def A(m0, t10): return np.array([
            [m0, t10],
            [t10, -m0]
        ])

        def Bx(phi0, t10, t20): return np.array([
            [np.exp(-1j * phi0) * t20, t10],
            [0, np.exp(1j * phi0) * t20]
        ])

        def By(phi0, t10, t20): return np.array([
            [np.exp(1j * phi0) * t2, t10],
            [0, np.exp(-1j * phi0) * t20]
        ])

        def Bxy(phi0, t20): return np.array([
            [np.exp(-1j * phi0) * t20, 0],
            [0, np.exp(1j * phi0) * t20]
        ])

        # basis vectors
        ax = np.array([1, 0])
        ay = np.array([0.5, np.sqrt(3) / 2])

        # these arrays keep track of which elements correspond to shifts in x, y and xy
        lx_limit = Lx if xedge is False else 2 * Lx
        ly_limit = Ly if yedge is False else 2 * Ly

        x_mask = np.array((mx_list[:, np.newaxis] - mx_list) % lx_limit == 1)
        x_mask *= np.array(my_list[:, np.newaxis] - my_list == 0)

        y_mask = np.array((my_list[:, np.newaxis] - my_list) % ly_limit == 1)
        y_mask *= np.array(mx_list[:, np.newaxis] - mx_list == 0)

        xy_mask = np.array((mx_list[:, np.newaxis] - mx_list) % lx_limit == (-1) % Lx)
        xy_mask *= np.array((my_list[:, np.newaxis] - my_list) % ly_limit == 1)
    '''
