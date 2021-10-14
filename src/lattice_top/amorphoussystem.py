
import numpy as np
import numpy.random as random
import scipy.spatial as spat
import matplotlib.pyplot as plt

from .latticesystem import *

#####################################################
######## functions for the spacing algorithm ########
#####################################################


def distance(displacement: list):
    """return the length of a vector

    Args:
        displacement (list): a vector in 2d space

    Returns:
        float: distance
    """
    d2 = displacement[0]**2 + displacement[1]**2
    return np.sqrt(d2)


def unit_vector(displacement: list):
    """returns displacement between two points

    Args:
        displacement (list): a list containing two vectors

    Returns:
        list: the vector displacement
    """
    dist = distance(displacement)
    return [displacement[0]/dist, displacement[1]/dist]


def pbc_displacement(cell1: list, cell2: list, lengths: tuple):
    """gives neares displacement between two points in periodic boundary

    Args:
        cell1 (list): first point
        cell2 (list): second point
        lengths (tuple): size of the system

    Returns:
        list: the vector displacement
    """
    dx = cell2[0] - cell1[0]
    dy = cell2[1] - cell1[1]
    dx_out = (dx + lengths[0] / 2) % lengths[0] - lengths[0] / 2
    dy_out = (dy + lengths[1] / 2) % lengths[1] - lengths[1] / 2
    return[dx_out, dy_out]


def force_function(dis: float):
    """gives the distance scaling of the repulsive potential as a function of distance

    Args:
        dis (float): distance

    Returns:
        float: potential energy
    """
    return -2/dis**3


def xy_part_pbc(cell1: list, cell2: list, lengths: tuple):
    """find the force repelling two points - and the step they take apart

    Args:
        cell1 (list): first point coordinates
        cell2 (list): second point coordinates
        lengths (tuple): system size

    Return:
        list: the displacement for that cell to move
    """

    difference = pbc_displacement(cell1, cell2, lengths)

    dist = distance(difference)
    unit = unit_vector(difference)

    # flatten extremes to stop too big a jump when points start too close together!
    force_flattened = np.tanh(force_function(dist))

    # force_flattened = force_function(dist)
    out = [force_flattened*unit[0],  force_flattened*unit[1]]

    return(out)
    # return np.tanh(out)


def dudx_individual_pbc(cells_in: list, index: int, lengths: tuple):
    """
    takes one of the cells of a list and calculates the total contribution to its displacement from every other cell
    TODO - add some kind of nearest neighbour thing to this so it vanishes when cells are too far!

    Args:
        cells_in (list): list of all the points in the system
        index (int): the index of the individual cell you want to shift
        lengths (tuple): system size

    Returns:
        list: a vector telling that cell how far to move
    """
    cells = cells_in.copy()
    cell1 = cells.pop(index)
    out = [0., 0.]
    for cell2 in cells:
        shift_vec = xy_part_pbc(cell1, cell2, lengths)
        out[0] -= shift_vec[0]
        out[1] -= shift_vec[1]
    return out


def dudx_pbc(cells: list, lengths: tuple):
    """takes a list of loads of cells and tells each one how to move based on where all the others are

    Args:
        cells (list): list of all the cells in the system
        lengths (tuple): system size

    Returns:
        list: a list of the right displacement for every cell in the system
    """
    out = []
    for k in range(len(cells)):
        shift_vec = dudx_individual_pbc(cells, k, lengths)
        out.append(shift_vec)
    return out


def spaced_points(lengths: tuple, number_of_sites: int, n_optimisation_steps: int, optimisation_rate: float):
    """
    creates a set of points randomly and then uses the spacing algorithm to arrange them nicely so they're all roughly spaced from one another

    Args:
        lengths (tuple): system size
        number_of_sites (int): how many states we want - defauly one for every area unit
        n_optimisation_steps (int): How many steps over which the system is optimised -  more steps = takes longer but more regular spacing
        optimisation_rate (float): how big each step jumps the points - bigger jumps = optimises faster but more erratic and prone to over shooting!!
    """

    cells = [[a*lengths[0], b*lengths[0]]
             for a, b in zip(random.random(number_of_sites), random.random(number_of_sites))]
    for i in range(n_optimisation_steps):

        dud = dudx_pbc(cells, lengths)
        for j, cell in enumerate(cells):
            cells[j][0] -= optimisation_rate*dud[j][0]
            cells[j][1] -= optimisation_rate*dud[j][1]

            # enforce periodic boundary conditions
            cells[j][0] = cells[j][0] % lengths[0]
            cells[j][1] = cells[j][1] % lengths[1]

    return cells


#############################################################
########### functions making the voronoi periodic ###########
#############################################################


def duplicate_edges(cells: list, lengths: tuple, extra_amount=0.2):
    """This takes a set of points and pads out the edges so that the voronoicells can be put in periodic boundaries

    Args:
        cells (list): all the cells of the system
        lengths (tuple): system size
        extra_amount (float, optional): How much extra system is added in each direction - increase this for small system size! Defaults to 0.2.

    Returns:
        list: A new set of slightly expanded points
    """
    shift = (0, 1, -1)

    # the list but with all the shifted bits added to the end
    cells_duplicated = []

    for x_s in shift:
        for y_s in shift:
            cells_moved = [[a[0] + x_s*lengths[0], a[1] +
                            y_s*lengths[1]] for a in cells]
            cells_duplicated += cells_moved

    # now remove some of the overspilled cells - we just want a small border around our region

    cells_truncated = []

    for cell in cells_duplicated:
        if cell[0] > -extra_amount*lengths[0] and cell[0] < (1+extra_amount) * lengths[0]:
            if cell[1] > -extra_amount*lengths[1] and cell[1] < (1+extra_amount) * lengths[1]:
                cells_truncated.append(cell)

    return cells_truncated


def which_region(x: float, y: float, lengths: tuple):
    """Tells you whether the point is internal, ot jumps the periodic boundary in +x, +y or both

    Args:
        x (float): x coordinate of point
        y (float): y coordinate of point
        lengths (tuple): system size

    Returns:
        tuple: descriptor for where the point is. eg (0,1) means the x is inside the cell and y is shifted positively over
    """

    if x < 0:
        x_pos = -1
    elif x >= lengths[0]:
        x_pos = 1
    else:
        x_pos = 0

    if y < 0:
        y_pos = -1
    elif y >= lengths[1]:
        y_pos = 1
    else:
        y_pos = 0

    return (x_pos, y_pos)


def point_close(point1, point2, tol=1e-10):
    """Tells you if a pair of points in 2d are close enough to be considered the same

    Args:
        point1 (list): a 2d point
        point2 (list): a 2d point
        tol (float, optional): tolerance. Defaults to 1e-10.

    Returns:
        bool: Are the two points the same
    """
    if abs(point1[0] - point2[0]) <= tol:
        if abs(point1[1] - point2[1]) <= tol:
            return True

    return False


def inside_list(points: list, which_region_list: list, lengths: tuple):
    """gives you a list of only the points that are inside the unit cell

    Args:
        points (list): all the points in your system
        which_region_list (list): list of which region each point is in
        lengths (tuple): length f system size

    Returns:
        list: a reduced list of all the points
    """
    inside_points = []
    for point, reg in zip(points, which_region_list):
        if reg == (0, 0):
            inside_points.append(point)
    return inside_points


def correspondence_list(points: list, which_region_list: list, inside_points: list, lengths: tuple):
    """takes a list of points and tells you which point in the unit cell list each gets remapped to

    Args:
        points (list): your long list of points
        which_region_list (list): the list of which region each point is in
        inside_points (list): the reduced list of points in the unit cell
        lengths (tuple): system size

    Returns:
        list: a list that tells you what each point in points gets remapped to
    """

    correspondence_list = []
    for point, reg in zip(points, which_region_list):
        new_point = [point[0] - reg[0]*lengths[0],
                     point[1] - reg[1]*lengths[1]]

        loop_worked = False
        for ind, point_compare in enumerate(inside_points):
            if point_close(new_point, point_compare):
                correspondence_list.append(ind)
                loop_worked = True
                break

        if loop_worked == False:
            # some points dont have an analog - they must be removed completely - we signify with None
            correspondence_list.append(None)

    return correspondence_list


def fix_edges(list_of_edges, correspondence):
    """Takes a list of edges and applies the correspondence list to them - changing the index in each edge so they are always inside the unit cell!

    Args:
        list_of_edges (list): list of edges - of form [index1, index2]
        correspondence (list): a list of what each point is remapped to

    Raises:
        Exception: Illegal edge - one of the end points of this edge has no corresponding point in the unit cell
    """

    for i, edge in enumerate(list_of_edges):
        ind1 = edge[0]
        ind2 = edge[1]
        remapped_1 = correspondence[ind1]
        remapped_2 = correspondence[ind2]
        if remapped_1 == None or remapped_2 == None:
            print(ind1, ind2)
            # raise Exception('illegal edge - this has no inside correspondence')

        list_of_edges[i] = [remapped_1, remapped_2]


#############################################################
###### functions for decorating points for Hamiltonian ######
#############################################################

def hopping_matrix_element(displacement: list):
    """takes a vector in 2d and returns a creation operator in that direction on bloch sphere!

    Args:
        displacement (list): [description]
    """
    mag = distance(displacement)
    vec_as_complex = (displacement[0] + 1j*displacement[1]) / mag

    M = np.array([
        [1, -np.conj(vec_as_complex)],
        [vec_as_complex, -1]
    ])

    return 0.5*M


def set_matrix_2x2(target_matrix, input_matrix, position):
    a, b = position
    target_matrix[a, b] = input_matrix[0, 0]
    target_matrix[a+1, b] = input_matrix[1, 0]
    target_matrix[a, b+1] = input_matrix[0, 1]
    target_matrix[a+1, b+1] = input_matrix[1, 1]

######################################################
################## the class itself ##################
######################################################


class Amorphous_System(Lattice_System):

    """
    The process for this kind of system is

    init: set up lengths, boundary conditions and how many sites
    create_amorphous_lattice creates the lattice, voronoi lattice and returns all the edges
    QWZ_decorate turns this lattice into a Hamiltonian

    """

    def __init__(self, lengths: tuple, number_of_sites=None):
        super().__init__(lengths)
        if number_of_sites == None:
            self._number_of_sites = lengths[0]*lengths[1]
        else:
            self._number_of_sites = number_of_sites

    def export_lattice(self):
        """exports the info for a lattice so you can save it for later

        Returns:
            dict: a dict containing the sites, connections and connection types
        """
        out_dict = {'sites': self._sites,
                    'connections': self._connections,
                    'connection_types': self._connection_types,
                    'lengths': self._lengths,
                    'n_sites': self._n_sites}
        return out_dict

    @classmethod
    def create_from_lattice(cls, lattice_info):
        """internally sets a lattice from an input dict - note this reinitialises the whole object! 

        Args:
            lattice_info (dict): dict containing the sites, connections and connection types, lengths, boundaries and number of sites
        """

        syst = cls(lattice_info['lengths'],  lattice_info['n_sites'])
        syst._sites = lattice_info['sites']
        syst._connections = lattice_info['connections']
        syst._connection_types = lattice_info['connection_types']

        return syst

    ############################################################
    ################ Lattice Creation Functions ################
    ############################################################

    def create_amorphous_lattice(self, lattice_type: str, n_optimisation_steps=20, optimisation_rate=0.1):
        """This creates an amorphous lattice - lattice is always created with periodic boundaries - but we keep a list
        of all the edges that cross the boundaries so they can be removed to create open boundaries!

        Args:
            lattice_type (str): 'voronoi' or 'dual'. What kind of lattice - voronoi cells or its dual, nearest neighbour connections
            n_optimisation_steps (int, optional): how many steps of spacing all the points out you want to do. Larger numbers means its more evenly spaced but takes longer Defaults to 20.
            optimisation_rate (float, optional): how big each step is. Larger numbers optimise faster but are more unstable Defaults to 0.1.

        Raises:
            Exception: Wrong lattice type name
        """

        t1 = time.time()

        # create the cell positions
        cell_locations = spaced_points(
            self._lengths, self._number_of_sites, n_optimisation_steps, optimisation_rate)

        # now we duplicate around the edges them to enforce periodic boundaries
        cells_duplicated_edge = duplicate_edges(cell_locations, self._lengths)

        # now we make the voronoi cells!!
        voronoi = spat.Voronoi(cells_duplicated_edge, furthest_site=False)

        #  pick what kind of cells we want
        if lattice_type == 'voronoi':
            sites = voronoi.vertices
            edges = voronoi.ridge_vertices
        elif lattice_type == 'dual':
            sites = voronoi.points
            edges = voronoi.ridge_points
        else:
            raise Exception('lattice type must be "voronoi" or "dual"')

        ### now we want to restrict back down to  periodic boundaries! ###

        # list of which region each point is in
        region_list = [which_region(a, b, self._lengths) for a, b in sites]

        # now separate all the connections into different lists based on whether they cross x or y
        # and fix what they index to make sure it's inside the main region!

        internal_connections = []
        x_connections = []
        y_connections = []
        px_py_connections = []
        mx_py_connections = []

        for edge in edges:

            # these dont connect to anything - kill em
            if edge[0] == -1 or edge[1] == -1:
                continue

            point1 = sites[edge[0]]
            region1 = region_list[edge[0]]

            point2 = sites[edge[1]]
            region2 = region_list[edge[1]]

            bad_regions = [(-1, 0), (-1, -1), (0, -1), (1, -1)]

            # these go to non positive regions - so are repeats of other hoppings
            one_is_bad = region1 in bad_regions or region2 in bad_regions
            if one_is_bad:
                continue

            fully_internal = region1 == (0, 0) and region2 == (0, 0)
            point1_outside = region1 != (0, 0) and region2 == (0, 0)
            point2_outside = region1 == (0, 0) and region2 != (0, 0)

            if fully_internal:
                internal_connections.append(edge)
            elif point1_outside:
                if region1 == (1, 0):
                    x_connections.append(edge)
                elif region1 == (0, 1):
                    y_connections.append(edge)
                elif region1 == (-1, 1):
                    mx_py_connections.append(edge)
                elif region1 == (1, 1):
                    px_py_connections.append(edge)
            elif point2_outside:
                if region2 == (1, 0):
                    x_connections.append(edge)
                elif region2 == (0, 1):
                    y_connections.append(edge)
                elif region2 == (-1, 1):
                    mx_py_connections.append(edge)
                elif region2 == (1, 1):
                    px_py_connections.append(edge)

        # now we want to make a list of every point in sites and the point they all correspond to inside the unit cell
        unit_cell_points = inside_list(sites, region_list, self._lengths)

        correspondence = correspondence_list(
            sites, region_list, unit_cell_points, self._lengths)

        # fix all the edges so they reference only inside points!
        fix_edges(x_connections, correspondence)
        fix_edges(y_connections, correspondence)
        # this line should be unnecessary...
        fix_edges(internal_connections, correspondence)
        fix_edges(mx_py_connections, correspondence)
        fix_edges(px_py_connections, correspondence)

        final_edges = internal_connections
        connection_types = [(0, 0)]*len(final_edges)

        # now, put it all together - include all the pbc edges

        final_edges += x_connections
        connection_types += [(1, 0)]*len(x_connections)

        final_edges += y_connections
        connection_types += [(0, 1)]*len(y_connections)

        if px_py_connections != []:
            final_edges += px_py_connections
            connection_types += [(1, 1)]*len(px_py_connections)

        if mx_py_connections != []:
            final_edges += mx_py_connections
            connection_types += [(-1, 1)]*len(mx_py_connections)

        # finally set the three internal things - set as tuple so they dont get changed
        self._sites = tuple(tuple(s) for s in unit_cell_points)
        self._connections = tuple(tuple(e) for e in final_edges)
        self._connection_types = tuple(tuple(ct) for ct in connection_types)

        dt = time.time() - t1
        print(f'Amorphous lattice created - This took {round_sig(dt)} seconds')

    def create_regular_lattice(self, position_jitter=0):
        t1 = time.time()

        Lx, Ly = self._lengths

        x_list = np.tile(np.arange(Lx), Ly) + 0.5
        y_list = np.kron(np.arange(Ly), np.ones(Lx, dtype='int')) + 0.5

        sites1 = list(zip(x_list, y_list))
        sites = [list(site) for site in sites1]
        connections = []
        connection_types = []

        for index, site in enumerate(sites):

            x_site = [(site[0] + 1) % Lx, site[1]]
            y_site = [site[0], (site[1]+1) % Ly]

            x_index = sites.index(x_site)
            y_index = sites.index(y_site)

            if site[0] + 1 >= Lx:
                x_type = (1, 0)
            else:
                x_type = (0, 0)

            if site[1] + 1 >= Ly:
                y_type = (0, 1)
            else:
                y_type = (0, 0)

            connections.append((index, x_index))
            connection_types.append(x_type)

            connections.append((index, y_index))
            connection_types.append(y_type)

        if position_jitter != 0:
            for site in sites:
                site[0] += np.random.normal(loc=0, scale=position_jitter)
                site[1] += np.random.normal(loc=0, scale=position_jitter)

        # finally set the three internal things - set as tuple so they dont get changed
        self._sites = tuple(tuple(s) for s in sites)
        self._connections = tuple(tuple(c) for c in connections)
        self._connection_types = tuple(tuple(ct) for ct in connection_types)

    ####################################################
    #### decorate the lattice to make a Hamiltonian ####
    ####################################################

    def QWZ_decorate(self, edges: tuple, u1: float, u2: float, u_type: str, angle=(0., 0.)):

        t1 = time.time()

        x_edge, y_edge = edges
        self._edges = edges

        # set up the on site energy terms
        if u_type == 'uniform':
            u_vals = np.array([u1]*len(self._sites))

        # make the hopping terms
        N = 2*len(self._sites)

        hamiltonian = np.zeros((N, N), dtype='complex')
        x_dif_hamiltonian = np.zeros((N, N), dtype='complex')
        y_dif_hamiltonian = np.zeros((N, N), dtype='complex')

        # set u values
        for j, site in enumerate(self._sites):
            hamiltonian[j*2, j*2] = u_vals[j]
            hamiltonian[j*2+1, j*2+1] = -u_vals[j]

        # set hoppings
        for edge, edge_type in zip(self._connections, self._connection_types):

            if x_edge:
                if edge_type[0] != 0:
                    continue
            if y_edge:
                if edge_type[1] != 0:
                    continue

            n_1 = edge[0]
            n_2 = edge[1]

            p1 = self._sites[n_1]
            p2 = self._sites[n_2]

            edge_displacement = pbc_displacement(p1, p2, self._lengths)
            hopping_matrix = hopping_matrix_element(edge_displacement)

            # add magnetic field terms!
            x_phase = np.exp(1j*angle[0]*edge_displacement[0]/self._lengths[0])
            y_phase = np.exp(1j*angle[1]*edge_displacement[1]/self._lengths[1])
            hopping_phase = x_phase*y_phase
            hopping_matrix = hopping_matrix*hopping_phase
            
            # set hopping in main hamiltonian
            set_matrix_2x2(hamiltonian, hopping_matrix,  (2*n_1, 2*n_2))

            set_matrix_2x2(x_dif_hamiltonian, hopping_matrix *
                           edge_displacement[0]*1j, (2*n_1, 2*n_2))
            set_matrix_2x2(y_dif_hamiltonian, hopping_matrix *
                           edge_displacement[1]*1j, (2*n_1, 2*n_2))

            # set conjugate hopping
            set_matrix_2x2(hamiltonian, hopping_matrix.conj().T,
                           (2*n_2, 2*n_1))
            set_matrix_2x2(x_dif_hamiltonian, - hopping_matrix.conj().T *
                           edge_displacement[0]*1j, (2*n_2, 2*n_1))
            set_matrix_2x2(y_dif_hamiltonian, - hopping_matrix.conj().T *
                           edge_displacement[1]*1j, (2*n_2, 2*n_1))

        self._hamiltonian = hamiltonian
        self._x_dif_hamiltonian = x_dif_hamiltonian
        self._y_dif_hamiltonian = y_dif_hamiltonian

        x_list, y_list = zip(*self._sites)

        self._x_list = np.kron(np.array(x_list), [1, 1])
        self._y_list = np.kron(np.array(y_list), [1, 1])
        self._n_sites = N

        dt = time.time() - t1
        print(f'Hamiltonian created - This took {round_sig(dt)} seconds')


    ####################################################
    ################## plotting stuff ##################
    ####################################################

    def plot_lattice(self):

        plt.xlim(-1.5, self._lengths[0]+1.5)
        plt.ylim(-1.5, self._lengths[1]+1.5)
        plt.axvline(0, color='grey', linestyle='--')
        plt.axhline(0, color='grey', linestyle='--')
        plt.axvline(self._lengths[0], color='grey', linestyle='--')
        plt.axhline(self._lengths[1], color='grey', linestyle='--')

        c_dict = {(0, 0): 'black', (1, 0): 'green', (0, 1)                  : 'red', (-1, 1): 'grey', (1, 1): 'grey'}

        for connect, connection_type in zip(self._connections, self._connection_types):
            point1 = list(self._sites[connect[0]])
            point2 = list(self._sites[connect[1]])
            c = c_dict[connection_type]

            if connection_type == (0, 0):
                plt.plot([point1[0], point2[0]], [
                         point1[1], point2[1]], color=c)

            elif connection_type == (1, 0):

                if point1[0] < self._lengths[0]/2:
                    point1[0] = point1[0] + self._lengths[0]
                else:
                    point2[0] = point2[0] + self._lengths[0]

                plt.plot([point1[0], point2[0]], [
                         point1[1], point2[1]], color=c)
                plt.plot([point1[0] - self._lengths[0], point2[0] - self._lengths[0]], [
                         point1[1], point2[1]], color=c)

            elif connection_type == (0, 1):

                if point1[1] < self._lengths[1]/2:
                    point1[1] = point1[1] + self._lengths[1]
                else:
                    point2[1] = point2[1] + self._lengths[1]

                plt.plot([point1[0], point2[0]], [
                         point1[1], point2[1]], color=c)
                plt.plot([point1[0], point2[0]], [
                         point1[1] - self._lengths[1], point2[1] - self._lengths[1]], color=c)

            elif connection_type == (1, 1):

                if point1[1] < self._lengths[1]/2:
                    point1[0] = point1[0] + self._lengths[0]
                    point1[1] = point1[1] + self._lengths[1]
                else:
                    point2[0] = point2[0] + self._lengths[0]
                    point2[1] = point2[1] + self._lengths[1]

                plt.plot([point1[0], point2[0]], [
                         point1[1], point2[1]], color=c)
                plt.plot([point1[0] - self._lengths[0], point2[0] - self._lengths[0]], [
                         point1[1] - self._lengths[1], point2[1] - self._lengths[1]], color=c)

            elif connection_type == (-1, 1):

                if point1[1] < self._lengths[1]/2:
                    point1[0] = point1[0] - self._lengths[0]
                    point1[1] = point1[1] + self._lengths[1]
                else:
                    point2[0] = point2[0] - self._lengths[0]
                    point2[1] = point2[1] + self._lengths[1]

                plt.plot([point1[0], point2[0]], [
                         point1[1], point2[1]], color=c)
                plt.plot([point1[0] + self._lengths[0], point2[0] + self._lengths[0]], [
                         point1[1] - self._lengths[1], point2[1] - self._lengths[1]], color=c)

        plt.show()

    def cmap_state(self, state_id, show=True):
        Z = self._states[:, state_id]
        Z = abs(Z)[::2] + abs(Z)[1::2]
        x_values = self._x_list[::2]
        y_values = self._y_list[::2]
        state_name = f'State number: {state_id}, Energy = {self._energies[state_id]}'
        cmap_triangulation(x_values, y_values, Z,
                           title=state_name, xy_labels=('x', 'y'), show=show)

    def plot_state(self, state_id, show=True):
        """
        this plots an arbitrary state in our system
        :param state_id: the number of the state you want to plot
        :return: nowt
        """

        Z = self._states[:, state_id]
        Z = abs(Z)[::2] + abs(Z)[1::2]
        x_values = self._x_list[::2]
        y_values = self._y_list[::2]
        state_name = f'State number: {state_id}, Energy = {self._energies[state_id]}'
        plot_triangulation(x_values, y_values, Z,
                           title=state_name, xy_labels=('x', 'y'), show=show)

    def plot_index(self, index, title=None, range=None, show=True):
        x_values = self._x_list[::2]
        y_values = self._y_list[::2]
        plot_triangulation(x_values, y_values, index,
                           xy_labels=('x', 'y'), range=range, show=show)

    def cmap_index(self, index, title=None, range=None, show=True):
        x_values = self._x_list[::2]
        y_values = self._y_list[::2]
        cmap_triangulation(x_values, y_values, index, title=title,
                           xy_labels=('x', 'y'), range=range, show=show)
