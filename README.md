# **lattice_top**
A package for lattice-based topological insulator calculations. This repo is mostly bult around a single class - the lattice system class. The class contains a bunch of protocols for creating a quantum system on a lattice, and then calculating various local markers.


## **Lattice System Class Structure**
The class methods can be divided into a few sections.

### Initialising the system:

The parent class does not initialise the system - this is left to a few subclasses to allow for a wider variety of quantum systems to be included. Whatever the system we have initialised, the same few internal parameters need to be set:

 - **_lengths**: system size, set by initialisation.
 - **_edges**: boundary conditions, set by initialisation.

Then the initialiser must set a few internal parameters:
 - **_x_list**: list of the x position of every site.
 - **_y_list**: list of the y position of every site.
 - **_hamiltonian**: the hamiltonian for the system.
 - **_x_dif_hamiltonian**: derivative of the hamiltonian with respect to a uniform magnetic vector potential in the x direction - used to calculate the adiabatic bott index and local kubo.
 - **_y_dif_hamiltonian**: derivative of the hamiltonian with respect to a uniform magnetic vector potential in the xy direction - used to calculate the adiabatic bott index and local kubo.
 - **_n_sites**: number of sites in the system

Individual initialisers may also set their own internal parameters, but they shouldnt be used for general calculations - only for methods specific to that individual type of system.

#### QWZ_System
This initialises a Qi-Wu-Zhang model on a grid. Sets the additional internal parameter:
- **_u_vals**: gives you the u internal parameter for every site.

#### Amorphous_System
This sets up an amorphous lattice and builds a hamiltonian on that lattice. The first part is to set up the lattice. The second is to decorate it with a Hamiltonian
This sets the additional parameters:
- **_sites**: list of the coordinates of every site
- **_connections**: list of pairs of indices of sites that are connected by an edge
- **_connection_types**: a list of which boundary each connection hops over, eg if it hops over the PBC in the x or y direction
