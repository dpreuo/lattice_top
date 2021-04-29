from numpy import random
import lattice_top

lengths = (20, 21)
edges = (False, False)

qwz_system = lattice_top.QWZ_System(lengths, edges)

qwz_system.setup_QWZ_system(
    1.1, 2.7, 'strip', u_noise_sigma=0., e_noise_sigma=0.)

qwz_system.solve_Hamiltonian()

qwz_system.plot_u_values()

state_number = random.randint(0, qwz_system._n_sites)
qwz_system.plot_state(state_number)
qwz_system.cmap_state(state_number)

qwz_system.find_bott_index()
qwz_system.plot_index(qwz_system.bott_index, 'Bott Index')

qwz_system.find_chern_marker()
qwz_system.plot_index(qwz_system.chern_marker, 'Chern Marker')

qwz_system.find_adiabatic_bott_index()
qwz_system.plot_index(qwz_system.adiabatic_bott_index, 'Adiabatic Bott Index')

qwz_system.find_local_kubo()
qwz_system.plot_index(qwz_system.local_kubo, 'Local Kubo')
