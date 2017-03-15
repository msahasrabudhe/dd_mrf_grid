import dd_mrf_grid as ddmrf

np = ddmrf.np

if __name__ == '__main__':
	lat = ddmrf.Lattice(10, 10 ,2)

	for i in range(lat.n_nodes):
		t = np.random.rand()
		lat.set_node_energies(i, [t, 1-t])

	for e in lat._find_empty_attributes()[1]:
		t = np.random.rand()
		lat.set_edge_energies(e[0], e[1], [[t, 1-t], [1-t, t]])

	lat._create_slaves()
	lat._slaves_to_solve = np.arange(lat.n_slaves)

	lat._optimise_slaves()

	for s in lat.slave_list:
		print s.labels
