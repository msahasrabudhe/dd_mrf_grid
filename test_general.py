import dd_mrf_grid as ddmrf
import matplotlib.pyplot as plt

np = ddmrf.np

n_labels = 2
grid_size = [20, 20]
lat = ddmrf.Lattice(grid_size[0], grid_size[1],n_labels)

for i in range(lat.n_nodes):
	ne	= np.random.standard_normal(size=(n_labels,))
	lat.set_node_energies(i, ne)

for e in lat._find_empty_attributes()[1]:
	ne	= np.random.standard_normal(size=(n_labels, n_labels))
	ne	= 2*ne			# Make sigma = 2
	lat.set_edge_energies(e[0], e[1], ne)

if __name__ == '__main__':
	lat.optimise(a_start=0.1, max_iter=5000)
	print np.reshape(lat.labels, grid_size)
	lat.plot_costs()
	plt.show()

