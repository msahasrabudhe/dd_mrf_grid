import dd_mrf_grid as ddmrf
import matplotlib.pyplot as plt

np = ddmrf.np

n_labels = 2
lat = ddmrf.Lattice(10, 10 ,n_labels)

for i in range(lat.n_nodes):
	ne	= np.random.standard_normal(size=(n_labels,))
	lat.set_node_energies(i, ne)

for e in lat._find_empty_attributes()[1]:
	ne	= np.random.standard_normal(size=(n_labels, n_labels))
	ne	= 2*ne			# Make sigma = 1
	lat.set_edge_energies(e[0], e[1], ne)

if __name__ == '__main__':
	lat.optimise(a_start=0.1, max_iter=5000)
	print np.reshape(lat.labels, [10, 10])
	plt.plot(lat.dual_costs, 'r-')
	plt.show()

