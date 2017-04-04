import scipy.misc
import dd_mrf_grid as ddmrf

import matplotlib.pyplot as plt

# Borrow Numpy
np = ddmrf.np

im = scipy.misc.imread('X_c.pgm')
#im = np.floor(scipy.misc.imresize(im, [150, 100], 'bicubic'))
im = im*1.0
rows, cols = im.shape

lat = ddmrf.Lattice(rows, cols, 2)

K = 200

for i in range(lat.n_nodes):
	x = i/cols
	y = i%cols
	lat.set_node_energies(i, [im[x,y], 255.0 - im[x,y]])
			
K = K*1.0
edge_energy = [[0.0, 255.0-K],[255.0-K, 0.0]]

for e in lat._find_empty_attributes()[1]:
	lat.set_edge_energies(e[0], e[1], edge_energy)

if __name__ == '__main__':
	if lat.check_completeness():
		lat.optimise(a_start=1.0, max_iter=10000, strategy='adaptive', decomposition='row_col')

	labels = lat.labels.astype(np.int)
				
	labels = np.reshape(labels, [rows, cols])
	print labels.tolist()

	f = plt.figure()
	plt.plot(lat.primal_costs, 'r-')
	plt.plot(lat.dual_costs, 'b-')
	plt.show()
