import scipy.misc
import dd_mrf_grid as ddmrf

import matplotlib.pyplot as plt

# Borrow Numpy
np = ddmrf.np

im = scipy.misc.imread('X_c.pgm')
im = im*1.0/255
rows, cols = im.shape

lat = ddmrf.Lattice(rows, cols, 2)

K = 125

for i in range(lat.n_nodes):
	x = i/cols
	y = i%cols
	lat.set_node_energies(i, [im[x,y], 1.0 - im[x,y]])
			
K = K*1.0/255
edge_energy = [[1.0, K], [K, 1.0]]

for e in lat._find_empty_attributes()[1]:
	lat.set_edge_energies(e[0], e[1], edge_energy)

if lat.check_completeness():
	lat.optimise(a_start=0.1, max_iter=200)

labels = lat.labels.astype(np.int)
				
labels = np.reshape(labels, [rows, cols])
print labels.tolist()

f = plt.figure()
plt.plot(lat.primal_costs, 'r-')
plt.plot(lat.dual_costs, 'b-')
plt.show()
