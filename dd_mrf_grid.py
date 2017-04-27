#!/usr/bin/python

# This library calculates an approximation to the optimal of an artibrary energy on a 2-D lattice
#	by splitting the lattice into sub-graphs formed by the smallest possible loops,
#	that is, the smallest loops of four vertices (nodes) forming a square. 

import numpy as np
import multiprocessing
from joblib import Parallel, delayed, cpu_count
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Max product belief propagation on trees. 
import bp			


# The dtype to use to store energies. 
e_dtype = np.float64

class Slave:
	'''
	A class to store a slave. An instance of this class stores
	the list of nodes it contains, the list of edges it contains, 
	and the energies corresponding to all of them. The nodes are 
	arranged as

			0 ----- 1
			|       |
			|       |
			2 ----- 3.
	
	if it is a cell, or as 

	        0 --- 1 --- 2 --- 3 ...

	if it is a tree. 

	Members:
		node_list:			List of nodes in this slave.
		edge_list: 			List of edges in this slave. 
		node_energies:		Energies for every label in the slave.
							List of length 4 for cells. 
		n_labels:			The number of labels for each node.
							shape: (4,) for cells. 
		edge_energies:		Energies for each edge arranged in the order
							0-1, 0-2, 1-3, 2-3, obeying the vertex order 
							as well (for cells).
							List of length 4 for cells. 

	Notes
	=====
	The node and edge lists have indices corresponding to the global indexing
	of the lattice. The global indexing starts by labelling the top-left node 0
	(that is, the node at [0,0]), then increments the node index in a row-major
	fashion. The edge index follows the node index: as an edge is always from a 
	lower node index to a higher node index, we greedily label edges starting
	the 0-th node index. Hence, nodex index 0 has edges 0 and 1, node index 1 
	has edges 2 and 3, and so on. One should note that node index (cols-1) has
	only one edge: 2*(cols-1). Hence, all rows but the last one take up 
	(2*cols - 1) edges. The last row takes only (cols - 1) edges, since 
	it has no "down" edges. The total number of edges is according
	to this numbering is, hence, 
	
		(rows - 1)*(2*cols - 1) + (cols - 1)
	  = 2*rows*cols - rows - 2*cols + 1 + cols - 1
	  = 2*rows*cols - rows - cols
	  = rows*cols - rows + cols*rows - cols
	  = rows*(cols - 1) + cols*(rows - 1), 

	which matches the number of edges there are in the lattice. This implies
	the edge between node x and node y has the index

		t*(2*cols - 1) + 2*u + (y-x == 1)?1:0,

	where t = floor(x/cols) and u = mod(x, cols). The last term implies
	the "down" edge gets the lower index, while the "righ" edge gets
	the higher index. Going in the other direction, an edge with index e 
	is between two nodes x and y, where

		t = floor(e/(2*cols - 1))
		u = floor(mod(e, 2*cols - 1)/2)
		x = t*cols + u
		y = x + ((mod(e,2*cols - 1) - 2*u == 1)?1:cols).

	'''
	def __init__(self, node_list=None, edge_list=None, 
			node_energies=None, n_labels=None, edge_energies=None, graph_struct=None, struct='cell'):
		'''
		Slave.__init__(): Initialise parameters for this slave, if given. 
						  Parameters are None by default. 
		'''
		if struct not in ['cell', 'tree']:
			print 'Slave struct not recognised: %s.' %(struct)
			raise ValueError

		self.node_list		= node_list
		self.edge_list		= edge_list
		self.node_energies	= node_energies
		self.n_labels		= n_labels
		self.edge_energies	= edge_energies
		self.graph_struct	= graph_struct
		self.struct			= struct				# Whether a cell or a tree. 
								
	def set_params(self, node_list, edge_list, 
			node_energies, n_labels, edge_energies, graph_struct, struct):
		'''
		Slave.set_params(): Set parameters for this slave.
							Parameters must be specified.
		'''
		if struct not in ['cell', 'tree']:
			print 'Slave struct not recognised: %s.' %(struct)
			raise ValueError

		self.node_list		= node_list
		self.edge_list		= edge_list
		self.node_energies	= node_energies
		self.n_labels		= n_labels
		self.edge_energies	= edge_energies
		self.graph_struct	= graph_struct
		self.struct			= struct

		# These dictionaries enable to determine easily at which 
		#    index in node_list or edge_list, a particular node
		#    or edge is. 
		self.node_map		= {}
		self.edge_map		= {}
		for i in range(self.node_list.size):
			self.node_map[node_list[i]] = i
		for i in range(self.edge_list.size):
			self.edge_map[edge_list[i]] = i

		# Generate all possible labellings for this slave. This might take some time everytime, 
		#    so it is better to generate them once and store them in memory. 
		if struct is 'cell':
			[ni, nj, nk, nl]    = self.n_labels
			self.all_labellings = np.array([[i,j,k,l] for i in range(ni) for j in range(nj) 
								for k in range(nk) for l in range(nl)])

	def get_params(self):
		'''
		Slave.get_params(): Return parameters of this slave
		'''
		return self.struct, self.node_list, self.edge_list, self.node_energies, self.n_labels, self.edge_energies

	def set_labels(self, labels):
		'''
		Slave.set_labels():	Set the labelling for a slave
		'''
		self.labels	= np.array(labels, dtype=np.int)

		# Also maintain a dictionary to easily fetch the label 
		#	given a node ID.
		self.label_from_node	= {}
		for i in range(self.n_labels.size):
			n_id = self.node_list[i]
			self.label_from_node[n_id] = self.labels[i]

	def get_node_label(self, n_id):
		'''
		Retrieve the label of a node in the current labelling. 
		The node ID is relative to the Lattice, and not the Slave. 
		'''
		if n_id not in self.node_list:
			print self.node_list,
			print 'Node %d is not in this slave.' %(n_id)
			raise ValueError
		return self.label_from_node[n_id]


	def optimise(self):
		'''
		Optimise this slave. 
		'''
		if self.struct == 'cell':
			return _optimise_4node_slave(self)
		elif self.struct == 'tree':
			return _optimise_tree(self)
		else:
			print 'Slave struct not recognised: %s.' %(self.struct)
			raise ValueError

	def _compute_energy(self):
		'''
		Slave._compute_energy(): Computes the energy corresponding to
								 the labels. 
		'''
		if self.struct == 'cell':
			self._energy	= _compute_4node_slave_energy(self.node_energies, self.edge_energies, self.labels)
		elif self.struct == 'tree':
			self._energy	= _compute_tree_slave_energy(self.node_energies, self.edge_energies, self.labels, self.graph_struct)
		else:
			print 'Slave struct not recognised: %s.' %(self.struct)
			raise ValueError
		return self._energy

# ---------------------------------------------------------------------------------------


class Lattice:
	'''
	A class which serves as an API to create the required lattice. 
	The class requires as input the height and width of the lattice when it is created. 
	Node energies and edge energies can be added later. Only after the addition of 
	node and edge energies, can the user proceed to its optimisation. The optimisation
	is done by first breaking the lattice into slaves (sub-graphs), and then iteratively solving
	them according to the DD-MRF algorithm. 

	Members:
		rows:			The number of rows in the lattice. 
		cols:			The number of columns in the lattice.

	Notes
	=====
	The node and edge lists have indices corresponding to the global indexing
	of the lattice. The global indexing starts by labelling the top-left node 0
	(that is, the node at [0,0]), then increments the node index in a row-major
	fashion. The edge index follows the node index: as an edge is always from a 
	lower node index to a higher node index, we greedily label edges starting
	the 0-th node index. Hence, nodex index 0 has edges 0 and 1, node index 1 
	has edges 2 and 3, and so on. One should note that node index (cols-1) has
	only one edge: 2*(cols-1). Hence, all rows but the last one take up 
	(2*cols - 1) edges. The last row takes only (cols - 1) edges, since 
	it has no "down" edges. The total number of edges is according
	to this numbering is, hence, 
	
		(rows - 1)*(2*cols - 1) + (cols - 1)
	  = 2*rows*cols - rows - 2*cols + 1 + cols - 1
	  = 2*rows*cols - rows - cols
	  = rows*cols - rows + cols*rows - cols
	  = rows*(cols - 1) + cols*(rows - 1), 

	which matches the number of edges there are in the lattice. This implies
	the edge between node x and node y has the index

		t*(2*cols - 1) + 2*u + (y-x == 1)?1:0,

	where t = floor(x/cols) and u = mod(x, cols). The last term implies
	the "down" edge gets the lower index, while the "right" edge gets
	the higher index. Going in the other direction, an edge with index e 
	is between two nodes x and y, where

		t = floor(e/(2*cols - 1))
		u = floor(mod(e, 2*cols - 1)/2)
		x = t*cols + u
		y = x + ((mod(e,2*cols - 1) - 2*u == 1)?1:cols).

	'''

	def __init__(self, rows, cols, n_labels):
		'''
		Lattice.__init__(): Initialise the lattice to be of shape (rows, cols), with 
                            a node taking a maximum of n_labels. 
		'''
		# The rows, columns, and node indexing. 
		self.rows		= rows
		self.cols		= cols
		self.n_nodes	= rows*cols
		self.n_edges	= rows*(cols-1) + cols*(rows-1)

		# An array to store the final labels. 
		self.labels		= np.zeros(self.n_nodes)

		# To set the maximum number of labels, we consider what kind of input is n_labels. 
		# If n_labels is an integer, we assume that all nodes should get the same max_n_label.
		# Another option is to specify a list of max_n_labels. 
		if type(n_labels) == np.int:
			self.n_labels	= n_labels + np.zeros(self.n_nodes).astype(np.int)
		elif np.array(n_labels).size == self.n_nodes:
			self.n_labels	= np.array(n_labels).astype(np.int)
		# In any case, the max n labels for this Lattice is np.max(self.n_lables)
		self.max_n_labels	= np.max(self.n_labels)
		
		# Initialise the node and edge energies. 
		self.node_energies	= np.zeros((self.n_nodes, self.max_n_labels))
		self.edge_energies	= np.zeros((self.n_edges, self.max_n_labels, self.max_n_labels))

		# Flags set to ensure that node and edge energies have been set. If any energies
		# 	have not been set, we cannot proceed to optimisation as the lattice is not complete. 
		self.node_flags	= np.zeros(self.n_nodes, dtype=np.bool)
		self.edge_flags	= np.zeros(self.n_edges, dtype=np.bool)

		# A variable which tells us how to estimate primal solutions. 
		self._est_prim = 'bp'


	def set_node_energies(self, i, energies):
		'''
		Lattice.set_node_energies(): Set the node energies for node i. 
		'''
		# Convert the energy to a numpy array
		energies = np.array(energies, dtype=e_dtype)

		if energies.size != self.n_labels[i]:
			print 'Lattice.set_node_energies(): The supplied node energies do not agree',
			print '(%d) on the number of labels required (%d).' %(energies.size, self.n_labels[i])
			raise ValueError

		# Make the assignment: set the node energies. 
		self.node_energies[i, 0:self.n_labels[i]] = energies
		# Set flag for this node to True.
		self.node_flags[i]		= True

	def set_edge_energies(self, i, j, energies):
		'''
		Lattice.set_edge_energies(): Sets the edge energies for edge (i,j). The
		function first checks for the possibility of an edge between i and j, 
		and makes the assignment only if such an edge is possible.
		'''
		# Convert indices to int, just in case ...
		i = np.int(i)
		j = np.int(j)

		# Convert the energy to a numpy array
		energies = np.array(energies, dtype=e_dtype)

		# Check that the supplied energy has the correct shape. 
		input_shape		= list(energies.shape)
		reqd_shape		= [self.n_labels[i], self.n_labels[j]]
		if input_shape != reqd_shape:
			print 'Lattice.set_edge_energies(): The supplied energies have invalid shape:',
			print '(%d, %d). It must be (%d, %d).' \
                         %(energies.shape[0], energies.shape[1], self.n_labels[i], self.n_labels[j])
			raise ValueError

		# Check that indices are not out of range. 
		if i >= self.n_nodes or j >= self.n_nodes:
			print 'Lattice.set_edge_energies(): At least one of the supplied edge indices is invalid.'
			raise IndexError
		# Check for the possibility of an edge. 
		if (j - i != 1) and (j - i != self.cols):
			# This signifies j is neither the node to the right of i, nor the node below i. 
			print 'Lattice.set_edge_energies(): The supplied edge indices are not consistent - this 2D',
			print 'lattice does not have an edge from %d to %d.' %(i, j)
			raise ValueError

		# We can proceed - everything is okay. 
		edge_id	= self._edge_id_from_node_ids(i, j)

		# Make assignment: set the edge energies. 
		self.edge_energies[edge_id,:self.n_labels[i],:self.n_labels[j]]	= energies
		self.edge_flags[edge_id]	= True


	def check_completeness(self):
		'''
		Lattice.check_completeness(): Check whether all attributes of the lattice have been set.
        This must return True before we proceed to optimisation.
		'''
		# Check whether all nodes have been set. 
		if np.sum(self.node_flags) < self.n_nodes:
			return False
		# Check whether all edges have been set. 
		if np.sum(self.edge_flags) < self.n_edges:
			return False
		# Everything is okay. 
		return True

	
	def _create_slaves(self, decomposition='cell'):
		'''
		Lattice._create_slaves(): Create slaves for this particular lattice.
		The default decomposition is 'cell'. If 'row_col' is specified, create a set of trees
		instead - one for every row and every column. 
		If 'rook' is specified, create one slave for every vertex - defined by permitted rook moves
		from that vertex. That is, for a point (i,j) we have a tree that extends in all directions from 
		(i, j). 
		'''

		# self._max_nodes_in_slaves, and self._max_edges_in_slaves are used
		#   to simplify node and edge updates. They shall be computed by the
		#   _create_*_slaves() functions, whichever is called. 
		self._max_nodes_in_slave = 0 
		self._max_edges_in_slave = 0

		# Functions to call depending on which slave is chosen
		_slave_funcs = {
			'cell':    self._create_cell_slaves,
			'row_col': self._create_row_col_slaves,
			'rook':    self._create_rook_slaves
		}
		
		if decomposition not in _slave_funcs.keys():
			print 'decomposition must be one of', _slave_funcs.keys()
			raise ValueError

		# Create slaves depending on what decomposition is requested.
		_slave_funcs[decomposition]()

		# Two variables to hold how many slaves each node and edge is contained in (instead
		#   of computing the size of the corresponding vector each time. 
		self._n_slaves_nodes = np.array([self.nodes_in_slaves[n].size for n in range(self.n_nodes)], dtype=np.int)
		self._n_slaves_edges = np.array([self.edges_in_slaves[e].size for e in range(self.n_edges)], dtype=np.int)
		# Now we need only check those nodes and edges which associate with at least two slaves. 
		self._check_nodes    = np.where(self._n_slaves_nodes > 1)[0]
		self._check_edges    = np.where(self._n_slaves_edges > 1)[0]

		# Finally, we must modify the energies for every edge or node depending on 
		#   how many slaves it is a part of. The energy for a node/edge is distributed
		#   equally among all slaves. 
		for n_id in np.where(self._n_slaves_nodes > 1)[0]:
			# Retrieve all the slaves this node is part of.
			s_ids	= self.nodes_in_slaves[n_id]
			# Distribute this node's energy equally between all slaves.
			for s in s_ids:
				n_id_in_slave	= self.slave_list[s].node_map[n_id]
				self.slave_list[s].node_energies[n_id_in_slave] /= 1.0*s_ids.size

		# Doing the same for edges ...
		for e_id in np.where(self._n_slaves_edges > 1)[0]:
			# Retrieve all slaves this edge is part of.
			s_ids	= self.edges_in_slaves[e_id]
			# Distribute this edge's energy equally between all slaves. 
			for s in s_ids:
				e_id_in_slave	= self.slave_list[s].edge_map[e_id]
				self.slave_list[s].edge_energies[e_id_in_slave] /= 1.0*s_ids.size

		# That is it. The slaves are ready. 

	def _create_cell_slaves(self):
		'''
		Lattice._create_cell_slaves(): Create a set of cell slaves - one slave for every 
		cell in the lattice. There are (rows-1)*(cols-1) slaves. 
		Convergence for this decomposition is slow, but the relaxation is tighter. 
		'''
		# The number of slaves.
		self.n_slaves		= (self.rows - 1)*(self.cols - 1)
		# Create empty slaves initially. 
		self.slave_list		= [Slave() for i in range(self.n_slaves)]
		
		# The following lists record for every node and edge, to which 
		#	slaves it belongs.
		self.nodes_in_slaves	= [[] for i in range(self.n_nodes)]
		self.edges_in_slaves	= [[] for i in range(self.n_edges)]
		# This also means we can assign these values. 
		self._max_nodes_in_slave = 4
		self._max_edges_in_slave = 4

		# Iterate over the number of slaves to create each one. 
		slave_ids	= [[i,j] for i in range(self.rows-1) for j in range(self.cols-1)]

		# Iterate over slave IDs to create slaves. 
		for id_pair in slave_ids:
			[i, j]	= id_pair
			# The slave id. 
			s_id	= i*(self.cols - 1) + j
	
			# A slave with id s_id, slave coordinates (i,j) 
			# 	contains nodes with coordinates (i,j), (i+1,j), (i,j+1), and (i+1,j+1).
			i_list		= np.array([i, i, i+1, i+1], dtype=np.int)
			j_list		= np.array([j, j+1, j, j+1], dtype=np.int)
			node_list	= i_list*self.cols + j_list

			# The number of labels can be easily extracted from self.n_labels.
			n_labels	= np.zeros(4, dtype=np.int)
			n_labels[:]	= self.n_labels[node_list].astype(np.int)

			# The included edges are (i,j)-(i,j+1), (i,j)-(i+1,j), 
			#	(i,j+1)-(i+1,j+1), and (i+1,j)-(i+1,j+1), in that order. 
			# We will make four IDs, e1, ..., e4, for the four edges in this slave. 
			# These edges MUST be in the order specified above. 
			e1			= self._edge_id_from_node_ids(node_list[0], node_list[1])			#	(i,j)-(i,j+1)
			e2			= self._edge_id_from_node_ids(node_list[0], node_list[2])			#	(i,j)-(i+1,j)
			e3			= self._edge_id_from_node_ids(node_list[1], node_list[3]) 			#	(i,j+1)-(i+1,j+1)
			e4			= self._edge_id_from_node_ids(node_list[2], node_list[3])			#	(i+1,j)-(i+1,j+1)
			edge_list	= np.array([e1, e2, e3, e4])
	
			# The node energies for this slave
			node_energies	= np.zeros((4, self.max_n_labels), dtype=e_dtype) 
			node_energies[:] = self.node_energies[node_list,:]
	
			# We now extract the edge energies, which are easy to extract as well, as we know
			# 	the edge IDs for all edges in this slave. 
			# e_label_list stores the required size of the energy matrix for edge in edge_list
			e_label_list 	 = [(n_labels[0], n_labels[1]), \
						   	    (n_labels[0], n_labels[2]), \
							    (n_labels[1], n_labels[3]), \
							    (n_labels[2], n_labels[3])]
			edge_energies    = np.zeros((4, self.max_n_labels, self.max_n_labels), dtype=e_dtype)
			edge_energies[:] = self.edge_energies[edge_list,:,:]
	
			# Make assignments for this slave. 
			self.slave_list[s_id].set_params(node_list, edge_list, node_energies, n_labels, edge_energies, None, 'cell')

			# Finally, add this slave the appropriate nodes_in_slaves, and edges_in_slaves.
			for n_id in node_list:
				self.nodes_in_slaves[n_id] 	+= [s_id]
			for e_id in edge_list:
				self.edges_in_slaves[e_id]	+= [s_id]

		# For convenience, turn the individual lists in nodes_in_slaves and edges_in_slaves into numpy arrays. 
		self.nodes_in_slaves	= [np.array(t) for t in self.nodes_in_slaves]
		self.edges_in_slaves	= [np.array(t) for t in self.edges_in_slaves]

			

	def _create_row_col_slaves(self):
		'''
		Lattice._create_row_col_slaves(): Create a set of slaves - one for each row and each column.
		There are hence (row + col) slaves. 
		Convergence for this decomposition is faster but the relaxation is not as tight as 
		the cell decomposition.
		'''
		# The number of slaves.
		self.n_slaves		= self.rows + self.cols
		# Create empty slaves initially. 
		self.slave_list		= np.array([Slave() for i in range(self.n_slaves)])
		
		# The following lists record for every node and edge, to which 
		#	slaves it belongs. A little-bit of hard-coding here. We know 
		#   each node is in two slaves, while each edge is in just one. 
		# More efficient than adding to lists. 
		self.nodes_in_slaves	= [np.zeros(2,dtype=np.int) for i in range(self.n_nodes)]
		self.edges_in_slaves	= [np.zeros(1,dtype=np.int) for i in range(self.n_edges)]
		# This also means we can automatically assign these values:
		self._max_nodes_in_slave = np.max([self.rows, self.cols])
		self._max_edges_in_slave = self._max_nodes_in_slave - 1

		# Iterate over the number of slaves to create each one. 
		# The rows get preference in terms of ID. Slaves numbered 0 through self.rows-1 are 
		#    row slaves, while those numbered self.rows to self.n_slaves-1 are column
		#    slaves. 
		slave_ids	= np.arange(self.n_slaves)

		# Create adjacency matrix for row slaves. 
		adj_row	= np.eye(self.cols, dtype=np.bool)			# A row slave as self.cols verticies. 
		# Roll this matrix two ways so that it indicates edges between adjacent vertices. 
		adj_row = np.roll(adj_row, 1, axis=1)
		# Add to its transpose to make it symmetric. Then fix corner elements. 
		adj_row = adj_row + adj_row.T
		adj_row[0, -1]  = adj_row[-1, 0] = False

		# Create adjacency matrix for column slaves. 
		adj_col = np.eye(self.rows, dtype=np.bool)			# A column slave has self.rows vertices. 	
		# Roll this matrix two ways so that it indicates edges between adjacent vertices. 
		adj_col = np.roll(adj_col, 1, axis=1)
		# Add to transpose to make it symmetric. Then fix corner elements. 
		adj_col = adj_col + adj_col.T
		adj_col[0, -1] = adj_col[-1, 0] = False

		for s_id in slave_ids[:self.rows]:
			# Create row slaves. 
			node_list	= np.arange(self.cols*s_id, self.cols*(s_id+1))
			edge_list	= np.zeros(self.cols - 1, dtype=np.int)

			# The number of labels for each node in this slave. 
			n_labels	= np.zeros(self.cols, dtype=np.int)
			n_labels[:]	= self.n_labels[node_list][:]

			node_energies = np.zeros((self.cols, self.max_n_labels), dtype=e_dtype)
			edge_energies = np.zeros((self.cols-1, self.max_n_labels, self.max_n_labels), dtype=e_dtype)

			# Extract node energies. 
			node_energies[:] = self.node_energies[node_list,:]

			# Extract edge energies. 
			for i in range(self.cols-1):
				x, y = node_list[i], node_list[i+1]
				# ID of this edge. 
				e_id			= self._edge_id_from_node_ids(x, y)
				edge_list[i]	= e_id
			edge_energies[:] = self.edge_energies[edge_list,:,:]

			# Make graph structure. 
			row_gs	= bp.make_graph_struct(adj_row, n_labels)
			# Set parameters for this slave. 
			self.slave_list[s_id].set_params(node_list, edge_list, node_energies, \
					n_labels, edge_energies, row_gs, 'tree')
			
			# Finally, add this slave to the lists of all nodes in node_list
			for n_id in node_list:
				# This is the first time each node is being assigned a slave. Hence, the "[0]".
				self.nodes_in_slaves[n_id][0]	= s_id
			for e_id in edge_list:
				# This is the first time these edges are being assigned to a slave. 
				self.edges_in_slaves[e_id][0]	= s_id


		for s_id in slave_ids[self.rows:]:
			# Create column slaves. 	
			node_list	= np.arange(0, self.rows)*self.cols + s_id - self.rows
			edge_list	= np.zeros(self.rows - 1, dtype=np.int)

			# The number of labels for each node in this slave. 
			n_labels	= np.zeros(self.rows, dtype=np.int)
			n_labels[:]	= self.n_labels[node_list][:]

			node_energies = np.zeros((self.rows, self.max_n_labels), dtype=e_dtype)
			edge_energies = np.zeros((self.rows-1, self.max_n_labels, self.max_n_labels), dtype=e_dtype)

			# Extract node energies.
			node_energies[:] = self.node_energies[node_list,:]

			# Extract edge energies. 
			for i in range(self.rows-1):
				x, y = node_list[i], node_list[i+1]
				# ID of this edge. 
				e_id			= self._edge_id_from_node_ids(x, y)
				edge_list[i]	= e_id 

			edge_energies[:] = self.edge_energies[edge_list,:,:]

			# Make graph structure. 
			col_gs = bp.make_graph_struct(adj_col, n_labels)
			# Set parameters for this slave. 
			self.slave_list[s_id].set_params(node_list, edge_list, node_energies, \
					n_labels, edge_energies, col_gs, 'tree')

			# Finally, add this slave to the lists of all nodes in the node list
			for n_id in node_list:
				# Again - some hard-coding here. This is the second time a slave is being added to each node. 
				# Hence, the "[1]".
				self.nodes_in_slaves[n_id][1]	= s_id
			for e_id in edge_list:
				# This is the first time these edges are being assigned to a slave. 
				self.edges_in_slaves[e_id][0]	= s_id

	
	def _create_rook_slaves(self):
		'''
		Lattice._create_rook_slaves(): Create slaves defined by all possible rook moves from a vertex, 
		for every vertex. That is, for every vertex, create a slave that includes its entire row and its
		entire column.
		'''
		# The number of slaves.
		self.n_slaves		= self.rows*self.cols
		# Create empty slaves initially. 
		self.slave_list		= np.array([Slave() for i in range(self.n_slaves)])
		
		# The following lists record for every node and edge, to which 
		#	slaves it belongs. A little-bit of hard-coding here. We know 
		#   each node is in (self.rows+self.cols-1) slaves.
		# More efficient than adding to lists. 
		self.nodes_in_slaves	= [[] for i in range(self.n_nodes)]
		self.edges_in_slaves	= [[] for i in range(self.n_edges)]
		# This also means we can automatically assign these values:
		self._max_nodes_in_slave = self.rows + self.cols - 1
		self._max_edges_in_slave = self.rows + self.cols - 2

		# Iterate over the number of slaves to create each one. 
		# Slave IDs correspond to node IDs.
		slave_ids	= np.arange(self.n_nodes)

		# Create each slave. 
		for s_id in slave_ids:
			# Get the centre point. 
			x, y = s_id/self.cols, s_id%self.cols

			# Creating node and edge lists: 
			# We add nodes on the same row first, then nodes on the same column. 
			node_list = x*self.cols + np.arange(self.cols)
			node_list = np.concatenate((node_list, np.array(range(x)+range(x+1,self.rows))*self.cols + y))

			# Add edges in this row.
			edge_list = [self._edge_id_from_node_ids(node_list[i], node_list[i+1]) for i in range(np.min([y+1,self.cols-1]))]
			# Add edges from id_xy to its top and bottom neighbours. This is done because of the ordering restriction 
			#    required by max-product BP. When adj_mat is flattened, the order in which True appears must signify
			#    the order in which edges appear in edge_pot. Hence, we must add these edges here, and, consequently, 
			#    add the edge energies also in the same order as these edges. 
			if x > 0:
				tn_id     = self.cols + x - 1
				edge_list += [self._edge_id_from_node_ids(node_list[tn_id], node_list[y])]		# Top neighbour. 
			if x + 1 < self.rows:
				bn_id     = self.cols + x
				edge_list += [self._edge_id_from_node_ids(node_list[y], node_list[bn_id])]		# Bottom neighbour. 

			edge_list += [self._edge_id_from_node_ids(node_list[i], node_list[i+1]) for i in range(y+1,self.cols-1)]
			# Add edges in this column. 
			_edges_column = [[i, i+1] for i in range(self.cols, node_list.size-1)
	                          if (node_list[i+1] - node_list[i] == self.cols)]
			edge_list += [self._edge_id_from_node_ids(node_list[e[0]], node_list[e[1]]) for e in _edges_column]
			edge_list = np.array(edge_list, dtype=np.int)

			# The number of labels for each node in this slave
			n_labels    = np.zeros(node_list.size, dtype=np.int)
			n_labels[:] = self.n_labels[node_list]	

			# Create node and edge energies
			node_energies = np.zeros((node_list.size, self.max_n_labels))
			edge_energies = np.zeros((edge_list.size, self.max_n_labels, self.max_n_labels))

			# Extract node and edge energies. 
			node_energies[:] = self.node_energies[node_list,:]
			edge_energies[:] = self.edge_energies[edge_list,:,:]

			# Create adjacency matrix. 
			adj_mat = np.zeros((node_list.size, node_list.size), dtype=np.bool)
			# Add edges for the "row part" of the rook. 
			adj_mat[:self.cols,:self.cols] = np.roll(np.eye(self.cols,dtype=np.bool), 1, axis=1)
			adj_mat[self.cols-1, 0]        = False 			# Fix the incorrect `True` introduced by np.roll
			# Add top neighbour of id_xy.
			if x > 0:
				adj_mat[y, tn_id] = True
			# Add bottom neighbour of id_xy.
			if x + 1 < self.rows:
				adj_mat[y, bn_id] = True
			# Add edges for the "column part" of the rook.
			for e in _edges_column:
				adj_mat[e[0],e[1]] = True
			# Finally, make adjacency matrix symmetric.
			adj_mat = adj_mat + adj_mat.T

			# TODO: Issues so far:
			# 1. create adjacency matrix. 
			# 2. bp.py assumes that edge_pot are received so that they agree with the ordering in adj_mat,
			#    that is, if adj_mat is row-first flattened, the `True`s will appear in the same order as
			#    the specified edge_energies. 

			# Make graph structure. 
			row_gs	= bp.make_graph_struct(adj_mat, n_labels)
			# Set parameters for this slave. 
			self.slave_list[s_id].set_params(node_list, edge_list, node_energies, \
					n_labels, edge_energies, row_gs, 'tree')

			# Finally, add this slave to the lists of all nodes in the node list
			for n_id in node_list:
				self.nodes_in_slaves[n_id] += [s_id]
			for e_id in edge_list:
				self.edges_in_slaves[e_id] += [s_id]

		# For convenience, turn the individual lists in nodes_in_slaves and edges_in_slaves into numpy arrays. 
		self.nodes_in_slaves	= [np.array(t) for t in self.nodes_in_slaves]
		self.edges_in_slaves	= [np.array(t) for t in self.edges_in_slaves]


	def _create_custom_slaves(self, slave_list):
		'''
		Create a custom decomposition of the Graph. This function allows the user to 
		create a custom decomposition of the graph and apply this decomposition on 
		an instance of Graph. 

		Inputs
		======
	        slave_list: A series of instances of type Slave. Could 
                        be a list or a Numpy array. Each member of 'slave_list' 
	                    must be of type Slave, and have all the required members
	                    initialised. 
	
		'''

		# Convert to Numpy array. 
		slave_list = np.array(slave_list)

		# Assign to self.slave_list
		self.slave_list = slave_list

		# The number of slaves. 
		self.n_slaves = slave_list.size

		# Create empty lists for nodes_in_slaves and edges_in_slaves. 
		self.nodes_in_slaves = [[] for i in range(self.n_nodes)]
		self.edges_in_slaves = [[] for i in range(self.n_edges)]

		# Initialise _max_*_in_slave
		self._max_nodes_in_slave = 0
		self._max_edges_in_slave = 0

		for s_id in range(self.n_slaves):
			# Get node and edge lists. 
			node_list = slave_list[s_id].node_list 
			edge_list = slave_list[s_id].edge_list

			# Number of nodes and edges. 
			n_nodes   = node_list.size
			n_edges   = edge_list.size

			# Update self._max_nodes_in_slave, and self._max_edges_in_slave
			if self._max_nodes_in_slave < n_nodes:
				self._max_nodes_in_slave = n_nodes
			if self._max_edges_in_slave < n_edges:
				self._max_edges_in_slave = n_edges

			# Add them to nodes_in_slaves and edges_in_slaves. 
			for n_id in node_list:
				self.nodes_in_slaves[n_id] += [s_id]
			for e_id in edge_list:
				self.edges_in_slaves[e_id] += [s_id]

		# Convert lists in self.nodes_in_slaves and self.edges_in_slaves to 
		#    Numpy arrays for convenience. 
		self.nodes_in_slaves = [np.array(t) for t in self.nodes_in_slaves]
		self.edges_in_slaves = [np.array(t) for t in self.edges_in_slaves]


	def optimise(self, a_start=1.0, max_iter=1000, decomposition='cell', strategy='step', _momentum=0.0, _verbose=True):
		'''
		Lattice.optimise(): Optimise the set energies over the lattice and return a labelling. 

		Takes as input a_start, which is a float and denotes the starting value of \\alpha_t in
		the DD-MRF algorithm. 

		struct specifies the type of decomposition to use. struct must be in ['cell', 'row_col']. 
		'cell' specifies a decomposition in which the lattice in broken into 2x2 cells - each 
		being a slave. 'row_col' specifies a decomposition in which the lattice is broken into
		rows and columns - each being a slave. 

		The strategy signifies what values of \\alpha to use at iteration t. Permissible 
		values are 'step' and 'adaptive'. The step strategy simply sets 

		      \\alpha_t = a_start/sqrt(t).

		The adaptive strategy sets 
		 
		      \\alpha_t = a_start*\\frac{Approx_t - Dual_t}{norm(\\nabla g_t)**2},

		where \\nabla g_t is the subgradient of the dual at iteration t. 
		'''

		# Check if a permissible decomposition is used. 
		if decomposition not in ['cell', 'row_col', 'rook']:
			print 'Permissible values for decomposition are \'cell\' and \'row_col\'.'
			raise ValueError

		# Check if a permissible strategy is being used. 
		if strategy not in ['step', 'step_ss', 'step_sg', 'adaptive', 'adaptive_d']:
			print 'Permissible values for strategy are \'step\', \'step_sg\', \'adaptive\', and \'adaptive_d\''
			print '\'step\'        Use diminshing step-size rule: a_t = a_start/sqrt(it).'
			print '\'step_ss\'     Use a square summable but not summable sequence: a_t = a_start/(1.0 + t).'
			print '\'step_sg\'     Use subgradient in combination with diminishing step-size rule: a_t = a_start/(sqrt(it)*||dg||**2).'
			print '\'adaptive\'    Use adaptive rule given by the difference between the estimated PRIMAL cost and the current DUAL cost: a_t = a_start*(PRIMAL_t - DUAL_t)/||dg||**2.'
			print '\'adaptive_d\'  Use adaptive rule with diminishing step-size rule: a_t = a_start*(PRIMAL_t - DUAL_t)/(sqrt(it)*||dg||**2).'
			raise ValueError
		# If strategy is adaptive, we would like a_start to be in (0, 2).
		if strategy is 'adaptive' and (a_start <= 0 or a_start > 2):
			print 'Please use 0 < a_start < 2 for an adaptive strategy.'
			raise ValueError

		# Momentum must be in [0, 1)
		if _momentum < 0 or _momentum >= 1:
			print 'Momentum must be in [0, 1).'
			raise ValueError

		# First check if the lattice is complete. 
		if not self.check_completeness():
			n_list, e_list	= self._find_empty_attributes()
			print 'Lattice.optimise(): The lattice is not complete.'
			print 'The following nodes are not set:', n_list
			print 'The following edges are not set:', e_list
			raise AssertionError

		# Set the optimisation strategy. 
		self._optim_strategy = strategy

		self.decomposition = decomposition

		_naive_search = False

		# TEST: 
		# Restrict node and edge energies to be in [0, 1]
#		n_min, n_max = np.min(self.node_energies), np.max(self.node_energies)
#		e_min, e_max = np.min(self.edge_energies), np.max(self.edge_energies)
#		g_min, g_max = np.min([n_min, e_min]), np.max([n_max, e_max])
#		self.node_energies = (self.node_energies - g_min)/(g_max - g_min)
#		self.edge_energies = (self.edge_energies - g_min)/(g_max - g_min)
		# Scale a_start if any of the 'step' strategies are being used. 
#		a_start = a_start/(g_max - g_min)

		# Find the least "step" size in energy. This is given by the smallest difference
		#   between any two energy energies in the Lattice. 
		# HACK: If the difference between primal and dual energies at 
		#   an iteration is smaller than this step size, we can safely 
		#   perform a naive search on the conflicting nodes, and choose the smallest
		#   energy. 
		_all_ergs = np.concatenate((self.node_energies.flatten(), self.edge_energies.flatten()))
		_all_ergs = np.unique(_all_ergs)
		_erg_difs = [np.abs(_all_ergs[i] - _all_ergs[i+1]) for i in range(_all_ergs.size-1)]
		_erg_step = np.min(_erg_difs)
		self._erg_step = _erg_step

		# Create slaves. This creates a list of slaves and stores it in 
		# 	self.slave_list. The numbering of the slaves starts from the top-left,
		# 	and continues in row-major fashion. For example, there are 
		#   (self.rows-1)*(self.cols-1) slaves if the 'cell' decomposition is used. 
		self._create_slaves(decomposition=self.decomposition)

		# Create update variables for slaves. Created once, reset to zero each time
		#   _apply_param_updates() is called. 
		self._slave_node_up	= np.zeros((self.n_slaves, self._max_nodes_in_slave, self.max_n_labels))
		self._slave_edge_up	= np.zeros((self.n_slaves, self._max_edges_in_slave, self.max_n_labels, self.max_n_labels))
		# Create a copy of these to hold the previous state update. Akin to momentum
		#   update used in NNs. 
		self._prv_node_sg   = np.zeros_like(self._slave_node_up)
		self._prv_edge_sg   = np.zeros_like(self._slave_edge_up)
		# Array to mark slaves for updates. 
		# The first row corresponds to node updates, while the second to edge updates. 
		self._mark_sl_up	= np.zeros((2,self.n_slaves), dtype=np.bool)

		# How much momentum to use. Must be in [0, 1)
		self._momentum = _momentum
	
		# Whether converged or not. 
		converged = False

		# The iteration
		it = 1

		# Set all slaves to be solved at first. 
		self._slaves_to_solve	= np.arange(self.n_slaves)

		# Two lists to record the primal and dual cost progression
		self.dual_costs			= []
		self.primal_costs		= []
		self.subgradient_norms	= []

		self._best_primal_cost	= np.inf
		self._best_dual_cost	= -np.inf

		# Loop till not converged. 
		while not converged and it <= max_iter:
			if _verbose:
				print 'Iteration %5d. Solving %5d subproblems ...' %(it, self._slaves_to_solve.size),
			# Solve all the slaves. 
			# The following optimises the energy for each slave, and stores the 
			#    resulting labelling as a member in the slaves. 
			self._optimise_slaves()
			if _verbose:
				print 'done.',
			sys.stdout.flush()

			# Find the number of disagreeing points. 
			disagreements = self._find_conflicts()

			# Get the primal cost at this iteration
			primal_cost     = self._compute_primal_cost()
			if self._best_primal_cost > primal_cost:
				self._best_primal_cost = primal_cost
			self.primal_costs += [primal_cost]

			# Get the dual cost at this iteration
			dual_cost       = self._compute_dual_cost()
			if self._best_dual_cost < dual_cost:
				self._best_dual_cost = dual_cost
			self.dual_costs	+= [dual_cost]

			# Verify whether the algorithm has converged. If all slaves agree
			#    on the labelling of every node, we have convergence. 
			if self._check_consistency():
				print 'Converged after %d iterations!\n' %(it)
				print 'At convergence, PRIMAL = %.6f, DUAL = %.6f, Gap = %.6f.' %(primal_cost, dual_cost, primal_cost - dual_cost)
				# Finally, assign labels.
				self._assign_labels()
				# Break from loop.
				converged = True
				break


			# Test: #TODO
			# If disagreements are less than or equal to 2, we do a brute force
			#    to search for the solution. 
			# Not a very good hack. Does not have a good argument. _naive_search is False, 
			#    best not to use it.
			if _naive_search and primal_cost-dual_cost <= self._erg_step:
				print 'Forcing naive search as _naive_search is True, and difference in primal and dual costs <= _erg_step.'
				self.force_naive_search(disagreements, response='y')
				break

			# Apply updates to parameters of each slave. 
			alpha = self._apply_param_updates(a_start, it)

			# Print statistics. .
			if _verbose:
				print ' alpha = %10.6f. n_miss = %6d.' %(alpha, disagreements.size),
				print '||dg||**2 = %6.2f, PRIMAL = %12.6f. DUAL = %12.6f, P - D = %12.6f, min(P - D) = %12.6f' \
				%(self.subgradient_norms[-1], primal_cost, dual_cost, primal_cost-dual_cost, self._best_primal_cost - self._best_dual_cost)

			# Switch to step strategy if n_miss = disagreements.size < 5% of number of nodes. 
			if self._optim_strategy is 'adaptive' and  disagreements.size < 0.000*self.n_nodes:
				print 'Switching to step strategy as n_miss < 0.1% of the number of nodes.'
				a_start = alpha
				self._optim_strategy = 'step'

			# Increase iteration.
			it += 1

		print 'The final labelling is stored as a member \'labels\' in the object.'
		

	def _optimise_slaves(self):
		'''
		A function to optimise all slaves. This function distributes the job of
        optimising every slave to all but one cores on the machine. 
		'''
		# Extract the list of slaves to be optimised. This contains of all the slaves that
		# 	disagree with at least one other slave on the labelling of at least one node. 
		_to_solve	= [self.slave_list[i] for i in self._slaves_to_solve]
		# The number of cores to use is the number of cores on the machine minus 1. 
		# Also, use only as many cores as needed. 
		n_cores		= np.min([cpu_count() - 1, len(_to_solve)])

		# Optimise the slaves. 
		# Using Joblib. 
		optima		= Parallel(n_jobs=n_cores)(delayed(_optimise_slave)(s) for s in _to_solve)
# --- Comment the previous line, and uncomment the following three lines if you wish to solve
# ---   the slaves sequentially instead of parallelly.
#		optima = []
#		for s in _to_solve:
#			optima += [_optimise_slave(s)]

		# Reflect the result in slave list for our Lattice. 
		for i in range(self._slaves_to_solve.size):
			s_id = self._slaves_to_solve[i]
			self.slave_list[s_id].set_labels(optima[i][0])
			self.slave_list[s_id]._energy = optima[i][1]
			if self.slave_list[s_id].struct is 'tree':
				self.slave_list[s_id]._messages = optima[i][2]
				self.slave_list[s_id]._messages_in = optima[i][3]
#			self.slave_list[s_id]._compute_energy()

	# End of Lattice._optimise_slaves()

	
	def _apply_param_updates(self, a_start, it):
		'''
		Apply updates to energies after one iteration of the DD-MRF algorithm. 
		'''

		# Flags to determine whether to solve a slave.
		slave_flags	= np.zeros(self.n_slaves, dtype=bool)

		# Compute the L2-norm of the subgradient. 
		norm_gt	= 0.0

		# Change of strategy here: Instead of iterating over all labels, we create 
		#   vectors so that updates can be very easily calculated using array operations!
		# A huge speed up is expected. 
		# Con: Memory usage is increased. 
		# Reset update variables for slaves. 
		self._slave_node_up[:] = 0.0
		self._slave_edge_up[:] = 0.0

		# Mark which slaves need updates. A slave s_id needs update only if self._slave_node_up[s_id] 
		#   has non-zero values in the end.
		self._mark_sl_up[:]	= False
	
		# We iterate over nodes and edges which associate with at least two slaves
		#    and calculate updates to parameters of all slaves. 
		for n_id in self._check_nodes:
			# Retrieve the list of slaves that use this node. 
			s_ids			= self.nodes_in_slaves[n_id]
			n_slaves_nid	= s_ids.size
	
			# Retrieve labels assigned to this point by each slave, and make it into a one-hot vector. 
	#			ls_		= [self.slave_list[s].get_node_label(n_id) for s in s_ids]
			ls_		= np.array([
					    make_one_hot(self.slave_list[s].get_node_label(n_id), self.n_labels[n_id])
					    for s in s_ids])
			ls_avg_	= np.mean(ls_, axis=0, keepdims=True)
	
			# Check if all labellings for this node agree. 
			if np.max(ls_avg_) == 1:
				# As all vectors are one-hot, this condition being true implies that 
				#   all slaves assigned the same label to this node (otherwise, the maximum
				#   number in ls_avg_ would be less than 1).
				continue

			# The next step was to iterate over all slaves. We calculate the subgradient here
			#   given by 
			#   
			#    \delta_\lambda^s_p = x^s_p - ls_avg_
			#
			#   for all s. s here signifies slaves. 
			# This can be very easily done with array operations!
			_node_up	= ls_ - np.tile(ls_avg_, [n_slaves_nid, 1])
	
			# Find the node ID for n_id in each slave in s_ids. 
			sl_nids = [self.slave_list[s].node_map[n_id] for s in s_ids]
	
			# Mark this update to be done later. 
			self._slave_node_up[s_ids, sl_nids, :self.n_labels[n_id]]  = _node_up #:self.n_labels[n_id]] = _node_up
			# Mark this slave for node updates. 
			self._mark_sl_up[0, s_ids] = True
	
			# Iterate over all labels. The update to the energy of label l_id is given by
			#
			#		\delta_l_id	= \alpha*(x_s(l_id) - \frac{\sum_s' x_s'(l_id)}{\sum_s' 1})
			#
	#			for l_id in np.unique(ls_):#(self.n_labels[n_id]):
	#				# Find slaves that assign this point the label l_id. 
	#				slaves_with_this_l_id		= np.array([s_ids[i] for i in np.where(ls_ == l_id)[0]])
	#				slaves_without_this_l_id	= np.array([s_ids[i] for i in np.where(ls_ != l_id)[0]])
	#
	#				# Calculate updates, given by the equation above. 
	#				l_id_delta		= 1.0 - (slaves_with_this_l_id.size*1.0)/s_ids.size
	#				no_l_id_delta	= l_id_delta - 1.0 # alpha*(-1.0*(slaves_without_this_l_id.size*1.0)/s_ids.size)
	#
	#				# Mark the l_id_delta update ...
	#				_node_up = [[s_l_id, self.slave_list[s_l_id].node_map[n_id], l_id, l_id_delta] for s_l_id in slaves_with_this_l_id]
	#				node_updates += _node_up
	#				#   ... and the no_l_id_delta update. 
	#				_node_up = [[s_nl_id, self.slave_list[s_nl_id].node_map[n_id], l_id, no_l_id_delta] for s_nl_id in slaves_without_this_l_id]
	#				node_updates += _node_up
	#
	#				# Add these updates to the L2 norm of the subgradient. 
	#				norm_gt += len(slaves_with_this_l_id)*(l_id_delta**2)
	#				norm_gt += len(slaves_without_this_l_id)*(no_l_id_delta**2)
	
				# For all slaves which assign n_id the label l_id, we apply
				# 	the update l_id_delta for the node energy corresponding to the label l_id. 
				# For all other slaves, we apply the update no_l_id_delta. 
	#				for s_l_id in slaves_with_this_l_id:
	#					n_id_in_s = self.slave_list[s_l_id].node_map[n_id]
	#					self.slave_list[s_l_id].node_energies[n_id_in_s][l_id] += alpha*l_id_delta
	#					# Add to the current subgradient. 
	#					norm_gt += l_id_delta**2
	#				for s_nl_id in slaves_without_this_l_id:
	#					n_id_in_s = self.slave_list[s_nl_id].node_map[n_id]
	#					self.slave_list[s_nl_id].node_energies[n_id_in_s][l_id] += alpha*no_l_id_delta
	#					# Add to the current subgradient. 
	#					norm_gt += no_l_id_delta**2
	
				# Set flags for these slaves to True
	
		# That completes the updates for node energies. Now we move to edge energies. 
		for e_id in self._check_edges:
			# Retrieve the list of slaves that use this edge. 
			s_ids			= self.edges_in_slaves[e_id]
			n_slaves_eid	= s_ids.size
	
			# Retrieve labellings of this edge, assigned by each slave.
			x, y	= self._node_ids_from_edge_id(e_id)
			ls_		= np.array([
						make_one_hot((self.slave_list[s].get_node_label(x), self.slave_list[s].get_node_label(y)), self.n_labels[x], self.n_labels[y]) 
						for s in s_ids])
			ls_avg_	= np.mean(ls_, axis=0, keepdims=True)
	
			# Check if all labellings for this node agree. 
			if np.max(ls_avg_) == 1:
				# As all vectors are one-hot, this condition being true implies that 
				#   all slaves assigned the same label to this node (otherwise, the maximum
				#   number in ls_avg_ would be less than 1).
				continue

			# The next step was to iterate over all slaves. We calculate the subgradient here
			#   given by 
			#   
			#    \delta_\lambda^s_p = x^s_p - ls_avg_
			#
			#   for all s. s here signifies slaves. 
			# This can be very easily done with array operations!
			_edge_up	= ls_ - np.tile(ls_avg_, [n_slaves_eid, 1, 1])
	
			# Find the node ID for n_id in each slave in s_ids. 
			sl_eids = [self.slave_list[s].edge_map[e_id] for s in s_ids]
	
			# Mark this update to be done later. 
			self._slave_edge_up[s_ids, sl_eids, :self.n_labels[x] , :self.n_labels[y]] = _edge_up #:self.n_labels[x]*self.n_labels[y]] = _edge_up
			# Mark this slave for edge updates. 
			self._mark_sl_up[1, s_ids] = True

#			# If we reach this stage, we have an edge shared between two trees, with both of them
#			#    assigning it different labels. A little bit of hard-coding goes a long way in improving 
#			#    performance. 
#			lx_1, ly_1	= ls_[0]
#			lx_2, ly_2	= ls_[1]
#
#			s_1	= s_ids[0]
#			s_2	= s_ids[1]
#
#			e_id_s_1	= self.slave_list[s_1].edge_map[e_id]
#			e_id_s_2	= self.slave_list[s_2].edge_map[e_id]
#
#			# Mark updates. 
#			_edge_up = [[s_1, e_id_s_1, lx_2, ly_2, -0.5]]
#			_edge_up += [[s_2, e_id_s_2, lx_1, ly_1, -0.5]]
#			_edge_up += [[s_1, e_id_s_1, lx_1, ly_1, 0.5]]
#			_edge_up += [[s_2, e_id_s_2, lx_2, ly_2, 0.5]]
#
#			# Add to the norm of the subgradient. 
#			norm_gt += 4*(0.5**2)

#			self.slave_list[s_1].edge_energies[e_id_s_1][lx_2][ly_2]	-= alpha/2
#			self.slave_list[s_2].edge_energies[e_id_s_2][lx_1][ly_1]	-= alpha/2
#
#			self.slave_list[s_1].edge_energies[e_id_s_1][lx_1][ly_1]	+= alpha/2
#			self.slave_list[s_2].edge_energies[e_id_s_2][lx_2][ly_2]	+= alpha/2
#
#			# Add to the norm of the subgradient. 
#			norm_gt += 4*(0.5**2)

#		# Distribute the computation of node and edge updates, as they can be done independantly. 
#		n_cores = cpu_count() - 1
#
#		# Collect node updates in node_updates, as returned by _compute_node_updates().
#		# Create input list first. 
#		_node_up_inputs = [[i, self.nodes_in_slaves[i], self.slave_list[self.nodes_in_slaves[i]], self.n_labels[i]]
#		        for i in range(self.n_nodes) if self._n_slaves_nodes[i] > 1]
#		# Distribute.
#		node_updates = Parallel(n_jobs=n_cores)(delayed(_compute_node_updates)(i) for i in _node_up_inputs)
#
#		# Collect edge updates in edge_updates, as returned by _compute_edge_updates(). 
#		# Create input list first. 
#		_edge_up_inputs = [[e, self.nodes_in_slaves[e], self.slave_list[self.nodes_in_slaves[e]], self._node_ids_from_edge_id(e), 
#		                   [self.n_labels[self._node_ids_from_edge_id(e)[0]], self.n_labels[self._node_ids_from_edge_id(e)[1]]]]
#						   for e in range(self.n_edges) if self._n_slaves_edges[e] > 1]
#		# Distribute only if there are shared edges (some decompositions, for example, row_col, do not have any shared edges).
#		if len(_edge_up_inputs) > 0:
#			edge_updates = Parallel(n_jobs=n_cores)(delayed(_compute_edge_updates)(i) for i in _edge_up_inputs)
#
#		# Now store obtained values in self._slave_node_up, and self._slave_edge_up. 
#		# Meanwhile, also compute the total subgradient. 
#		for _nu in range(len(node_updates)):
#			_node_ret = node_updates[_nu]
#			# Check if any updates need to be made.
#			if not _node_ret[0]: 
#				continue
#
#			_node_in  = _node_up_inputs[_nu]
#
#			# Record slave updates. 
#			self._slave_node_up[_node_in[1], _node_ret[2], 0:n_labels[n_id]] = _node_ret[1]
#			# Add subgradient.
#			norm_gt += _node_ret[3]
#			# Mark updates. 
#			self._mark_sl_up[0, _node_in[1]] = True
#		
#		# Do the same for edge updates. 
#		for _eu in range(len(edge_updates)):
#			_edge_ret = edge_updates[_eu]
#			# Check if any updates need to be done. 
#			if not _edge_ret[0]:
#				continue
#
#			_edge_in = _edge_up_inputs[_eu]
#
#			# Record slave updates. 
#			lx, ly = _edge_in[4]
#			self._slave_edge_up[_edge_in[1], _edge_ret[2], 0:lx*ly] = _edge_ret[1]
#			# Add subgradient. 
#			norm_gt += _edge_ret[3]
#			# Mark updates. 
#			self._mark_sl_up[1, _edge_in[1]] = True

		# Reset the slaves to solve. 
		self._slaves_to_solve = np.where(np.sum(self._mark_sl_up, axis=0)!=0)[0]

		# The subgradient is the L2-norm of the update matrices. 
		# We want the square of the subgradient, in any case. 
		norm_gt  = np.sum(self._slave_node_up**2)
		norm_gt += np.sum(self._slave_edge_up**2)
		# Record the norm of the subgradient. 
		self.subgradient_norms += [np.sqrt(norm_gt)]

		# Add momentum.
		self._slave_node_up = (1.0 - self._momentum)*self._slave_node_up + self._momentum*self._prv_node_sg
		self._slave_edge_up = (1.0 - self._momentum)*self._slave_edge_up + self._momentum*self._prv_edge_sg

		# Compute the alpha for this step. 
		if self._optim_strategy is 'step':
			alpha	= a_start/np.sqrt(it)
		elif self._optim_strategy is 'step_ss':
			alpha   = a_start/(1 + it)
		elif self._optim_strategy is 'step_sg':
			alpha   = a_start/np.sqrt(it)
			alpha   = alpha*1.0/np.sqrt(norm_gt)
		elif self._optim_strategy in ['adaptive', 'adaptive_d']:
			approx_t	= self._best_primal_cost
			dual_t		= self.dual_costs[-1]
			alpha		= a_start*(approx_t - dual_t)/norm_gt
			if self._optim_strategy is 'adaptive_d':
				alpha   = alpha*1.0/np.sqrt(it)

		# Perform the marked updates. The slaves to be updates are also the slaves
		#   to be solved!
		for s_id in self._slaves_to_solve:
			if self._mark_sl_up[0, s_id]:
				# Node updates have been marked. 
				n_nodes_this_slave = self.slave_list[s_id].node_list.size
				self.slave_list[s_id].node_energies += alpha*self._slave_node_up[s_id,:n_nodes_this_slave,:]

			if self._mark_sl_up[1, s_id]:
			 	# Edge updates have been marked. 
				n_edges_this_slave = self.slave_list[s_id].edge_list.size
				self.slave_list[s_id].edge_energies += alpha*self._slave_edge_up[s_id, :n_edges_this_slave, :, :]
#				self.slave_list[s_id].edge_energies += alpha*np.reshape(self._slave_edge_up[s_id,:n_edges_this_slave,:], [n_edges_this_slave,self.max_n_labels,self.max_n_labels])

		self._prv_node_sg[:] = self._slave_node_up[:]
		self._prv_edge_sg[:] = self._slave_edge_up[:]

		return alpha


	def force_naive_search(self, disagreements, response='t'):
		''' 
		Force a naive search for the best solution by varying 
		the labelling of nodes in `disagreements`. Please use
		cautiously, specifying at most three nodes in `disagreements`.
		'''
		while response not in ['y', 'n']:
			print 'Naive search takes exponential time. Supplied disagreeing nodes',
			print 'are %d in number. Proceed? (y/n) ' %(disagreements.size),
			response = raw_input()
			print 'You said: ' + response + '.'
		if response is 'n':
			return
	
		# Get part of the primal solution. 
		labels_ = self._get_primal_solution()

		# Generate all possible labellings. 
		n_labels = self.n_labels[disagreements]
		labellings = _generate_label_permutations(n_labels)	
		
		# Find the minimum. 
		min_energy = np.inf
		min_labels = None
		for l_ in labellings:
			labels_[disagreements] = l_
			_energy = self._compute_primal_cost(labels=labels_)
			if _energy < min_energy:
				min_energy = _energy
				min_labels = l_

		# Set the best labels. 
		print 'Setting the best labels for disagreeing nodes ...'
		self.labels                = labels_
		self.labels[disagreements] = min_labels


	def plot_costs(self):
		f = plt.figure()
		pc, = plt.plot(self.primal_costs, 'r-', label='PRIMAL')
		dc, = plt.plot(self.dual_costs, 'b-', label='DUAL')
		plt.legend([pc, dc], ['PRIMAL', 'DUAL'])
		plt.show()


	def _check_consistency(self):
		'''
		A function to check convergence of the DD-MRF algorithm by checking if all slaves
		agree in their labels of the shared nodes. 
		It works by iterating over the list of subproblems over each node to make sure they 
		agree. If a disagreement is found, we do not have consistency
		'''
		for n_id in range(self.n_nodes):
			s_ids	= self.nodes_in_slaves[n_id]
			ls_		= [self.slave_list[s].get_node_label(n_id) for s in s_ids]
			ret_	= reduce(lambda x,y: x and (y == ls_[0]), ls_[1:], True)
			if not ret_:
				return False
		return True


	def _find_conflicts(self):
		'''
		A function to find disagreeing nodes at a step of the algorithm. 
		'''
		node_conflicts = np.zeros(self.n_nodes, dtype=bool)
		edge_conflicts = np.zeros(self.n_edges, dtype=bool)

		for n_id in range(self.n_nodes):
			s_ids	= self.nodes_in_slaves[n_id]
			ls_		= [self.slave_list[s].get_node_label(n_id) for s in s_ids]
			ret_	= map(lambda x: x == ls_[0], ls_[1:])
			if False in ret_:
				node_conflicts[n_id] = True

		# Update self._check_nodes to find only those nodes where a disagreement exists. 
		self._check_nodes = np.where(node_conflicts == True)[0].astype(np.int)
		# Find disagreeing edges. We iterate over self._check_nodes, and add all 
		#    neighbours of a node in _check_nodes. 
		for i in range(self._check_nodes.size - 1):
			n_id = self._check_nodes[i]
			neighs = [n_id + x for x in [-self.cols, -1, 1, self.cols]]
			neighs = [x for x in neighs if x >= 0 and x < self.n_nodes and \
		                                   not (n_id%self.cols == 0 and x-n_id == -1) and \
				                           not (x%self.cols == 0 and x-n_id == 1)]	# Ensure valid edges only.
			e_neighs = [self._edge_id_from_node_ids(n_id, x) for x in neighs]
			edge_conflicts[e_neighs] = True

		# Update self._check_edges to reflect to be only these edges. 
		self._check_edges = np.where(edge_conflicts == True)[0].astype(np.int)
		# Return disagreeing nodes. 
		return self._check_nodes


	def _assign_labels(self):
		'''
		Assign the final labels to all points. This function must be called if Lattice._check_consistency() returns 
		True. This function simply assigns to every node, the label assigned to it by the first
		slave in its own slave list. Thus, if called without checking consistency first, or even if
		Lattice._check_consistency() returned False, it is not guaranteed that this function
		will return the correct labels. 
		Also computes the primal cost for the final labelling. 
		'''
		# Assign labels now. 
		for n_id in range(self.n_nodes):
			s_id				= self.nodes_in_slaves[n_id][0]
			self.labels[n_id]	= self.slave_list[s_id].get_node_label(n_id)

		self.labels	= self.labels.astype(np.int)
		# Compute primal cost. 
		
		self.primal_cost = self._compute_primal_cost(labels=self.labels)
		return self.labels


	def _compute_dual_cost(self):
		'''
		Returns the dual cost at a given stage of the optimisation. 
		The dual cost is simply the sum of all energies of the slaves. 
		'''
		return reduce(lambda x, y: x + y, [s._compute_energy() for s in self.slave_list], 0)


	def _get_primal_solution(self):
	 	'''
		Estimate a primal solution from the obtained dual solutions. 
		This strategy uses the most voted label for every node. 
		'''
		labels = np.zeros(self.n_nodes, dtype=np.int)

		# Iterate over every node. 
		if self.decomposition is not 'cell' and self._est_prim is 'bp':
			# Use Max product messages to compute the best solution. 
		
			# Conflicts are in self._check_nodes. 
			# Assign non-conflicting labels first. 
			for n_id in np.setdiff1d(np.arange(self.n_nodes), self._check_nodes):
				s_id = self.nodes_in_slaves[n_id][0]
				labels[n_id] = self.slave_list[s_id].get_node_label(n_id)

			# Now traverse conflicting labels. 	
			node_order = self._check_nodes
			for i in range(node_order.size):
				n_id  = node_order[i]
				n_lbl = self.n_labels[n_id]
					
				# Check that an edge exists between n_id and (n_id + offset)
				# No top (bottom) edges for vertices in the top (bottom) row.
				# No left edges for vertices in the left-most column. 
				# No right edges for vertices in the right-most column. 
				neighs = [n_id + offset for offset in [-self.cols, -1, 1, self.cols] if (n_id + offset >= 0 and n_id + offset < self.n_nodes and \
				          not (n_id%self.cols == 0 and offset == -1) and \
				          not ((n_id+1)%self.cols == 0 and offset == 1))]
				neighs = [_n for _n in neighs if _n in node_order[:i]]
				
				node_bel = np.zeros(n_lbl)
				if len(neighs) == 0:
				# If there are no previous neighbours, take the maximum of the node belief. 
					for s_id in self.nodes_in_slaves[n_id]:
						n_id_in_s = self.slave_list[s_id].node_map[n_id]
						node_bel += self.slave_list[s_id]._messages_in[n_id_in_s, :n_lbl]
					labels[n_id] = np.argmax(node_bel)
				else:
				# Else, take the argmax decided by the sum of messages from its neighbours that
				#   have already appeared in node_order. 
					for _n in neighs:
						e_id = self._edge_id_from_node_ids(_n, n_id)
						for s_id in self.edges_in_slaves[e_id]:
							n_edges_in_s = self.slave_list[s_id].graph_struct['n_edges']
							_e_id = self.slave_list[s_id].edge_map[e_id]
							_e_id += n_edges_in_s if _n > n_id else 0
							node_bel += self.slave_list[s_id]._messages[_e_id, :n_lbl]

				labels[n_id] = np.argmax(node_bel)


#				for np_id in neighs:
#					e_id = self._edge_id_from_node_ids(n_id, np_id)
#
#					# Check if the node lies to the left or the right of n_id in node_order
#					if np_id in node_order[:i]:
#						ee_neigh = self.edge_energies[e_id,:n_lbl,labels[np_id]] if np_id > n_id else self.edge_energies[e_id,labels[np_id],:n_lbl]
#						cost += np.exp(-1*ee_neigh)
##						print 'pairwise ', self.edge_energies[e_id,:n_lbl,labels[np_id]] if np_id > n_id else self.edge_energies[e_id,labels[np_id],:n_lbl]
#					else:
#						# Get the slave ID in which this edge is. 
#						for s_id in self.edges_in_slaves[e_id]:
#							e_id_in_s = self.slave_list[s_id].edge_map[e_id]
#							n_edges_in_s = self.slave_list[s_id].graph_struct['n_edges']
#						
#							e_id_in_s += n_edges_in_s if np_id > n_id else 0
#							cost += self.slave_list[s_id]._messages[e_id_in_s, :n_lbl]
#						print 'msgs     ', self.slave_list[s_id]._messages[e_id_in_s, :n_lbl]

#				print '--'

		else:
			for n_id in range(self.n_nodes):
				# Retrieve the labels assigned by every slave to this node. 
				s_ids    = self.nodes_in_slaves[n_id]
				s_labels = [self.slave_list[s].label_from_node[n_id] for s in s_ids]
				# Find the most voted label. 
				labels[n_id] = np.int(stats.mode(s_labels)[0][0])

		# Return this labelling. 
		return labels

	def _compute_primal_cost(self, labels=None):
		'''
		Returns the primal cost given a labelling. 
		'''
		cost	= 0

		# Generate a labelling first, if not specified. 
		if labels is None:
			labels	= self._get_primal_solution()

		# Compute node comtributions.
		for n_id in range(self.n_nodes):
			cost += self.node_energies[n_id][labels[n_id]]

		# Compute the edge list
		edge_list	= [self._node_ids_from_edge_id(e_id) for e_id in range(self.n_edges)]
		# Compute edge contributions. 
		for e_id in range(self.n_edges):
			e = edge_list[e_id]
			x, y = e
			cost += self.edge_energies[e_id][labels[x]][labels[y]]

		# This is the primal cost corresponding to either the input labels, or the generated ones. 
		return cost


	def _edge_id_from_node_ids(self, x, y):
		'''
		Lattice._edge_id_from_node_ids(): Return the edge ID given the nodes it connects. 
		'''
		# If y is less than x, swap them, as we assume the edge from the lower ID to the 
		#   higher ID. 
		if y < x:
			x, y = y, x

		t		= x/self.cols
		u		= x % self.cols
		edge_id	= (2*self.cols - 1)*t

		# Boundary conditions: If this is an edge originating in the last row of the lattice, 
		#	it can only be a "right" edge. Hence, the "right" edge in this case gets the 
		#	0 index. We must adjust for this. 
		if t == self.rows - 1:
			edge_id	+= u			# There is only one edge for every node (except the last).
		else:
			edge_id	+= 2*u			# There are two edges for each node (except the last).
			edge_id	+= 1 if (y - x) == 1 else 0
		return edge_id

	
	def _primal_dual_gap(self):
		'''
		Return the primal dual gap at the current stage of the optimisation.
		'''
		return self._compute_primal_cost() - self._compute_dual_cost()


	def _node_ids_from_edge_id(self, e):
		'''
		Lattice._node_ids_from_edge_id(): Return the node IDs which are connected by a given edge.
		'''
		t	= e/(2*self.cols - 1)
		# The last row has only one edge per node. We must adjust for this in the
		#	expression for u below. 
		if t == self.rows - 1:
			u	= e%(2*self.cols-1)
			x	= t*self.cols + u
			y	= x + 1
		else:
			u	= (e%(2*self.cols - 1))/2
			x	= t*self.cols + u
			y	= x + (self.cols if (e%(2*self.cols - 1)) - 2*u == 0 else 1)
		return [x, y]


	def _find_empty_attributes(self):
		'''
		Lattice._find_empty_attributes(): Returns the list of attributes not set. 
		'''
		# Retrieve the indices for nodes and edges not set. 
		n	= np.where(self.node_flags == False)[0]
		e	= np.where(self.edge_flags == False)[0]

		# Compute the edge list
		edge_list	= [self._node_ids_from_edge_id(e_id) for e_id in e]

		# Compute the node_list
		node_list	= n.tolist()
		return node_list, edge_list
# ---------------------------------------------------------------------------------------


def _compute_node_updates(n_id, s_ids, slave_list, n_labels_nid):
	'''
	A function to handle parallel computation of node updates. 
	The entire lattice cannot be passed as a parameter to this function, 
	and so we must create a function that is not a member of the class Lattice.
	'''
	# The number of slaves.
	n_slaves_nid	= s_ids.size

	# If there is only one slave, we have nothing to do. However, to avoid the overhead
	#   of calling a function that does nothing, we will simply not call this function
	#   for those nodes that belong to only one slave. 

	# Retrieve labels assigned to this point by each slave, and make it into a one-hot vector. 
	ls_		= np.array([make_one_hot(slave_list[s].get_node_label(n_id), n_labels_nid) for s in range(n_slaves_nid)])
	ls_avg_	= np.mean(ls_, axis=0)

	# Check if all labellings for this node agree. 
	if np.max(ls_avg_) == 1:
		# As all vectors are one-hot, this condition being true implies that 
		#   all slaves assigned the same label to this node (otherwise, the maximum
		#   number in ls_avg_ would be less than 1).
		return False, None, None, 0.0

	# The next step was to iterate over all slaves. We calculate the subgradient here
	#   given by 
	#   
	#    \delta_\lambda^s_p = x^s_p - ls_avg_
	#
	#   for all s. s here signifies slaves. 
	# This can be very easily done with array operations!
	_node_up	= ls_ - np.tile(ls_avg_, [n_slaves_nid, 1])

	# Find the node ID for n_id in each slave in s_ids. 
	sl_nids = [slave_list[s].node_map[n_id] for s in range(n_slaves_nid)]

	# Add this value to the subgradient. 
	norm_gt	= np.sum(_node_up**2)

	return True, _node_up, sl_nids, norm_gt
# ---------------------------------------------------------------------------------------


def _compute_edge_updates(e_id, s_ids, slave_list, pt_coords, n_labels):
	'''
	A function to handle parallel computation of edge updates. 
	The entire lattice cannot be passed as a parameter to this function, 
	and so we must create a function that is not a member of the class Lattice.
	'''
	# The number of slaves that this edge belongs to. 
	n_slaves_eid	= s_ids.size

	# If there is only one slave, we have nothing to do. However, to avoid the overhead
	#   of calling a function that does nothing, we will simply not call this function
	#   for those edges that belong to only one slave. 

	# Retrieve labellings of this edge, assigned by each slave.
	x, y    = pt_coords
	ls_		= np.array([
				make_one_hot([slave_list[s].get_node_label(x), slave_list[s].get_node_label(y)], n_labels[0], n_labels[1]) 
				for s in s_ids])
	ls_avg_	= np.mean(ls_, axis=0)

	# Check if all labellings for this node agree. 
	if np.max(ls_avg_) == 1:
		# As all vectors are one-hot, this condition being true implies that 
		#   all slaves assigned the same label to this node (otherwise, the maximum
		#   number in ls_avg_ would be less than 1).
		return False, None, None, 0.0

	# The next step was to iterate over all slaves. We calculate the subgradient here
	#   given by 
	#   
	#    \delta_\lambda^s_p = x^s_p - ls_avg_
	#
	#   for all s. s here signifies slaves. 
	# This can be very easily done with array operations!
	_edge_up	= ls_ - np.tile(ls_avg_, [n_slaves_eid, 1])

	# Find the node ID for n_id in each slave in s_ids. 
	sl_eids = [slave_list[s].edge_map[e_id] for s in range(s_ids)]

	# Add this value to the subgradient. 
	norm_gt	= np.sum(_edge_up**2)

	# Mark this slave for edge updates, and return.
	return True, _edge_up, sl_eids, norm_gt

	
def _compute_4node_slave_energy(node_energies, edge_energies, labels):
	'''
	Compute the energy of a slave corresponding to the labels. 
	'''
	[i,j,k,l]	= labels
	
	# Add node energies. 
	total_e		= node_energies[0][i] + node_energies[1][j] + \
					node_energies[2][k] + node_energies[3][l]

	# Add edge energies. 
	total_e		+= edge_energies[0][i,j] + edge_energies[1][i,k] \
				    + edge_energies[2][j,l] + edge_energies[3][k,l]

	return total_e
# ---------------------------------------------------------------------------------------


def _compute_tree_slave_energy(node_energies, edge_energies, labels, graph_struct):
	''' 
	Compute the energy corresponding to a given labelling for a tree slave. 
	The edges are specified in graph_struct. 
	'''
	
	energy = 0
	for n_id in range(graph_struct['n_nodes']):
		energy += node_energies[n_id][labels[n_id]]
	for edge in range(graph_struct['edge_ends'].shape[0]):
		e0, e1 = graph_struct['edge_ends'][edge]
		energy += edge_energies[edge][labels[e0]][labels[e1]]

	return energy
# ---------------------------------------------------------------------------------------
	

def _optimise_4node_slave(slave):
	'''
	Optimise the smallest possible slave consisting of four vertices. 
	This is a brute force optimisation done by enumerating all possible
	states of the four points and finding the minimum energy. 
	The nodes are arranged as 

			0 ----- 1
			|       |
			|       |
			2 ----- 3,

	where these indices are the same as their indices in node_energies. 

	Input:
		An instance of class Slave which has the following members:
			node_energies: 		The node energies for every label,
								in shape (4, max_n_labels)
			n_labels:			The number of labels for each node. 
								shape: (4,)
			edge_energies:		Energies for each edge arranged in the order
								0-1, 0-2, 1-3, 2-3, obeying the vertex order 
								as well. 
								shape: (max_num_nodes, max_num_nodes, 4)

	Outputs:
		labels:				The labelling for vertices in the order [0, 1, 2, 3]
		min_energy:			The total energy corresponding to the labelling. 
	'''

	# Extract parameters from the slave. 
	node_energies 		= slave.node_energies
	n_labels			= slave.n_labels
	edge_energies		= slave.edge_energies

	# Use already generated all labellings. 
	all_labellings      = slave.all_labellings
	
	# Minimum energy. We set the minimum energy to four times the maximum node energy plus
	# 	four times the maximum edge energy. 
	min_energy		= 4*np.max(node_energies) + 4*np.max(edge_energies)

	# The optimal labelling. 
	labels			= np.zeros(4)

	# Record energies for every labelling. 
	for l_ in range(all_labellings.shape[0]):
		total_e		= _compute_4node_slave_energy(node_energies, edge_energies, all_labellings[l_,:])
		# Check if best. 
		if total_e < min_energy:
			min_energy	= total_e
			labels[:]	= all_labellings[l_,:]

	return labels, min_energy
# ---------------------------------------------------------------------------------------


def _optimise_tree(slave):
	''' 
	Optimise a tree-structured slave. We use max-product belief propagation for this optimisation. 
	The package bp provides a function max_prod_bp, which optimises a given tree based on supplied
	node and edge potentials. However, this function maximises the total potential on a tree. To 
	use it to minimise our energy, we apply exp(-x) on all node and edge energies before passing
	it to max_prod_bp
	'''
	node_pot 	= np.array([np.exp(-1*ne) for ne in slave.node_energies])
	edge_pot	= np.array([np.exp(-1*ee) for ee in slave.edge_energies])
	gs			= slave.graph_struct
	# Call bp.max_prod_bp
	labels, messages, messages_in = bp.max_prod_bp(node_pot, edge_pot, gs)
	# We return the energy. 
	energy = _compute_tree_slave_energy(slave.node_energies, slave.edge_energies, labels, slave.graph_struct)
	return labels, energy, messages, messages_in
# ---------------------------------------------------------------------------------------


def _update_slave_states(c):
	'''
	Update the states for slave (given by c[1]) and set its labelling to the specified one
	(given by c[2][0]). Also performs a sanity check on c[1]._compute_energy and c[2][1], which
	is the optimal energy returned by _optimise_4node_slave().
	'''
	i					= c[0]
	s					= c[1]
	[s_labels, s_min]	= c[2]

	# Set the labels in s. 
	s.set_labels(s_labels)
	s._compute_energy()

	# Sanity check. The two energies (s_min and s._energy) must agree.
	if s._energy != s_min:
		print '_update_slave_states(): Consistency error. The minimum energy returned \
by _optimise_4node_slave() for slave %d is %g and does not match the one computed \
by Slave._compute_energy(), which is %g. The labels are [%d, %d, %d, %d]' \
				%(i, s_min, s._energy, s_labels[0], s_labels[1], s_labels[2], s_labels[3])
		return False, None

	# Everything is okay.
	return True, s

# ---------------------------------------------------------------------------------------


def optimise_all_slaves(slaves):
	'''
	A function to optimise all slaves. This function distributes the job of
	optimising every slave to all but one cores on the machine. 
	Inputs:
		slaves:		A list of objects of type Slave. 
	'''
	# The number of cores to use is the number of cores on the machine minum 1. 
	n_cores		= cpu_count() - 1
	optima		= Parallel(n_jobs=n_cores)(delayed(_optimise_4node_slave)(s) for s in slaves)

	# Update the labelling in slaves. 
	success		= np.array(Parallel(n_jobs=n_cores)(delayed(_update_slave_states)(c) for c in zip(range(len(slaves)), slaves, optima)))
	if False in success:
		print 'Update of slave states failed for ',
		print np.where(success == False)[0]
		raise AssertionError

# ---------------------------------------------------------------------------------------


def _optimise_slave(s):
	'''
	Function to handle optimisation of any random slave. 
	'''
	if s.struct == 'cell':
		return _optimise_4node_slave(s)
	elif s.struct == 'tree':
		return _optimise_tree(s)
	else:
		print 'Slave structure not recognised: %s.' %(s.struct)
		raise ValueError
# ---------------------------------------------------------------------------------------


def make_one_hot(label, s1, s2=None):
	'''
	Make one-hot vector for the given label depending on the number of labels
	specified by s1 and s2. 
	'''
	# Make sure the input label conforms with the input dimensions. 
	if s2 is None:
		if type(label) == list:
			print 'Please specify an int label for unary energies.'
			raise ValueError
		label = int(label)

	# Number of labels in the final vector. 
	size = s1 if s2 is None else (s1,s2)	

	# Make final vector. 
	oh_vec = np.zeros(size, dtype=np.bool)
	
	# Set label.
	oh_vec[label] = True
	# Return 
	return oh_vec
# ---------------------------------------------------------------------------------------


def _generate_label_permutations(n_labels):
	if n_labels.size == 1:
		return [[i] for i in range(n_labels[0])]

	_t   = _generate_label_permutations(n_labels[1:])

	_ret = []
	for i in range(n_labels[0]):
		_ret += [[i] + _tt for _tt in _t]

	return _ret
# ---------------------------------------------------------------------------------------


def generate_trees(adj_mat, max_depth=2):
	'''
	Generate a set of trees from a given adjacency matrix. The number of trees is
	equal to the number of nodes in the graph. Each tree has diameter at most 2*max_depth.
	'''

	n_nodes = adj_mat.shape[0]
	n_trees = n_nodes

	tree_adjmats = np.zeros((n_trees, n_nodes, n_nodes), dtype=np.bool)
	
	for i in range(n_nodes):
		tree_adjmats[i,:,:] = generate_tree_with_root(adj_mat, i, max_depth=max_depth)

	return tree_adjmats
# ---------------------------------------------------------------------------------------


def generate_tree_with_root(adj_mat, root, max_depth=2):
	'''
	Generate a tree of max depth specified by max_depth, and with root specified by root. 
	'''
		
	# Create a queue to traverse the graph in a bredth-first manner
	# Each element is a pair, where the first of the pair specified the vertex, and the second 
	#    specifies the depth. 
	queue = [[root, 0]]

	# The current depth of the tree. 
	c_depth = 0

	# The number of nodes. 
	n_nodes = adj_mat.shape[0]
	
	# Create the output adjacency matrix. 
	tree_adjmat = np.zeros((n_nodes, n_nodes), dtype=np.bool)

	# Record whether we already visited a node. 
	visited = np.zeros(n_nodes, dtype=np.bool)

	# We have alredy visited root. 
	visited[root] = True

	while len(queue) > 0:
		# The current root in the traversal. 
		_v, _d = queue[0]
		# Pop this vertex from the queue. 
		queue = queue[1:]

		# If we have reached the maximum allowed depth, stop, and backtrack.
		if _d == max_depth:
			continue
		
		# Neighbours of _v that we have not already visited. 
		neighbours = [i for i in np.where(adj_mat[_v, :] == True)[0] if not visited[i]]

		# If we have no more possible neighbours, stop and backtrack. 
		if len(neighbours) == 0:
			continue

		# Mark all neighbours as visited. 
		visited[neighbours] = True

		# Add these edges to the adjacency matrix. 
		tree_adjmat[_v, neighbours] = True

		# Insert these in the queue. 
		_next_nodes = [[_n, _d + 1] for _n in neighbours]
		queue += _next_nodes	
		
	# Make adjacency matrix symmetric. 
	tree_adjmat = tree_adjmat + tree_adjmat.T
	# Return this adjcency matrix. 
	return tree_adjmat
