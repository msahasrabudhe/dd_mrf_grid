# This library approximation to the optimal of an artibrary energy on a 2-D lattice
#	by splitting the lattice into sub-graphs formed by the smallest possible loops,
#	that is, the smallest loops of four vertices (nodes) forming a square. 

import numpy as np
from joblib import Parallel, delayed, cpu_count

# Epsilon to determine at most how far two floats can be to be considered equal. 
_FLOAT_EPSILON	= 1e-10

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

	Members:
		node_list:			List of nodes in this slave.
		edge_list: 			List of edges in this slave. 
		node_energies:		Energies for every label in the slave.
							List of length 4. 
		n_labels:			The number of labels for each node.
							shape: (4,)
		edge_energies:		Energies for each edge arranged in the order
							0-1, 0-2, 1-3, 2-3, obeying the vertex order 
							as well. 
							List of length 4

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
			node_energies=None, n_labels=None, edge_energies=None):
		'''
		Slave.__init__(): Initialise parameters for this slave, if given. 
						  Parameters are None by default. 
		'''
		self.node_list		= node_list
		self.edge_list		= edge_list
		self.node_energies	= node_energies
		self.n_labels		= n_labels
		self.edge_energies	= edge_energies

	def set_params(self, node_list, edge_list, 
			node_energies, n_labels, edge_energies):
		'''
		Slave.set_params(): Set parameters for this slave.
							Parameters must be specified.
		'''
		self.node_list		= node_list
		self.edge_list		= edge_list
		self.node_energies	= node_energies
		self.n_labels		= n_labels
		self.edge_energies	= edge_energies

	def set_labels(self, labels):
		'''
		Slave.set_labels():	Set the labelling for a slave
		'''
		self.labels		= labels

		# Also maintain a dictionary to easily fetch the label 
		#	given a node ID.
		self.label_from_node	= {}
		for i in range(4):
			n_id = self.node_list[i]
			self.label_from_node[n_id] = self.labels[i]

	def get_node_label(self, n_id):
		'''
		Retrieve the label of a node in the current labelling
		'''
		if n_id not in self.node_list:
			print 'Node %d is not in this slave.' %(n_id)
			raise ValueError
		return self.label_from_node[n_id]

	def _compute_energy(self):
		'''
		Slave._compute_energy(): Computes the energy corresponding to
								 the labels. 
		'''
		self._energy	= _compute_slave_energy(self.node_energies, self.edge_energies, self.labels)

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
			self.n_labels	= n_labels*np.ones(self.n_nodes)
		elif np.array(n_labels).size == self.n_nodes:
			self.n_labels	= np.array(n_labels)
		# In any case, the max n labels for this Lattice is np.max(self.n_lables)
		self.max_n_labels	= np.max(self.n_labels)
		
		# Initialise the node and edge energies. 
		self.node_energies	= [None]*self.n_nodes		#np.zeros((self.n_nodes, self.max_n_labels))
		self.edge_energies	= [None]*self.n_edges		#np.zeros((self.n_edges, self.max_n_labels, self.max_n_labels))

		# Flags set to ensure that node and edge energies have been set. If any energies
		# 	have not been set, we cannot proceed to optimisation as the lattice is not complete. 
		self.node_flags	= np.zeros(self.n_nodes, dtype=np.bool)
		self.edge_flags	= np.zeros(self.n_edges, dtype=np.bool)


	def set_node_energies(self, i, energies):
		'''
		Lattice.set_node_energies(): Set the node energies for node i. 
		'''
		# Convert the energy to a numpy array
		energies = np.array(energies)

		if energies.size > self.n_labels[i]:
			print 'Lattice.set_node_energies(): The supplied node energies have more labels',
			print '(%d) than the permissible number (%d)' %(energies.size, self.n_labels[i])
			raise ValueError

		# Make the assignment: set the node energies. 
		self.node_energies[i] = energies
		# Set flag for this node to True.
		self.node_flags[i]		= True

	def set_edge_energies(self, i, j, energies):
		'''
		Lattice.set_edge_energies(): Sets the edge energies for edge (i,j). The
		function firstly checks for the possibility of an edge between i and j, 
		and makes the assignment only if such an edge is possible.
		'''
		# Convert the energy to a numpy array
		energies = np.array(energies)

		# Convert indices to int, just in case ...
		i = np.int(i)
		j = np.int(j)
		# The edge is always from the lower index to the higher index. Hence, 
		#	find the lower and higher coordinate
		x, y = [np.min([i,j]), np.max([i,j])]
		# Check that the supplied energy has the correct shape. 
		input_shape		= np.sort(energies.shape).tolist()
		reqd_shape		= np.sort([self.n_labels[x], self.n_labels[y]]).tolist()
		if input_shape != reqd_shape:
			print 'Lattice.set_edge_energies(): The supplied energies have invalid shape:',
			print '(%d, %d). It must be (%d, %d) or (%d, %d).' \
						%(energies.shape[0], energies.shape[1], self.n_labels[i], self.n_labels[j], self.n_labels[i], self.n_labels[j])
			raise ValueError

		# Check that indices are not out of range. 
		if x >= self.n_nodes or y >= self.n_nodes:
			print 'Lattice.set_edge_energies(): At least one of the supplied edge indices is invalid.'
			raise IndexError
		# Check for the possibility of an edge. 
		if (y - x != 1) and (y - x != self.cols):
			# This signifies y is neither the node to the right of x, nor the node below x. 
			print 'Lattice.set_edge_energies(): The supplied edge indices are not consistent - a 2D',
			print 'lattice does not have an edge between %d and %d.' (i, j)
			raise ValueError

		# Correct the shape of the matrix, if required. If the user specified the matrix in a column-major
		#	fashion, it is possible it was specified of shape (self.n_labels[y], self.n_labels[x]), instead of the
		#	the other way around. 
		if energies.shape[0] == self.n_labels[y]:
			energies = energies.transpose()

		# We can proceed - everything is okay. 
		edge_id	= self._edge_id_from_node_ids(x,y)

		# Make assignment: set the edge energies. 
		self.edge_energies[edge_id]	= energies
		self.edge_flags[edge_id]	= True


	def check_completeness(self):
		'''
		Lattice.check_completeness(): Check whether all attributes of the lattice have been set.
									  This must return True in order to proceed to optimisation. 
		'''
		# Check whether all nodes have been set. 
		if np.sum(self.node_flags) < self.n_nodes:
			return False
		# Check whether all edges have been set. 
		if np.sum(self.edge_flags) < self.n_edges:
			return False
		# Everything is okay. 
		return True

	
	def _create_slaves(self):
		'''
		Lattice._create_slaves(): Create slaves for this particular lattice.
		'''
		# The number of slaves.
		self.n_slaves		= (self.rows - 1)*(self.cols - 1)
		# Create empty slaves initially. 
		self.slave_list		= [Slave() for i in range(self.n_slaves)]
		
		# The following lists record for every node and edge, to which 
		#	slaves it belongs.
		self.nodes_in_slaves	= [[]]*self.n_nodes
		self.edges_in_slaves	= [[]]*self.n_edges

		# We also make a slave list for nodes and edges. For each node and edge, 
		#	its list contains the IDs of all slaves that contain it. Initially, 
		# 	these lists are empty. We will fill them up. 
		self.slave_list_for_nodes	= [[]]*self.n_nodes
		self.slave_list_for_edges	= [[]]*self.n_edges

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
			node_energies	= [self.node_energies[i] for i in node_list]
	
			# The number of labels can be easily extracted from self.n_labels.
			n_labels	= self.n_labels[node_list].astype(np.int)
	
			# We now extract the edge energies, which are easy to extract as well, as we know
			# 	the edge IDs for all edges in this slave. 
			edge_energies	= [self.edge_energies[i] for i in edge_list]
	
			# Make assignments for this slave. 
			self.slave_list[s_id].set_params(node_list, edge_list, node_energies, n_labels, edge_energies)

			# Finally, add this slave the appropriate nodes_in_slaves, and edges_in_slaves
			for n_id in node_list:
				self.nodes_in_slaves[n_id] 	+= [s_id]
			for e_id in edge_list:
				self.edges_in_slaves[e_id]	+= [s_id]

		# For convenience, turn the individual lists in nodes_in_slaves and edges_in_slaves into numpy arrays. 
		self.nodes_in_slaves	= [np.array(t) for t in self.nodes_in_slaves]
		self.edges_in_slaves	= [np.array(t) for t in self.edges_in_slaves]


	def optimise(self, a_start=1.0):
		'''
		Lattice.optimise(): Optimise the set energies over the lattice and return a labelling. 

		Takes as input a_start, which is a float and denotes the starting value of \alpha_t in
		the DD-MRF algorithm. 
		'''
		# First check if the lattice is complete. 
		if not self.check_completeness():
			n_list, e_list	= self._find_empty_attributes()
			print 'Lattice.optimise(): The lattice is not complete.'
			print 'The following nodes are not set:', n_list
			print 'The following edges are not set:', e_list
			return 

		# Create slaves. This creates a list of slaves and stores it in 
		# 	self.slave_list. The numbering of the slaves starts from the top-left,
		# 	and continues in row-major fashion. There are (self.rows-1)*(self.cols-1)
		# 	slaves. 
		self._create_slaves()

		# Whether converged or not. 
		converged = False

		# The iteration
		it = 1

		# Set all slaves to be solved at first. 
		self._slaves_to_solve	= np.arange(self.n_slaves)

		# Loop till not converged. 
		while not converged:
			alpha	= a_start/sqrt(it)
			self._optimise_slaves()

			if self._check_consistency():
				print 'Converged after %d iterations!' %(it)
				print 'alpha_t at convergence iteration is %g.' %(alpha)
				self._assign_labels()

			self._apply_param_updates(alpha)
		

	def _optimise_slaves(self):
		'''
		A function to optimise all slaves. This function distributes the job of
        optimising every slave to all but one cores on the machine. 
		'''
		# Extract the list of slaves to be optimised. This contains of all the slaves that
		# 	disagree with at least one other slave on the labelling of at least one node. 
		_to_solve	= [self.slave_list[i] for i in self._slaves_to_solve]
		# The number of cores to use is the number of cores on the machine minus 1. 
		n_cores		= cpu_count() - 1
		optima		= Parallel(n_jobs=n_cores)(delayed(_optimise_4node_slave)(s) for s in _to_solve)
	
		# Update the labelling in slaves. 
		result		= Parallel(n_jobs=n_cores)(delayed(_update_slave_states)(c) for c in zip(self._slaves_to_solve, _to_solve, optima))
		success		= [result[i][0] for i in range(len(result))]
		if False in success:
			print 'Update of slave states failed for ',
			print np.where(success == False)[0]
			raise AssertionError
		# Reflect the result in slave list for our Lattice. 
		for i in range(self._slaves_to_solve.size):
			self.slave_list[i]	= result[i][1]

	
	def _apply_param_updates(self, alpha):
		'''
		Apply updates to energies after one iteration of the DD-MRF algorithm. 
		'''

		# Flags to determine whether to solve a slave.
		slave_flags	= np.zeros(self.n_slaves, dtype=bool)

		# We iterate over all nodes and edges and calculate updates to parameters of all slaves. 
		for n_id in range(self.n_nodes):
			# Retrieve the list of slaves that use this node. 
			s_ids	= self.nodes_in_slaves[n_id]
			# If there is only one such node, we have nothing to do. 
			if s_ids.size == 1:
				continue

			ls_		= [self.slave_list[s].get_node_label(n_id) for s in s_ids]
			ret_	= reduce(lambda x,y: x == y, ls_)

			# If True, no need to update parameters here. 
			if ret_:
				continue

			# Iterate over all labels. The update to the energy of label l_id is given by
			#
			#		\delta_l_id	= \alpha*(x_s(l_id) - \frac{\sum_s' x_s'(l_id)}{\sum_s' 1})
			#
			for l_id in range(self.n_labels[n_id]):
				# Find slaves that assign this point the label l_id. 
				slaves_with_this_l_id		= [s_ids[i] for i in np.where(ls_ == l_id)[0]]
				slaves_without_this_l_id	= np.setdiff1d(s_ids, slaves_with_this_l_id)

				# Calculate updates, given by the equation above. 
				l_id_delta		= alpha*(1.0 - (slaves_with_this_l_id.size*1.0)/s_ids.size)
				no_l_id_delta	= alpha*(-1.0*(slaves_without_this_l_id.size*1.0)/s_ids.size)

				# For all slaves which assign n_id the label l_id, we apply
				# 	the update l_id_delta for the node energy corresponding to the label l_id. 
				# For all other slaves, we apply the update no_l_id_delta. 
				for s_l_id in slaves_with_this_l_id:
					self.slave_list[s_l_id].node_energies[l_id] += l_id_delta
				for s_nl_id in slaves_without_this_l_id:
					self.slave_list[s_nl_id].node_energies[l_id] += no_l_id_delta

		# That completes the updates for node energies. Now we move to edge energies. 
		for e_id in range(self.n_edges):
			# Retrieve the list of slaves that use this edge. 
			s_ids	= self.edges_in_slaves[e_id]
			# If there is only one such slave, we have nothing to do. 
			if s_ids.size == 1:
				continue


			



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
			ret_	= reduce(lambda x,y: x == y, ls_)
			if not ret_:
				return False
		return True


	def _assign_labels(self):
		'''
		Assign the final labels to all points. This function must be called if Lattice._check_consistency() returns 
		True. This function simply assigns to every node, the label assigned to it by the first
		slave in its own slave list. Thus, if called without checking consistency first, or even if
		Lattice._check_consistency() returned False, it is not guaranteed that this function
		will return the correct labels. 
		'''
		for n_id in range(self.n_nodes):
			s_id				= self.nodes_in_slaves[n_id][0]
			self.labels[n_id]	= s_id.get_node_label(n_id)
		return self.labels


	def _edge_id_from_node_ids(self, x, y):
		'''
		Lattice._edge_id_from_node_ids(): Return the edge ID given the nodes it connects. 
		'''
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

	
def _compute_slave_energy(node_energies, edge_energies, labels):
	'''
	Compute the energy of a slave corresponding to the labels. 
	'''
	[i,j,k,l]	= labels
	
	# Add node energies. 
	total_e		= node_energies[0][i] + node_energies[1][j] + \
					node_energies[2][k] + node_energies[3][l]

	# Add edge energies. 
	total_e		+= edge_energies[0][i,j]
	total_e		+= edge_energies[1][i,k]
	total_e		+= edge_energies[2][j,l]
	total_e		+= edge_energies[3][k,l]
	return total_e

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

	# Generate all labellings. 
	[ni, nj, nk, nl]	= n_labels
	all_labellings		= np.array([[i,j,k,l] for i in range(ni) for j in range(nj) 
								for k in range(nk) for l in range(nl)])
	
	# Minimum energy. We set the minimum energy to four times the maximum node energy plus
	# 	four times the maximum edge energy. 
	min_energy		= 4*np.max(node_energies) + 4*np.max(edge_energies)

	# The optimal labelling. 
	labels			= all_labellings[0,:]

	# Record energies for every labelling. 
	for l_ in range(all_labellings.shape[0]):
		total_e		= _compute_slave_energy(node_energies, edge_energies, all_labellings[l_,:])
		# Check if best. 
		if total_e < min_energy:
			min_energy	= total_e
			labels		= all_labellings[l_,:]

	return labels, min_energy
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

	# Sanity check. The two energies (s_min and s._energy must agree)
	if s._energy != s_min:
		print 'optimise_all_slaves(): Consistency error. The minimum energy returned \
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


