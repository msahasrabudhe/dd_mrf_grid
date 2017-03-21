# This library calculates an approximation to the optimal of an artibrary energy on a 2-D lattice
#	by splitting the lattice into sub-graphs formed by the smallest possible loops,
#	that is, the smallest loops of four vertices (nodes) forming a square. 

import numpy as np
from joblib import Parallel, delayed, cpu_count
import scipy.stats as stats


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

		# These dictionaries enable to determine easily at which 
		#    index in node_list or edge_list, a particular node
		#    or edge is. 
		self.node_map		= {}
		self.edge_map		= {}
		for i in range(4):
			self.node_map[node_list[i]] = i
			self.edge_map[edge_list[i]] = i

	def get_params(self):
		'''
		Slave.get_params(): Return parameters of this slave
		'''
		return self.node_list, self.edge_list, self.node_energies, self.n_labels, self.edge_energies

	def set_labels(self, labels):
		'''
		Slave.set_labels():	Set the labelling for a slave
		'''
		self.labels	= np.array(labels, dtype=np.int)

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
			print self.node_list,
			print 'Node %d is not in this slave.' %(n_id)
			raise ValueError
		return self.label_from_node[n_id]


	def optimise(self):
		'''
		Optimise this slave. 
		'''
		return _optimise_4node_slave(self)

	def _compute_energy(self):
		'''
		Slave._compute_energy(): Computes the energy corresponding to
								 the labels. 
		'''
		self._energy	= _compute_slave_energy(self.node_energies, self.edge_energies, self.labels)
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
			self.n_labels	= n_labels*np.ones(self.n_nodes).astype(np.int)
		elif np.array(n_labels).size == self.n_nodes:
			self.n_labels	= np.array(n_labels).astype(np.int)
		# In any case, the max n labels for this Lattice is np.max(self.n_lables)
		self.max_n_labels	= np.max(self.n_labels)
		
		# Initialise the node and edge energies. 
		self.node_energies	= [None for i in range(self.n_nodes)]		#np.zeros((self.n_nodes, self.max_n_labels))
		self.edge_energies	= [None for i in range(self.n_edges)]		#np.zeros((self.n_edges, self.max_n_labels, self.max_n_labels))

		# Flags set to ensure that node and edge energies have been set. If any energies
		# 	have not been set, we cannot proceed to optimisation as the lattice is not complete. 
		self.node_flags	= np.zeros(self.n_nodes, dtype=np.bool)
		self.edge_flags	= np.zeros(self.n_edges, dtype=np.bool)


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
		energies = np.array(energies, dtype=e_dtype)

		# Convert indices to int, just in case ...
		i = np.int(i)
		j = np.int(j)
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
			print 'lattice does not have an edge between %d and %d.' %(i, j)
			raise ValueError

		# We can proceed - everything is okay. 
		edge_id	= self._edge_id_from_node_ids(i, j)

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
		self.nodes_in_slaves	= [[] for i in range(self.n_nodes)]
		self.edges_in_slaves	= [[] for i in range(self.n_edges)]

		# We also make a slave list for nodes and edges. For each node and edge, 
		#	its list contains the IDs of all slaves that contain it. Initially, 
		# 	these lists are empty. We will fill them up. 
		self.slave_list_for_nodes	= [[] for i in range(self.n_nodes)]
		self.slave_list_for_edges	= [[] for i in range(self.n_edges)]

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
			node_energies	= [np.zeros(nl, dtype=e_dtype) for nl in n_labels]
			for i in range(4):
				node_energies[i][:] = self.node_energies[node_list[i]][:]
#			node_energies	= [self.node_energies[i] for i in node_list]
	
			# We now extract the edge energies, which are easy to extract as well, as we know
			# 	the edge IDs for all edges in this slave. 
			# e_label_list stores the required size of the energy matrix for edge in edge_list
			e_label_list 	= [(n_labels[0], n_labels[1]), \
						   	   (n_labels[0], n_labels[2]), \
							   (n_labels[1], n_labels[3]), \
							   (n_labels[2], n_labels[3])]
			edge_energies	= [np.zeros(sz) for sz in e_label_list]
			for i in range(4):
				edge_energies[i][:] = self.edge_energies[edge_list[i]][:]
#			edge_energies	= [self.edge_energies[i] for i in edge_list]
	
			# Make assignments for this slave. 
			self.slave_list[s_id].set_params(node_list, edge_list, node_energies, n_labels, edge_energies)

			# Finally, add this slave the appropriate nodes_in_slaves, and edges_in_slaves.
			for n_id in node_list:
				self.nodes_in_slaves[n_id] 	+= [s_id]
			for e_id in edge_list:
				self.edges_in_slaves[e_id]	+= [s_id]

		# For convenience, turn the individual lists in nodes_in_slaves and edges_in_slaves into numpy arrays. 
		self.nodes_in_slaves	= [np.array(t) for t in self.nodes_in_slaves]
		self.edges_in_slaves	= [np.array(t) for t in self.edges_in_slaves]

		# Finally, we must modify the energies for every edge or node depending on 
		#   how many slaves it is a part of. The energy for a node/edge is distributed
		#   equally among all slaves. 
		for n_id in range(self.n_nodes):
			# Retrieve all the slaves this node is part of.
			s_ids	= self.nodes_in_slaves[n_id]
			# If there is only one slave, no need to do anything.
			if s_ids.size == 1:
				continue
			# Distribute this node's energy equally between all slaves.
			for s in s_ids:
				n_id_in_slave	= self.slave_list[s].node_map[n_id]
				self.slave_list[s].node_energies[n_id_in_slave] /= 1.0*s_ids.size

		# Doing the same for edges ...
		for e_id in range(self.n_edges):
			# Retrieve all slaves this edge is part of.
			s_ids	= self.edges_in_slaves[e_id]
			# Do nothing if this edge is in only one slave
			if s_ids.size == 1:
				continue
			# Distribute this edge's energy equally between all slaves. 
			for s in s_ids:
				e_id_in_slave	= self.slave_list[s].edge_map[e_id]
				self.slave_list[s].edge_energies[e_id_in_slave] /= 1.0*s_ids.size

		# That is it. The slaves are ready. 
			


	def optimise(self, a_start=1.0, max_iter=1000, strategy='step'):
		'''
		Lattice.optimise(): Optimise the set energies over the lattice and return a labelling. 

		Takes as input a_start, which is a float and denotes the starting value of \\alpha_t in
		the DD-MRF algorithm. 

		The strategy signifies what values of \\alpha to use at iteration t. Permissible 
		values are 'step' and 'adaptive'. The step strategy simply sets 

		      \\alpha_t = a_start/sqrt(t).

		The adaptive strategy sets 
		 
		      \\alpha_t = a_start*\\frac{Approx_t - Dual_t}{norm(\\nabla g_t)**2},

		where \\nabla g_t is the subgradient of the dual at iteration t. 
		'''

		# Check if a permissible strategy is being used. 
		if strategy not in ['step', 'adaptive']:
			print 'Permissible values for strategy are \'step\', and \'adaptive\''
			return
		# If strategy is adaptive, we would like a_start to be in (0, 2).
		if strategy is 'adaptive' and (a_start <= 0 or a_start >= 2):
			print 'Please use 0 < a_start < 2 for an adaptive strategy.'
			return

		# Set the optimisation strategy. 
		self._optim_strategy = strategy

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

		# Two lists to record the primal and dual cost progression
		self.dual_costs			= []
		self.primal_costs		= []
		self.subgradient_norms	= []

		# Loop till not converged. 
		while not converged and it <= max_iter:
			if self._optim_strategy is 'step':
				alpha	= a_start/np.sqrt(it)
			elif self._optim_strategy is 'adaptive':
				if it == 1:
					alpha = 1
				else:
					approx_t	= np.min(self.primal_costs)
					dual_t		= self.dual_costs[-1]
					norm_gt		= self.subgradient_norms[-1]
					alpha		= a_start*(approx_t - dual_t)/norm_gt

			print 'Iteration %d. %d subproblems to be solved. Optimising ...' %(it, self._slaves_to_solve.size),
			# Solve all the slaves. 
			# The following optimises the energy for each slave, and stores the 
			#    resulting labelling as a member in the slaves. 
			self._optimise_slaves()
			print 'done.',

			# Verify whether the algorithm has converged. If all slaves agree
			#    on the labelling of every node, we have convergence. 
			if self._check_consistency():
				print 'Converged after %d iterations!' %(it)
				print 'alpha_t at convergence iteration is %g.' %(alpha)
				# Finally, assign labels.
				self._assign_labels()
				# Break from loop.
				converged = True
				break

			# Find the number of disagreeing points. 
			disagreements = self._find_disagreeing_nodes()
			print ' alpha = %g. n_miss = %d.' %(alpha, disagreements.size),

			# Apply updates to parameters of each slave. 
			self._apply_param_updates(alpha)

			# Get the primal cost at this iteration
			primal_cost		= self._compute_primal_cost()
			self.primal_costs += [primal_cost]

			# Get the dual cost at this iteration
			dual_cost		= self._compute_dual_cost()
			self.dual_costs	+= [dual_cost]
			print 'Primal cost = %g. Dual cost = %g' %(primal_cost, dual_cost)

			# Increase iteration.
			it += 1

		print 'The final labelling is stored as a member \'label\' in the object.'
		

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
			s_id = self._slaves_to_solve[i]
			self.slave_list[s_id].node_energies = result[i][1].node_energies
			self.slave_list[s_id].edge_energies = result[i][1].edge_energies
			self.slave_list[s_id].set_labels(result[i][1].labels)
			self.slave_list[s_id]._energy		= result[i][1]._energy
#			self.slave_list[s_id]	= result[i][1]

	
	def _apply_param_updates(self, alpha):
		'''
		Apply updates to energies after one iteration of the DD-MRF algorithm. 
		'''

		# Flags to determine whether to solve a slave.
		slave_flags	= np.zeros(self.n_slaves, dtype=bool)

		# Compute the L2-norm of the subgradient. 
		norm_gt	= 0.0

	
		_subgrad_norm_vec = np.zeros(self.max_n_labels)

		# We iterate over all nodes and edges and calculate updates to parameters of all slaves. 
		for n_id in range(self.n_nodes):
			# Retrieve the list of slaves that use this node. 
			s_ids	= self.nodes_in_slaves[n_id]
			# If there is only one such slave, we have nothing to do. 
			if s_ids.size == 1:
				continue

			# Retrieve labels assigned to this point by each slave. 
			ls_		= [self.slave_list[s].get_node_label(n_id) for s in s_ids]
			ret_	= reduce(lambda x,y: x and (y == ls_[0]), ls_[1:], True)

			# If True, no need to update parameters here. 
			if ret_:
				continue

			# Iterate over all labels. The update to the energy of label l_id is given by
			#
			#		\delta_l_id	= \alpha*(x_s(l_id) - \frac{\sum_s' x_s'(l_id)}{\sum_s' 1})
			#
			for l_id in np.unique(ls_):#(self.n_labels[n_id]):
				# Find slaves that assign this point the label l_id. 
				slaves_with_this_l_id		= np.array([s_ids[i] for i in np.where(ls_ == l_id)[0]])
				slaves_without_this_l_id	= np.array([s_ids[i] for i in np.where(ls_ != l_id)[0]])

				# Calculate updates, given by the equation above. 
				l_id_delta		= 1.0 - (slaves_with_this_l_id.size*1.0)/s_ids.size
				no_l_id_delta	= l_id_delta - 1.0 # alpha*(-1.0*(slaves_without_this_l_id.size*1.0)/s_ids.size)

				# For all slaves which assign n_id the label l_id, we apply
				# 	the update l_id_delta for the node energy corresponding to the label l_id. 
				# For all other slaves, we apply the update no_l_id_delta. 
				for s_l_id in slaves_with_this_l_id:
					self.slave_list[s_l_id].node_energies[l_id] += alpha*l_id_delta
					# Add to the current subgradient. 
					norm_gt += l_id_delta**2
				for s_nl_id in slaves_without_this_l_id:
					self.slave_list[s_nl_id].node_energies[l_id] += alpha*no_l_id_delta
					# Add to the current subgradient. 
					norm_gt += no_l_id_delta**2

				# Set flags for these slaves to True
				slave_flags[s_ids] = True

		# That completes the updates for node energies. Now we move to edge energies. 
		for e_id in range(self.n_edges):
			# Retrieve the list of slaves that use this edge. 
			s_ids	= self.edges_in_slaves[e_id]
			# If there is only one such slave, we have nothing to do. 
			if s_ids.size == 1:
				continue

			# Retrieve labellings of this edge, assigned by each slave.
			x, y	= self._node_ids_from_edge_id(e_id)
			ls_		= [(self.slave_list[s].get_node_label(x), self.slave_list[s].get_node_label(y)) for s in s_ids]
			ret_	= reduce(lambda x,y: x and (y == ls_[0]), ls_[1:], True)

			# If True, no need to update parameters here. 
			if ret_:
				continue

			# If we reach this stage, we have an edge shared between two trees, with both of them
			#    assigning it different labels. A little bit of hard-coding goes a long way in improving 
			#    performance. 
			lx_1, ly_1	= ls_[0]
			lx_2, ly_2	= ls_[1]

			s_1	= s_ids[0]
			s_2	= s_ids[1]

			e_id_s_1	= self.slave_list[s_1].edge_map[e_id]
			e_id_s_2	= self.slave_list[s_2].edge_map[e_id]

			self.slave_list[s_1].edge_energies[e_id_s_1][lx_2][ly_2]	-= alpha/2
			self.slave_list[s_2].edge_energies[e_id_s_2][lx_1][ly_1]	-= alpha/2

			self.slave_list[s_1].edge_energies[e_id_s_1][lx_1][ly_1]	+= alpha/2
			self.slave_list[s_2].edge_energies[e_id_s_2][lx_2][ly_2]	+= alpha/2

			norm_gt += 4*(0.5**2)
			# Create a matrix which records which label pairs occurred how many times
#			label_freq_mat	= np.zeros((self.n_labels[x], self.n_labels[y]))
#			for up in ls_:
#				lx, ly = up
#				label_freq_mat[lx][ly] += 1
#
#			# Now we can make updates. 
#			non_zero_labels	= np.where(label_freq_mat != 0)
#			for lpairs in zip(non_zero_labels[0], non_zero_labels[1]):
#				for s in range(s_ids.size):
#					s_id = s_ids[s]
#					e_id_in_slave	= np.where(self.slave_list[s_id].edge_list == e_id)[0]
#					if ls_[s] == lpairs:
#						self.slave_list[s_id].edge_energies[e_id_in_slave][lpairs[0],lpairs[1]] += alpha*(1.0 - label_freq_mat[lpairs[0],lpairs[1]]*1.0/s_ids.size)
#					else:
#						self.slave_list[s_id].edge_energies[e_id_in_slave][lpairs[0],lpairs[1]] += alpha*(-1.0*(s_ids.size - label_freq_mat[lpairs[0],lpairs[1]])/s_ids.size)

			# Set flags for these slaves to True, which means they should be solved again. 
			slave_flags[s_ids] = True

		# Reset the slaves to solve. 
		self._slaves_to_solve = np.where(slave_flags == True)[0]

		# Record the norm of the subgradient. 
		self.subgradient_norms += [norm_gt]


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


	def _find_disagreeing_nodes(self):
		'''
		A function to find disagreeing nodes at a step of the algorithm. 
		'''
		disagreements = np.zeros(self.n_nodes, dtype=bool)

		for n_id in range(self.n_nodes):
			s_ids	= self.nodes_in_slaves[n_id]
			ls_		= [self.slave_list[s].get_node_label(n_id) for s in s_ids]
			ret_	= reduce(lambda x,y: x and (y == ls_[0]), ls_[1:], True)
			if not ret_:
				disagreements[n_id] = True
		return np.where(disagreements == True)[0]


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
		
		self.primal_cost = _compute_primal_cost(labels=self.labels)
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
		for n_id in range(self.n_nodes):
			# Retrieve the labels assigned by every slave to this node. 
			s_labels = [self.slave_list[s].label_from_node[n_id] for s in self.nodes_in_slaves[n_id]]
			# Find the most voted label. 
			labels[n_id] = stats.mode(s_labels)[0][0]

		# Return this labelling. 
		return labels

	def _compute_primal_cost(self, labels=None):
		'''
		Returns the primal cost at a given stage of the optimisation.
		For each stage of the optimisation, this function takes the labelling from every
		slave, and adds the part of the energy corresponding to that labelling
		from each component of the energy function. 
		'''
		cost	= 0

		# Generate a labelling first. 
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
	total_e		+= edge_energies[0][i,j] + edge_energies[1][i,k] \
				    + edge_energies[2][j,l] + edge_energies[3][k,l]

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
	labels			= np.zeros(4)

	# Record energies for every labelling. 
	for l_ in range(all_labellings.shape[0]):
		total_e		= _compute_slave_energy(node_energies, edge_energies, all_labellings[l_,:])
		# Check if best. 
		if total_e < min_energy:
			min_energy	= total_e
			labels[:]	= all_labellings[l_,:]

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


