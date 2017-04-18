# Max product belief propagation

import numpy as np

def max_prod_bp(node_pot, edge_pot, graph_struct):
	adj_mat			= graph_struct['adj_mat']
	n_nodes			= graph_struct['n_nodes']
	n_edges			= graph_struct['n_edges']
	edge_ends		= graph_struct['edge_ends']
	node_degrees	= graph_struct['node_degrees']
	e_ids			= graph_struct['e_ids']
	n_states		= graph_struct['n_states']

	# Normalise node_pot and edge_pot to lie in [0,1] 
	#   so that messages don't explode. 
#	mnp = np.max(node_pot)
#	mep = np.max(edge_pot)
#	mmx = np.max([mnp, mep])
#	node_pot /= mmx
#	edge_pot /= mmx

	# The maximum state
	max_state	= np.max(n_states)

	# The final labelling
	labels = np.zeros(n_nodes, dtype=np.int)

	# Copy the adjacency matrix. 
	adj_mat_copy = np.zeros_like(adj_mat)
	adj_mat_copy[:] = adj_mat[:]
	# Copy the node degrees. 
	node_degrees_copy		= np.zeros_like(node_degrees)
	node_degrees_copy[:]	= node_degrees

	# Stage I: Propagate from leaves to root. 

	# Find leaf nodes. 
	queue = np.where(node_degrees_copy == 1)[0].tolist()

	# Record if node is already visited. 
	visited = np.zeros(n_nodes, dtype=np.bool)
	# Record if node is in queue. 
	in_queue = np.zeros(n_nodes, dtype=np.bool)

	# Place initial nodes in queue. 
	in_queue[queue] = True

	# Messages to be received. Each element i stores the message received
	#   by node i. 
	messages = np.zeros((n_edges*2, max_state))
	# Stores the product of in-coming messages into a node. 
	messages_in = np.ones((n_nodes, max_state))
	sent = np.zeros(n_edges*2, dtype=np.bool)

	# Store the path we took for these messages
	path = []

	while len(queue) != 0:
		# The first node in queue. 
		_from	= queue[0]
		queue 	= queue[1:]

		# Mark this as visited. 
		visited[_from] = True
		# The node to send message to
		_find_next = np.where(adj_mat_copy[_from,:] == True)[0]
		# If there are no next nodes, we have reached the root. 
		if _find_next.size == 0:
			_root = _to	
			break
		_to = _find_next[0]

		path = [[_from, _to]] + path

		# Update node degrees by removing this edge. 
		node_degrees_copy[_from]	-= 1
		node_degrees_copy[_to]	-= 1
		# Update adj_mat by removing this edge. 
		adj_mat_copy[_from, _to] = False
		adj_mat_copy[_to, _from] = False

		# Node ID for this edge. 
		e_id = e_ids[_from, _to]

		# This is needed to slice edge_pot properly. 
		_mfrom, _mto = np.min([_from,_to]), np.max([_from,_to])
		# edge_pot for this edge. 
		_ep = edge_pot[e_id, 0:n_states[_mfrom], 0:n_states[_mto]]
		# If we are sending a message the other way, it is necessary to flip edge_pot. 
		if _from > _to:
			_ep = _ep.T

		# node_pot for this node
		_np = node_pot[_from, 0:n_states[_from]]

		# Send messages. 
		_t = _np*messages_in[_from, 0:n_states[_from]]
		_t = np.tile(np.reshape(_t, [-1, 1]), [1, n_states[_to]])	# Max-product

		# The message ID, m_id, is the same as e_id if _from < _to, 
		#   else it is e_id + n_edges.
		m_id	= e_id if _from < _to else e_id + n_edges
		messages[m_id, :n_states[_to]] = np.max(_t*_ep, axis=0)				# Max-product
		sent[m_id] = True
		# Normalise.
		messages[m_id, :n_states[_to]] /= np.sum(messages[m_id,:n_states[_to]])

		messages_in[_to, :n_states[_to]] *= messages[m_id, :n_states[_to]]

		# If the degree of the node _to is 0, it is supposed to be the root. 
		# We break from the while loop
		if node_degrees_copy[_to] == 0:
			_root = _to
			break
		# If the degree of the node _to is 1, insert it to the end of the queue. 
		if node_degrees_copy[_to] == 1:
			queue += [_to]
	# Make sure that queue is empty after we break. 

	# Now propagate back to the leaves. 
	for _e in path:
		# The inverse edge.
		_to, _from = _e
		# Edge id. 
		e_id = e_ids[_to, _from]

		# This is needed to slice edge_pot properly. 
		_mfrom, _mto = np.min([_from,_to]), np.max([_from,_to])
		# Edge potentials for this edge. 
		_ep = edge_pot[e_id, :n_states[_mfrom], :n_states[_mto]]
		# If we are sending a message the other way, it is necessary to flip edge_pot. 
		if _from > _to:
			_ep = _ep.T

		neighs = np.setdiff1d(np.where(adj_mat[_from,:] == True)[0], [_to])
		n_mids = [e_ids[_from,n] if n < _from else e_ids[_from,n] + n_edges for n in neighs]

		# Get the product of messages into _from, excluding those from _to, and 
		#    remove this edge, i.e., _to->_from, from the product
		tm_id = e_id if _to < _from else e_id + n_edges
		mesg_prod = messages_in[_from,:n_states[_from]]/messages[tm_id,:n_states[_from]]
		# Compute and send message. 
		_t = node_pot[_from, :n_states[_from]]*mesg_prod
		_t = np.tile(np.reshape(_t, [-1, 1]), [1, n_states[_to]])	# Max-product

		m_id = e_id if _from < _to else e_id + n_edges
		messages[m_id,:n_states[_to]] = np.max(_t*_ep, axis=0)		
		sent[m_id] = True

		# Normalise.
		messages[m_id,:n_states[_to]] /= np.sum(messages[m_id,:n_states[_to]])

		messages_in[_to,:n_states[_to]] *= messages[m_id,:n_states[_to]]

#	messages_in /= np.tile(np.sum(messages_in, axis=1, keepdims=True), [1, max_state])

	for _n in range(n_nodes):
		_t = messages_in[_n,:n_states[_n]]*node_pot[_n,:n_states[_n]]
		# If there are multiple random values, randomly choose one. 
		_max_at    = np.where(_t == np.max(_t))[0]
		_n_max_at  = _max_at.size
		labels[_n] = _max_at[np.random.randint(_n_max_at)]
#		labels[_n] = np.argmax(_t)
		
	return labels, messages, messages_in


def make_graph_struct(adj_mat, n_states):
	''' 
	Make graph struct from adjacency matrix.
	Input adj_mat is a numpy.ndarray with dtype
	np.bool. It must be symmetric. 
	'''
	# Number of nodes in the graph.
	n_nodes = adj_mat.shape[0]

	# Number of edges in the graph. 
	n_edges = np.sum(adj_mat)/2
	
	# Edge ends for every edge. 
	edge_ends = np.zeros((n_edges, 2), dtype=np.int)
	# Edge ends can be easily found using np.where.
	(e1, e2) = np.where(np.tril(adj_mat).T == True)
	edge_ends[:,0] = e1
	edge_ends[:,1] = e2

	# The degree of each node. 
	node_degrees = np.sum(adj_mat, axis=1)

	# e_ids: a mapping between the nodes and edges.
	# The (i,j)-th element stores the ID of the edge between
	#   the nodes i and j. 
	e_ids = np.zeros((n_nodes, n_nodes), dtype=np.int)
	for e in range(n_edges):
		e0, e1 = edge_ends[e,:]
		e_ids[e0, e1] = e

	# Make it symmetric because the edges are undirected. 
	e_ids = e_ids + e_ids.T

	if type(n_states) is int:
		n_states = n_states*np.ones(n_nodes, dtype=np.int)
	else:
		n_states = np.array(n_states, dtype=np.int)

	graph_struct 				= {}
	graph_struct['n_nodes'] 	= n_nodes
	graph_struct['n_edges'] 	= n_edges
	graph_struct['adj_mat'] 	= adj_mat
	graph_struct['edge_ends']	= edge_ends
	graph_struct['e_ids'] 		= e_ids
	graph_struct['node_degrees'] = node_degrees
	graph_struct['n_states']	= n_states

	return graph_struct

