# dd_mrf_grid

DD-MRF on a regular 2-D grid
============================
(Code not yet complete)

This Python library implements the Dual Decomposition algorithm by Komodakis, Paragios, and Tziritas [1] on a regular
2-D grid by dividing the graph into sub-problems (slaves), each sub-problem being a unique loop of four points. 

Requirements
------------
The Python packages `numpy` and `joblib` are required. `numpy` is used extensively for mathematics and `joblib`
allows distributing operations over several cores of the machine. 

Usage
-----
To use, simply include the library in your Python code:
```python
import dd_mrf_grid as ddmrf
```      
Graphs can be built incrementally using the Lattice class provided by the module. 
```python
lat = ddmrf.Lattice(rows, cols, n_labels)
```      
where `rows` and `cols` signify the number of rows and columns in the grid, and n_labels is the number of labels 
each node takes. n_labels can be an integer (in which case all nodes can take the same number of labels), or a 
numpy array or list of integers of size `rows*cols` to specify how many labels each node must take.

The nodes are indexed in a row-major fashion, starting at the top-left. Thus, the top-left node (also `[0,0]` in
the Numpy array) is index `0`, and the index increments along the row first. 
Edges can be specified by indicating the two nodes they are between. Nodes are specified by indices, as indicated
above. 

To add energies to nodes and edges, use
```python
lat.set_node_energies(i, E_node)
lat.set_edge_energies(i, j, E_edge)
```
where `i` and `j` are node indices, and `E_node` and `E_edge` are, respectively, the node and edge energies. `E_node` 
must have `n_labels[i]` elements, while `E_edge` must have shape `(n_labels[x], n_labels[y])`, where `x` and `y` are 
defined as `x, y = min(i,j), max(i,j)`.

All nodes and edges must be assigned energies before the overall energy can be minimised. To verify whether all 
energies have been defined, use
```python
lat.check_completeness()
```

A complete grid can be optimised using 
```python
lat.optimise(a_start=1.0)
```
 
After the algorithm converges, the obtained labelling is stored in `lat.labels`. 

References
----------
[1] MRF Energy Minimization and Beyond via Dual Decomposition, N. Komodakis, N. Paragios and G. Tziritas, PAMI 2011. 
