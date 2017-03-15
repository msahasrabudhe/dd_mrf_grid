# dd_mrf_grid

DD-MRF on a regular 2-D grid
============================

This Python library implements the Dual Decomposition algorithm by Komodakis, Paragios, and Tziritas [1] on a regular
2-D grid by dividing the graph into sub-problems (slaves), each sub-problem being a unique loop of four points. 

To use, simply include the library in your Python code:
```python
import dd_mrf_grid as ddmrf
```      
Graphs can be built incrementally using the Lattice class provided by the module. 
```python
lat = ddmrf.Lattice(rows, cols, n_labels)
```      
where rows and cols signify the number of rows and columns in the grid, and n_labels is the number of labels 
each node takes. n_labels can be an integer (in which case all nodes can take the same number of labels), or a 
numpy array or list of integers of size `rows*cols` to specify how many labels each node must take.

The nodes are indexed in a row-major fashion, starting at the top-left. Thus, the top-left node (also `[0,0]` in
the Numpy array) is index `0`, and the index increments along the row first. 
Edges can be specified by indicating the two nodes they are between. Nodes are specified by indices, as indicated
above. 


