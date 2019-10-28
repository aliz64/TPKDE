"""
This file contains the GPU implementations of algorithms calculating the
min-max closure of a set.
"""
import numpy as np
import math

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import cuda_code_strings as cu

VERBOSE = True

def get_grid(X):
    """
    Get a grid representation of numpy array X.
    X is meant to represent a dataset of real vectors dimension of D and dataset
    size N, and this will have shape (N, D).

    A grid representation of X will be a D-hypercube with side length N,
    represented as an int8 numpy array. This grid will be sparse and a gridpoint
    represents a possible D-dimensional vector with each coordinate picked from
    one of the corresponding coordinates of the N vectors in X. The indexing is
    done after the vectors in X are sorted. Thus Grid[0,..,0] represents the
    elementwise minimum of all elements of all vectors in X.
    
    This method also returns a list of dimensions of X, sorted independently.
    The grid and sorted dimension list can be used to recreate X.

    There is an assumption that the points in X are unique in every dimension -
    this is not unreasonable to assume if X is a randomly generated real vector
    set.

    """
    # 'N' is the number of samples, 'D' is the dimensionality.
    N, D = X.shape

    # Sort each dimension
    sort_dims = [ np.sort(X.T[i]) for i in range(D) ]

    # The grid is an 'D'-dimensional hypercube of side length 'N'.
    grid_shape = tuple(N for i in range(D))

    # Fill grid with zeros initially.
    grid = np.zeros(grid_shape).astype(np.int8)

    # 'sort_dim_revs' maps from a dimension index 'i' to a dictionary, which
    # is a map from a value to its sorted position in that dimension.
    sort_dims_rev = [{} for i in range(D)]
    for i in range(D):
        for j in range(N):
                sort_dims_rev[i][sort_dims[i][j]] = j

    # Populate the grid with 1 where points exist.
    for i in range(N):
        index_in_grid = []
        for d in range(D):
            index_in_grid.append(sort_dims_rev[d][X[i][d]])
        grid[tuple(index_in_grid)] = 1

    return N, D, grid, sort_dims

def construct_output(present, sort_dims):
    """
    present:   a list of indexes in a grid representation of real vector sample
                 set X
    sort_dims: a map from dimension number to a sorted list of the values in the
                 sample set X corresponding to the dimension number

    Construct output from the present points and sorted dimensions map. This can
    be used to translate a grid representation of a real vector set back into
    its original numpy array representation.
    """
    D = len(sort_dims)
    outlist = []
    for p in present:
        pt = np.array([])
        for d in range(D):
            pt = np.append(pt, sort_dims[d][p[d]])
        outlist.append(pt)
    outlist = np.array(outlist)
    return outlist

def get_map_num_to_pair(N):
    """
    Derive a mapping from an index to a unique pair from the set {0,1,...(N-1)}
    
    Pairs are stored in a numpy array with pair i being
    (map_array[i], map_array[i+N*(N-1)/2])

    This is useful if we have an array of size N and want to run a thread for 
    each pair of elements in the array, with each thread needing the indices of
    a unique pair.
    """
    x_cords = []
    y_cords = []

    # Iterate over triangle j < i
    i = 0
    j = 0
    for i in range(N):
        for j in range(i):
            x_cords.append(i)
            y_cords.append(j)
    
    # Length of array is 2* [ N*(N-1)/2 ]
    map_array = np.array(x_cords + y_cords).astype(np.int32)
    return map_array

def gpu_mm(X, thres=float('inf')):
    """
    This is a simple GPU based algorithm for calculation of MM(X).
    This operates on the grid representation of the vector set X.
    It calculates the list of present points in the grid and the number of pairs
    of indexes in this present array is essentially being parallelized over.
    Each gpu thread is given 'tile_size' number of pairs of points in the
    present array which it calculates the min and max of.

    This improves significantly on CPU algorithm but fails in most large sample
    cases due to a large amount of memory being needed for index maps.
    """
    # 'N'     : The number of samples
    # 'D'     : The dimensionality of sample points
    # 'grid'    : Sample in (Z+)^D polyhedral representation (detail in paper)
    # 'sort_dims' : Map from dimension index to the sorted dimension
    # NOTE: grid is np.int8
    N, D, grid, sort_dims = get_grid(X)

    # Get the CUDA code this this instance
    func = SourceModule(eval(f"cu.get_code_pair({D})")).get_function("f")
    
    # the actual non-zero locations in the grid. Must be N.
    # i.e. present is a DxN array.
    present = np.argwhere(grid == 1).astype(np.int32)
    assert len(present) == N

    # Allocate and copy 'grid' and 'present' arrays to GPU
    grid_gpu = cuda.mem_alloc(grid.nbytes)
    pres_gpu = cuda.mem_alloc(present.nbytes)
    cuda.memcpy_htod(grid_gpu, grid)
    cuda.memcpy_htod(pres_gpu, present)

    cursize = N
    i = 0
    while i < thres:
        i += 1

        # Get np.int32 array mapping from thread index to a pair of sample index
        idx_to_pair = get_map_num_to_pair(cursize)
        idx_to_pair_gpu = cuda.mem_alloc(idx_to_pair.nbytes)
        cuda.memcpy_htod(idx_to_pair_gpu, idx_to_pair)

        # The number of unique pairs of {0,1,...(N-1)}
        # Need to multiply by 2 since idx_to_pair stores both coordinates of the
        # pair.
        num_pairs = int(cursize*(cursize-1)/2)
        assert idx_to_pair.shape[0] == num_pairs*2

        # CUDA block and grid size
        tile_size = 1000
        block = (1024, 1, 1)
        cugrid = (num_pairs//(1024*tile_size) + 1, 1)

        if VERBOSE:
            print("Step {} : size={} npairs={} block={} grid={}".format(
                i, cursize, num_pairs, block, cugrid))

        # Execute the CUDA kernel

        # f(int8_t* grid, int32_t* present, int32_t* idx_to_pair,
        #   int32_t ns, int32_t N, int32_t tile_size) {
        func(grid_gpu, pres_gpu, idx_to_pair_gpu,
             np.int32(cursize), np.int32(N), np.int32(tile_size),
             block=block, grid=cugrid)
        
        # copy the gird back and calculate the new present array
        # free the present array on the GPU
        cuda.memcpy_dtoh(grid, grid_gpu)
        present = np.argwhere(grid == 1).astype(np.int32)
        pres_gpu.free()
        idx_to_pair_gpu.free()
        
        # break if nothing was added to the grid i.e. size of present array is
        # unchanged
        if cursize == len(present):
            break

        # Update present on the GPU and its current size
        pres_gpu = cuda.mem_alloc(present.nbytes)
        cuda.memcpy_htod(pres_gpu, present)
        cursize = len(present)

    # Free grid at the end.
    grid_gpu.free()

    outlist = construct_output(present, sort_dims)
    if VERBOSE:
        print(f"Finished simple GPU algorithm in {i} iterations.")
    return outlist

def gpu_mm_hyper(X, thres=float('inf')):
    """
    This is an improved (over gpu_mm) GPU based algorithm for calculation of
    MM(X). This operates on the grid representation of the vector set X.
    
    It calculates the list of present points in the grid and parallelizes over
    the set of present points (as opposed to pairs of points as in gpu_mm). Each
    thread however will do more work and go over all the other points
    (optimized to go over unique points only) and calculate the min and max.

    This seems to improve significantly over gpu_mm method.
    """
    # 'N'     : The number of samples
    # 'D'     : The dimensionality of sample points
    # 'grid'    : Sample in (Z+)^D polyhedral representation (detail in paper)
    # 'sort_dims' : Map from dimension index to the sorted dimension
    # NOTE: grid is np.int8
    N, D, grid, sort_dims = get_grid(X)

    # Get the CUDA code this this instance
    func = SourceModule(eval(f"cu.get_code_hyper({D})")).get_function("f")

    # the actual non-zero locations in the grid. Must be N initially.
    # i.e. present is a DxN array.
    present = np.argwhere(grid == 1).astype(np.int32)
    assert len(present) == N

    # Allocate and copy 'grid' and 'present' arrays to GPU
    grid_gpu = cuda.mem_alloc(grid.nbytes)
    pres_gpu = cuda.mem_alloc(present.nbytes)
    cuda.memcpy_htod(grid_gpu, grid)
    cuda.memcpy_htod(pres_gpu, present)

    cursize = N
    i = 0
    while i < thres:
        i += 1

        # The number of unique pairs of {0,1,...(N-1)}
        num_pairs = int(cursize*(cursize-1)/2)

        # CUDA block and grid size
        block = (1024, 1, 1)
        cugrid = (cursize//1024 + 1, 1)

        if VERBOSE:
            print("Step {} : cursize={} npairs={} block={} grid={}".format(
                i, cursize, num_pairs, block, cugrid))

        # Execute the CUDA kernel
        # f(int8_t* grid, int32_t* present, int32_t ns, int32_t N)
        func(grid_gpu, pres_gpu, np.int32(cursize), np.int32(N),
             block=block, grid=cugrid)
        
        # Copy the grid back and calculate the new present
        # free present on the GPU
        cuda.memcpy_dtoh(grid, grid_gpu)
        present = np.argwhere(grid == 1).astype(np.int32)
        pres_gpu.free()
        
        # If nothing was added to the grid i.e. length of present is unchanged,
        # the algorithm has finished.
        if cursize == len(present):
            break

        # Update present on the GPU and its current size
        pres_gpu = cuda.mem_alloc(present.nbytes)
        cuda.memcpy_htod(pres_gpu, present)
        cursize = len(present)

    # Free grid at the end.
    grid_gpu.free()

    outlist = construct_output(present, sort_dims)
    if VERBOSE:
        print("Finished hyper GPU algorithm in {} iterations.".format(i))
    return outlist

def gpu_mm_grid(X, thres=float('inf'), K=5):
    """
    Another improved (over gpu_mm) GPU based algorithm for calculation of
    MM(X). This operates on the grid representation of the vector set X.

    Compared to gpu_mm_hyper, this method aims to reduce memory transfers
    between the CPU host memory and GPU. This is done by not using a present
    array which need the grid to be copied to and from the GPU in every single
    iteration.
    This method parallelizes over every single possible vector i.e. N^D vectors.
    This is a lot of extra work, but can be done entirely on GPU without needing
    to transfer any memory. However after a fixed number of iterations, the grid
    does need to be copied back to check for convergence.

    This (gpu_mm_hyper) improves significantly over gpu_mm method.
    """
    assert K > 0
    # 'N'     : The number of samples
    # 'D'     : The dimensionality of sample points
    # 'grid'    : Sample in (Z+)^D polyhedral representation (detail in paper)
    # 'sort_dims' : Map from dimension index to the sorted dimension
    # NOTE: grid is np.int8
    N, D, grid, sort_dims = get_grid(X)

    # Get the CUDA code this this instance
    func = SourceModule(eval(f"cu.get_code_grid({D})")).get_function("f")

    # Allocate and copy grid to GPU
    grid_gpu = cuda.mem_alloc(grid.nbytes)
    cuda.memcpy_htod(grid_gpu, grid)

    # The block and grid for CUDA call
    block = (1024, 1, 1)
    cugrid = (N**D//1024 + 1, 1) # Work is done for all N^D possibilities.

    converge = False
    cursize = N
    i = 0
    while i < thres:
        i += 1

        # If on a 'K'-multiple iteration, copy grid back and record its 'size'.
        if thres == float('inf') and (i % K) == 0:
            cuda.memcpy_dtoh(grid, grid_gpu)
            cursize = len(np.argwhere(grid == 1))
        
        if VERBOSE:
            print("Step {} : N^D={} cursize={} block={} grid={}".format(
                i, int(N**D), cursize, block, cugrid))

        # Execute the CUDA kernel
        # void f(int8_t* grid, int32_t N, int32_t N_D)
        func(grid_gpu, np.int32(N), np.int32(int(N**D)),
             block=block, grid=cugrid)
        
        # wait for kernel completion
        cuda.Context.synchronize()

        # Check if the size has been updated; break loop if so.
        if thres == float('inf') and (i % K) == 0:
            cuda.memcpy_dtoh(grid, grid_gpu)
            if len(np.argwhere(grid == 1)) == cursize:
                converge = True
                break

    # If grid was not updated, i.e. if convergence was not reached but threshold
    # was reached, then copy grid (otherwise it has already been copied)
    if not converge:
        cuda.memcpy_dtoh(grid, grid_gpu)
    
    # Free grid at the end.
    grid_gpu.free()
    
    # Calculate the present array and output MM(X)
    present = np.argwhere(grid == 1)
    outlist = construct_output(present, sort_dims)

    if VERBOSE:
        print("Finished grid GPU algorithm in {} iterations.".format(i))
    return outlist

def mmc_classic(X, iters=float("inf")):
    """
    Calculates MM(X) using naive CPU iterative algorithm.
    """

    # N is the number of original points, D is the dimensionality.
    N = len(X)
    D = len(X[0])

    # Add all the points as tuples to a list, and a set
    x_set = set()
    x_list = []
    for i in range(N):
        x_i = tuple(X[i])
        x_set.add(x_i)
        x_list.append(x_i)

    # 'c' keeps track of iterations of the algorithm (each iteration of the
    # algorithm is 1 iteration over the Cartesian product of the current list
    # with itself).
    c = 0
    # 'boundary' tracks the highest index in the current list, smaller than
    # which all indices have been compared in pairs (it is actually length of
    # active/current list in previous iteration).
    boundary = 0
    while c < iters:
        c += 1
        to_add = set()
        old_size = len(x_set)
        for i in range(boundary,old_size):
            for j in range(i):
                min_i_j = tuple( map(lambda x : min(x[0],x[1]), zip(x_list[i],
                                                                    x_list[j])))
                max_i_j = tuple( map(lambda x : max(x[0],x[1]), zip(x_list[i],
                                                                    x_list[j])))
                if (min_i_j not in x_set):
                    to_add.add(min_i_j)
                if (max_i_j not in x_set):
                    to_add.add(max_i_j)
        if len(to_add) == 0:
            break
        boundary = old_size
        for pt in to_add:
            x_set.add(pt)
            x_list.append(pt)
    if VERBOSE:
        print("Finished CPU algorithm in", c, "iterations!")
    return x_list

