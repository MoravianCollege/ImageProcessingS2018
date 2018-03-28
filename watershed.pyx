cdef long INIT = -1 # initial value of output pixels
cdef long MASK = -2 # initial value of a threshold level
cdef long INQUEUE = -3 # value assigned to pixels put into the queue
cdef long WSHED = 0 # value of pixels belonging to watersheds

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list create_queue(double[:,::1] im, long[:,::1] lbls, double height):
    cdef Py_ssize_t i, j, ni, nj
    cdef list queue = []
    for i in range(1, im.shape[0]-1):
        for j in range(1, im.shape[1]-1):
            if im[i,j] != height: continue
            # If a neighbor is in a watershed or labeled add it to pixel to the queue instead
            if lbls[i, j-1] >= WSHED or lbls[i, j+1] >= WSHED or lbls[i-1, j] >= WSHED or lbls[i+1, j] >= WSHED:
                lbls[i, j] = INQUEUE
                queue.append((i, j))
            else: lbls[i, j] = MASK # default to be placed as MASK
    return queue

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long process_queue_point_neighbor(list queue, long[:,::1] lbls, Py_ssize_t ni, Py_ssize_t nj, long lbl, bint* flag):
    cdef long n_lbl = lbls[ni, nj]
    if n_lbl > 0: # the pixel belongs to an already labeled basin
        if lbl == INQUEUE or lbl == WSHED and flag[0]:
            return n_lbl
        elif lbl > 0 and lbl != n_lbl:
            flag[0] = False
            return WSHED
    elif n_lbl == WSHED and lbl == INQUEUE:
        flag[0] = True
        return WSHED
    elif n_lbl == MASK:
        lbls[ni, nj] = INQUEUE
        queue.append((ni, nj))
    return lbl

@cython.boundscheck(False)
@cython.wraparound(False)
cdef process_queue(list queue, long[:,::1] lbls):
    cdef Py_ssize_t i, j, ni, nj
    cdef long lbl
    cdef bint flag = False
    while queue:
        i,j = queue.pop(0)
        lbl = lbls[i,j]
        # Label pixel by inspecting neighbors
        lbl = process_queue_point_neighbor(queue, lbls, i, j-1, lbl, &flag)
        lbl = process_queue_point_neighbor(queue, lbls, i, j+1, lbl, &flag)
        lbl = process_queue_point_neighbor(queue, lbls, i-1, j, lbl, &flag)
        lbl = process_queue_point_neighbor(queue, lbls, i+1, j, lbl, &flag)
        lbls[i,j] = lbl
        
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long update_labels(long[:,::1] lbls, long nlbls):
    cdef Py_ssize_t i, j, pi, pj, ni, nj
    cdef list stack
    for i in range(1, lbls.shape[0]-1):
        for j in range(1, lbls.shape[1]-1):
            if lbls[i,j] != MASK: continue
            nlbls += 1
            lbls[i,j] = nlbls
            stack = [(i,j)]
            while stack:
                pi,pj = stack.pop()
                if lbls[pi,pj-1] == MASK: lbls[pi,pj-1] = nlbls; stack.append((pi,pj-1))
                if lbls[pi,pj+1] == MASK: lbls[pi,pj+1] = nlbls; stack.append((pi,pj+1))
                if lbls[pi-1,pj] == MASK: lbls[pi-1,pj] = nlbls; stack.append((pi-1,pj))
                if lbls[pi+1,pj] == MASK: lbls[pi+1,pj] = nlbls; stack.append((pi+1,pj))
    return nlbls

def watershed(im, hMin=0.0, hMax=1.0):
    """
    Apply fast watersheds using flooding simulations, as described
    by Soille, Pierre, and Luc M. Vincent. "Determining watersheds 
    digital pictures via flooding simulations." Lausanne-DL 
    tentative. International Society for Optics and Photonics, 1990.
    NOTE: this algorithm may have plateaus in the dams.
    """
    from numpy import pad, ones, unique, long as np_long
    from skimage import img_as_float

    # We want at most 1024 distinct values or this is going to take forever
    im = (img_as_float(im) * 1024).round() / 1024
    
    cdef double[:,::1] im_ = pad(im, 1, 'constant', constant_values=-1) # pad the image to make things easier
    cdef long[:,::1] lbls = INIT*ones((im_.shape[0], im_.shape[1]), np_long)
    cdef long nlbls = 0

    # Get all heights, excluding heights above and below the min/max
    heights = unique(im)
    heights = heights[(hMin <= heights) & (heights <= hMax)]

    for height in heights:
        # Find the points ot be processed at the current height
        queue = create_queue(im_, lbls, height)

        # Process the queue until empty
        process_queue(queue, lbls)
                        
        # Assign labels to new minima
        nlbls = update_labels(lbls, nlbls)

    # Set background (unlabeled) pixels
    lbls.base[lbls.base == INIT] = 0
    return lbls.base[1:-1,1:-1], nlbls
