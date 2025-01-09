import numpy as np

## QUALITY METRICS 

def jaccard(gt, rec, tol):
    """
    Compute Jaccard index and related metrics considering tolerance.

    Parameters:
    gt.x (list): Ground truth positions set (list of elements).
    rec.x (list): Reconstructed positions set (list of elements).
    tol (float): Tolerance for set element comparison.

    Returns:
    true positives = common elements within tolerance
    false negatives = elements in gt not in rec
    false positives = elements in rec not in gt
    tuple: Jaccard index, true positives, false negatives, false positives.
    """
    
    # Helper function to determine if elements are within tolerance
    within_tol = lambda a, b: abs(a - b) <= tol

    # Convert lists to sets
    set_gt = set(gt.x)
    set_rec = set(rec.x)

    # Compute correctly detected molecules
    intersection = 0
    for a in set_gt:
        for b in set_rec:
            if within_tol(a, b):
                intersection = intersection + 1
                set_rec.remove(b)  # Remove b from set_rec in order to 
                                    # count a reconstructed molecule 
                                    # only once as correctly detected
                break  # Break the loop when a satisfying b is found for this a

    # Compute false negatives and false positives
    set_rec = set(rec.x) # I need the inital lenght of set_rec to count false positives
    true_positives = intersection
    false_negatives = len(set_gt) - intersection
    false_positives = len(set_rec) - intersection
    jaccard_index = true_positives / (true_positives + false_negatives + false_positives)

    return jaccard_index, true_positives, false_negatives, false_positives

def jaccard_rmse(gt, rec, tol):
    """
    Compute Jaccard index and related metrics.
    Save in CD_rec_ampl and CD_rec_pos the amplitudes and positions 
    of the reconstructed molecules classified as correctly detected;
    and save in CD_gt_ampl and CD_gt_pos the amplitudes and positions 
    of the ground truth molecules classified as correctly detected.
    Compute RMSE for amplitudes and positions, only using the correctly 
    detected molecules saved as above. 

    Parameters:
    gt.x (list): Ground truth positions set (list of elements).
    rec.x  (list): Reconstructed positions set (list of elements).
    gt.a (list): Ground truth amplitudes set (list of elements).
    rec.a  (list): Reconstructed amplitudes set (list of elements).
    tol (float): Tolerance for set element comparison.

    Returns:
    tuple: Jaccard index, true positives, false negatives, false positives,
           RMSE for amplitudes, RMSE for positions.
    """
    
    # Helper function to determine if elements are within tolerance
    within_tol = lambda a, b: abs(a - b) <= tol

    # Convert lists to sets
    set_gt = set(gt.x)
    set_rec = set(rec.x)
    
    CD_gt_pos = []
    CD_rec_pos = []
    
    while len(set_gt) > 0 and len(set_rec) > 0:
        # Compute the absolute differences
        dist_array = np.abs(np.array(list(set_gt))[:, np.newaxis] - np.array(list(set_rec)))
        min_indices = np.unravel_index(np.argmin(dist_array), dist_array.shape)
        min_index_gt, min_index_rec = min_indices
        
        a = list(set_gt)[min_index_gt]
        b = list(set_rec)[min_index_rec]
        
        if within_tol(a, b):
            CD_gt_pos.append(a)
            CD_rec_pos.append(b)
            set_gt.remove(a)
            set_rec.remove(b)
        else:
            break  # None of the pairings has a distance smaller than the tolerance

    CD_gt_pos = np.array(CD_gt_pos)
    CD_rec_pos = np.array(CD_rec_pos)
    
    # Save correctly detected amplitudes 
    CD_rec_ampl = np.array([rec.a[rec.x.index(pos)] for pos in CD_rec_pos])
    CD_gt_ampl = np.array([gt.a[gt.x.index(pos)] for pos in CD_gt_pos])
    
    # Compute correctly detected molecules
    true_positives = len(CD_gt_pos)
    false_negatives = len(gt.x) - true_positives
    false_positives = len(rec.x) - true_positives
    
    jaccard_index = true_positives / (true_positives + false_negatives + false_positives)

    # Compute RMSE for amplitudes and positions
    RMSE_ampl = np.sqrt(np.nanmean((CD_rec_ampl - CD_gt_ampl) ** 2))
    RMSE_pos = np.sqrt(np.nanmean((CD_rec_pos - CD_gt_pos) ** 2))

    return jaccard_index, true_positives, false_negatives, false_positives, RMSE_ampl, RMSE_pos

def mse_measures(gt,rec):
    if gt.N==rec.N:
        mse = mean_squared_error(gt.x,rec.x) + mean_squared_error(gt.a,rec.a)
    elif gt.N>rec.N:
        mse = 0
        for i in range(rec.N):
            mse = mse+(gt.x[i]-rec.x[i])**2+(gt.a[i]-rec.a[i])**2
        for i in range(rec.N, gt.N):
            mse = mse+(gt.x[i])**2+(gt.a[i])**2
    elif rec.N>gt.N:
        mse = 0
        for i in range(gt.N):
            mse = mse+(gt.x[i]-rec.x[i])**2+(gt.a[i]-rec.a[i])**2
        for i in range(gt.N, rec.N):
            mse = mse+(rec.x[i])**2+(rec.a[i])**2
    return mse/max(gt.N,rec.N)
    

