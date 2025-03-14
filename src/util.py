import os
import numpy as np
from scipy.interpolate import UnivariateSpline
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.stats as stats

def p_lis(gamma_1, threshold=0.1, label=None, savepath=None):
    '''
    Rejection of null hypothesis are shown as 1, consistent with online BH, Q-value, smoothFDR methods.
    # LIS = P(theta = 0 | x)
    # gamma_1 = P(theta = 1 | x) = 1 - LIS
    '''
    gamma_1 = gamma_1.ravel()
    dtype = [('index', int), ('value', float)]
    size = gamma_1.shape[0]

    # flip
    lis = np.zeros(size, dtype=dtype)
    lis[:]['index'] = np.arange(0, size)
    lis[:]['value'] = 1-gamma_1 

    # get k
    lis = np.sort(lis, order='value')
    cumulative_sum = np.cumsum(lis[:]['value'])
    k = np.argmax(cumulative_sum > (np.arange(len(lis)) + 1)*threshold)

    signal_lis = np.zeros(size)
    signal_lis[lis[:k]['index']] = 1

    if savepath is not None:
        np.save(os.path.join(savepath, 'gamma.npy'), gamma_1)
        np.save(    os.path.join(savepath, 'lis.npy'), signal_lis)

    if label is not None:
        # GT FDP
        rx = k
        sigx = np.sum(1-label[lis[:k]['index']])
        fdr = sigx / rx if rx > 0 else 0

        # GT FNR
        rx = size - k
        sigx = np.sum(label[lis[k:]['index']]) 
        fnr = sigx / rx if rx > 0 else 0

        # GT ATP
        atp = np.sum(label[lis[:k]['index']]) 
        return fdr, fnr, atp, signal_lis

def qvalue(pvals, threshold=0.05, verbose=False):
    """Function for estimating q-values from p-values using the Storey-
    Tibshirani q-value method (2003).

    Input arguments:
    ================
    pvals       - P-values corresponding to a family of hypotheses.
    threshold   - Threshold for deciding which q-values are significant.

    Output arguments:
    =================
    significant - An array of flags indicating which p-values are significant.
    qvals       - Q-values corresponding to the p-values.
    """

    """Count the p-values. Find indices for sorting the p-values into
    ascending order and for reversing the order back to original."""
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    rev_ind = np.argsort(ind)
    pvals = pvals[ind]

    # Estimate proportion of features that are truly null.
    kappa = np.arange(0, 0.96, 0.01)
    pik = [sum(pvals > k) / (m*(1-k)) for k in kappa]
    cs = UnivariateSpline(kappa, pik, k=3, s=None, ext=0)
    pi0 = float(cs(1.))
    if (verbose):
        print('The estimated proportion of truly null features is %.3f' % pi0)

    """The smoothing step can sometimes converge outside the interval [0, 1].
    This was noted in the published literature at least by Reiss and
    colleagues [4]. There are at least two approaches one could use to
    attempt to fix the issue:
    (1) Set the estimate to 1 if it is outside the interval, which is the
        assumption in the classic FDR method.
    (2) Assume that if pi0 > 1, it was overestimated, and if pi0 < 0, it
        was underestimated. Set to 0 or 1 depending on which case occurs.

    I'm choosing second option 
    """
    if pi0 < 0:
        pi0 = 0
    elif pi0 > 1:
        pi0 = 1

    # Compute the q-values.
    qvals = np.zeros(np.shape(pvals))
    qvals[-1] = pi0*pvals[-1]
    for i in np.arange(m-2, -1, -1):
        qvals[i] = min(pi0*m*pvals[i]/float(i+1), qvals[i+1])

    # Test which p-values are significant.
    significant = np.zeros(np.shape(pvals), dtype='bool')
    significant[ind] = qvals<threshold

    """Order the q-values according to the original order of the p-values."""
    qvals = qvals[rev_ind]
    return significant, qvals
