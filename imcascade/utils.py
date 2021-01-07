import numpy as np
from scipy.special import gamma
import itertools


def get_med_errors(arr, lo = 16,hi = 84):
    med,lo,hi = np.percentile(arr, [50.,lo,hi])
    return np.array([med, med - lo, hi - med])

def b(n):
    """ Simple function to approximate b(n) when evaluating a Sersic profile
    following Capaccioli (1989). Valid for 0.5 < n < 10

    Parameters
    ----------
    n: float or array
        Sersic index

    Returns
    -------
    b(n): float or array
        Approximation to Gamma(2n) = 2 gamma(2n,b(n))
"""
    return 1.9992*n - 0.3271

def sersic(r,n,re,Ltot):
    """Calculates the surface brightness profile for a Sersic profile

    Parameters
    ----------
    r: array
        Radii at which to evaluate surface brightness profile
    n: float
        Sersic index of profile
    re: float
        Half-light radius of profile
    Ltot: float
        Total flux of Sersic profile
    Returns
    -------

    float or array
        Surface brightness profile evaluate along the semi-major axis at 'r'

"""
    Ie = Ltot / (re*re* 2* np.pi*n * np.exp(b(n))* gamma(2*n) ) * b(n)**(2*n)
    return Ie*np.exp ( -b(n)*( (r/re)**(1./n) - 1. ) )

def min_diff_array(arr):
    """ Function used to calculate the minimum difference between any  two elements
    in a given array_like
    Parameters
    ----------
    arr: 1-D array
        Array to be searched
    Returns
    -------
    min_diff: Float
        The minimum difference between any two elements of the given array
"""
    min_diff = 1e6
    for combo in itertools.combinations(arr,2):
        diff = np.abs(combo[0] - combo[1])
        if diff < min_diff:
            min_diff = diff
    return min_diff
