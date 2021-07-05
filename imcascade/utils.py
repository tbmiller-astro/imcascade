import numpy as np
from scipy.special import gamma
import itertools
from astropy.convolution import convolve, Gaussian2DKernel

def guess_weights(sig, re, flux):
    """ Method to guess the weights of gaussian componenets given an re and flux.
    Based on a polynomial fit to the exp fits of Hogg & Lang 2013

    Parameters
    ----------
    sig: array
        List of gaussian widths for imcascade model
    re: Float
        Estimate of effective radius
    flux:
        Estimate of flux

    Returns
    -------
    a_i: Array
        Inital estimate of weights based on re and flux
"""
    P = [-0.82022178, -2.74810102,  0.0210647,   0.50427881]
    fitf = np.poly1d(P)
    #Their findings are normalized to re
    a_i = 10**fitf(np.log10(sig/re))
    a_i = a_i /np.sum(a_i) *flux
    return a_i

def expand_mask(mask, radius = 5, threshold = 0.001):
    """ Expands mask by convolving it with a Gaussians

    Parameters
    ----------
    Mask: 2D array
        inital mask with masked pixels equal to 1
    radius: Float
        width of gaussian used to convolve mask. default 5, set larger for more aggresive masking
    threshold: Float
        threshold to generate new mask from convolved mask. Default is 1e-3, set lower for more aggresive mask

    Returns
    -------
    new_mask: 2D-Array
        New, expanded mask
"""
    mask_conv = convolve(mask, Gaussian2DKernel(radius) )
    mask_conv[mask_conv>threshold] = 1
    mask_conv[mask_conv<=threshold] = 0
    return mask_conv

def asinh_scale(start,end,num):
    """Simple wrapper to generate list of numbers equally spaced in asinh space

    Parameters
    ----------
    start: floar
        Inital number
    end: Float
        Final number
    num: Float
        Number of number in the list

    Returns
    -------
    list: 1d array
        List of number spanning start to end, equally space in asinh space
"""
    temp = np.linspace(np.arcsinh(start), np.arcsinh(end), num = num )
    return np.sinh(temp)

def log_scale(start,end,num):
    """Simple wrapper to generate list of numbers equally spaced in logspace

    Parameters
    ----------
    start: floar
        Inital number
    end: Float
        Final number
    num: Float
        Number of number in the list

    Returns
    -------
    list: 1d array
        List of number spanning start to end, equally space in log space
"""
    return np.logspace(np.log10(start), np.log10(end), num = num)

def dict_add(dict_use, key, obj):
    """Simple wrapper to add obj to dictionary if it doesn't exist. Used in fitter.Fitter when defining defaults

    Parameters
    ----------
    dict_use: Dictionary
        dictionary to be, possibly, updated
    key: str
        key to update, only updated if the key doesn't exist in dict_use already
    obj: Object
        Object to be added to dict_use under key

    Returns
    -------
    dict_add: Dictionary
        updated dictionary
"""
    dict_res = dict_use.copy()
    if key not in dict_res:
        dict_res[key] = obj
    return dict_res

def get_med_errors(arr, lo = 16,hi = 84):
    """Simple function to find percentiles from distribution

    Parameters
    ----------
    arr: array
        Array containing in the distribution of intrest
    lo: float (optional)
        percentile to define lower error bar, Default 16
    hi: float (optional)
        percentile to define upper error bar, Default 84

    Returns
    -------
    (med,err_lo,err_hi): array
        Array containg the median and errorbars of the distiribution
"""
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
