import numpy as np
from scipy.special import gamma

def get_sbp(r,a_list,sig, return_ind = False):
    """Calculates the surface brightness profile for a given set of Gaussians
    widths and weights

    Parameters
    ----------

    r: array
        Radii at which to evaluate surface brightness profile
    a_list: array
        Gaussian weights
    sig: array:
        Gaussian widths
    return_ind: Bool, optional
        If True will return profiles of each individual gaussian along with
        the sum

    Returns
    -------

    float or array
        Surface brightness profile evaluate along the semi-major axis at 'r'

"""
    ans = np.zeros(r.shape)
    ind_prof = []
    for i in range(len(a_list)):
        prof_cur = a_list[i]/(2*np.pi*sig[i]**2) * np.exp(-r**2/ (2*sig[i]**2))
        ans += prof_cur
        ind_prof.append(prof_cur)
    if return_ind:
        return ans, np.asarray(ind_prof).transpose()
    else:
        return ans

def b(n):
    """ Simple function to approximate b(n) when evaluating the Sersic profile
    following Capaccioli (1989). Valid for 0.5 < n < 10

    Parameters
    ----------
    n: float or array
        Sersic index

    Returns
    -------
    b(n): flor or array
        Approximation to Gamma(2n) = 2 gamma(2n,b(n))
""""
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
