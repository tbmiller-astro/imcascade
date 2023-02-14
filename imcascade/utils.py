import numpy as np
import jax.numpy as jnp
from scipy.special import gamma,comb
import itertools
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.stats.biweight import biweight_location as bwl

def reg_resid(x):
    x_expec = (x[:-2] + x[2:])/2
    return (x_expec - x[1:-1])

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


def get_sersic_amps(sigmas, re, n, percision = 10):
        # Calculation of MoG representation of Sersic Profiles based on lenstrometry implementation and Shajib (2019)
        kes = np.arange(2 * percision + 1)
        betas = np.sqrt(2 * percision * np.log(10) / 3. + 2. * 1j * np.pi * kes)
        epsilons = np.zeros(2 * percision + 1)

        epsilons[0] = 0.5
        epsilons[1:percision + 1] = 1.
        epsilons[-1] = 1 / 2. ** percision

        for k in range(1, percision):
            epsilons[2 * percision - k] = epsilons[2 * percision - k + 1] + 1 / 2. ** percision * comb(
                percision, k)

        etas = jnp.array( (-1.) ** kes * epsilons * 10. ** (percision / 3.) * 2. * np.sqrt(2*np.pi) )
        betas = jnp.array(betas)


        f_sigmas = jnp.sum(etas * sersic(jnp.outer(sigmas,betas), 1.,re,n).real,  axis=1)

        del_log_sigma = jnp.abs(jnp.diff(jnp.log(sigmas)).mean())

        amps = f_sigmas * del_log_sigma / jnp.sqrt(2*np.pi)

        # weighting for trapezoid method integral
        amps = amps.at[0].multiply(0.5)
        amps = amps.at[-1].multiply(0.5)

        amps = amps*2*np.pi*sigmas*sigmas
        return amps

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


def parse_input_dicts(bounds_dict,init_dict, fitter_class):
    init_dict = dict_add(init_dict, 're', 5.)

    init_dict = dict_add(init_dict, 'xc',int(fitter_class.img.shape[0]/2) )
    init_dict = dict_add(init_dict, 'yc',int(fitter_class.img.shape[1]/2) )
    bounds_dict = dict_add(bounds_dict, 'xc',[init_dict['xc'] - 10,init_dict['xc'] + 10])
    bounds_dict = dict_add(bounds_dict, 'yc',[init_dict['yc'] - 10,init_dict['yc'] + 10])

    init_dict = dict_add(init_dict, 'phi', np.pi/2.)
    bounds_dict = dict_add(bounds_dict, 'phi', [0,np.pi])


    init_dict = dict_add(init_dict,'q', 0.5)
    bounds_dict = dict_add(bounds_dict, 'q', [0.2,1.])

    if fitter_class.sky_model:
        #Try to make educated guesses about sky model

        #estimate background using median
        sky0_guess = bwl(fitter_class.img[np.where(fitter_class.mask == 0)], ignore_nan = True)
        if np.abs(sky0_guess) < 1e-6:
            sky0_guess = 1e-4*np.sign(sky0_guess+1e-12)
        init_dict = dict_add(init_dict, 'sky0', sky0_guess)
        bounds_dict = dict_add(bounds_dict, 'sky0', [-np.abs(sky0_guess)*10, np.abs(sky0_guess)*10])

        #estimate X and Y slopes using edges
        use_x_edge = np.where(fitter_class.mask[:,-1]*fitter_class.mask[:,0] == 0)
        sky1_guess = bwl(fitter_class.img[:,1][use_x_edge] - fitter_class.img[:,0][use_x_edge], ignore_nan =True)/fitter_class.img.shape[0]
        if np.abs(sky1_guess) < 1e-8:
            sky1_guess = 1e-6*np.sign(sky1_guess+1e-12)
        init_dict = dict_add(init_dict, 'sky1', sky1_guess)
        bounds_dict = dict_add(bounds_dict, 'sky1', [-np.abs(sky1_guess)*10, np.abs(sky1_guess)*10])

        use_y_edge = np.where(fitter_class.mask[-1,:]*fitter_class.mask[0,:] == 0)
        sky2_guess = bwl(fitter_class.img[-1,:][use_y_edge] - fitter_class.img[0,:][use_y_edge], ignore_nan =True)/fitter_class.img.shape[1]
        if np.abs(sky2_guess) < 1e-8:
            sky2_guess = 1e-6*np.sign(sky2_guess+1e-12)
        init_dict = dict_add(init_dict, 'sky2', sky2_guess)
        bounds_dict = dict_add(bounds_dict, 'sky2', [-np.abs(sky2_guess)*10, np.abs(sky2_guess)*10])

        if fitter_class.sky_type == 'tilted-plane':
            init_sky_model =  fitter_class.get_sky_model([init_dict['sky0'],init_dict['sky1'],init_dict['sky2']] )
        else:
            init_sky_model =  fitter_class.get_sky_model([init_dict['sky0'],] )

        A_guess = np.sum( (fitter_class.img - init_sky_model )[np.where(fitter_class.mask == 0)]  )
    else:
        A_guess = np.sum(fitter_class.img)

    #Below assumes all gaussian have same A
    init_dict = dict_add(init_dict, 'flux', A_guess )
    

    a_guess = guess_weights(fitter_class.sig, init_dict['re'], init_dict['flux'])

    init_dict = dict_add(init_dict, 'a_max', init_dict['flux'])
    init_dict = dict_add(init_dict, 'a_min', 0)

    for i in range(fitter_class.Ndof_gauss):
        a_guess_cur = a_guess[i]
        if a_guess_cur < init_dict['flux']/1e5:
            a_guess_cur = init_dict['flux']/1e4

        init_dict = dict_add(init_dict,'a%i'%i, a_guess_cur )
        bounds_dict = dict_add(bounds_dict,'a%i'%i,  [init_dict['a_min'], init_dict['a_max'] ])

    #Now set initial and boundry values once defaults or inputs have been used
    lb = [bounds_dict['xc'][0], bounds_dict['yc'][0], ]
    ub = [bounds_dict['xc'][1], bounds_dict['yc'][1], ]

    param_init = []
    param_init.append(init_dict['xc'] )
    param_init.append(init_dict['yc'])
    

    param_init.append(init_dict['q'])
    lb.append(bounds_dict['q'][0])
    ub.append(bounds_dict['q'][1])

    
 
    param_init.append( init_dict['phi'] )
    lb.append(bounds_dict['phi'][0])
    ub.append(bounds_dict['phi'][1])


    for i in range(fitter_class.Ndof_gauss):
        param_init.append(init_dict['a%i'%i])
        lb.append( bounds_dict['a%i'%i][0] )
        ub.append(bounds_dict['a%i'%i][1] )

    for i in range(fitter_class.Ndof_sky):
        param_init.append( init_dict['sky%i'%i] )
        lb.append(bounds_dict['sky%i'%i][0] )
        ub.append(bounds_dict['sky%i'%i][1] )

    lb = np.asarray(lb)
    ub = np.asarray(ub)
    param_init = np.asarray(param_init)

    return bounds_dict, init_dict, lb,ub,param_init 
