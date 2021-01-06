import numpy as np
from scipy.special import gamma
import itertools
from astropy.io import fits
import sep
from scipy.optimize import least_squares


def get_med_errors(arr, lo = 16,hi = 84):
    med,lo,hi = np.percentile(arr, [50.,lo,hi])
    return np.array([med, med - lo, hi - med])

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


class PSFFitter():
    """A Class used to fit Gaussian models to a PSF image"""
    def __init__(self,psf_img,oversamp = 1.):
        """ Initialize a PSFFitter instance
        Paramaters
        ----------
        psf_img: str or 2D array
        PSF data to be fit. If a string is given will assume it is a fits file
        and load the Data in the first HDU. If it is an array then will use
        as the PSF image. Either way it is assumed the PSF is centered and image
        is square.
        oversamp: Float, optional
        Factor by which the PSF image is oversampled. Default is 1.
"""
        if type(psf_img) == str:
            fits_file = fits.open(psf_img)
            self.psf_data = np.array(fits_file[0].data, dtype = '<f4')
        elif type(psf_img) == np.ndarray:
            self.psf_data = np.array(psf_img, dtype = '<f4')
        self.oversamp = oversamp

        #calculate 1-D circular profile
        _,_ = self.calc_profile()

    def calc_profile(self):
        """Calculates the 1-D PSF profile in 1 pixel steps assuming it is circular
        Returns
        -------
        intens: 1-D array
            Intensity profile
        radius:
            radii at which the intensity measuremnts are made
"""
        cent_pix = self.psf_data.shape[0]/2.
        maxr = int(cent_pix)
        r_in = np.arange(0, maxr - 1)
        r_out = np.arange(1,maxr)
        area = np.pi*(r_out**2 - r_in**2)
        prof_sum,_,_ = sep.sum_circann(self.psf_data, cent_pix,cent_pix, r_in,r_out)
        self.intens = prof_sum/area
        self.radius = (r_in + r_out)/2.
        return self.intens, self.radius

    def multi_gauss_1d(self,r,*params, mu = 0):
        """ Function used to evaluate a 1-D multi Gaussian model with any number
        of components

        Paramaters
        ----------
        r: float or array
            Radii at which the profile is to  be evaluated

        params: 1D Array
            List of parameters to define model. The length should be twice
            the number of components in the following pattern:
            [ a_1, sig_1, a_2,sig_2, ....]. Here a_i is the weight of the i'th
            component and sig_i is the width of the i'th component
         mu: float, Optional
            The centre of the gaussian distribution, default is 0.

        Returns
        -------
        prof: 1D array
            The multi-gaussian profile evaluated at 'r'
"""

        num_gauss = int(len(params[0])/2)
        prof = np.zeros(r.shape)
        for i in range(num_gauss):
            a_cur = params[0][2*i]
            sig_cur = params[0][2*i+1]
            prof += a_cur/(np.abs(sig_cur)*np.sqrt(2*np.pi)) *np.exp(-1.*(r-mu)**2/(2*sig_cur**2))
        return prof

    def multi_gauss_1d_ls(self,*params, x_data = np.zeros(10),y_data = np.zeros(10),mu = 0):
        """ Wrapper for multi_gauss_1d function, to be used in fitting the profile
        Paramaters
        ----------
        params: 1D Array
            List of parameters to define model. The length should be twice
            the number of components in the following pattern:
            [ a_1, sig_1, a_2,sig_2, ....]. Here a_i is the weight of the i'th
            component and sig_i is the width of the i'th component
        x_data: 1-D array
            x_data to be fit, generally self.radius
        y_data: 1-D array
            y_data to be fit, genrally self.intens
         mu: float, Optional
            The centre of the gaussian distribution, default is 0.
        Returns
        -------
        log_resid: 1D array
            log scale residuals between the models specified by 'params' and
            the given y_data
"""
        prof = self.multi_gauss_1d(x_data, *params, mu = mu)
        prof[prof<= 1e-5] = 1e-5
        return np.log10(prof) - np.log10(y_data)

    def fit_1D(self,N, init_guess = None,frac_cutoff = 1e-4):
        """ Fit a 1-D Multi Gaussian Model to the psf profile

        Paramaters
        ----------
        N: Int
            Number of gaussian to us in fit
        Init guess: array, optional
            Initial guess at parameters, if None will set Default based on N
        frac_cutoff: float
            Fraction of max, below which to not fit. This is done to focus
            on the center of the PSF and not the edges. Important because
            we using the log-residuals
        Returns
        -------
        a_fit: 1-D array
            Best fit Weights
        sig_fit: 1-D array
            Best fit widths
        Chi2: Float
            The overall chi squared of the fit
"""
        if init_guess == None:
            x0 = np.array([[1./ (2**i),2**i] for i in range(N)]).flatten()
        else:
            x0 = np.copy(init_guess)
        w_fit = self.intens > (np.max(self.intens)*frac_cutoff)

        kwargs_cur = {'x_data':self.radius[w_fit], 'y_data':self.intens[w_fit], 'mu':0.}

        ls_res_cur = least_squares(self.multi_gauss_1d_ls, x0,bounds = (0,150), kwargs = kwargs_cur)
        self.ls_res = ls_res_cur
        a_fit = ls_res_cur.x[::2]
        sig_fit = ls_res_cur.x[1::2]
        return a_fit, sig_fit, np.sum((ls_res_cur.fun)**2)

    def auto_fit(self, N_max = 5, frac_cutoff = 1e-4,norm_a = True, show_fig = True):
        """ Function used for automatic fitting of PSF. First using a 1-D fit to find
    the smallest acceptable number of Gaussians and the corresponding widths,
    then using these widths to fit in 2D and find the weights.
    Paramaters
    ----------
    N: Int
        Number of gaussian to us in fit
    Init guess: array, optional
        Initial guess at parameters, if None will set Default based on N
    frac_cutoff: float
        Fraction of max, below which to not fit. This is done to focus
        on the center of the PSF and not the edges. Important because
        we using the log-residuals

    Returns
    -------
    a_fit: 1-D array
        Best fit Weights
    sig_fit: 1-D array
        Best fit widths
"""
        chi2_min = 1e6
        num_min = 0
        a1D_min = 0
        sig_min = 0

        for num_fit in range(1,N_max+1):
            a_cur,sig_cur,chi2_cur = self.fit_1D(num_fit, frac_cutoff = frac_cutoff)
            if min_diff_array(sig_cur) < 0.5:
                print ( "Skipping %i, two sigma's close together"%num_fit)
                continue
            print (num_fit, chi2_cur )
            if chi2_cur < chi2_min:
                a1D_min = a_cur
                sig_min = sig_cur
                chi2_min = chi2_cur
                num_min = num_fit

        #Moving on to 2D fit
        psf_task = Fitter(self.psf_data,sig_min, None,None, sky_model = False, log_weight_scale=False)
        min_res = psf_task.run_ls_min()
        a2D_min = min_res.x[4:]

        #Add Show fig

        if norm_a:
            return sig_min / self.oversamp, a2D_min / np.sum(a2D_min)
        else:
            return sig_min / self.oversamp, a2D_min / self.oversamp**2
