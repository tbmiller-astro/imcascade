import numpy as np
from astropy.io import fits
import sep
from scipy.optimize import least_squares
from imcascade.fitter import Fitter
from imcascade.utils import min_diff_array

class PSFFitter():
    """A Class used to fit Gaussian models to a PSF image

Parameters
----------
psf_img : str or 2D array
    PSF data to be fit. If a string is given will assume it is a fits file
    and load the Data in the first HDU. If it is an array then will use
    as the PSF image. Either way it is assumed the PSF is centered and image
    is square.
oversamp : Float, optional
    Factor by which the PSF image is oversampled. Default is 1.

Attributes
----------
psf_data: 2D array
    pixelized PSF data to be fit
intens: 1D array
    1D sbp of PSF
radii: 1D array
    Radii corresponding to ``intens``

"""
    def __init__(self,psf_img,oversamp = 1.):
        """ Initialize a PSFFitter instance
"""

        if type(psf_img) == str:
            fits_file = fits.open(psf_img)
            self.psf_data = np.array(fits_file[0].data, dtype = '<f4')
        elif type(psf_img) == np.ndarray:
            self.psf_data = np.array(psf_img, dtype = '<f4')
        self.oversamp = oversamp

        self.cent_pix_x = np.where(self.psf_data == np.max(self.psf_data) )[0][0]
        self.cent_pix_y = np.where(self.psf_data == np.max(self.psf_data) )[1][0]

        #calculate 1-D circular profile
        _,_ = self.calc_profile()

    def calc_profile(self):
        """Calculates the 1-D PSF profile in 1 pixel steps assuming it is circular

        Returns
        -------
        intens: 1D array
            Intensity profile.
        radius: 1D array
            radii at which the intensity measuremnts are made.
"""

        maxr = int(np.min([self.cent_pix_x,self.cent_pix_y]) )
        r_in = np.arange(0, maxr - 1)
        r_out = np.arange(1,maxr)
        area = np.pi*(r_out**2 - r_in**2)
        prof_sum,_,_ = sep.sum_circann(self.psf_data, [self.cent_pix_x+0.5]*len(r_in),[self.cent_pix_y+0.5]*len(r_in), r_in,r_out)
        self.intens = prof_sum/area
        self.radius = (r_in + r_out)/2.
        return self.intens, self.radius

    def multi_gauss_1d(self,r,*params, mu = 0):
        """ Function used to evaluate a 1-D multi Gaussian model with any number
        of components

        Parameters
        ----------
        params: 1D array
            List of parameters to define model. The length should be twice the
            number of components in the following pattern:[ a_1, sig_1, a_2,sig_2, ....]
            where a_i is the weight of the i'th  component and sig_i is the width of the i'th component.
        r: float, array
            Radii at which the profile is to  be evaluated
        mu: float, optional
            The centre of the gaussian distribution, default is 0

        Returns
        -------
        1D array
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

        Parameters
        ----------
        params: 1D Array
            List of parameters to define model. The length should be twice
            the number of components in the following pattern:
            [ a_1, sig_1, a_2,sig_2, ....]. Here a_i is the weight of the i'th
            component and sig_i is the width of the i'th component.
        x_data : 1-D array
            x_data to be fit, generally self.radius.
        y_data: 1-D array
            y_data to be fit, genrally self.intens
        mu: float, optional
            The centre of the gaussian distribution, default is 0.

        Returns
        -------
        resid: 1D array
            log scale residuals between the models specified by 'params' and
            the given y_data
"""
        prof = self.multi_gauss_1d(x_data, *params, mu = mu)
        prof[prof<= 1e-5] = 1e-5
        return np.log10(prof) - np.log10(y_data)

    def fit_N(self, N,frac_cutoff = 1e-4, plot = False):
        """ 'Fully' Fit a Multi Gaussian Model with a given number of gaussians
        to the psf profile. Start with 1D to find the best fit widths and then
        us evaluate chi2 in 2D

        Parameters
        ----------
        N: Int
            Number of gaussian to us in fit
        frac_cutoff: float, optional
            Fraction of max, below which to not fit. This is done to focus
            on the center of the PSF and not the edges. Important because
            we using the log-residuals
        plot: bool
            Whether or not to show summary plot

        Returns
        -------
        a_fit: 1-D array
            Best fit Weights, corrected for oversamping
        sig_fit: 1-D array
            Best fit widths, corrected for oversamping
        Chi2: Float
            The overall chi squared of the fit, computed using the best fit 2D model
"""
        a_1d,sig_1d, chi2_1d = self.fit_1D(N,frac_cutoff = frac_cutoff)
        a2D_cur = a_1d*sig_1d*np.sqrt(2*np.pi)
        tow = np.copy(self.psf_data)
        tow[np.where(tow< 0)] = 0
        eps = 1e-4
        w = 1/ (tow + eps)
        w[np.where(np.isinf(w))] = 0
        w[np.where(np.isnan(w))] = 0

        fitter_cur = Fitter(self.psf_data, sig_1d, None,None, weight = w, sky_model = False, log_weight_scale=False,
             verbose = False, render_mode = 'erf', init_dict = {'x0':self.cent_pix_x, 'y0': self.cent_pix_y, 'q':1, 'phi':0}, bounds_dict={'q':[0.99,1.01], 'phi':[-1e-4,1e-4]})
        #min_res = fitter_cur.run_ls_min()
        #a2D_cur = min_res.x[4:]
        param = np.copy(fitter_cur.param_init)
        param[4:] = a2D_cur
        chi2_cur = fitter_cur.chi_sq(param)
        if plot:
            import matplotlib.pyplot as plt
            fig, (ax1,ax2,cax) = plt.subplots(1,3, figsize = (9,4), gridspec_kw={'width_ratios':[1.,1.,0.05,]})

            ax1.plot(self.radius, np.log10(self.intens), 'C0-', lw = 2, label = '1D profile')

            min_i = np.min(self.intens[self.intens > 0])
            max_i = np.max(self.intens)
            rplot = np.linspace(self.radius[0],self.radius[-1], num  = 200)
            full_p = []
            for i in range(N):
                ax1.plot(rplot, np.log10(self.multi_gauss_1d(rplot, [a_1d[i],sig_1d[i]])), 'k--')
                full_p.append(a_1d[i])
                full_p.append(sig_1d[i])
            ax1.plot(rplot, np.log10( self.multi_gauss_1d(rplot, full_p) ), 'k-', label = 'Best fit model')
            ax1.set_ylim([np.log10(0.8*min_i), np.log10(2*max_i)])
            ax1.set_ylabel('log ( Intensity)')
            ax1.set_xlabel('Radius (pix)')
            ax1.set_aspect(1./ax1.get_data_ratio())
            ax1.legend(fontsize = 12, frameon = False)

            mod = fitter_cur.make_model(param)
            resid = (self.psf_data - mod)/mod

            im2 = ax2.imshow(resid, vmin = -0.5, vmax = 0.5, cmap = 'RdGy')
            ax2.axis('off')

            ax2.set_title('Residual: (data-model)/model')

            fig.colorbar(im2, cax=cax, orientation='vertical',fraction=0.046, pad=0.04)
            fig.subplots_adjust(wspace = 0.01)

            return sig_1d/self.oversamp, a2D_cur/self.oversamp**2, chi2_cur,fig

        else:
            return sig_1d/self.oversamp, a2D_cur/self.oversamp**2, chi2_cur

    def fit_1D(self,N, init_guess = None,frac_cutoff = 1e-4):
        """ Fit a 1-D Multi Gaussian Model to the psf profile

        Parameters
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
            Best fit Weights, Not corrected for oversampling
        sig_fit: 1-D array
            Best fit widths, Not corrected for oversampling
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

    def auto_fit(self, N_max = 5, frac_cutoff = 1e-4,norm_a = True):
        """ Function used for automatic fitting of PSF. First using a 1-D fit to find
    the smallest acceptable number of Gaussians and the corresponding widths,
    then using these widths to fit in 2D and find the weights.

    Parameters
    ----------
    N: Int
        Number of gaussian to us in fit
    frac_cutoff: float
        Fraction of max, below which to not fit. This is done to focus
        on the center of the PSF and not the edges. Important because
        we using the log-residuals
    norm_a: Bool
        Wheter or not to normize the resulting weight so that the sum is unity

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
        a2D_min = 0
        for num_fit in range(1,N_max+1):
            sig_cur,a_cur,chi2_cur = self.fit_N(num_fit, frac_cutoff = frac_cutoff)
            if min_diff_array(sig_cur) < 0.5:
                #print ( "Skipping %i, two sigma's close together"%num_fit)
                continue

            #print (num_fit, '%.3e'%chi2_cur )
            if chi2_cur < chi2_min:
                a_min = a_cur
                sig_min = sig_cur
                chi2_min = chi2_cur
                num_min = num_fit
        #Add Show fig

        if norm_a:
            return sig_min / self.oversamp, a_min / np.sum(a_min)
        else:
            return sig_min / self.oversamp, a_min / self.oversamp**2
