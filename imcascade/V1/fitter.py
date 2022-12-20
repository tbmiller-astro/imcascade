
import numpy as np
import asdf
import logging
from scipy.optimize import least_squares, minimize, Bounds
from scipy.stats import norm, truncnorm

import dynesty
from dynesty import utils as dyfunc
from .mgm import MultiGaussModel
from imcascade.V1.results import ImcascadeResults
from imcascade.utils import parse_input_dicts,reg_resid

class Fitter(MultiGaussModel):
    """A Class used fit images with MultiGaussModel

    This is the main class used to fit ``imcascade`` models

    Parameters
    ----------
    img: 2D Array
        Data to be fit, it is assumed to be a cutout with the object of interest
        in the center of the image
    sig: 1D Array
        Widths of Gaussians to be used in MultiGaussModel
    psf_sig: 1D array, None
        Width of Gaussians used to approximate psf
    psf_a: 1D array, None
        Weights of Gaussians used to approximate psf
        If both psf_sig and psf_a are None then will run in Non-psf mode
    weight: 2D Array, optional
        Array of pixel by pixel weights to be used in fitting. Must be same
        shape as 'img' If None, all the weights will be set to 1.
    mask: 2D Array, optional
        Array with the same shape as 'img' denoting which, if any, pixels
        to mask  during fitting process. Values of '1' or 'True' values for
        the pixels to be masked. If set to 'None' then will not mask any
        pixels. In practice, the weights of masked pixels is set to '0'.
    sky_model: bool, optional
        If True will incorperate a tilted plane sky model. Reccomended to be set
        to True
    sky_type: str, 'tilted-plane' or 'flat'
        Function used to model sky. Default is tilted plane with 3 parameters, const bkg
        and slopes in each directin. 'flat' uses constant background model with 1 parameter.
    render_mode: 'hybrid', 'erf' or 'gauss'
        Option to decide how to render models. 'erf' analytically computes
        the integral over the pixel of each profile therefore is more accurate
        but more computationally intensive. 'gauss' assumes the center of a pixel
        provides a reasonble estimate of the average flux in that pixel. 'gauss'
        is faster but far less accurate for objects which vary on O(pixel size),
        so use with caution. 'hybrid' is the defualt, uses 'erf' for components with width < 5
        to ensure accuracy and uses 'gauss' otherwise as it is accurate enough and faster. Also
        assumes all flux > 5 sigma for components is 0.
    log_weight_scale: bool, optional
        Wether to treat weights as log scale, Default True
    verbose: bool, optional
        If true will log and print out errors
    psf_shape: dict, Optional
        Dictionary containg at 'q' and 'phi' that define the shape of the PSF.
        Note that this slows down model rendering significantly so only
        reccomended if neccesary.
    init_dict: dict, Optional
        Dictionary specifying initial guesses for least_squares fitting. The code
        is desigined to make 'intelligent' guesses if none are provided
    bounds_dict: dict, Optional
        Dictionary specifying boundss for least_squares fitting and priors. The code
        is desigined to make 'intelligent' guesses if none are provided
"""
    def __init__(self, img, sig, psf_sig, psf_a, weight = None, mask = None,\
      sky_model = True,sky_type = 'tilted-plane',reg_k = 0, render_mode = 'hybrid', log_weight_scale = True, verbose = True,
      psf_shape = None,init_dict = {}, bounds_dict = {}, log_file = None):
        """Initialize a Task instance"""
        self.img  = img
        self.verbose = verbose

        if log_file is None:
            handler = logging.StreamHandler()
        else:
            handler = logging.FileHandler(log_file)

        logging.basicConfig(format = "%(asctime)s - %(message)s", level = logging.INFO,
            handlers = [handler,])
        self.logger = logging.getLogger()

        if not sky_type in ['tilted-plane', 'flat']:
            if verbose: self.logger.info("Incompatible sky_type, must choose 'tilted-plane' or 'flat'! Setting to 'tilted-plane'")
            sky_type = 'tilted-plane'

        if not render_mode in ['gauss', 'erf','hybrid']:
            if verbose: self.logger.info("Incompatible render mode, must choose 'gauss','erf' or 'hybrid'! Setting to 'hybrid'")
            render_mode = 'hybrid'

        if psf_sig is None or psf_a is None:
            if verbose: self.logger.info('No PSF input, running in non-psf mode')

        if weight is None:
            self.weight = np.ones(img.shape)
        else:
            if weight.shape != self.img.shape:
                raise ValueError("'weight' array must have same shape as 'img'")
            self.weight = weight


        if mask is not None:
            if self.weight.shape != self.img.shape:
                raise ValueError("'mask' array must have same shape as 'img' ")
            self.weight[np.where(mask == 1)] = 0
            self.mask = mask
        else:
            self.mask = np.zeros(self.img.shape)


        if np.sum(np.isnan(self.img) + np.sum(np.isnan(self.weight))) > 0:
            where_inf = np.where(np.isnan(self.img) + np.isnan(self.weight))
            self.logger.info("Masking nan values at locations:")
            self.logger.info(where_inf)
            self.img[where_inf] = 0
            self.weight[where_inf] = 0
            self.mask[where_inf] = 1

        self.log_weight = np.zeros(self.weight.shape)
        self.log_weight[self.weight > 0 ] = np.log(self.weight[self.weight > 0])
        self.loglike_const = 0.5*(np.sum(self.log_weight) - np.log10(2*np.pi)*(np.sum(self.mask == 0)) )
        self.reg_k = reg_k

        MultiGaussModel.__init__(self,self.img.shape,sig, psf_sig, psf_a, \
          verbose = verbose, sky_model = sky_model,sky_type = sky_type, render_mode = render_mode, log_weight_scale = log_weight_scale, psf_shape = psf_shape)

        self.npix = self.img.shape[0]*self.img.shape[1] - np.sum(self.mask)

        bounds_dict, init_dict, lb,ub, param_init = parse_input_dicts(bounds_dict,init_dict, self)
        
        self.param_init = param_init
        self.lb = lb
        self.ub = ub
        self.bnds = Bounds(self.lb,self.ub)

    def resid_1d(self,params):
        """ Given a set of parameters returns the 1-D flattened residuals
        when compared to the Data, to be used in run_ls_min Function

        Parameters
        ----------
        params: Array
            List of parameters to define model

        Returns
        -------
        resid_flatten: array
            1-D array of the flattened residuals
        """
        model = self.make_model(params)
        resid = (self.img - model)*np.sqrt(self.weight)
        A_cur = params[4:4+self.Ndof_gauss]
        reg_resid_cur = np.abs(reg_resid(A_cur[:int(self.Ndof_gauss/2)+1]))

        return np.append(resid.flatten()/(self.npix - self.Ndof),reg_resid_cur*self.reg_k)

    def run_ls_min(self, ls_kwargs = {}):
        """ Function to run a least_squares minimization routine using pre-determined
        inital guesses and bounds.

        Utilizes the scipy least_squares routine (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

        Parameters
        ----------
        ls_kwargs: dict, optional
            Optional list of arguments to be passes to least_squares routine

        Returns
        -------
        min_param: 1D array
            Returns a 1D array containing the optimized parameters that describe
            the best fit model.
"""
        if self.verbose: self.logger.info('Running least squares minimization')
        min_res = least_squares(self.resid_1d, self.param_init, bounds = [self.bnds.lb,self.bnds.ub], **ls_kwargs)
        self.min_res = min_res
        self.min_param = min_res.x
        if self.verbose: self.logger.info('Finished least squares minimization')

        return self.min_param

    def set_up_express_run(self, set_params = None):
        """ Function to set up 'express' run using pre-rendered images with a
        fixed x0,y0, phi and q. Sets class attribute 'express_gauss_arr' which
        is needed to run dynesty or emcee in express mode

        Parameters
        ----------
        set_params: len(4) array-like, optional
            Parameters (x0,y0,q,phi) to use to per-render images. If None will
            call ``run_ls_min()`` or use stored ``min_res`` to find parameters.

        Returns
        -------
        express_gauss_arr: array (shape[0],shape[1], Ndof_gauss)
            Returns a 3-D with pre-rendered images based on input parameters
"""
        if self.log_weight_scale:
            p_to_use = np.zeros(self.Ndof)
        else:
            p_to_use = np.ones(self.Ndof)

        p_to_use[-self.Ndof_sky:] = 0

        if not set_params is None:
            p_to_use[0] = set_params[0]
            p_to_use[1] = set_params[1]
            p_to_use[2] = set_params[2]
            p_to_use[3] = set_params[3]

        else:
            if not hasattr(self, 'min_res'):
                self.run_ls_min()

            p_to_use[0]= self.min_res.x[0]
            p_to_use[1] = self.min_res.x[1]
            p_to_use[2] = self.min_res.x[2]
            p_to_use[3] = self.min_res.x[3]

        self.exp_set_params = p_to_use[:4]

        if self.verbose:
            self.logger.info('Parameters to be set for pre-rendered images:')
            self.logger.info('\t Galaxy Center: %.2f,%.2f'%(p_to_use[0],p_to_use[1]) )
            self.logger.info('\t Axis Ratio: %.5f'%p_to_use[2])
            self.logger.info('\t PA: %.5f'%p_to_use[3])


        #### Render images best on best fit structural paramters

        stack_raw = self.make_model(p_to_use, return_stack = True)

        if self.has_psf:
            #Sum over the components of the PSF if neccesary
            stack =  np.moveaxis(np.array([stack_raw[:,:,i::self.Ndof_gauss].sum(axis = -1) for i in range(self.Ndof_gauss)] ),0,-1)
        else:
            stack = stack_raw.copy()

        self.express_gauss_arr = np.copy(stack)
        means = np.zeros(self.Ndof_gauss)

        if hasattr(self,'min_res'):
            self._exp_pri_locs = self.min_res.x[4:].copy()

            jac_cur = self.min_res.jac
            cov = np.linalg.inv(jac_cur.T.dot(jac_cur))
            sig = np.sqrt(np.diag(cov))[4:]


            #Set width to reasonable value if really big
            sig = sig*2.
            sig[sig>1] = 1.
            self._exp_pri_scales = sig.copy()
        return stack

    def make_express_model(self, exp_params):
        """Function to generate a model for a given set of paramters,
        specifically using the pre-renedered model for the 'express' mode

        Parameters
        ----------
        exo_params: Array
            List of parameters to define model. Length is Ndof_gauss + Ndof_sky
            since the structural parameters (x0,y0,q, PA) are set

        Returns
        -------
        model: 2D-array
            Model image based on input parameters
"""
        if self.sky_model:
            final_a = exp_params[:-self.Ndof_sky]
        else:
            final_a = np.copy(exp_params)

        if self.log_weight_scale: final_a = 10**final_a

        model = np.sum(final_a*self.express_gauss_arr, axis = -1)

        if self.sky_model:
            model += self.get_sky_model(exp_params[-self.Ndof_sky:])

        return model

    def chi_sq(self,params):
        """Function to calculate chi_sq for a given set of paramters

        Parameters
        ----------
        params: Array
            List of parameters to define model
        Returns
        -------
        chi^2: float
            Chi squared statistic for the given set of parameters
"""
        model = self.make_model(params)
        return np.sum( (self.img - model)**2 *self.weight - self.log_weight + np.log(2*np.pi) )

    def log_like(self,params):
        """Function to calculate the log likeliehood for a given set of paramters

        Parameters
        ----------
        params: Array
            List of parameters to define model

        Returns
        -------
        log likeliehood: float
            log likeliehood for a given set of paramters, defined as -0.5*chi^2
"""
        return -0.5*self.chi_sq(params)

    def ptform(self, u):
        """Prior transformation function to be used in dynesty 'full' mode

        Parameters
        ----------
        u: array
            array of random numbers from 0 to 1

        Returns
        -------
        x: array
            array containing distribution of parameters from prior
"""
        x = np.zeros(len(u))
        x[0] = norm.ppf(u[0], loc = self.param_init[0], scale = 1)
        x[1] = norm.ppf(u[1], loc = self.param_init[1], scale = 1)

        #For q
        m, s = 1, 0.25 # mean and standard deviation
        low, high = 0, 1. # lower and upper bounds
        low_n, high_n = (low - m) / s, (high - m) / s  # standardize
        x[2] = truncnorm.ppf(u[2], low_n, high_n, loc=m, scale=s)

        #Uniform between 0 and pi for phi
        x[3] = u[3]*np.pi

        #Uniform for gaussian weights and sky
        x[4:] = u[4:]*(self.ub[4:] - self.lb[4:]) + self.lb[4:]

        return x

    def log_like_exp(self,exp_params):
        """Function to calculate the log likeliehood for a given set of paramters,
        specifically using the pre-renedered model for the 'express' mode

        Parameters
        ----------
        exo_params: Array
            List of parameters to define model. Length is Ndof_gauss + Ndof_sky
            since the structural parameters (x0,y0,q, PA) are set

        Returns
        -------
        log likeliehood: float
            log likeliehood for a given set of paramters, defined as -0.5*chi^2
"""
        model = self.make_express_model(exp_params)
        return -0.5*np.sum( (self.img - model)**2 *self.weight ) + self.loglike_const

    def ptform_exp_ls(self, u):
        """Prior transformation function to be used in dynesty 'express' mode using
        gaussian priors defined by the results of the least_squares minimization

        Parameters
        ----------
        u: array
            array of random numbers from 0 to 1

        Returns
        -------
        x: array
            array containing distribution of parameters from prior
"""
        return norm.ppf(u, loc = self._exp_pri_locs, scale = self._exp_pri_scales)

    def ptform_exp_unif(self, u):
        """Prior transformation function to be used in dynesty 'express' mode using
        unifrom priors defined by self.lb and self.ub

        Parameters
        ----------
        u: array
            array of random numbers from 0 to 1

        Returns
        -------
        x: array
            array containing distribution of parameters from prior
"""
        x = np.zeros(len(u))
        x = u*(self.ub[4:] - self.lb[4:]) + self.lb[4:]
        return x


    def run_dynesty(self,method = 'full', sampler_kwargs = {}, run_nested_kwargs = {}, prior = 'min_results'):
        """Function to run dynesty to sample the posterior distribution using either the
        'full' methods which explores all paramters, or the 'express' method which sets
        the structural parameters.

        Parameters
        ----------
        method: str: 'full' or 'express'
            Which method to use to run dynesty
        sampler_kwargs: dict
            set of keyword arguments to pass the the dynesty DynamicNestedSampler call, see:
            https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynesty.DynamicNestedSampler
        run_nested_kwargs: dict
            set of keyword arguments to pass the the dynesty run_nested call, see:
            https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynamicsampler.DynamicSampler.run_nested
        prior: 'min_results' or 'uniform'
            Which of the two choices of priors to use. The `min_results` priors are Gaussian,
            with centers defined by the best fit paramters and variance equal to 4 times
            the variance estimated using the Hessian matrix from the run_ls_min() run.
            `uniform` is what it sounds like, uniform priors based on the the lower and upper bounds
            Defualt is `min_results`

        Returns
        -------
        Posterior: Array
            posterior distribution derrived. If method is 'express', the first 4 columns,
            containg x0, y0, PA and q, are all the same and equal to values used to pre-render the images
"""
        if self.verbose: self.logger.info('Running dynesty using the %s method'%method)
        if method == 'full':
            ndim = self.Ndof
            sampler = dynesty.DynamicNestedSampler( self.log_like, self.ptform, ndim= ndim, **sampler_kwargs)
        if method == 'express':
            ndim = self.Ndof_gauss + self.Ndof_sky
            if not hasattr(self, 'express_gauss_arr'):
                if self.verbose: self.logger.info('Setting up pre-rendered images')
                _ = self.set_up_express_run()
            if prior == 'min_results':
                sampler = dynesty.DynamicNestedSampler( self.log_like_exp, self.ptform_exp_ls, ndim= ndim, **sampler_kwargs)
            elif prior == 'uniform':
                sampler = dynesty.DynamicNestedSampler( self.log_like_exp, self.ptform_exp_unif, ndim= ndim, **sampler_kwargs)
            else:
                raise ("Chosen prior must be either 'min_results' or 'uniform' ")
        sampler.run_nested(**run_nested_kwargs)

        if self.verbose: self.logger.info('Finished running dynesty, calculating posterior')
        self.dynesty_sampler = sampler
        res_cur = sampler.results
        self.logz = res_cur.logz[-1]
        self.logz_err = res_cur.logzerr[-1]
        dyn_samples, dyn_weights = res_cur.samples, np.exp(res_cur.logwt - res_cur.logz[-1])
        post_samp = dyfunc.resample_equal(dyn_samples, dyn_weights)

        if method == 'full':
            self.posterior = np.copy(post_samp)
        else:
            set_arr = self.exp_set_params[:,np.newaxis] *np.ones((4,post_samp.shape[0] ) )
            set_arr = np.transpose(set_arr)
            self.posterior = np.hstack([set_arr, post_samp])
        self.post_method = 'dynesty-'+method
        return self.posterior


    def save_results(self,file_name, run_basic_analysis = True, thin_posterior = 1, zpt = None, cutoff = None, errp_lo = 16, errp_hi =84):
        """Function to save results after run_ls_min, run_dynesty and/or run_emcee is performed. Will be saved as an ASDF file.

        Parameters
        ----------
        file_name: str
            Str defining location of where to save data
        run_basic_analysis: Bool (default True)
            If true will run ImcascadeResults.run_basic_analysis

        Returns
        -------
        Posterior: Array
            posterior distribution derrived. If method is 'express', the first 4 columns,
            containg x0,y0,PA and q, are all the same and equal to values used to pre-rended the images
"""
        if run_basic_analysis:
            if self.verbose: self.logger.info('Saving results to: %s'%file_name)
            if self.verbose: self.logger.info('Running basic morphological analysis')

            res = ImcascadeResults(self, thin_posterior = thin_posterior)
            res_dict = res.run_basic_analysis(zpt = zpt, cutoff = cutoff, errp_lo = errp_lo, errp_hi =errp_hi,\
              save_results = True , save_file = file_name)
            if self.verbose:
                for key in res_dict:
                    self.logger.info('%s = '%(key) + str(res_dict[key]))
            return res
        else:
            if self.verbose: self.logger.info('Saving results to: %s'%file_name)
            #If analysis is not to be run then simply save the important contents of the class
            dict_to_save = {}
            for key in vars_to_use:
                try:
                    dict_to_save[key] = vars(self)[key]
                except:
                    continue

            file = asdf.AsdfFile(dict_to_save)
            file.write_to(file_name)

            return dict_to_save