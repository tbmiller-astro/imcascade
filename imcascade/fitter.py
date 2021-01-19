import numpy as np
import asdf
from scipy.optimize import least_squares
from scipy.stats import norm, truncnorm

from imcascade.mgm import MultiGaussModel
from imcascade.results import ImcascadeResults,vars_to_use
from imcascade.utils import dict_add
import dynesty
from dynesty import utils as dyfunc
import emcee

def Fitter_from_asdf(file_name, init_dict= {}, bounds_dict = {}):
    af = asdf.open(file_name,copy_arrays=True)
    dict = af.tree.copy()

    keys_for_func = ['weight','mask','sky_model','render_mode','log_weight_scale',
    'verbose']
    kwargs = {}
    for key in dict:
        if key in keys_for_func:
            kwargs[key] = dict[key]

    img = np.copy(dict.pop('img'))
    sig = dict.pop('sig')
    psf_sig = dict.pop('psf_sig')
    psf_a = dict.pop('psf_a')
    inst = Fitter(img,sig,psf_sig,psf_a,init_dict = init_dict, bounds_dict = bounds_dict **kwargs)
    return inst

class Fitter(MultiGaussModel):
    """A Class used fit images with MultiGaussModel"""
    def __init__(self, img, sig, psf_sig, psf_a, weight = None, mask = None,\
      sky_model = True,render_mode = 'erf', log_weight_scale = True, verbose = True,
      init_dict = {}, bounds_dict = {}):
        """Initialize a Task instance
        Paramaters
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
            pixels.In practice, the weights of masked pixels is set to '0'.
        sky_model: bool, optional
            If True will incorperate a tilted plane sky model. Reccomended to be set
            to True
        render_mode: 'gauss' or 'erf'
            Option to decide how to render models. Default is 'erf' as it computes
            the integral over the pixel of each profile therefore is more accurate
            but more computationally intensive. 'gauss' assumes the center of a pixel
            provides a reasonble estimate of the average flux in that pixel. 'gauss'
            is faster but far less accurate for objects which vary on O(pixel size),
            so use with caution.
        log_weight_scale: bool, optional
            Wether to treat weights as log scale, Default True
        verbose: bool, optional
            If true will log and print out errors
        init_dict: dict, Optional
            Dictionary specifying initial guesses for least_squares fitting. The code
            is desigined to make 'intelligent' guesses if none are provided
        bounds_dict: dict, Optional
            Dictionary specifying boundss for least_squares fitting and priors. The code
            is desigined to make 'intelligent' guesses if none are provided
"""
        self.img  = img
        self.verbose = verbose
        if weight is None:
            self.weight = np.ones(self.img.shape)
        else:
            self.weight = weight

        if self.weight.shape != self.img.shape:
            raise ValueError("'weight' array must have same shape as 'img'")

        if mask is not None:
            if self.weight.shape != self.img.shape:
                raise ValueError("'mask' array must have same shape as 'img' ")
            self.weight[mask] = 0
            self.mask = mask
        else:
            self.mask = np.zeros(self.img.shape)

        self.mean_weight = np.mean(self.weight[self.weight>0])
        self.sum_weight = np.sum(self.weight[self.weight>0])
        self.avg_noise = np.mean(1./np.sqrt(self.weight[self.weight>0]))

        MultiGaussModel.__init__(self,self.img.shape,sig, psf_sig, psf_a, \
          verbose = verbose, sky_model = sky_model, render_mode = render_mode, \
          log_weight_scale = log_weight_scale)

        #Measuring shape of img and w

        #init_dict['x0'] = self.x_mid
        #init_dict['y0'] = self.y_mid
        #bounds_dict['x0'] = [init_dict['x0'] - 10, init_dict['x0'] + 10]
        #bounds_dict['y0'] = [init_dict['y0'] - 10,init_dict['y0'] + 10]

        init_dict = dict_add(init_dict, 'x0',self.x_mid)
        init_dict = dict_add(init_dict, 'y0',self.y_mid)
        bounds_dict = dict_add(bounds_dict, 'x0',[init_dict['x0'] - 10,init_dict['x0'] + 10])
        bounds_dict = dict_add(bounds_dict, 'y0',[init_dict['y0'] - 10,init_dict['y0'] + 10])

        init_dict = dict_add(init_dict, 'phi', np.pi/2.)
        bounds_dict = dict_add(bounds_dict, 'phi', [0,np.pi])

        init_dict = dict_add(init_dict,'q', 0.5)
        bounds_dict = dict_add(bounds_dict, 'q', [0,1.])

        if sky_model:
            #Try to make educated guesses about sky model
            sky0_guess = np.median(self.img[np.where(self.mask == 0)])
            init_dict = dict_add(init_dict, 'sky0', sky0_guess)
            bounds_dict = dict_add(bounds_dict, 'sky0', [-np.abs(sky0_guess)*5, np.abs(sky0_guess)*5])

            #estimate X and Y slopes using edges
            use_x_edge = np.where(self.mask[:,-1]*self.mask[:,0] == 0)
            sky1_guess = np.median(self.img[:,1][use_x_edge] - img[:,0][use_x_edge])/img.shape[0]

            init_dict = dict_add(init_dict, 'sky1', sky1_guess)
            bounds_dict = dict_add(bounds_dict, 'sky1', [-np.abs(sky1_guess)*5, np.abs(sky1_guess)*5])

            use_y_edge = np.where(self.mask[-1,:]*self.mask[0,:] == 0)
            sky2_guess = np.median(self.img[-1,:][use_y_edge] - img[0,:][use_y_edge])/img.shape[1]

            init_dict = dict_add(init_dict, 'sky2', sky2_guess)
            bounds_dict = dict_add(bounds_dict, 'sky2', [-np.abs(sky2_guess)*5, np.abs(sky2_guess)*5])

            init_sky_model =  self.get_sky_model([init_dict['sky0'],init_dict['sky1'],init_dict['sky2']] )

            A_guess = np.sum( (self.img - init_sky_model )[np.where(self.mask == 0)]  )
        else:
            A_guess = np.sum(img)

        #Below assumes all gaussian have same A
        a_norm = np.ones(self.Ndof_gauss)*A_guess/self.Ndof_gauss

        #If using log scale then adjust initial guesses
        if self.log_weight_scale:
            init_dict = dict_add(init_dict, 'flux', np.log10(A_guess) )
            init_dict = dict_add(init_dict, 'a_unif', np.log10(init_dict['flux']/self.Ndof_gauss) )
            #set minimum possible weight value
            init_dict = dict_add(init_dict, 'a_min', -9)
        else:
            init_dict = dict_add(init_dict, 'flux', A_guess )
            init_dict = dict_add(init_dict, 'a_unif', init_dict['flux']/self.Ndof_gauss )
            init_dict = dict_add(init_dict, 'a_min', 0)

        for i in range(self.Ndof_gauss):
            init_dict = dict_add(init_dict,'a%i'%i, init_dict['a_unif'] )
            bounds_dict = dict_add(bounds_dict,'a%i'%i,  [init_dict['a_min'], init_dict['flux'] ])

        #Now set initial and boundry values once defaults or inputs have been used
        self.lb = [bounds_dict['x0'][0], bounds_dict['y0'][0], bounds_dict['q'][0], bounds_dict['phi'][0]]
        self.ub = [bounds_dict['x0'][1], bounds_dict['y0'][1], bounds_dict['q'][1], bounds_dict['phi'][1]]

        self.param_init = np.ones(self.Ndof)
        self.param_init[0] = init_dict['x0']
        self.param_init[1] = init_dict['y0']
        self.param_init[2] = init_dict['q']

        self.param_init[3] = init_dict['phi']

        for i in range(self.Ndof_gauss):
            self.param_init[4+i] = init_dict['a%i'%i]
            self.lb.append( bounds_dict['a%i'%i][0] )
            self.ub.append(bounds_dict['a%i'%i][1] )

        for i in range(self.Ndof_sky):
            self.param_init[4+self.Ndof_gauss+i] = init_dict['sky%i'%i]
            self.lb.append(bounds_dict['sky%i'%i][0] )
            self.ub.append(bounds_dict['sky%i'%i][1] )

        self.lb = np.asarray(self.lb)
        self.ub = np.asarray(self.ub)
        self.bnds = (self.lb,self.ub)

    def resid_1d(self,params):
        """ Given a set of parameters returns the 1-D flattened residuals
        when compared to the Data, to be used in run_ls_min Function

        Paramaters
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
        return resid.flatten()

    def run_ls_min(self, ls_kwargs = {}):
        """ Function to run a least_squares minimization routine using pre-determined
        inital guesses and bounds. Utilizes the scipy least_squares routine
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

        Paramaters
        ----------
        ls_kwargs: dict, optional
            Optional list of arguments to be passes to least_squares routine
        Returns
        -------
        min_res: scipy.optimize.OptimizeResult
            Returns an Optimize results class containing the optimized parameters
            along with error and status messages
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html)
"""
        min_res = least_squares(self.resid_1d, self.param_init, bounds = self.bnds, **ls_kwargs)
        self.min_res = min_res
        self.min_param = min_res.x
        return min_res

    def set_up_express_run(self, set_params = None):
        """ Function to set up 'express' run using pre-rendered images with a
        fixed x0,y0, phi and q. Sets class attribute 'express_gauss_arr' which
        is needed to run dynesty or emcee in express mode

        Paramaters
        ----------
        set_params: len(4) array-like, optional
            Parameters (x0,y0,q,phi) to use to per-render images. If None will
            call run_ls_min to find parameters.
        Returns
        -------
        express_gauss_arr: array (shape[0],shape[1], Ndof_gauss)
            Returns a 3-D with pre-rendered images based on input parameters
"""
        if not set_params is None:
            x0= set_params[0]
            y0 = set_params[1]
            q_in = set_params[2]
            phi = set_params[3]
        else:
            if not hasattr(self, 'min_res'):
                if self.verbose: print ('Running least squares minimization')
                self.run_ls_min()

            x0= self.min_res.x[0]
            y0 = self.min_res.x[1]
            q_in = self.min_res.x[2]
            phi = self.min_res.x[3]

        if self.verbose:
            print ('Parameters to be set:')
            print ('\t Galaxy Center: %.2f,%.2f'%(x0,y0) )
            print ('\t Axis Ratio: %.5f'%q_in)
            print ('\t PA: %.5f'%phi)

        self.exp_set_params = np.array([x0,y0,q_in,phi])

        mod_final = []
        for i,var_cur in enumerate( self.var ):
            final_var = self.psf_var + var_cur
            final_q = np.sqrt( (var_cur*q_in*q_in + self.psf_var ) / (final_var) )
            final_a = np.copy(self.psf_a)
            mod_cur = self.get_erf_stack(x0, y0, phi,final_q, final_a, final_var)
            mod_final.append(mod_cur)
        gauss_arr = np.moveaxis( np.asarray(mod_final),0,-1)
        self.express_gauss_arr = np.copy(gauss_arr)
        means = np.zeros(self.Ndof_gauss)

        if hasattr(self,'min_res'):
            J = self.min_res.jac
            cov = np.linalg.inv(J.T.dot(J))
            s_sq = self.chi_sq(self.min_res.x) / (np.shape(self.img)[0] *np.shape(self.img)[1] - self.Ndof)*self.avg_noise**2
            err = np.sqrt(np.diagonal(cov)*s_sq)
            self.min_err = err
        return gauss_arr

    def chi_sq(self,params):
        """Function to calculate chi_sq for a given set of paramters

        Paramaters
        ----------
        params: Array
            List of parameters to define model
        Returns
        -------
        chi^2: float
            Chi squared statistic for the given set of parameters
"""
        model = self.make_model(params)
        return np.sum( (self.img - model)**2 *self.weight)

    def log_like(self,params):
        """Function to calculate the log likeliehood for a given set of paramters

        Paramaters
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

        Paramaters
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

        Paramaters
        ----------
        exo_params: Array
            List of parameters to define model. Length is Ndof_gauss + Ndof_sky
            since the structural parameters (x0,y0,q, PA) are set

        Returns
        -------
        log likeliehood: float
            log likeliehood for a given set of paramters, defined as -0.5*chi^2
"""

        if self.sky_model:
            final_a = exp_params[:-self.Ndof_sky]
        else:
            final_a = np.copy(exp_params)
        if self.log_weight_scale: final_a = 10**final_a
        model = np.sum(final_a*self.express_gauss_arr, axis = -1)

        if self.sky_model:
            model += self.get_sky_model(exp_params[-self.Ndof_sky:])

        return -0.5*np.sum( (self.img - model)**2 *self.weight)

    def ptform_exp(self, u):
        """Prior transformation function to be used in dynesty 'express' mode.
        By default they are all unifrom priors defined by self.lb and self.ub

        Paramaters
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

    def log_prior_express(self,params):
        """Prior function to be used in emcee 'express' mode.
        By default they are all unifrom priors defined by self.lb and self.ub

        Paramaters
        ----------
        u: array
            array of random numbers from 0 to 1

        Returns
        -------
        log prior: float
            Value of prior for given set of params
"""
        inside_prior = (params > self.lb[4:])*(params < self.ub[4:])
        if inside_prior.all():
            return 0
        else:
            return -np.inf

    def log_prob_express(self,params):
        """Probability function to be used in emcee 'express' mode.
        By default they are all unifrom priors defined by self.lb and self.ub

        Paramaters
        ----------
        u: array
            array of random numbers from 0 to 1

        Returns
        -------
        log prob: float
            Value of log Probability for given set of params
"""
        logp = self.log_prior_express(params)
        return self.log_like_exp(params) + logp


    def run_dynesty(self,method = 'full', sampler_kwargs = {}, run_nested_kwargs = {}):
        """Function to run dynesty to sample the posterier distribution using either the
        'full' methods which explores all paramters, or the 'express' method which sets
        the structural parameters.

        Paramaters
        ----------
        method: str: 'full' or 'express'
            Which method to use to run dynesty
        sampler_kwargs: dict
            set of keyword arguments to pass the the dynesty DynamicNestedSampler call, see:
            https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynesty.DynamicNestedSampler
        run_nested_kwargs: dict
            set of keyword arguments to pass the the dynesty run_nested call, see:
            https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynamicsampler.DynamicSampler.run_nested

        Returns
        -------
        Posterier: Array
            Posterier distribution derrived. If method is 'express', the first 4 columns,
            containg x0,y0,PA and q, are all the same and equal to values used to pre-rended the images
"""
        if method == 'full':
            ndim = self.Ndof
            sampler = dynesty.DynamicNestedSampler( self.log_like, self.ptform, ndim= ndim, **sampler_kwargs)
        if method == 'express':
            ndim = self.Ndof_gauss + self.Ndof_sky
            if not hasattr(self, 'express_gauss_arr'):
                if self.verbose: print ('Setting up Express params')
                _ = self.set_up_express_run()
            sampler = dynesty.DynamicNestedSampler( self.log_like_exp, self.ptform_exp, ndim= ndim, **sampler_kwargs)
        sampler.run_nested(**run_nested_kwargs)

        self.dynesty_sampler = sampler
        res_cur = sampler.results
        dyn_samples, dyn_weights = res_cur.samples, np.exp(res_cur.logwt - res_cur.logz[-1])
        post_samp = dyfunc.resample_equal(dyn_samples, dyn_weights)

        if method == 'full':
            self.posterier = np.copy(post_samp)
        else:
            set_arr = self.exp_set_params[:,np.newaxis] *np.ones((4,post_samp.shape[0] ) )
            set_arr = np.transpose(set_arr)
            self.posterier = np.hstack([set_arr, post_samp])
        self.post_method = 'dynesty-'+method
        return self.posterier


    def run_emcee(self,method = 'express', nwalkers = 32, max_it = int(1e6), check_freq = int(500), print_progress = False):
        "Not sure if it will stay"
        if method == 'full':
            print(" 'full' not yet implemented")
            return 0
        elif method == 'express':
            if not hasattr(self, 'express_gauss_arr') or not hasattr(self, 'min_err') or not hasattr(self, 'min_res'):
                if self.verbose: print ('Setting up express run')
                self.set_up_express_run()

            ndim = self.Ndof_gauss + self.Ndof_sky
            init_arr = self.min_param[4:]

            #pos = init_arr + 0.05 * np.random.randn(nwalkers, ndim)
            pos = (self.ub[4:] - self.lb[4:])*np.random.uniform(size = (nwalkers,ndim) )  + self.lb[4:]

            #Check to see if initial positions are valid
            for ex in pos:
                if np.isinf(self.log_prior_express(ex) ):
                    w_lo = ex < self.lb[4:]
                    ex[w_lo] = self.lb[4:][w_lo] + 1e-5
                    w_hi = ex > self.ub[4:]
                    ex[w_hi] = self.ub[4:][w_hi] - 1e-5
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob_express)
        else:
            print("Must choose either 'full' or 'express' method")
            return 0

        old_tau = 0
        for sample in sampler.sample(pos, iterations=max_it):
            #only check every check_freq number of steps
            if sampler.iteration % check_freq: continue

            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)


            if print_progress:
                print (r"Iter: %.2e - 50xtau: %.2e - d tau: %.3f"%(sampler.iteration, np.max(tau*50), np.max(np.abs(old_tau - tau) / tau) ) )

            if converged:
                break
            old_tau = tau

        self.emcee_sampler = sampler
        self.emcee_tau = tau
        post_samp = sampler.get_chain(discard=int(3 * np.max(tau)), thin = int(0.3 * np.min(tau)),  flat=True)

        if method == 'express':
            set_arr = self.exp_set_params[:,np.newaxis] *np.ones((4,post_samp.shape[0] ) )
            set_arr = np.transpose(set_arr)
            self.posterier = np.hstack([set_arr, post_samp])
        else:
            self.posterier = np.copy(post_samp)

        self.post_method = 'emcee-'+method
        return sampler

    def save_results(self,file_name, run_basic_analysis = True, thin_posterier = 1, zpt = None, cutoff = None, errp_lo = 16, errp_hi =84):
        """Function to save results after run_ls_min, run_dynesty and/or run_emcee is performed. Will be saved as an ASDF file.

        Paramaters
        ----------
        file_name: str
            Str defining location of where to save data
        run_basic_analysis: Bool (default true)
            If true will run ImcascadeResults.run_basic_analysis

        Returns
        -------
        Posterier: Array
            Posterier distribution derrived. If method is 'express', the first 4 columns,
            containg x0,y0,PA and q, are all the same and equal to values used to pre-rended the images
"""
        if run_basic_analysis:
            res = ImcascadeResults(self, thin_posterier = thin_posterier)
            res.run_basic_analysis(zpt = zpt, cutoff = cutoff, errp_lo = errp_lo, errp_hi =errp_hi,\
              save_results = True , save_file = file_name)
            return res
        else:
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
