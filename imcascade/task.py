import numpy as np
from scipy.optimize import least_squares
import sep
from figpy.mgm import MultiGaussModel

import dynesty
from scipy.stats import norm, truncnorm
from dynesty import utils as dyfunc
import emcee

class Task(MultiGaussModel):
    """A Class used fit images with MultiGaussModel"""
    def __init__(self, img, sig, psf_sig, psf_a, weight = None, mask = None,\
      sky_model = True,render_mode = 'erf', log_weight_scale = True, verbose = True):
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

        bounds_dict = {}
        init_dict = {}
        #Measuring shape of img and w
        x_mid = img.shape[0]/2.
        y_mid = img.shape[0]/2.
        bounds_dict['x0'] = [x_mid - 5, x_mid + 5]
        bounds_dict['y0'] = [y_mid - 5,y_mid + 5]
        init_dict['x0'] = x_mid
        init_dict['y0'] = y_mid

        bounds_dict['phi'] = [0,np.pi]
        init_dict['phi'] = np.pi/2.
        bounds_dict['q'] = [0,1]
        init_dict['q'] = 0.5


        if sky_model:
            bkg_init = sep.Background(img,bw = 16, bh = 16)
            obj_init,seg_init = sep.extract(img - bkg_init.back(), 5, err = bkg_init.rms(), segmentation_map=True)

            seg_init[seg_init>1] = 1
            sep_mask = np.copy(seg_init)

            bkg = sep.Background(img,mask = sep_mask,  maskthresh=0.,bw = 16, bh = 16)
            obj,seg = sep.extract(img - bkg.back(), 5, err = bkg.rms(), segmentation_map=True)
            init_dict['sky0'] = bkg.globalback
            bounds_dict['sky0'] = [-25,25]

            init_dict['sky1'] = 0
            init_dict['sky2'] = 0

            bounds_dict['sky1'] = [-1., 1.]
            bounds_dict['sky2'] = [-1., 1.]
            self.A_guess = np.max(obj['flux'])*1.25
            self.sep_bkg = bkg

        else:
            self.A_guess = np.sum(img)

        #Below assumes all gaussian have same A
        a_norm = np.ones(self.Ndof_gauss)*self.A_guess/self.Ndof_gauss

        #If using log scale then adjust initial guesses
        if self.log_weight_scale:
            self.A_guess = np.log10(self.A_guess)
            a_norm = np.log10(a_norm)
            #set minimum possible weight value
            a_min = -9
        else:
            a_min = 0

        for i in range(self.Ndof_gauss):
            init_dict['a%i'%i] = a_norm[i]
            bounds_dict['a%i'%i] = [a_min, self.A_guess]

        #Add option to specificy your own initial conditions and bounds
        self.lb = [bounds_dict['x0'][0], bounds_dict['y0'][0], bounds_dict['q'][0], bounds_dict['phi'][0]]
        self.ub = [bounds_dict['x0'][1], bounds_dict['y0'][1], bounds_dict['q'][1], bounds_dict['phi'][1]]

        self.param_init = np.ones(self.Ndof)
        self.param_init[0] = init_dict['x0']
        self.param_init[1] = init_dict['x0']
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
        return min_res

    def set_up_express_run(self, set_params = None):
        """ Funciton to set up 'express' run using pre-rendered images with a
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
            phi = self.min_res.x[3]
            q_in = self.min_res.x[2]

        if self.verbose:
            print ('Parameters to be set:')
            print ('\t Galaxy Center: %.2f,%.2f'%(x0,y0) )
            print ('\t PA: %.5f'%phi)
            print ('\t Axis Ratio: %.5f'%q_in)

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
        model = self.make_model(params)
        return np.sum( (self.img - model)**2 *self.weight)

    def log_like(self,params):
        return -0.5*self.chi_sq(params)

    def ptform(self, u):
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
        final_a = xp_params[:-self.Ndof_sky]

        if self.log_weight_scale: final_a = 10**final_a
        model = np.sum(final_a*self.express_gauss_arr, axis = -1)

        if self.sky_model:
            model += self.get_sky_model(exp_params[-self.Ndof_sky:])

        return -0.5*np.sum( (self.img - model)**2 *self.weight)

    def ptform_exp(self, u):
        x = np.zeros(len(u))
        x = u*(self.ub[4:] - self.lb[4:]) + self.lb[4:]
        return x

    def log_prior_express(self,params):
        inside_prior = (params > self.lb[4:])*(params < self.ub[4:])
        if inside_prior.all():
            return 0
        else:
            return -np.inf

    def log_prob_express(self,params):
        logp = self.log_prior_express(params)
        return self.log_like_exp(params) + logp


    def run_dynesty(self,method = 'full', dynesty_kwargs = {}):
        if method == 'full':
            ndim = self.Ndof
            sampler = dynesty.DynamicNestedSampler( self.log_like, self.ptform, ndim= ndim, **dynesty_kwargs)
        if method == 'express':
            ndim = self.Ndof_gauss + self.Ndof_sky
            if not hasattr(self, 'express_gauss_arr'):
                if self.verbose: print ('Setting up Express params')
                _ = self.set_up_express_run()
            sampler = dynesty.DynamicNestedSampler( self.log_like_exp, self.ptform_exp, ndim= ndim, **dynesty_kwargs)
        sampler.run_nested()
        self.dynesty_res = sampler.results

        dyn_samples, dyn_weights = self.dynesty_res.samples, np.exp(self.dynesty_res.logwt - self.dynesty_res.logz[-1])
        post_samp = dyfunc.resample_equal(dyn_samples, dyn_weights)
        return post_samp


    def run_emcee(self,method = 'express', nwalkers = 32, max_it = int(1e6), check_freq = int(500), print_progress = False):

        if method == 'full':
            print(" 'full' not yet implemented")
            return 0
        elif method == 'express':
            if not hasattr(self, 'express_gauss_arr') or not hasattr(self, 'min_err') or not hasattr(self, 'min_res'):
                if self.verbose: print ('Setting up express run')
                self.set_up_express_run()

            ndim = self.Ndof - 4
            init_arr = np.ones(self.Ndof_gauss+self.Ndof_sky)
            init_arr[:-self.Ndof_sky] = np.log10(self.min_res.x[4:-self.Ndof_sky])
            init_arr[-self.Ndof_sky:] = self.min_res.x[-self.Ndof_sky:]

            pos = init_arr + 0.05 * np.random.randn(nwalkers, ndim)

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
        return sampler.get_chain(discard=int(3 * np.max(tau)), thin = int(0.3 * np.min(tau)),  flat=True)