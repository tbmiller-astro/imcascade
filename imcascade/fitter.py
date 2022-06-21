import jax
import numpy as np
import asdf
import logging
from scipy.optimize import least_squares, minimize, Bounds
from scipy.stats import norm, truncnorm

from imcascade.mgm import V2MGM, MultiGaussModel
from imcascade.results import ImcascadeResults,vars_to_use
from imcascade.utils import dict_add, log_scale,expand_mask,reg_resid,parse_input_dicts
from astropy.io import fits

import jax.numpy as jnp

import dynesty
from dynesty import utils as dyfunc
import pymc3 as pm
import theano.tensor as tt

log2pi = np.log(2.*np.pi)

def initialize_fitter(im, psf, mask = None, err = None, x0 = None,y0 = None, re = None, flux = None, psf_oversamp = 1, sky_model = True, log_file = None, readnoise = None, gain = None, exp_time = None):
    """Function used to help Initialize Fitter instance from simple inputs

    Parameters
    ----------
    im: str or 2D Array
        The image or cutout to be fit with imcascade. If a string is given, it is
        interpretted as the location of a fits file with the cutout in it's first HDU.
        Otherwise is a 2D numpy array of the data to be fit
    psf: str, 2D Array or None
        Similar to above but for the PSF. If not using a PSF, the use None
    mask: 2D array (optional)
        Sources to be masked when fitting, if none is given then one will be derrived
    err: 2D array (optional)
        Pixel errors used to calculate the weights when fitting. If none is given will
        use readnoise, gain and exp_time if given, or default to sep derrived rms
    x0: float (optional)
        Inital guess at x position of center, if not will assume the center of the image
    y0: float (optional)
        Inital guess at y position of center, if not will assume the center of the image
    re: float (optional)
        Inital guess at the effective radius of the galaxy, if not given will estimate
        using sep kron radius
    flux: float (optional)
        Inital guess at the flux of the galaxy, if not given will estimate
        using sep flux
    psf_oversamp: float (optional)
        Oversampling of PSF given, default is 1
    sky_model: boolean (optional)
        Whether or not to model sky as tilted-plane, default is True
    log_file: str (optional)
        Location of log file
    readnoise,gain,exp_time: float,float,float (all optional)
        The read noise (in electrons), gain and exposure time of image that is
        used to calculate the errors and therefore pixel weights. Only used if
        ``err = None``. If these parameters are also None, then will estimate
        pixel errors using sep rms map.

    Returns
    -------
    Fitter: imcascade.fitter.Fitter
        Returns intialized instance of imcascade.fitter.Fitter which can then
        be used to fit galaxy and analyze results.

"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if log_file is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_file)

    logging.basicConfig(format = "%(asctime)s - %(message)s", level = logging.INFO,
        handlers = [handler,])
    logger = logging.getLogger()

    if x0 is None:
        x0 = im.shape[0]/2.
    if y0 is None:
        y0 = im.shape[0]/2.

    #Fit PSF
    if type(psf) == str:
        psf_data = fits.open(psf)
        from imcascade.psf_fitter import PSFFitter
        pfitter = PSFFitter(psf_data[0].data, oversamp = psf_oversamp)
    elif type(psf) == np.ndarray:
        from imcascade.psf_fitter import PSFFitter
        pfitter = PSFFitter(psf, oversamp = psf_oversamp)
    elif psf is None:
        logger.info("No PSF given")
        psf_sig = None
        psf_a = None

    if psf is not None:
        #Find best fit gaussian decomposition of PSF
        psf_sig,psf_a = pfitter.auto_fit()
        logger.info("Fit PSF with %i components"%len(psf_sig))
        logger.info("Widths: "+ ','.join(map(str,np.round(psf_sig,2))) )
        logger.info("Fluxes: "+ ','.join(map(str,np.round(psf_a,2))))

        #Calculate hwhm
        psf_mp = np.ones(4+len(psf_sig))
        psf_mp[4:] = psf_a
        rdict = {'sig':psf_sig, 'Ndof':4+len(psf_sig), 'Ndof_gauss':len(psf_sig), 'Ndof_sky':0, 'log_weight_scale':False,'min_param':psf_mp, 'sky_model':False}
        psf_res = ImcascadeResults(rdict)
        psf_hwhm = psf_res.calc_iso_r(psf_res.calc_sbp(np.array([0,]))/2)

        if psf_hwhm > 1:
            sig_min = psf_hwhm*0.5
        else:
            sig_min = 0.5
    else:
        sig_min = 0.75

    #Load image data
    if type(im) == str:
        im_fits = fits.open(im)
        im_data = im_fits[0].data
    elif type(im) == np.ndarray:
        im_data = im.copy()

    if im_data.dtype.byteorder == '>':
        im_data = im_data.byteswap().newbyteorder()

    #Use sep to estimate object properties and rms depending
    import sep
    bkg = sep.Background(im_data, bw = 16,bh = 16)
    obj,seg = sep.extract(im_data, 2., err = bkg.globalrms, segmentation_map = True, deblend_cont=1e-4)
    seg_obj = seg[int(x0), int(y0)]

    if re is None:
        a_guess = obj['a'][seg_obj-1]
    else:
        a_guess = re

    if flux is None:
        fl_guess = obj['flux'][seg_obj-1]
    else:
        fl_guess = None
    sig_max = a_guess*9

    init_dict = {'re':a_guess, 'flux':fl_guess, 'sky0':bkg.globalback, 'x0':x0, 'y0':y0}

    #Calculate widths of components
    if im_data.shape[0] > 100:
        num_sig = 9
    else:
        num_sig = 7
    sig_use = log_scale(sig_min,sig_max, num_sig)
    logger.info("Using %i components with logarithmically spaced widths to fit galaxy"%num_sig)
    logger.info(", ".join(map(str,np.round(sig_use,2))))

    #Use sep results to estimate mask
    if mask is None:
        logger.info("No mask was given, derriving one using sep")
        mask = np.copy(seg)
        mask[np.where(seg == seg_obj)] = 0
        mask[mask>=1] = 1
        mask = expand_mask(mask, threshold = 0.01, radius = 2.5)

    if err is not None:
        #Calculate weights based on error image given
        pix_weights = 1./err**2
        logger.info("Using given errors to calculate pixel weights")
    elif (gain is not None and readnoise is not None and exp_time is not None):
        #Calculate based on exp time, gain and readnoise
        var_calc = im_data/ (gain*exp_time) + readnoise**2/(exp_time**2)  #Maybe gain too?
        pix_weights = 1/var_calc
        logger.info("Using gain, exp_time and readnoise to calculate pixel weights")
    else:
        #If no info is given then default to sep results which do pretty well
        logger.info("Using sep rms map to calculate pixel weights")
        pix_weights = 1./bkg.rms()**2

    #Initalize Fitter
    return Fitter(im_data,sig_use, psf_sig,psf_a, weight = pix_weights, mask = mask, init_dict = init_dict, sky_model = sky_model, log_file = log_file)


def fitter_from_ASDF(file_name, init_dict= {}, bounds_dict = {}):
    """ Function used to initalize a fitter from a saved asdf file

    This can be useful for re-running or for initializing a series
    of galaxies beforehand and then transferring to somewhere else or running in
    parallel

    Parameters
    ----------
    file_name: str
        location of asdf file containing saved data. Often this is a file created
        by Fitter.save_results
    init_dict: dict (optional)
        Dictionary specifying initial guesses for least_squares fitting to be passed
        to Fitter instance.
    bounds_dict: dict (optional)
        Dictionary specifying bounds for least_squares fitting to be passed
        to Fitter instance.

"""
    af = asdf.open(file_name,copy_arrays=True)
    dict = af.tree.copy()

    keys_for_func = ['weight','mask','sky_model','render_mode','log_weight_scale',
    'verbose', 'log_file','psf_shape']
    kwargs = {}
    for key in dict:
        if key in keys_for_func:
            kwargs[key] = dict[key]

    img = np.copy(dict.pop('img'))
    sig = dict.pop('sig')
    psf_sig = dict.pop('psf_sig')
    psf_a = dict.pop('psf_a')
    inst = Fitter(img,sig,psf_sig,psf_a,init_dict = init_dict, bounds_dict = bounds_dict, **kwargs)
    return inst

 # TODO Implement pymc3 model set up
 #? How well does Hessian approx posterior?

class V2Fitter(V2MGM):
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
      sky_model = True,sky_type = 'tilted-plane', log_weight_scale = True, q_profile = False,phi_profile = False, verbose = True, psf_shape = None,init_dict = {}, bounds_dict = {}, log_file = None):
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

        self.inv_mask = np.abs(self.mask-1.)

        if np.sum(np.isnan(self.img) + np.sum(np.isnan(self.weight))) > 0:
            where_inf = np.where(np.isnan(self.img) + np.isnan(self.weight))
            self.logger.info("Masking nan values at locations:")
            self.logger.info(where_inf)
            self.img[where_inf] = 0
            self.weight[where_inf] = 0
            self.mask[where_inf] = 1


        super().__init__(self.img.shape,sig, psf_sig, psf_a,
          verbose = verbose, sky_model = sky_model,sky_type = sky_type, log_weight_scale = log_weight_scale, psf_shape = psf_shape, q_profile=q_profile, phi_profile=phi_profile)

        self.npix = self.img.shape[0]*self.img.shape[1] - np.sum(self.mask)

        bounds_dict, init_dict, lb,ub, param_init = parse_input_dicts(bounds_dict,init_dict, self)
        
        self.init_dict = init_dict
        self.bounds_dict = bounds_dict
        self.param_init = param_init
        self.lb = lb
        self.ub = ub
        self.bnds = Bounds(self.lb,self.ub)
    
    def run_ls_min(self, ls_kwargs = {}, lasso_k = 1e-4, lasso_inds = np.arange(3)):
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
        if self.verbose: self.logger.info('Compiling JAX functions')

        def loss_base(param):
            chi_2 = jnp.sum( (self.img - self.make_model(param))**2 *self.weight )
            # mean_lasso = jnp.mean(param[self.Ndof_struct+lasso_inds])
            # lasso_loss =  lasso_k * jnp.abs(param[self.Ndof_struct+lasso_inds] - mean_lasso).sum() *jnp.sum(self.inv_mask) 
            return chi_2# + lasso_loss
        
        loss_jit = jax.jit(loss_base)
        _ = loss_jit(self.param_init)

        loss_jac = jax.jacfwd(loss_base)
        _ = loss_jac(self.param_init)

        loss_hess = jax.jacfwd(jax.jacfwd(loss_base) )
        _ = loss_hess(self.param_init)

        if self.verbose: self.logger.info('Running least squares minimization')
        
        ls_kwargs = dict_add(ls_kwargs,'method','TNC')
        min_res = minimize(loss_jit, self.param_init,jac = loss_jac,hess = loss_hess, bounds = self.bnds, **ls_kwargs)
        self.min_res = min_res
        self.min_param = min_res.x
        if self.verbose: self.logger.info('Finished least squares minimization')
        
        hess_min = loss_hess(self.min_param)
        self.hess_min = hess_min

        return self.min_param

    def sample_min_param_covariance(self, num = 10000):
        assert 'hess_min' in dir(self), "Need to run 'run_ls_min' to calculate min_param"

        #Find Full covariance matrix from hessian
        cov_full =  jnp.linalg.inv(self.hess_min) 
        # Perfrom Cholesky decomp to find triangular covariance
        cov = jnp.linalg.cholesky(cov_full) 

        #Sample from covariance
        samps = jnp.matmul(cov,np.random.normal(size = (self.Ndof, num))).T + self.min_param

        return samps

    def q_prof(self, q_params, calc = jnp):
        if calc is jnp:
            arr_func = jnp.array
        else:
            arr_func = tt.as_tensor_variable
        
        return self.lin_interp(self.sig, q_params[:3], arr_func([q_params[3],q_params[4], 4*self.init_dict['re']] ) , calc = calc )
    
    def phi_prof(self, phi_params):
        phi0,phi1,phi2,r_phi1 = phi_params
        return self.lin_interp(self.sig,  jnp.array([phi0,phi1,phi2]),  jnp.array([0,r_phi1, 3*self.init_dict['re']] ) )
        

    def lin_interp(self,r, x_list,r_list, calc = jnp):

        if calc is jnp:
            x_interp = jnp.interp(r, r_list,x_list,left = x_list[0], right = x_list[-1])
        
        if calc is pm.math:
            x0,x1,x2 = x_list
            r0,r1,r2 = r_list
            x_interp = pm.math.switch(pm.math.lt(r,r0), x0,
                pm.math.switch(
                pm.math.lt(r,r1), (x0*(r1-r) + x1*(r - r0))/(r1-r0), 
                pm.math.switch(
                pm.math.lt(r,r2),(x1*(r2-r) + x2*(r - r1))/(r2-r1) , x2 ) 
                ) )
        
        return x_interp



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
        self.loglike_const = 0.5*(np.sum(self.log_weight) - log2pi*(np.sum(self.mask == 0)) )
        self.reg_k = reg_k

        MultiGaussModel.__init__(self,self.img.shape,sig, psf_sig, psf_a, \
          verbose = verbose, sky_model = sky_model,sky_type = sky_type, render_mode = render_mode, log_weight_scale = log_weight_scale, psf_shape = psf_shape)

        self.npix = self.img.shape[0]*self.img.shape[1] - np.sum(self.mask)

        bounds_dict, init_dict, lb,ub, param_init = parse_input_dicts(bounds_dict,init_dict, self)
        
        self.param_init = param_init
        self.lb = lb
        self.ub = ub
        self.bnds = (self.lb,self.ub)

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
        min_res = least_squares(self.resid_1d, self.param_init, bounds = self.bnds, **ls_kwargs)
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
