import numpy as np
import logging
from .mgm import MGM, render_gauss_model
from imcascade.results import ImcascadeResults,vars_to_use
from imcascade.utils import dict_add, log_scale,expand_mask,reg_resid,parse_input_dicts, get_sersic_amps
from astropy.io import fits
import asdf
import arviz as az 
import optax

from typing import Optional,Union, Callable
import jax.numpy as jnp
from jax.random import PRNGKey
from jax import jit
import numpyro
from numpyro.handlers import seed, trace, reparam
import numpyro.distributions as dist
from numpyro import infer
from numpyro.optim import Adam
from numpyro.infer.reparam import TransformReparam, LocScaleReparam


import dynesty
from dynesty import utils as dyfunc

log2pi = np.log(2.*np.pi)


def initialize_fitter(im, psf, mask = None, err = None, x0 = None,y0 = None, re = None, flux = None,
 psf_oversamp = 1, sky_model = True, log_file = None, readnoise = None, gain = None, exp_time = None, num_components = None,
 component_widths = None, log_weight_scale = True):
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

class Fitter(MGM):
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
      sky_model = True,sky_type = 'tilted-plane', verbose = True, init_dict = {}, bounds_dict = {}, log_file = None):
        """Initialize a Task instance"""
        self.img  = img
        self.verbose = verbose
        self.rkey = PRNGKey(42)
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
            self.sig_img = 1./(np.sqrt(weight)+ 1e-8)


        if mask is not None:
            if self.weight.shape != self.img.shape:
                raise ValueError("'mask' array must have same shape as 'img' ")
            self.mask = mask
        else:
            self.mask = np.zeros(self.img.shape)

        where_inf = np.logical_or(np.isnan(self.img),
            np.isnan(self.weight),
            self.sig_img > 1e7)
        
        if np.sum(where_inf) > 0:
            where_inf = np.where(where_inf)
            self.logger.info("Masking nan values at locations:")
            self.logger.info(where_inf)
            self.mask[where_inf] = 1
        
        self.inv_mask = np.abs(self.mask-1).astype(bool)

        super().__init__(self.img.shape,sig, psf_sig, psf_a,
          verbose = verbose, sky_model = sky_model,sky_type = sky_type)

        self.npix = self.img.shape[0]*self.img.shape[1] - np.sum(self.mask)

        bounds_dict, init_dict, lb,ub, param_init = parse_input_dicts(bounds_dict,init_dict, self)
        
        self.init_dict = init_dict
        self.bounds_dict = bounds_dict
        self.param_init = param_init
        self.lb = lb
        self.ub = ub
        self.bnds = self.lb,self.ub
        
    def validate_generators(self)-> bool:
        xc_yc_trace = trace(seed(self.generate_xc_yc, self.rkey)).get_trace()
        assert 'xc' in xc_yc_trace.keys()
        assert 'yc' in xc_yc_trace.keys()
        
        phi_trace = trace(seed(self.generate_phi, self.rkey)).get_trace()   
        assert ('phi' in phi_trace.keys() or 'phi_unwrapped' in phi_trace.keys() )

        q_trace = trace(seed(self.generate_q, self.rkey)).get_trace()   
        assert 'q' in q_trace.keys()

        comp_fluxes_trace = trace(seed(self.generate_comp_fluxes, self.rkey)).get_trace()

        assert 'comp_fluxes' in comp_fluxes_trace.keys()
        assert comp_fluxes_trace['comp_fluxes']['value'].shape == (self.N_gauss,)

        return True


    def build_model(self,) -> Callable:

        @jit
        def render_full_model(x_mid,y_mid, var,a, phi,q):
            mod_no_sub = render_gauss_model(self.X[:,:,None], 
                self.Y[:,:,None], 
                x_mid,
                y_mid, 
                var[self.w_no_sub], 
                a[self.w_no_sub], 
                phi[self.w_no_sub],
                q[self.w_no_sub] )
            
            mod_sub = render_gauss_model(self.X_sub[:,:,:,:,None], 
                self.Y_sub[:,:,:,:,None], 
                x_mid,
                y_mid, 
                var[self.w_sub], 
                a[self.w_sub], 
                phi[self.w_sub],
                q[self.w_sub] ).mean(axis = (2,3))

            model_image = mod_no_sub.at[self.sub_cen[0] - self.sub_render_size : self.sub_cen[0] + self.sub_render_size, self.sub_cen[1] - self.sub_render_size: self.sub_cen[1] + self.sub_render_size].add(mod_sub)

            return model_image
        
        def model():
            xc,yc = self.generate_xc_yc()
            phi = self.generate_phi()
            q = self.generate_q()
            comp_fluxes = self.generate_comp_fluxes()
            
            if self.has_psf:
                var_render = (self.var + self.psf_var[:,None]).flatten()
                q_render = jnp.sqrt( (self.var*q**2 + self.psf_var[:,None]).flatten()/var_render ) 
                flux_render = (comp_fluxes*self.psf_a[:,None]).flatten()
            else:
                var_render = self.var
                q_render = q
                flux_render = comp_fluxes
            
            model_image = render_full_model(xc,yc,var_render,flux_render,phi,q_render) 
            sky_image = self.generate_sky()

            final_image =  model_image + sky_image

            with numpyro.handlers.mask(mask=self.inv_mask):
                numpyro.sample('obs', dist.Normal(final_image, self.sig_img), obs =self.img )

        return model
    
    def sample(self, sampler: Optional[infer.MCMC] = infer.NUTS,
        sampler_kwargs: Optional[dict] = {},
        mcmc_kwargs: Optional[dict] = {'num_warmup':500, 'num_samples':500, 'num_chains':1},
        neutra_reparam: Optional[bool] = False
        ) -> az.InferenceData:
        assert self.validate_generators()

        model_use = self.build_model()
        if neutra_reparam:
            assert hasattr(self, 'guide') and hasattr(self,'svi_result')
            neutra = infer.reparam.NeuTraReparam(self.guide, self.svi_result.params)
            model_use = neutra.reparam(model_use)

        
        mcmc_sampler = sampler(model_use, **sampler_kwargs)
        mcmc_kernel = infer.MCMC(mcmc_sampler,**mcmc_kwargs)

        mcmc_kernel.run(self.rkey)

        self.arvizID = az.from_numpyro(mcmc_kernel)
        self.posterior = az.extract(self.arvizID)
        return self.posterior

    def fit(self,
        use_posterior: Optional[bool] = False,
        guide_model: Optional[infer.autoguide.AutoGuide] = infer.autoguide.AutoLaplaceApproximation,
        guide_kwargs: Optional[dict] = {},
        run_kwargs: Optional[dict] = {'num_steps': 2500,},
        optimizer_kwargs: Optional[dict] = {'step_size': 0.05}
    )-> Union[dict, az.InferenceData]:
        assert self.validate_generators()
        
        model_use = self.build_model()
        optimizer = numpyro.optim.Adam(**optimizer_kwargs)
        guide = guide_model(model_use, **guide_kwargs)
        svi_kernel = infer.SVI(model_use, guide, optimizer, loss = infer.Trace_ELBO(num_particles=3))
        
        self.svi_result = svi_kernel.run(self.rkey, **run_kwargs)
        self.guide = guide
        
        if use_posterior:
            #Sample guide posterior
            post_raw = guide.sample_posterior(self.rkey, self.svi_result.params, sample_shape = ((1000,)))
            #Convert to arviz
            post_dict = {}
            for key in post_raw:
                post_dict[key] = post_raw[key][jnp.newaxis,]
            self.arvizID = az.from_dict(post_dict)
            self.posterior = az.extract(self.arvizID)
            return self.posterior
        else:
            median_dict = guide.median(self.svi_result.params)
            for key in median_dict:
                if 'raw' in median_dict:
                    median_dict.pop(key)
            pred = infer.Predictive(model_use, guide = self.guide,num_samples=1)
            pred_dict = pred(self.rkey)
            for k in pred_dict.keys():
                pred_dict[k] = pred_dict[k].squeeze()
            median_dict.update(pred_dict)
            self.result_dict = median_dict.copy()

            return self.result_dict

    def generate_xc_yc(self):
        xc_raw = numpyro.sample('xc_base', dist.Normal())
        xc = numpyro.deterministic('xc', xc_raw + self.init_dict['xc'])
        
        yc_raw = numpyro.sample('yc_base', dist.Normal())
        yc = numpyro.deterministic('yc', yc_raw + self.init_dict['yc'])
        return xc,yc
    
    def generate_phi(self):

        phi = numpyro.sample('phi', dist.Uniform(low = 0., high = jnp.pi/2.))
        
        return phi*jnp.ones(self.N_gauss)

    def generate_q(self):
        q = numpyro.sample('q', dist.Uniform(0.1,1.))
        return q*jnp.ones(self.N_gauss)
    
    def generate_comp_fluxes(self):
        #Uniform Priors on linear fluxes
        comp_fluxes_raw =  numpyro.sample('comp_fluxes_raw',
           dist.Uniform(low = -0.1, 
              high = 1),
              sample_shape = (self.N_gauss,))
        comp_fluxes = numpyro.deterministic('comp_fluxes', comp_fluxes_raw*self.init_dict['flux'])
        return comp_fluxes
    
    def generate_sky(self):
        if not self.sky_model: return 0

        if self.sky_type == 'flat':
            sky_back_raw = numpyro.sample('sky0_raw', dist.Normal())
            sky_back = numpyro.deterministic('sky0', 
              sky_back_raw*self.init_dict['sky0']/2. + self.init_dict['sky0'])
            return sky_back

        if self.sky_type == 'tilted-plane':
            sky_back_raw = numpyro.sample('sky0_raw', dist.Normal())
            sky_back = numpyro.deterministic('sky0', 
              sky_back_raw*1e-3 + self.init_dict['sky0'])
            
            sky_xsl_raw = numpyro.sample('sky1_raw', dist.Normal())
            sky_xsl = numpyro.deterministic('sky1', 
              sky_xsl_raw*1e-4)

            sky_ysl_raw = numpyro.sample('sky2_raw', dist.Normal())
            sky_ysl = numpyro.deterministic('sky2', 
              sky_ysl_raw*1e-4)

            return sky_back + (self.X- self.x_mid)*sky_xsl + (self.Y- self.y_mid)*sky_ysl

def get_priors_results(fitter: Fitter) -> ImcascadeResults:
    model_use = fitter.build_model()
    pred = infer.Predictive(model_use, num_samples= 1000, batch_ndims=2)
    prior_pred = pred(fitter.rkey)
    var_dict = vars(fitter)
    prior_arviz = az.from_dict(prior_pred)
    var_dict['posterior'] = az.extract(prior_arviz)
    return ImcascadeResults(var_dict)

import copy

def add_sersic_flux_priors(fitter: Fitter, re: float, n: float):
    fitter = copy.deepcopy(fitter)
    med_prior_frac = get_sersic_amps(fitter.sig, re, n)
    med_prior_frac = fitter.init_dict['flux']*med_prior_frac/np.sum(med_prior_frac)
    sig_prior_frac = 0.05*fitter.init_dict['flux']
        
    def generate_comp_fluxes():
        reparam_config = {"comp_fluxes": TransformReparam()}
        with numpyro.handlers.reparam(config = reparam_config):
            comp_fluxes = numpyro.sample('comp_fluxes',
                dist.TransformedDistribution(
                    dist.Normal(loc = 0, scale = jnp.ones(fitter.N_gauss)),
                    dist.transforms.AffineTransform(med_prior_frac,sig_prior_frac)
                ),
            )
        return comp_fluxes
    
    fitter.__setattr__('generate_comp_fluxes', generate_comp_fluxes)
    assert fitter.validate_generators()
    return fitter

def add_q_prof_lin_interp(fitter: Fitter,r_low: float, r_high: float):
    fitter = copy.deepcopy(fitter)

    def generate_q_lin_interp():
        q_med = numpyro.sample('q_med', dist.Uniform(0.1,1.))
        tau_q = 0.1#numpyro.sample('tau_q', dist.HalfCauchy(scale=0.25))
        with numpyro.plate('q_plate', 3):
            lambda_q = numpyro.sample('lambda_q', dist.HalfCauchy(scale=1))
            horseshoe_sigma = tau_q*lambda_q**2
            q_nodes = numpyro.sample('q_nodes', dist.TruncatedNormal(loc=q_med, scale=horseshoe_sigma, low = 0.1, high = 1))
        
        r_mid = numpyro.sample('r_q_node_mid', dist.TruncatedNormal(loc=10, scale=3, low = 3, high = 20) )

        r_nodes = jnp.array([r_low,r_mid, r_high])
        q = numpyro.deterministic('q', jnp.interp(fitter.sig, r_nodes, q_nodes))
        return q

    fitter.__setattr__('generate_q', generate_q_lin_interp)
    assert fitter.validate_generators()
    return fitter

def add_q_all_free(fitter: Fitter):
    fitter = copy.deepcopy(fitter)

    def generate_q_all():
        q_med = numpyro.sample('q_med', dist.Uniform(0.1,1.))
        tau_q = 0.1#numpyro.sample('tau_q', dist.HalfCauchy(scale=0.1))
        with numpyro.plate('q_plate', fitter.N_gauss):
            lambda_q = numpyro.sample('lambda_q', dist.HalfCauchy(scale=1.))
            horseshoe_sigma = tau_q *lambda_q**2
            q = numpyro.sample('q', dist.TruncatedNormal(loc=q_med, scale=horseshoe_sigma, low = 0.1, high = 1))

        return q

    fitter = copy.deepcopy(fitter)
    fitter.__setattr__('generate_q', generate_q_all)
    assert fitter.validate_generators()
    return fitter
