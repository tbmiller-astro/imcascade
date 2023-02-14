from abc import ABC
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial

class BaseMultiGaussModel(ABC):
    def __init__(self, shape, sig, psf_sig, psf_a, verbose = True, \
      sky_model = True,sky_type = 'tilted-plane'):
        """ Initialize a MultiGaussModel instance"""
        if psf_sig is not None or psf_a is not None:
            self.psf_sig = psf_sig
            self.psf_var = psf_sig*psf_sig
            self.psf_a = psf_a


            self.has_psf = True
        else:
            self.has_psf = False

        self.shape = shape

        self.x_mid = shape[0]/2.
        self.y_mid = shape[1]/2.

        self.x_pix = np.arange(0,shape[0])
        self.y_pix = np.arange(0,shape[1])

        X,Y = np.meshgrid(self.x_pix,self.y_pix, indexing = 'ij')
        self.X = X
        self.Y = Y

        self.sig = sig
        self.var = sig*sig
        
        self.sky_model = sky_model
        self.sky_type = sky_type
        if sky_model:
            if sky_type == 'flat':
                self.Ndof_sky = 1
                self.get_sky_model = self.get_sky_model_flat

            elif sky_type == 'tilted-plane':
                self.Ndof_sky = 3
                self.get_sky_model = self.get_sky_model_tp
        else:
            self.Ndof_sky = 0
        
        self.N_gauss = len(self.sig)

        #TODO Deprecate below
        self.Ndof_gauss = len(self.sig)


        self.Ndof_q = 1
        self.Ndof_phi = 1

        
        self.Ndof_struct = 2 + self.Ndof_q + self.Ndof_phi

        self.Ndof = self.Ndof_struct + self.Ndof_gauss + self.Ndof_sky

    def get_sky_model_flat(self,args):
        """ Function used to calculate flat sky model

        Parameters
                    ----------
        args: (a,) (float,float,float)

        Returns
        -------
        sky_model: 2D Array
            Model for sky background based on given parameters, same shape as 'shape'
"""
        a = args[0]

        return a
            
    def get_sky_model_tp(self,args):
        """ Function used to calculate tilted-plane sky model

        Parameters
        ----------
        args: (a,b,c) (float,float,float)
        a - overall normalization
        b - slope in x direction
        c - slope in y direction

        Returns
        -------
        sky_model: 2D Array
            Model for sky background based on given parameters, same shape as 'shape'
"""
        a,b,c = args

        return a + (self.X - self.x_mid)*b + (self.Y - self.y_mid)*c   

class MGM(BaseMultiGaussModel):
    """A class used to generate models based series of Gaussians

    Parameters
    ----------
    shape: 2x1 array_like
        Size of model image to generate
    sig: 1-D array
        Widths of Gaussians used to genrate model
    psf_sig: 1-D array, None
        Width of Gaussians used to approximate psf
    psf_a: 1-D array, None
        Weights of Gaussians used to approximate psf, must be same length
        as 'psf_sig'. If both psf_sig and psf_a are None then will run in
        Non-psf mode
    verbose: bool, optional
        If true will print out errors
    sky_model: bool, optional
        If True will incorperate a tilted plane sky model
    log_weight_scale: bool, optional
        Wether to treat weights as log scale, Default True
"""

    def __init__(self, shape, sig, psf_sig, psf_a, verbose = True, \
      sky_model = True,sky_type = 'tilted-plane'):

        """ Initialize a MultiGaussModel instance"""
        super().__init__(shape, sig, psf_sig, psf_a, verbose, sky_model,sky_type)
        self.X = jnp.array(self.X)
        self.Y = jnp.array(self.Y)

        self.sub_render_cut = 4
        self.sub_render_size = 20
        self.os_factor = 3
        frac = 1/(self.os_factor-1)
        os_ax = jnp.linspace(-frac,frac, num = self.os_factor)
        os_ax_x,os_ax_y = jnp.meshgrid(os_ax,os_ax)

        self.sub_cen = ( int((self.shape[0]) /2), int( (self.shape[1])/2 ) )

        self.X_sub = self.X[self.sub_cen[0]-self.sub_render_size : self.sub_cen[0]+self.sub_render_size, self.sub_cen[1]-self.sub_render_size : self.sub_cen[1]+self.sub_render_size ][:,:,None,None] + os_ax_x

        self.Y_sub = self.Y[self.sub_cen[0]-self.sub_render_size : self.sub_cen[0]+self.sub_render_size, self.sub_cen[1]-self.sub_render_size : self.sub_cen[1]+self.sub_render_size ][:,:,None,None] + os_ax_y


        
        if self.has_psf:
            var = (self.var + self.psf_var[:,None]).ravel()
        else:
            var = self.var
        
        self.Nrender = len(var)
        self.w_sub = jnp.where( jnp.sqrt(var) <= self.sub_render_cut )
        self.w_no_sub = jnp.where( jnp.sqrt(var) > self.sub_render_cut )

@jit
def render_gauss_model(X,Y, x_mid,y_mid, var,a, phi,q):
    X_mmid = (X - x_mid)
    Y_mmid = (Y - y_mid)
    xp = X_mmid*jnp.cos(phi) + Y_mmid*jnp.sin(phi)
    yp = X_mmid*-1*jnp.sin(phi) + Y_mmid*jnp.cos(phi)

    r_sq = xp**2 + (yp / q )**2
    const = a/ (2*jnp.pi*var*q)
    mod_3d = const* jnp.exp(-r_sq/(2*var) )
    return mod_3d.sum(axis = -1)

