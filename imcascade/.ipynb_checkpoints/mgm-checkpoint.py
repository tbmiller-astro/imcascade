import numpy as np
from scipy.special import erf
from scipy.ndimage import rotate,shift
from numba import njit

class MultiGaussModel():
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
    render_mode: 'gauss' or 'erf'
        Option to decide how to render models. Default is 'erf' as it computes
        the integral over the pixel of each profile therefore is more accurate
        but more computationally intensive. 'gauss' assumes the center of a pixel
        provides a reasonble estimate of the average flux in that pixel. 'gauss'
        is faster but far less accurate for objects with size O(pixel size),
        so use with caution.
    log_weight_scale: bool, optional
        Wether to treat weights as log scale, Default True
"""

    def __init__(self, shape, sig, psf_sig, psf_a, verbose = True, \
      sky_model = True,sky_type = 'tilted-plane', render_mode = 'hybrid', log_weight_scale = True, \
      psf_shape = None):
        """ Initialize a MultiGaussModel instance"""
        if psf_sig is not None or psf_a is not None:
            self.psf_sig = psf_sig
            self.psf_var = psf_sig*psf_sig
            self.psf_a = psf_a


            self.has_psf = True
        else:
            self.has_psf = False

        self.psf_shape = psf_shape
        self.shape = shape
        self.render_mode = render_mode
        self.log_weight_scale = log_weight_scale

        self.x_mid = shape[0]/2.
        self.y_mid = shape[1]/2.

        x_pix = np.arange(0,shape[0])
        y_pix = np.arange(0,shape[1])
        X,Y = np.meshgrid(x_pix,y_pix, indexing = 'ij')
        self.X = X
        self.Y = Y

        #Get larger grid for enlarged erf_stack calculation
        x_pix_lg = np.arange(0,int(shape[0]*1.41)+2 )
        y_pix_lg = np.arange(0,int(shape[1]*1.41)+2 )
        X_lg,Y_lg = np.meshgrid(x_pix_lg,y_pix_lg, indexing = 'ij')
        self._lg_fac_x = int( (x_pix_lg[-1] - x_pix[-1])/2.)
        self._lg_fac_y = int( (y_pix_lg[-1] - y_pix[-1])/2.)
        self.X_lg = X_lg
        self.Y_lg = Y_lg

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

        self.Ndof_gauss = len(self.sig)

        self.Ndof = 4 + self.Ndof_gauss + self.Ndof_sky

    def get_gauss_stack(self, x0,y0, q_arr, a_arr,var_arr):
        """ Function used to calculate render model using the 'Gauss' method

        Parameters
        ----------
        x0: float
            x position of center
        y0: float
            y position of center
        q_arr: Array
            Array of axis ratios
        a_arr:
            Array of Gaussian Weights
        var_arr:
            Array of Gassian widths, note this the variance so sig^2

        Returns
        -------
        Gauss_model: array
                Array representing the model image, same shape as 'shape'
"""

        Xp = self.X - x0
        Yp = self.Y - y0
        Rp_sq = (Xp*Xp)[:,:,None] + ((Yp*Yp)[:,:,None] / (q_arr*q_arr))

        gauss_stack = a_arr / (2*np.pi*var_arr * q_arr) * np.exp( -1.*Rp_sq/ (2*var_arr) )

        return gauss_stack

    def get_erf_stack(self,x0, y0, final_q, final_a, final_var):
        """ Function used to calculate render model using the 'erf' method

        Parameters
        ----------
        x0: float
            x position of center
        y0: float
            y position of the center
        final_q: Array
            Array of axis ratios
        final_a: Array
            Array of Gaussian Weights
        final_var: Array
            Array of Gassian widths, note this the variance so sig^2
        Returns
        -------
        erf_model: array
            Array representing each rendered component
"""
        X_use = self.X_lg[:,:,None] - (x0 + self._lg_fac_x)
        Y_use = self.Y_lg[:,:,None] - (y0 + self._lg_fac_y)
        c_x = 1./(np.sqrt(2*final_var))
        c_y = 1./(np.sqrt(2*final_var)*final_q)

        unrotated_stack = final_a/4.*( ( erf(c_x*(X_use-0.5)) - erf(c_x*(X_use+0.5)) )* ( erf(c_y*(Y_use-0.5)) - erf(c_y*(Y_use+0.5)) ) )
        return unrotated_stack

    def get_hybrid_stack(self,x0, y0, final_q, final_a, final_var):
        """ Function used to calculate render model using the hybrid method, which uses erf where neccesary to ensure accurate integration and gauss otherwise. Also set everything >5 sigma away to 0.

        Parameters
        ----------
        x0: float
            x position of center
        y0: float
            y position of the center
        final_q: Array
            Array of axis ratios
        final_a: Array
            Array of Gaussian Weights
        final_var: Array
            Array of Gassian widths, note this the variance so sig^2

        Returns
        -------
        erf_model: 3D array
            Array representing each rendered component
"""
        im_args = (self.X_lg,self.Y_lg,self._lg_fac_x,self._lg_fac_y, self.shape )

        return _get_hybrid_stack(x0, y0,final_q, final_a, final_var, im_args)

    def make_model(self,param,return_stack = False):
        """ Function to generate model image based on given paramters array.
        This version assumaes the gaussian weights are given in linear scale

        Parameters
        ----------
        param: array
            1-D array containing all the Parameters

        Returns
        -------
        model_image: 2D Array
            Generated model image as the sum of all components plus sky, if included
"""
        x0= param[0]
        y0 = param[1]
        q_in = param[2]
        phi = param[3]


        if self.log_weight_scale:
            a_in = 10**param[4:4+self.Ndof_gauss]
        else:
            a_in = param[4:4+self.Ndof_gauss]

        if not self.has_psf:
            final_var = np.copy(self.var)
            final_q = np.array([q_in]*len(final_var))
            final_a = a_in
            final_phi = phi

        else:
            if self.psf_shape == None:
                final_var = (self.var + self.psf_var[:,None]).ravel()
                final_q = np.sqrt( (self.var*q_in*q_in+ self.psf_var[:,None]).ravel() / (final_var) )
                final_a = (a_in*self.psf_a[:,None]).ravel()
                final_phi = phi
            else:
                final_var, final_phi, final_q = get_ellip_conv_params(self.var, q_in, phi, self.psf_var,self.psf_shape['q'],self.psf_shape['phi'])
                final_a = (a_in*self.psf_a[:,None]).ravel()

        ## Render unrotated stack of components
        if self.render_mode == 'hybrid':
            unrot_stack = self.get_hybrid_stack(x0, y0,final_q, final_a, final_var)
        elif self.render_mode == 'erf':
            unrot_stack = self.get_erf_stack(x0, y0,final_q, final_a, final_var)
        elif self.render_mode == 'gauss':
            unrot_stack = self.get_gauss_stack(x0,y0, final_q, final_a, final_var)

        #If circular PSF, sum to create img then rotate
        if self.psf_shape == None:
            if return_stack:
                stack = np.array([rot_im(unrot_stack[:,:,i], final_phi, x0+self._lg_fac_x,y0+self._lg_fac_y) for i in range(len(final_a))])
                stack =  np.moveaxis(stack,0,-1)
                return stack[self._lg_fac_x:self._lg_fac_x + self.shape[0], self._lg_fac_y:self._lg_fac_y + self.shape[1], :]

            unrot_im_lg = unrot_stack.sum(axis = -1)
            im_lg = rot_im(unrot_im_lg, final_phi, x0+self._lg_fac_x,y0+self._lg_fac_y)

        #Else rotate each component indvidually, much slower so not advised unless neccesarry
        else:
            stack = np.array([rot_im(unrot_stack[:,:,i], final_phi[i], x0 + self._lg_fac_x,y0 + self._lg_fac_y) for i in range(len(final_phi))])
            stack = np.moveaxis(stack,0,-1)
            if return_stack:
                return stack[self._lg_fac_x:self._lg_fac_x + self.shape[0], self._lg_fac_y:self._lg_fac_y + self.shape[1], :]

            im_lg = stack.sum(axis = -1)

        model_im = im_lg[self._lg_fac_x:self._lg_fac_x + self.shape[0], self._lg_fac_y:self._lg_fac_y + self.shape[1]]


        if not self.sky_model:
            return model_im
        else:
            return model_im + self.get_sky_model(param[-self.Ndof_sky:])

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

def rot_im(img,phi,x0,y0):
    """Function to rotate image around a given point
    
    Parameters
    ----------
    img: 2D array
        Image to be rotated
    phi: Float
        angle to rotate image
    x0: Float
        x coordinate to rotate image around
    y0: Float
        y coordinate to rotate image around

    Returns
    -------
    2D array
        rotated image
"""
    xc,yc = img.shape
    xc *= 0.5
    yc *= 0.5
    to_shiftx = xc - x0
    to_shifty = yc - y0
    #shift to center
    shifted = shift(img, (to_shiftx,to_shifty))
    #rotate around center
    rot_shifted = rotate(shifted,phi*180/np.pi, reshape = False)
    #shift back
    final = shift(rot_shifted,(-to_shiftx,-to_shifty))
    return final

@njit
def get_ellip_conv_params(var_all, q, phi, psf_var_all,psf_q,psf_phi):
    """Function used to derrive the observed Gaussian Parameters for a non-circular PSF

        Parameters
        ----------
        var: array
            Variances of Gaussian components
        q: Float
            Axis ratio of Galaxy
        phi: Float
            PA of galaxy
        psf_var_all: array
            Variances of PSF gaussian decomposition
        psf_q: float
            Axis ratio of PSF
        psf_phi: PA of PSF

        Returns
        -------
        obs_var: array
            Array of variances for the components of the convolved gaussian model
        obs_phi: array
            Array of position angles for the components of the convolved gaussian model
        obs_q: array
            Array of axis ratios for the components of the convolved gaussian model
"""

    size = len(var_all)*len(psf_var_all)

    var_final = np.zeros(size)
    phi_final = np.zeros(size)
    q_final = np.zeros(size)

    num = 0
    for psf_var in psf_var_all:
        for var in var_all:
            x_temp = (var*(1-q**2)*np.sin(2*phi) + psf_var*(1-psf_q**2)*np.sin(2*psf_phi) )
            y_temp = (var*(1-q**2)*np.cos(2*phi) + psf_var*(1-psf_q**2)*np.cos(2*psf_phi) )
            phi_cur = 0.5*np.arctan2(x_temp,y_temp)

            var_cur =  var  *(np.cos(phi-phi_cur)**2 + q**2*np.sin(phi-phi_cur)**2 ) + psf_var *(np.cos(psf_phi-phi_cur)**2 + psf_q**2 *np.sin(psf_phi-phi_cur)**2 )
            q_cur =  np.sqrt(( var*(np.sin(phi-phi_cur)**2 + q**2*np.cos(phi-phi_cur)**2 ) + psf_var*(np.sin(psf_phi-phi_cur)**2 + psf_q**2 *np.cos(psf_phi-phi_cur)**2 ) ) / var_cur )

            var_final[num] = var_cur
            phi_final[num] = phi_cur
            q_final[num] = q_cur
            num += 1
    return var_final,phi_final,q_final

@njit
def _erf_approx(x):
    """ Approximate erf function for use with numba

    Parameters
    ----------
    x: scalar
        value

    Returns
    -------
        Approximation of erf(x)
"""
    a1 = 0.0705230784
    a2 = 0.0422820123
    a3 = 0.0092705272
    a4 = 0.0001520143
    a5 = 0.0002765672
    a6 = 0.0000430638
    if x > 0:
        return 1. - np.power(1. + a1*x + a2*np.power(x,2.) + a3*np.power(x,3.) + a4*np.power(x,4.) + a5*np.power(x,5.) + a6*np.power(x,6.), -16.)
    else:
        return -1 + np.power(1. + a1*np.abs(x) + a2*np.power(np.abs(x),2.) + a3*np.power(np.abs(x),3.) + a4*np.power(np.abs(x),4.) + a5*np.power(np.abs(x),5.) + a6*np.power(np.abs(x),6.), -16.)

@njit
def _get_hybrid_stack(x0, y0,final_q, final_a, final_var, im_args):
    """ Wrapper Function used to calculate render model using the hybrid method

        Parameters
        ----------
        x0: float
            x position of center
        y0: float
            y position of the center
        final_q: Array
            Array of axis ratios
        final_a:
            Array of Gaussian Weights
        final_var:
            Array of Gassian widths, note this the variance so sig^2
        return_stack: Bool, optional
            If True returns an image for each individual gaussian
        Returns
        -------
        erf_model: array
            Array representing the model image, same shape as 'shape'
"""
    X_lg,Y_lg,_lg_fac_x,_lg_fac_y, shape = im_args
    X_use = X_lg - (x0 + _lg_fac_x)
    Y_use = Y_lg - (y0 + _lg_fac_y)
    stack_full = np.zeros((X_use.shape[0],X_use.shape[1], len(final_q)))
    num_g = final_q.shape[0]

    for k in range(num_g):
        q,a,var = final_q[k],final_a[k],final_var[k]
        use_gauss = (var*q*q > 25.)
        R2_use = np.square(X_use) + np.square(Y_use)/(q*q)

        for i in range(X_use.shape[0]):
            for j in range(X_use.shape[1]):
                #If outside 5sigma then keep as 0
                if R2_use[i,j]/var > 25.:
                    continue
                elif use_gauss:
                    #If sigma>5 no benefit to using erf so go with quicker simple calc
                    stack_full[i,j,k] = a / (2*np.pi*var * q) * np.exp( -1.*(X_use[i,j]*X_use[i,j] + Y_use[i,j]*Y_use[i,j]/(q*q) )/ (2*var) )
                else:
                    c_x = 1./(np.sqrt(2*var))
                    c_y = 1./(np.sqrt(2*var)*q)
                    stack_full[i,j,k] = a/4 *( ( _erf_approx(c_x*(X_use[i,j]-0.5)) - _erf_approx(c_x*(X_use[i,j]+0.5)) )* ( _erf_approx(c_y*(Y_use[i,j] -0.5)) - _erf_approx(c_y*(Y_use[i,j]+0.5)) ) )

    return stack_full
