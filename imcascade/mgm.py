import numpy as np
from scipy.special import erf
from scipy.ndimage import rotate

class MultiGaussModel():
    """A class used to generate models based series of Gaussians """

    def __init__(self, shape, sig, psf_sig, psf_a, verbose = True, \
      sky_model = True, render_mode = 'erf', log_weight_scale = True):
        """ Initialize a MultiGaussModel instance
        Paramaters
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
        if psf_sig is not None or psf_a is not None:
            self.psf_sig = psf_sig
            self.psf_var = psf_sig*psf_sig
            self.psf_a = psf_a
            self.has_psf = True
        else:
            if verbose: print('No PSF input, running in non-psf mode')
            self.has_psf = False
        
        self.shape = shape
        self.render_mode = render_mode
        self.log_weight_scale = log_weight_scale

        if not render_mode in ['gauss', 'erf']:
            if verbose: print("Incompatible render mode, must choose 'gauss' or 'erf'! Setting to 'erf'")
            self.render_mode = 'erf'

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
        if sky_model:
            self.Ndof_sky = 3
        else:
            self.Ndof_sky = 0

        self.Ndof_gauss = len(self.sig)

        self.Ndof = 4 + self.Ndof_gauss + self.Ndof_sky

    def get_prime_coord(self, args):
        """ Function used to calculate the prime coordiantes based on a given
        central position and position angle

        Parameters
        ----------
        args: (xo,y0,phi) (float,float,float)
            Tuple containing the location of the centre and position angle
        Returns
        -------
        Xp: array
            Array of primed X coordiantes, same shape as 'X'
        Yp: array
            Array of primed Y coordiantes, same shape as 'Y'

"""
        xo,yo, phi = args
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        X_centered = self.X - xo
        Y_centered = self.Y - yo
        Yp = X_centered * cos_phi + Y_centered * sin_phi
        Xp = -1*X_centered* sin_phi + Y_centered * cos_phi

        return Xp, Yp

    def get_gauss_stack(self, Xp,Yp, q_arr, a_arr,var_arr):
        """ Function used to calculate render model using the 'Gauss' method

        Parameters
        ----------
        Xp: array
            Array of primed X coordiantes
        Yp: array
            Array of primed Y coordiantes,
        q_arr: Array
            Array of axis ratios
        a_arr:
            Array of Gaussian Weights
        var_arr:
            Array of Gassian widths, note this the variance so sig^2

        Returns

        Gauss_model: array
                Array representing the model image, same shape as 'shape'
"""

        Rp_sq = (Xp*Xp)[:,:,None] + ((Yp*Yp)[:,:,None] / (q_arr*q_arr))

        gauss_stack = a_arr / (2*np.pi*var_arr * q_arr) * np.exp( -1.*Rp_sq/ (2*var_arr) )

        return gauss_stack.sum(axis = -1)

    def get_erf_stack(self,x0, y0, phi,final_q, final_a, final_var, return_stack = False):
        """ Function used to calculate render model using the 'erf' method
        Parameters
        ----------
        x0: float
            x position of center
        y0: float
            y position of the center
        phi: float
            Position angle
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
        X_use = self.X_lg[:,:,None] - (x0 + self._lg_fac_x)
        Y_use = self.Y_lg[:,:,None] - (y0 + self._lg_fac_y)
        c_x = 1./(np.sqrt(2*final_var))
        c_y = 1./(np.sqrt(2*final_var)*final_q)

        unrotated_stack = final_a/4.*( ( erf(c_x*(X_use-0.5)) - erf(c_x*(X_use+0.5)) )* ( erf(c_y*(Y_use-0.5)) - erf(c_y*(Y_use+0.5)) ) )

        if return_stack:
            im_lg = rotate(unrotated_stack, phi*180./np.pi,reshape = False)
            return im_lg[self._lg_fac_x:self._lg_fac_x + self.shape[0], self._lg_fac_y:self._lg_fac_y + self.shape[1], :] 
        unrotated_im = unrotated_stack.sum(axis = -1)
        im_lg = rotate(unrotated_im, phi*180./np.pi,reshape = False)
        return im_lg[self._lg_fac_x:self._lg_fac_x + self.shape[0], self._lg_fac_y:self._lg_fac_y + self.shape[1]] 

    def get_sky_model(self,args):
        """ Function used to calculate tilted-plane sky model
        Parameters
        ----------
        args: (a,b,c) (float,float,float)
            a: overall normalization
            b: slope in x direction
            c: slope in y direction

        Returns
        -------
        sky_model: Array
        Model for sky background based on given parameters, same shape as 'shape'
"""
        a,b,c = args

        return a + (self.X - self.x_mid)*b + (self.Y - self.y_mid)*c

    def make_model(self,param):
        """ Function to generate model image based on given paramters array.
        This version assumaes the gaussian weights are given in linear scale
        Paramaters
        ----------
        param: array
            1-D array containing all the Parameters
        Returns
        -------
        model_image: Array
            Generated model image
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
            final_q = q_in
            final_a = a_in
        else:
            final_var = (self.var + self.psf_var[:,None]).ravel()
            final_q = np.sqrt( (self.var*q_in*q_in+ self.psf_var[:,None]).ravel() / (final_var) )
            final_a = (a_in*self.psf_a[:,None]).ravel()


        if self.render_mode == 'gauss':
            Xp,Yp = self.get_prime_coord( (x0, y0, phi) )
            model_im = self.get_gauss_stack(Xp,Yp, final_q, final_a, final_var)

        if self.render_mode == 'erf':
            model_im = self.get_erf_stack(x0, y0, phi,final_q, final_a, final_var)

        if not self.sky_model:
            return model_im
        else:
            return model_im + self.get_sky_model(param[-self.Ndof_sky:])
