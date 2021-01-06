import imcascade
from imcascade.utils import get_med_errors
import numpy as np
import asdf
from scipy.optimize import root_scalar

def calc_flux_input(weights,sig, cutoff = None):
    if cutoff == None:
        return np.sum(weights, axis = -1)
    else:
        return np.sum(weights*(1-np.exp(-1*cutoff**2/ (2*sig**2) ) ), axis = -1 )

def r_root_func(r,f_L, weights,sig,cutoff):
    return 1.- f_L - np.sum(weights*np.exp(-1 * r**2 / (2*sig**2)),axis = -1 ) / calc_flux_input(weights,sig,cutoff = cutoff)

vars_to_use = ['img', 'weight', 'mask', 'sig', 'Ndof', 'Ndof_sky', 'Ndof_gauss', 'has_psf', 'psf_a','psf_sig', 'log_weight_scale','min_param','sky_model', 'posterier', 'post_method']

class ImcascadeResults():
    """A class used for collating imcascade results and performing analysis"""
    def __init__(self, Obj, thin_posterier = 1):
        if type(Obj) == imcascade.fitter.Fitter:
            self.obj_type = 'class'
            dict_obj = vars(Obj)
        if type(Obj) == dict:
            self.obj_type = 'dict'
            dict_obj = Obj
        if type(Obj) == str:
            self.obj_type = 'file'
            file = asdf.open(Obj)
            dict_obj = file.tree

        self.input = Obj

        #To-do write better function to do this that can handle missing values
        for var_name in vars_to_use:
            try:
                setattr(self, var_name, dict_obj[var_name] )
            except:
                print ('Could not load -', var_name)

        if hasattr(self, 'posterier'):
            self.x0 = self.posterier[::int(thin_posterier),0]
            self.y0 = self.posterier[::int(thin_posterier),1]
            self.q = self.posterier[::int(thin_posterier),2]
            self.pa = self.posterier[::int(thin_posterier),3]
            self.weights = self.posterier[::int(thin_posterier),4:4+self.Ndof_gauss]

            if self.sky_model: self.sky_params = self.posterier[::int(thin_posterier),4+self.Ndof_gauss:]

        elif hasattr(self, 'min_param'):
            self.x0 = self.min_param[0]
            self.y0 = self.min_param[1]
            self.q = self.min_param[2]
            self.pa = self.min_param[3]
            self.weights = self.min_param[4:4+self.Ndof_gauss]
            if self.sky_model: self.sky_params = self.min_param[4+self.Ndof_gauss:]

        #Account for if weights are in log_scale
        if self.log_weight_scale:
            self.weights = 10**self.weights

    def calc_flux(self, cutoff = None):
        if cutoff == None:
            flux_cur = np.sum(self.weights, axis = -1)
        else:
            flux_cur - np.sum(self.weights*(1-np.exp(-1*cutoff**2/ (2*self.sig**2) ) ), axis = -1 )
        self.flux = flux_cur
        return flux_cur

    def calc_rX(self,X,cutoff = None):
        f_cur = lambda w: root_scalar(r_root_func, bracket = [0,5*np.max(self.sig)], args = (X/100.,w,self.sig,cutoff) ).root

        if self.weights.ndim == 1:
            return f_cur(self.weights)
        if self.weights.ndim == 2:
            return np.array(list(map(f_cur, self.weights)) )

    def calc_r90(self,cutoff = None):
        r90_cur = self.calc_rX(90.)
        self.r90 = r90_cur
        return r90_cur

    def calc_r80(self,cutoff = None):
        r80_cur = self.calc_rX(80.)
        self.r80 = r80_cur
        return r80_cur

    def calc_r50(self,cutoff = None):
        r50_cur = self.calc_rX(50.)
        self.r50 = r50_cur
        return r50_cur

    def calc_r20(self,cutoff = None):
        r20_cur = self.calc_rX(20.)
        self.r20 = r20_cur
        return r20_cur

    def calc_sbp(self,r, return_ind = False):
        #r needs to be an array to work with np.newaxis below
        if type(r) == float:
            r = np.array([r,])

        prof_all = self.weights/(2*np.pi*self.sig**2) * np.exp(-r[:,np.newaxis,np.newaxis]**2/ (2*self.sig**2))
        prof_all = prof_all.squeeze()

        if return_ind:
            return prof_all
        else:
            return np.sum(prof_all, axis = -1)

    def run_basic_analysis(self, zpt = None, cutoff = None, errp_lo = 16, errp_hi =84,\
      save_results = False, save_file = './imcascade_results.asdf'):

        #Run all basic calculations neccesary
        if not hasattr(self,'flux'): self.calc_flux(cutoff = cutoff)
        if not hasattr(self,'r20'): self.calc_r20(cutoff = cutoff)
        if not hasattr(self,'r50'): self.calc_r50(cutoff = cutoff)
        if not hasattr(self,'r80'): self.calc_r80(cutoff = cutoff)
        if not hasattr(self,'r90'): self.calc_r90(cutoff = cutoff)

        res_dict = {}
        if self.weights.ndim == 1:
            res_dict['flux'] = self.flux
            if zpt != None: res_dict['mag'] = -2.5*np.log10(self.flux) + zpt

            #Save All Radii
            res_dict['r20'] = self.r20
            res_dict['r50'] = self.r50
            res_dict['r80'] = self.r80
            res_dict['r90'] = self.r90

            #Save concentration indexes
            res_dict['C80_20'] = self.r80 / self.r20
            res_dict['C90_50'] = self.r90 / self.r50
        else:
            res_dict['flux'] = get_med_errors(self.flux,lo = errp_lo, hi = errp_hi)
            if zpt != None:
                res_dict['mag'] = get_med_errors(-2.5*np.log10(self.flux) + zpt,lo = errp_lo, hi = errp_hi)

            res_dict['r20'] = get_med_errors(self.r20,lo = errp_lo, hi = errp_hi)
            res_dict['r50'] = get_med_errors(self.r50,lo = errp_lo, hi = errp_hi)
            res_dict['r80'] = get_med_errors(self.r80,lo = errp_lo, hi = errp_hi)
            res_dict['r90'] = get_med_errors(self.r90,lo = errp_lo, hi = errp_hi)

            res_dict['C80_20'] = get_med_errors(self.r80 / self.r20,lo = errp_lo, hi = errp_hi)
            res_dict['C90_50'] = get_med_errors(self.r90 / self.r50,lo = errp_lo, hi = errp_hi)

            res_dict['x0'] = get_med_errors(self.x0,lo = errp_lo, hi = errp_hi)
            res_dict['y0'] = get_med_errors(self.y0,lo = errp_lo, hi = errp_hi)
            res_dict['q'] = get_med_errors(self.q,lo = errp_lo, hi = errp_hi)
            res_dict['pa'] = get_med_errors(self.pa,lo = errp_lo, hi = errp_hi)


        if save_results:
            if self.obj_type == 'file':
                input_asdf = asdf.open(self.input)
                input_asdf.tree.update(res_dict)
                input_asdf.write_to(self.input)
            else:
                dict_to_save = vars(self)
                dict_to_save.pop('input')
                dict_to_save.pop('obj_type')

                if hasattr(self, 'posterier'):
                    for key in ['flux','r20','r50','r80','r90','x0','y0','q','pa']:
                        dict_to_save[key+'_post'] = dict_to_save.pop(key)

                dict_to_save.update(res_dict)
                file = asdf.AsdfFile(dict_to_save)
                file.write_to(save_file)

        return res_dict
