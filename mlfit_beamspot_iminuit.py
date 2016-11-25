#!/usr/bin/python

import sympy
import iminuit
import numpy as np
from scipy.stats import multivariate_normal

class MultivariateGaussianFitterNLL():
    '''
    Fit 3D gaussian cloud.
    '''
    def __init__(self, events, verbose=False):
        self.events    = events
        self.verbose   = verbose # should use a logger...
      
    @staticmethod
    def _compute_covariance_matrix(theta_x, theta_y, theta_z, sigma_x, sigma_y, sigma_z):
        '''
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function
        '''
  
        rot_x = np.array([
            [ 1., 0.             ,  0.             ],
            [ 0., np.cos(theta_x), -np.sin(theta_x)],
            [ 0., np.sin(theta_x),  np.cos(theta_x)],
        ]).astype(np.float64)
        
        rot_y = np.array([
            [  np.cos(theta_y), 0., np.sin(theta_y)],
            [  0.             , 1., 0.             ],
            [ -np.sin(theta_y), 0., np.cos(theta_y)],
        ]).astype(np.float64)

        rot_z = np.array([
            [ np.cos(theta_z), -np.sin(theta_z), 0.],
            [ np.sin(theta_z),  np.cos(theta_z), 0.],
            [ 0.             ,  0.             , 1.],
        ]).astype(np.float64)
        
        widths = np.array([
            [ np.power(sigma_x, 2), 0.                  , 0.                  ],
            [ 0.                  , np.power(sigma_y, 2), 0.                  ],
            [ 0.                  , 0.                  , np.power(sigma_z, 2)],
        ]).astype(np.float64)
        
        M_rot_x  = sympy.Matrix(rot_x)
        M_rot_y  = sympy.Matrix(rot_y)
        M_rot_z  = sympy.Matrix(rot_z)
        M_widths = sympy.Matrix(widths)
        cov = (M_rot_x * (M_rot_y * (M_rot_z * widths * M_rot_z.T) * M_rot_y.T) * M_rot_x.T)
                
        return cov
                
    def nll(self, x, y, z, theta_x, theta_y, theta_z, sigma_x, sigma_y, sigma_z):
        '''
        '''
        
        if self.verbose:
            print '\n=========='
            print 'x      :\t', x      , '[cm]'
            print 'y      :\t', y      , '[cm]'
            print 'z      :\t', z      , '[cm]'
            print 'theta_x:\t', theta_x, '[rad]'
            print 'theta_y:\t', theta_y, '[rad]'
            print 'theta_z:\t', theta_z, '[rad]'
            print 'sigma x:\t', sigma_x, '[cm]'
            print 'sigma y:\t', sigma_y, '[cm]'
            print 'sigma z:\t', sigma_z, '[cm]'

        cov = self._compute_covariance_matrix(theta_x, theta_y, theta_z, sigma_x, sigma_y, sigma_z)
        
        if self.verbose:
            print 'covariance matrix', np.matrix(cov)
            print 'determinant: ', cov.det()
        
        # check singularity / inveritbility
        if cov.det() > 1.e-14:
            nll = -multivariate_normal.logpdf(self.events,
                                              mean=np.array([x, y, z]).astype(np.float64),
                                              cov=np.matrix(cov)).astype(np.float64).sum().astype(np.float64)
        else:
            print 'WARNING! Singular covariance matrix, cannot invert!'
            return float('nan')

        if self.verbose:
            print 'nLL: ', nll
        
        return nll



if __name__ == '__main__':

    # ---------- GENERATE EVENTS -----------
    # generate events with somewhat realistic parameters
    ntoys = 10000
    
    # centroid position
    pos = np.array([0.067, 0.109, .805,])
    
    # build the covariance matrix from angles and widths,
    # easier to read
    cov = MultivariateGaussianFitterNLL._compute_covariance_matrix(
        theta_x=170.e-6, 
        theta_y=170.e-6, 
        theta_z=0., 
        sigma_x=2.e-3, 
        sigma_y=2.e-3, 
        sigma_z=4.
    )
    
    # convert the covaraince matrix into a numpy array
    cov = np.array(cov).astype(np.float)                 
    
    # fix random seed
    rng = np.random.RandomState(1986)
    
    # generate multivariate normal
    mvg = rng.multivariate_normal(pos, cov, ntoys)

    print 'generated %d toys' %ntoys
    
    # create a multivariate gaussian likelihood for the given dataset
    bs = MultivariateGaussianFitterNLL(mvg, verbose=False)

    print 'find positions only'
    # instantiate a minimizer, fix all but the positions
    minimizer_pos = iminuit.Minuit(
        bs.nll,
        pedantic=False,
        x=0.,
        y=0.,
        z=0.,
        theta_x=0.,
        theta_y=0.,
        theta_z=0.,
        sigma_x=0.002,
        sigma_y=0.002,
        sigma_z=4.,
        fix_theta_x=True,      
        fix_theta_y=True,      
        fix_theta_z=True,      
        fix_sigma_x=True,      
        fix_sigma_y=True,      
        fix_sigma_z=True,      
    ) 
    
    # make it NOT verbose
    minimizer_pos.set_print_level(-1)
    
    # run the minimization            
    minimizer_pos.migrad()        

    print 'find sigmas only'
    # instantiate a minimizer, fix all the positions to the values found before, relax widths
    minimizer_sigma = iminuit.Minuit(
        bs.nll,
        pedantic=False,
        x=minimizer_pos.values['x'],
        y=minimizer_pos.values['y'],
        z=minimizer_pos.values['z'],
        theta_x=0.,
        theta_y=0.,
        theta_z=0.,
        sigma_x=0.002,
        sigma_y=0.002,
        sigma_z=4.,
        fix_x=True,      
        fix_y=True,      
        fix_z=True,      
        fix_theta_x=True,      
        fix_theta_y=True,      
        fix_theta_z=True,      
    ) 
    
    # run the minimization            
    minimizer_sigma.migrad()        

    print 'find tilts only'
    # instantiate a minimizer, fix all but the thetas to the values found before
    minimizer_tilt = iminuit.Minuit(
        bs.nll,
        pedantic=False,
        x=minimizer_pos.values['x'],
        y=minimizer_pos.values['y'],
        z=minimizer_pos.values['z'],
        theta_x=0.,
        theta_y=0.,
        theta_z=0.,
        sigma_x=minimizer_sigma.values['sigma_x'],
        sigma_y=minimizer_sigma.values['sigma_y'],
        sigma_z=minimizer_sigma.values['sigma_z'],
        fix_x=True,      
        fix_y=True,      
        fix_z=True,      
        fix_theta_z=True, # no tilt along the z axis   
        fix_sigma_x=True,      
        fix_sigma_y=True,      
        fix_sigma_z=True,      
    ) 
        
    # run the minimization            
    minimizer_tilt.migrad()        

    print 'find all parameters'
    # instantiate a minimizer, all free
    minimizer_tot = iminuit.Minuit(
        bs.nll,
        pedantic=False,
        x=minimizer_pos.values['x'],
        y=minimizer_pos.values['y'],
        z=minimizer_pos.values['z'],
        theta_x=minimizer_tilt.values['theta_x'],
        theta_y=minimizer_tilt.values['theta_y'],
        theta_z=0.,
        sigma_x=minimizer_sigma.values['sigma_x'],
        sigma_y=minimizer_sigma.values['sigma_y'],
        sigma_z=minimizer_sigma.values['sigma_z'],
        fix_theta_z=True, # no tilt along the z axis   
    ) 
    
    # run the minimization            
    minimizer_tot.migrad()        

    # print results
    print '\n========== FIT RESULTS ============'
    for k in ['x', 'y', 'z', 'theta_x', 'theta_y', 'theta_z', 
              'sigma_x', 'sigma_y', 'sigma_z']:
        print '%s:\t %.5f +/- %.6f [cm]' %(k, minimizer_tot.values[k], minimizer_tot.errors[k])

