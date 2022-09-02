import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling.models import Sersic1D,Sersic2D
from scipy.optimize import curve_fit


def sersic(
    r,
    amplitude,
    r_eff,
    n):
    
    '''
    Calculates the Surface Brightness (SB) 
    at a given radius r for a sersic profile
    (Can be used for anything, but for galaxy research it is SB)
    More info at: https://en.wikipedia.org/wiki/Sersic_profile
                  https://arxiv.org/pdf/astro-ph/0503176.pdf
    
    Parameters
    ----------
    r:         array_like, radius at which you are measuring the SB 
    amplitude: float, Effective Intensity
                SB at r_eff 
    r_eff:     float, Effective Radius
                radius that encoses half the light, same unit at r
    n:         Sersic index, unitless
                for the bn appoximation you need 0.5 < n < 10
    
    Returns
    -------
    Sersic Value at r: array_like, shape(r), same unit as amplitude
    '''
    
    bn = 2*n - 0.327 # approximation for 0.5 < n < 10
    return amplitude * np.exp ( -bn*( (r/r_eff)**(1/n) -1 ) )


def sersic1D_forfit(
    r,
    amplitude,
    r_eff,
    n):

    '''
    The Sersic1D creates a model for the given amp, r_eff, n
    To fit model using curve_fit to data, 
    the function needs a 1d array input (r)

    Parameters
    ----------
    r:      array_like, radii to calculate Sersic value
    kwargs: from Sersic1D

    Returns
    -------
    Sersic Value at r: array_like, shape(r), same unit as amplitude

    Example
    -------
    r = np.arange(0, 100, .01)
    sersic_value = sersic_1D_forfit(r,amplitude,r_eff,n)
    '''

    sersic_model = Sersic1D(amplitude=amplitude, 
                            r_eff=r_eff, 
                            n=n)

    return sersic_model(r)


def sersic2D_forfit(
    mesh1D,
    amplitude,
    r_eff,
    n,
    x_0,
    y_0,
    ellip,
    theta):
    
    '''
    The Sersic2D takes in a meshgrid to return an image. 
    This function takes in the compressed mesh so that 
    it can be fit with curve_fit

    Parameters
    ----------
    mesh1D: array_like, shape (2,pixel**2), 
                stacked the unraveled x and y values of the mesh (example below)
    kwargs: from Sersic2D

    Returns
    -------
    Sersic Value in each pixel: array_like, shape (pixel,pixel)
                the intensity value at each pixel 

    Example
    -------
    How to set up the 1D mesh grid
    x, y = np.linspace(-FOV, FOV, pixel), np.linspace(-FOV, FOV, pixel)
    X, Y = np.meshgrid(x, y)
    mesh1D = np.vstack((X.ravel(), Y.ravel()))
    '''

    sersic_model = Sersic2D(amplitude=amplitude, 
                            r_eff=r_eff,
                            n=n,
                            x_0=x_0, 
                            y_0=y_0, 
                            ellip=ellip, 
                            theta=theta)


    x, y = mesh1D
    return sersic_model(x, y)


def fit_sersic(
    r=None,
    sb=None,
    image=None,
    pixel=1000,
    FOV=20,
    sersic_type='sersic',
    p0=None,
    ax_sersic=None):
    
    '''
    Fits a Sersic SB Profile to input image and returns the 
    best fit vales for the sersit fit. If ax_sersic is specified,
    function returns loglog plot of data and sersic fit. 
    
    Parameters
    ----------
    r:          array_like, radii where SB is measured
                    None if doing 2D
    sb:         array_like, SB associated with radius r
                    None if doing 2D
    image:      array_like, shape (pixel,pixel), intesity in each pixel,
                    None if doing 1D
    sersic_type: str, which sersic function to fit, 
                    options: 'sersic', 'sersic1D', 'sersic2D'
    p0:         array_like, len is 3 or 6, guess parameters 
                    1d: [amplitude,r_eff,n]
                    2d: [amplitude,r_eff,n,x_0,y_0,ellip,theta]
    ax_sersic:  plot axis, ex: plt.gca()
    
    Returns
    -------
    popt,std: best fit sersic parameters and the standard deviation
                Uses units: r [kpc], sb [Lsun/kpc^2]
    '''
  
    # p0 is guess for paprameters, should not change outcome
    # [amplitude,r_eff,n]
    
    if p0 is None:
        if sersic_type == 'sersic2D':
            p0 = [10**6,1,0.7,0,0,0,0]
        else:
            p0 = [10**6,1,0.7]

    if sersic_type == 'sersic':
        popt, pcov = curve_fit(sersic, r, sb, p0=p0)
        std = np.sqrt(np.diag(pcov))  
        
    if sersic_type == 'sersic1D':        
        popt, pcov = curve_fit(sersic1D_forfit, r, sb, p0=p0)
        std = np.sqrt(np.diag(pcov))  
        
    if sersic_type == 'sersic2D':
        mid_pixel_FOV = FOV - FOV / pixel
        x, y = np.linspace(-mid_pixel_FOV, mid_pixel_FOV, pixel), np.linspace(-mid_pixel_FOV, mid_pixel_FOV, pixel)
        X, Y = np.meshgrid(x, y)
        mesh1D = np.vstack((X.ravel(), Y.ravel()))

        popt, pcov = curve_fit(sersic2D_forfit, mesh1D, image.ravel(), p0,
                               bounds=((0, 0, 0, -np.inf, -np.inf, 0, -np.inf),
                                       (np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf)))
        # this makes the angle given to be between 0,pi 
        # I tried doing this in the bounds of curve_fit and it messed it up
        popt[6] = popt[6]%(np.pi) 
        std = np.sqrt(np.diag(pcov))  
        
        
    if ax_sersic is not None:
        if sersic_type == 'sersic2D':
            sersic_model = Sersic2D(*popt)
            ax_sersic.imshow(np.log10(sersic_model(X,Y)+1),cmap='Greys_r')
            ax_sersic.text(75, 100, f'R$_e$: {popt[1]:.2f} kpc',color='white')
            ax_sersic.text(75, 200, f'n:  {popt[2]:.2f}',color='white')
            ax_sersic.get_xaxis().set_ticks([])
            ax_sersic.get_yaxis().set_ticks([])
            ax_sersic.get_figure().set_dpi(120) 
            
        else:         
            ax_sersic.loglog(r, sersic(r, *popt), label='Sersic',c='red')
            ax_sersic.loglog(r,sb,label='Data',c='k')
            ax_sersic.set_ylim(56000, np.max(sb)*1.1)
            ax_sersic.set_ylabel("'den' [L$_\odot$ kpc$^{-2}$]")
            ax_sersic.set_xlabel(" Radius [kpc]")
            ax_sersic.legend(frameon=False,loc=1)
            ax_sersic.text(r[0], 56000 +  np.log10(np.max(sb))/10*56000*2,f'R$_e$: {popt[1]:.2f} kpc',color='k')
            ax_sersic.text(r[0], 56000 +  np.log10(np.max(sb))/10*56000,f'n:  {popt[2]:.2f}',color='k')
            ax_sersic.get_figure().set_dpi(120)
    
    return popt, std

