import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def center_mass(coords,mass):
    '''
    Calculates the center of mass given coordinates and masses
    Works with any M dimensional coordinate system with N particles
    Does NOT like mass shape as (N,1)
    
    Parameters
    ----------
    coords: array_like, shape (N,M), coordinates, any unit
    mass:   array_like, shape (N,), mass, any unit
    
    Returns
    -------
    cm: array_like, shape (M,)
        center of mass in same units as coords
    
    Example
    -------
    If you have 3 seperate coords arrays use this set up:
    xcm, ycm, zcm = center_mass(np.transpose([xcoord,ycoord,zcoord]), mass)
    '''
    
    return mass.dot(coords) / np.sum(mass)


def absolute_mag(
    L, 
    band=None, 
    Lsun = 1):
    
    '''
    Calculates the absolute magnitude of the given luminosity given. 
    Its can account for different units of L, and different bands.
    
    Parameters
    ----------
    L:    Luminosity in any units
    band: Using the bands associated with sun_abs_mag function
    Lsun: Luminosity of the sun in the same units as L
            Lsun = 1 for L in solar lum units
            Lsun = 3.846e33 for L in ergs/s
    
    Returns
    -------
    Absolute_magnitude
    
    Example
    -------
    mag_andromeda_V = absolute_mag(2.6e10, band=8, Lsun=1)
    
    '''

    if band is None:
        Msun = 4.83
    else:
        Msun = sun_abs_mag(band)
    
    return Msun - 2.5 * np.log10(L / Lsun)


def lum_to_mag_SB(
    sb_lum, 
    band):
    '''
    Converts SB Lsun/kpc^2 to mag/arcsec^2 for a given filter band
    
    Parameters
    ----------
    sb_lum: array_like, SB Lsun/kpc^2 
    band:   index of band for sun_abs_mag
              Review sun_abs_mag documentation to identify 
              which index corresponds to which band
              Default value is 4.83 mag which corresponds to 
              
    Returns
    -------
    sb_mag: array_like, SB mag/arcsec^2
    '''
    
    if band is None:
        return 4.83 + 21.572 - 2.5 * np.log10(sb_lum / 10**6)
    else:
        sun_abs_mag(band) + 21.572 - 2.5 * np.log10(sb_lum / 10**6)


def mag_to_lum_SB(
    sb_mag, 
    band):
    '''
    Converts SB  mag/arcsec^2 to Lsun/kpc^2 for a given filter band
    
    Parameters
    ----------
    sb_mag: array_like, mag/arcsec^2
    band: index of band for sun_abs_mag
          Review sun_abs_mag documentation to identify 
          which index corresponds to which band
          Default value is 4.83 mag which corresponds to 
          
    
    Returns
    -------
    sb_mag: array_like, SB mag/arcsec^2
    '''
    
    if band is None:
        return 10 ** ((sb_mag - 4.83 - 21.572) / -2.5 + 6)
    else:
        10 ** ((sb_mag - sun_abs_mag(band) - 21.572) / -2.5 + 6)
        

def sersic(
    r,
    Re,
    Ie,
    n):
    '''
    Calculates the Surface Brightness (SB) 
    at a given radius r for a sersic profile 
    More info at: https://en.wikipedia.org/wiki/Sersic_profile
                  https://arxiv.org/pdf/astro-ph/0503176.pdf
    
    Parameters
    ----------
    r: radius at which you are measuring the SB 
    Re: Effective Radius, radius that encoses half the light, same unit at r
    Ie: Effective Intensity, SB at Re 
    n: Sersic index, unitless, for the bn appoximation you need 0.5 < n < 10
    
    Returns
    -------
    Surface Brightness (Intesntiy) at r, same unit as Ie
    '''
    
    bn = 2*n - 0.327 # approximation for 0.5 < n < 10
    return Ie * np.exp ( -bn*( (r/Re)**(1/n) -1 ) )


def measure_surfbright(
    image, 
    FOV, 
    center_mass=None, 
    nmeasure=100, 
    sb_lim=57650,
    return_type='surf_bright'):
    '''
    Calculates the azimuthally averaged SB at nmeasure different 
    radii equally spaced from the center of the galaxy to the FOV. 
     
    Parameters
    ----------
    image:       array_like, shape (N,N)
                    Image with NxN pixels, 
                    Each array value is the SB assoiated with that pixel     
    FOV:         Field of View, physical distance from the center of 
                    the galaxy to the edge of the image, often in kpc
    center_mass: Stellar center of mass of the galaxy.
                    Form of [xcm,ycm,zcm]. If None it assumes center of image [0,0,0] 
    nmeasure:    integer Number of radii where the SB is measured 
    sb_lim:      Impose a limit of observablility, units Lsun/kpc^2
                    Default of 57650 Lsun/kpc^2 = 29.5 mag, if no limit sb_lim=0
    return_type: str, 'surf_bright','shell_lum','cum_lum'
    
    Returns
    -------
    r,sb: returns an array of radii and the SB associated with them
        They are masked to only return the r,SB such that SB > sb_lim
    ''' 
    
    pixels = len(image)
    kpc_per_pixel = (FOV / (pixels/2))**2 # area 
    
    # Create distance array, the same shape as the image
    # Used to mask image based on physical location
    x_coord_kpc = np.linspace(-FOV,FOV,num=pixels)
    x_coord_kpc = np.array([x_coord_kpc,]*pixels)
    y_coord_kpc = np.linspace(FOV,-FOV,num=pixels) 
    y_coord_kpc = np.array([y_coord_kpc,]*pixels).transpose()
    
    # creat arrays
    radius = np.linspace(0,FOV,num=nmeasure)     
    sum_light = np.zeros(len(radius))
    circle_area = np.zeros(len(radius))
    
    # assume magnitude limited image,
    # 0 contribution from pixels less the the limit
    mag_lim_mask =  image < sb_lim
    image[mag_lim_mask] = 0
    
    if center_mass is None:
        center_mass = [0,0,0]
    
    for i in range(len(radius)):
        # mask that grabs pixels within give physical radius
        rmask = ((x_coord_kpc-center_mass[0])**2 + (y_coord_kpc-center_mass[1])**2  <  radius[i]**2)
        # Sum of the Luminosity with radius
        sum_light[i] = np.sum(image[rmask] * kpc_per_pixel) 
        # Area of within radius
        circle_area[i] = np.pi * radius[i]**2
    r = radius[1:]
    if return_type is 'cum_lum':
        return radius, sum_light
    
    # luminosity within annulus between radii
    light_shell = sum_light[1:]-sum_light[:-1]
    if return_type is 'shell_lum':
        return r, light_shell
    
    # Area within annulus between radii
    shell_area = circle_area[1:]-circle_area[:-1]
    # Average SB for a given annulus
    light_tot_shell = light_shell / shell_area
    # Mask out when ave SB drops below limit
    light_tot_shell_mask = light_tot_shell > sb_lim
    
    return r[light_tot_shell_mask], light_tot_shell[light_tot_shell_mask]


def measure_surfmass(
    star_coords, 
    star_mass, 
    FOV, 
    centermass=True, 
    nmeasure=100):
    '''
    Calculates the azimuthally averaged surface mass (SM) at nmeasure different 
    radii equally spaced from the center of the galaxy to the FOV. 
     
    Parameters
    ----------
    star_coords: array_like, shape (N,N)
    star_mass: array_like
    FOV: Field of View, physical distance from the center of 
        the galaxy to the edge of the image, often in kpc
    center_mass: Stellar center of mass of the galaxy.
        Form of [xcm,ycm,zcm]. If None it assumes center of is [0,0,0] 
    nmeasure: integer Number of radii where the SM is measured 
     
    Returns
    -------
    r,m: returns an array of radii and the SM associated with them
        
    Example
    -------
    measure_surfmass(np.transpose([xcoord,ycoord,zcoord]), mass)
    ''' 
    # creat arrays
    radius = np.linspace(0,FOV,num=nmeasure)     
    sum_mass = np.zeros(len(radius))
    circle_area = np.zeros(len(radius))
    
    if centermass is True:
        cm_coord = center_mass(np.transpose([star_coords[0],star_coords[1],star_coords[2]]), star_mass)    
    for i in range(len(radius)):
        # mask that grabs pixels within give physical radius
        rmask = ((star_coords[0]-cm_coord[0])**2 + (star_coords[1]-cm_coord[1])**2  <  radius[i]**2)
        # Sum of the Luminosity with radius
        sum_mass[i] = np.sum(star_mass[rmask]) 
        # Area of within radius
        circle_area[i] = np.pi * radius[i]**2
    
    # luminosity within annulus between radii
    mass_shell = sum_mass[1:]-sum_mass[:-1]
    # Area within annulus between radii
    shell_area = circle_area[1:]-circle_area[:-1]
    # Average SB for a given annulus
    mass_tot_shell = mass_shell / shell_area
    r = radius[1:]
    
    return r, mass_tot_shell


def fit_sersic(
    r, 
    sb, 
    ax_sersic=None):
    '''
    Fits a Sersic SB Profile to input data and returns the 
    best fit vales for the sersit fit. If ax_sersic is specified,
    function returns loglog plot of data and sersic fit. 
    
    Parameters
    ----------
    r: array_like, radii where SB is measured
    sb: array_like, SB associated with radius r
    ax_sersic: plot axis, ex: plt.gca()
    
    Returns
    -------
    Re,Ie,n,std: best fit sersic parameters and the standard deviation
    '''
  
    # p0 is guess for paprameters, should not change outcome
    popt, pcov = curve_fit(sersic, r, sb, p0=[1,10**6,0.7])
    Re,Ie,n = popt
    std = np.sqrt(np.diag(pcov))    
        
    if ax_sersic is not None:
        ax_sersic.loglog(r, sersic(r, *popt),label='Sersic',c='red')
        ax_sersic.loglog(r,sb,label='Data',c='k')
        ax_sersic.set_ylim(56000, np.max(sb)*1.1)
        ax_sersic.set_ylabel("'den' [L$_\odot$ kpc$^{-2}$]")
        ax_sersic.set_xlabel(" Radius [kpc]")
        ax_sersic.legend(frameon=False,loc=1)
        ax_sersic.text(r[0], 56000 +  np.log10(np.max(sb))/10*56000*2,f'R$_e$: {Re:.2f} kpc',color='k')
        ax_sersic.text(r[0], 56000 +  np.log10(np.max(sb))/10*56000,f'n:  {n:.2f}',color='k')
        ax_sersic.get_figure().set_dpi(120)
    
    return Re,Ie,n,std


def radius_of_param_limit(
    radii, 
    parameter, 
    limits, 
    limit_type='param_fraction'):
    
    '''
    This function sorts the radii and the parameters in order of radius. It then does a 
    cumulative sum of the parameter for the particles in order of smallest to largest radius.
    It will reutrn the radius and cumulative parameter values for the limits set. Examples of 
    parameters to use are mass or light. You can use a 3d or 2d radius. You can change the 
    type of limit using the limit_type, refer to Parameters for examples.
    
    Parameters
    ----------
    radii:       array_like, the radius from the center to the particle
    parameter:   array_like, the parameter value of the partice, ex: mass or luminosity
    limits:      list_like, list of values to evaluate, the value depends on the limit_type
    limit_type: 'param_fraction': returns radius that cantains the given fraction of the parameter
                                    ex: radius that contains 50% of the total mass
                'radius_fraction': returns parameter that cantains the given fraction of the radius 
                                    ex: the mass that is contained in half the total radius
                'param': returns the radius that conatins the param value, 
                                    ex: the radius that contains a luminosity of 1e10 Lsun
                'radius': returns the param value that is contained within the specificed radius 
                                    ex: the luminosity within 1 kpc
                
    
    Returns
    -------
    radius_measure: the radius at the specificed limits
    param_measure:  the cumulative parameter value of the specified limits
    
    Example
    -------
    find the half light radius, and the 90% light radius in the xy projection
    this would be the limit_type='param_fraction' with a limit =[.5,.9]
     
    radius_measure, param_measure = radius_of_param_limit(star_snapdict['r_xy'],
                                                          lums[0],
                                                          limits=[.5,.9],
                                                          limit_type='param_fraction'
                                                          )
    
    find the mass with a 3d 5 kpc radius
    this would be the limit_type='radius' with a limit =[5]
     
    radius_measure, param_measure = radius_of_param_limit(star_snapdict['r'],
                                                          star_snapdict['Masses'],
                                                          limits=[5],
                                                          limit_type='radius'
                                                          )
    '''
    
    
    sort_index = np.argsort(radii)
    
    radius = radii[sort_index]
    param_sort = parameter[sort_index]
    
    param_cum = np.cumsum(param_sort)
    param_tot = param_cum[-1]
    rad_tot = radius[-1]

    param_measure = []
    radius_measure = []
    
    for i in limits:
        if limit_type is 'param_fraction':
            mask = param_cum <= param_tot * i
        elif limit_type is 'radius_fraction':
            mask = radius <= rad_tot * i
        elif limit_type is 'param':
            mask = param_cum <= i
        elif limit_type is 'radius':
            mask = radius <= i
        
                
        p = param_cum[mask][-1]
        param_measure.append(p)
        
        r = radius[mask][-1]
        radius_measure.append(r)
        
    return radius_measure, param_measure

   
def sun_abs_mag(bands):
    '''
    Gives you the sun's absolute magnitude for a given filter band 
    Band indices are: 
    0  - Absolute mag
    1  - SDSS u (unprimed AB)
    2  - SDSS g (unprimed AB)
    3  - SDSS r (unprimed AB)
    4  - SDSS i (unprimed AB)
    5  - SDSS z (unprimed AB)
    6  - U (BESSEL)
    7  - B (BESSEL)
    8  - V (BESSEL)
    9  - R (KPNO)
    10 - I (KPNO)
    11 - J (BESSEL)
    12 - H (BESSEL)
    13 - K (BESSEL)
    
    Parameters
    ----------
    bands: array_like, the index corresponding to the filters wanted
    
    Returns
    -------
    mag_sun_ab: array_like, returns the sun's absolute magnitude for the given filters
    
    '''
    
    N_BANDS=14
    mag_sun_ab = np.zeros(N_BANDS,dtype=float)
    mag_sun_ab[0] = 4.74  # Bolemetric Mag 
    mag_sun_ab[1] = 6.75  #SDSS u (unprimed AB)
    mag_sun_ab[2] = 5.33  #SDSS g (unprimed AB)
    mag_sun_ab[3] = 4.67  #SDSS r (unprimed AB)
    mag_sun_ab[4] = 4.48  #SDSS i (unprimed AB)
    mag_sun_ab[5] = 4.42  #SDSS z (unprimed AB)
    mag_sun_ab[6] = 6.34  #U (BESSEL)
    mag_sun_ab[7] = 5.33  #B (BESSEL)
    mag_sun_ab[8] = 4.81  #V (BESSEL) 
    mag_sun_ab[9] = 4.65  #R (KPNO)
    mag_sun_ab[10] = 4.55 #I (KPNO)    
    mag_sun_ab[11] = 4.57 #J (BESSEL)
    mag_sun_ab[12] = 4.71 #H (BESSEL)  
    mag_sun_ab[13] = 5.19 #K (BESSEL)

    return mag_sun_ab[bands]

