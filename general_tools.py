import numpy as np
import matplotlib.pyplot as plt

def center_mass(
    coords,
    mass):
    
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
    Lsun=1):
    
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


def absolute_lum(
    M, 
    band=None, 
    Lsun = 1):
    
    '''
    Calculates the absolute magnitude of the given luminosity given. 
    Its can account for different units of L, and different bands.
    
    Parameters
    ----------
    M:    Luminosity in any units
    band: Using the bands associated with sun_abs_mag function
    
    Returns
    -------
    Luminosity in units Lsun
    
    Example
    -------
    
    '''

    if band is None:
        Msun = 4.83
    else:
        Msun = sun_abs_mag(band)
    
    return 10 ** ((M - Msun) / -2.5)

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
        return sun_abs_mag(band) + 21.572 - 2.5 * np.log10(sb_lum / 10**6)


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
        

def measure_surfbright(
    image, 
    FOV, 
    pixel=1000,
    major_axis=1,
    ellip=0, 
    theta=0,
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
    FOV:         float, Field of View, physical distance from the center of 
                    the galaxy to the edge of the image, often in kpc
    pixel:       int, number of pixels across the image
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
    
    pixel = len(image)
    mid_pixel_FOV = FOV - FOV / pixel
    kpc_per_pixel = (FOV / (pixel/2))**2 # area 
    
    # Create distance array, the same shape as the image
    # Used to mask image based on physical location
    #x_coord_kpc = np.linspace(-mid_pixel_FOV,mid_pixel_FOV,num=pixels)
    #x_coord_kpc = np.array([x_coord_kpc,]*pixels)
    #y_coord_kpc = np.linspace(-mid_pixel_FOV,mid_pixel_FOV,num=pixels) 
    #y_coord_kpc = np.array([y_coord_kpc,]*pixels).transpose()
    
    mid_pixel_FOV = FOV - FOV / pixel
    x, y = np.linspace(-mid_pixel_FOV, mid_pixel_FOV, pixel), np.linspace(-mid_pixel_FOV, mid_pixel_FOV, pixel)
    X, Y = np.meshgrid(x, y)
    
    
    if center_mass is None:
        center_mass = [0,0,0]
  
    if ellip > 0 :
        a, b = major_axis, (1 - ellip) * major_axis
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        X_rot = (X - center_mass[0]) * cos_theta + (Y - center_mass[1]) * sin_theta
        Y_rot = -(X - center_mass[0]) * sin_theta + (Y - center_mass[1]) * cos_theta
        z = np.sqrt((X_rot / a) ** 2 + (Y_rot / b) ** 2)
    else:
        z = None


    
    # creat arrays
    radius = np.linspace(0,FOV,num=nmeasure)     
    sum_light = np.zeros(len(radius))
    circle_area = np.zeros(len(radius))
 
    # assume magnitude limited image,
    # 0 contribution from pixels less the the limit
    mag_lim_mask =  image < sb_lim
    image[mag_lim_mask] = 0
    
    for i in range(len(radius)):
        # mask that grabs pixels within give physical radius
        if z is not None:
            #this makes the radius equivalent to the distance on the major axis
            #ex z = X_rot / a, so the distance within the major axis is X_rot = z * a
            rmask = (z * major_axis)**2  <  (radius[i] **2)
    
        else:
                     
            rmask = ((X-center_mass[0])**2 + (Y-center_mass[1])**2  <  radius[i]**2)
        # Sum of the Luminosity with radius

        sum_light[i] = np.sum(image[rmask] * kpc_per_pixel) 
        # Area of within radius
        circle_area[i] = np.pi * radius[i]**2 * (1 - ellip) 
    r = radius[1:]
    if return_type == 'cum_lum':
        return radius, sum_light
    
    # luminosity within annulus between radii
    light_shell = sum_light[1:]-sum_light[:-1]
    if return_type == 'shell_lum':
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
        if limit_type == 'param_fraction':
            mask = param_cum <= param_tot * i
        elif limit_type == 'radius_fraction':
            mask = radius <= rad_tot * i
        elif limit_type == 'param':
            mask = param_cum <= i
        elif limit_type == 'radius':
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


def obs_mass(
    lum, 
    mag_g, 
    mag_r):
     
    '''
    Calculates the estimated mass of the Galaxy using eq 3:
    https://iopscience.iop.org/article/10.3847/1538-4357/ac2581/pdf
    This is a color weighted mass to light ratio
    
    
    Parameters
    ----------
    lum: array_like, Total luminosity of galaxy, units Lsun
    mag_g, mag_r: array_like, Total mag in band g,r for galaxy
    
    Returns
    -------
    Mass: array_like, Estimated mass of the galaxy in Msun
    
    '''
    
    logMass_L =  1.774 * (mag_g - mag_r) - 0.783
    Mass = lum * 10 ** logMass_L
    return Mass


def re_from_mass(
    mass, 
    a,
    b,
    param_type='lange'):
    
    '''
    Calculates the estimated effective radius of the galaxy 
    based on mass and the mass-radius relationship used.
    Different a and b values used for different relationships.
    
    Two functionally forms of the relationship, 
    'local' is the form used in the local dwarfs paper eq 5:
    https://iopscience.iop.org/article/10.3847/1538-4357/ac2581/pdf
    
    'lange' is the form used in Lange et al. eq 2:
    https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1470L/abstract


    Parameters
    ----------
    mass: array_like, Measured mass of the galaxy, Msun
    a, b: float, relationship parameters found in the papers
    
    Returns
    -------
    re: array_like, estimated effective radius of galaxy, units kpc
    
    Example
    -------
    #Global trend from Lange
    a = 4.104
    b = 0.208
    re = re_from_mass(1e8,a,b)
    
    #Local Dwarf Trend 
    a = 1.077
    b = 0.246    
    re = re_from_mass(5e5,a,b,param_type='local')
    
    '''
    
    
    if param_type == 'local':
        # https://iopscience.iop.org/article/10.3847/1538-4357/ac2581/pdf
        # eq 5
        log_re_pc = a + b * np.log10(mass)
        return (10 ** log_re_pc) * 1e-3
    
    elif param_type == 'lange':
        # https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1470L/abstract
        # eq 2
        return a * ( mass * 1e-10) ** b




