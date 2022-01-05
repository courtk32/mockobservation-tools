#This file makes use of packages written by Alex Gurvich 
#To import FIRE_studio and abg_python go to https://github.com/agurvich


import numpy as np
import pandas as pd
import h5py 
import astropy
import copy
import glob

#from firestudio.utils.stellar_utils.colors_sps import lum_mag_conversions
from firestudio.utils.stellar_utils.load_stellar_hsml import get_particle_hsml

from firestudio.utils.stellar_utils import raytrace_projection
from firestudio.studios.star_studio import raytrace_ugr_attenuation
import firestudio.utils.stellar_utils.make_threeband_image as makethreepic

from astropy.cosmology import Planck13
from general_tools import center_mass,lum_to_mag_SB



def load_sim(pathtofolder, snapshot):
    '''
    Loads one snap shot of the FIRE Simulations
    
    Parameters
    ----------
    pathtofolder: str, path to the snapshot simulation directory, ex snapdir_600/
    nfiles:       int, number of files in the snapdir
    snapshot:     int, which snapshot you are loading
    
    Returns
    -------
    gas_snapdict: h,a,x,y,z,Masses,Metallicity,hsml,ParticleIDs
    star_snapdict:h,a,x,y,z,Masses,Metallicity,hsml,ParticleIDs,StellarFormationTime[scalefactor],StellarAge[Gyr] 
    
    units are:  positions [physical kpc], masses [M_sun], 
                Metallicity "total" metal mass (everything not H, He) [mass fraction – so “solar” would be ~0.02] 
    
    Example
    -------
    pathtofolder = '/data17/grenache/aalazar/FIRE/GVB/m12b_res7100/output/hdf5/snapdir_600/'
    star_snapdict, gas_snapdict = load_sim(pathtofolder,8,600)

    '''

    gas_snapdict  = {"h":np.array([]),"a":np.array([]),
                     "x":np.array([]),"y":np.array([]),"z":np.array([]),
                     "Masses":np.array([]),"Metallicity":np.array([]),
                     "hsml":np.array([]), "ParticleIDs":np.array([])}
    star_snapdict = {"h":np.array([]),"a":np.array([]),
                     "x":np.array([]),"y":np.array([]),"z":np.array([]),
                     "Masses":np.array([]),"Metallicity":np.array([]),
                     "hsml":np.array([]), "ParticleIDs":np.array([]),
                     "StellarFormationTime":np.array([]), "StellarAge":np.array([])}

    snap_files = glob.glob(pathtofolder +'*.hdf5')

    #...Loop through each snap file
    for i in snap_files:
        f = h5py.File(i, 'r')

        h = f['Header'].attrs['HubbleParam'] 
        z_snap = f['Header'].attrs['Redshift']
        a_snap = 1/(1+z_snap)
        
        gas_snapdict["x"] = np.append(gas_snapdict["x"],f['PartType0']['Coordinates'][:,0]*a_snap/h)
        gas_snapdict["y"] = np.append(gas_snapdict["y"],f['PartType0']['Coordinates'][:,1]*a_snap/h)
        gas_snapdict["z"] = np.append(gas_snapdict["z"],f['PartType0']['Coordinates'][:,2]*a_snap/h)
        
        gas_snapdict["Masses"]      = np.append(gas_snapdict["Masses"],f['PartType0']['Masses'][:]*(10**10)/h)
        gas_snapdict["Metallicity"] = np.append(gas_snapdict["Metallicity"],f['PartType0']['Metallicity'][:,0])
        gas_snapdict["hsml"]        = np.append(gas_snapdict["hsml"],f['PartType0']['SmoothingLength'][:])
        gas_snapdict["ParticleIDs"] = np.append(gas_snapdict["ParticleIDs"],f['PartType0']['ParticleIDs'][:])
        
        
        star_snapdict["x"] = np.append(star_snapdict["x"],f['PartType4']['Coordinates'][:,0]*a_snap/h)
        star_snapdict["y"] = np.append(star_snapdict["y"],f['PartType4']['Coordinates'][:,1]*a_snap/h)
        star_snapdict["z"] = np.append(star_snapdict["z"],f['PartType4']['Coordinates'][:,2]*a_snap/h)
        
        star_snapdict["Masses"] =      np.append(star_snapdict["Masses"],f['PartType4']['Masses'][:]*(10**10)/h)
        star_snapdict["Metallicity"] = np.append(star_snapdict["Metallicity"],f['PartType4']['Metallicity'][:,0])
        star_snapdict["StellarFormationTime"] = np.append(star_snapdict["StellarFormationTime"],f['PartType4']['StellarFormationTime'][:])
        star_snapdict["ParticleIDs"] = np.append(star_snapdict["ParticleIDs"],f['PartType4']['ParticleIDs'][:])

        f.close()
    
    z_form = 1/star_snapdict['StellarFormationTime'] - 1
    star_snapdict["StellarAge"] = np.array( ( Planck13.lookback_time( z_form ) ) ) 
        
    star_snapdict["h"] = h
    star_snapdict["a"] = a_snap
    
    gas_snapdict["h"] = h
    gas_snapdict["a"] = a_snap
    
    
    return star_snapdict, gas_snapdict


def load_halo(pathtofolder, snapshot, host=True, filetype='ascii'):
    '''
    Load the halo files, there are several different configurations of file set up
    
    Parameters
    ----------
    pathtofolder: str, path to the snapshot halo directory, ex /halo/rockstar_dm/hdf5/
    host: bool, If true only returns the host values, otherwise it will return all halo data
    filetype: str,  ascii or hdf5
    
    Returns
    -------
    halo: id,x,y,z,mvir,Halfmass_Radius,mbound_vir,rvir
    
    units are:  positions/distance [comoving kpc], masses [M_sun], 
                
    Example
    -------
    pathtofolder = '/data17/grenache/aalazar/FIRE/GVB/m12b_res7100/halo/rockstar_dm/hdf5'
    halo = load_halo(pathtofolder, host=True)
    
    '''
    
    if filetype=='ascii':
        # sizes are in sim code, need the 1/h factor
        h = 0.710000
        halo_files = glob.glob(pathtofolder +'*.ascii')
        halo = pd.read_csv(halo_files[0], skiprows=np.arange(1,20),sep=' ')
        if len(halo_files) > 1:
            for i in halo_files[1:]:
                halo_hold = pd.read_csv(i, skiprows=np.arange(1,20),sep=' ')
                halo = halo.append(halo_hold)
        halo = halo[['#id', 'x','y','z', 'mvir', 'Halfmass_Radius','mbound_vir', 'rvir']]    
        halo = halo.rename(columns={'#id': 'id'})
        halo = halo.set_index('id')
        halo['x'],halo['y'],halo['z'] = halo['x']*1e3/h,halo['y']*1e3/h,halo['z']*1e3/h
        halo['Halfmass_Radius'], halo['rvir'] = halo['Halfmass_Radius']/h, halo['rvir']/h
        halo['mvir'], halo['mbound_vir'] = halo['mvir']/h, halo['mbound_vir']/h
        
        if host is True:
            host_mask = halo['mvir'] == np.max(halo['mvir'])
            return halo.loc[host_mask]
        else:
            return halo

    if filetype=='hdf5':
        # Sizes are loaded as comoving, no 1/h factor
        # I think masses are are in M_sun
        halo_load = h5py.File(pathtofolder +'halo_'+ str(snapshot) +'.hdf5')
        data = {'id':halo_load['id'][:],
                'x':halo_load['position'][:,0],
                'y':halo_load['position'][:,1],
                'z':halo_load['position'][:,2], 
                'mvir':halo_load['mass.vir'][:], 
                'Halfmass_Radius':halo_load['radius'][:] ,
                'mbound_vir':halo_load['mass.bound'][:], 
                'rvir':halo_load['radius'][:]
               }
        halo = pd.DataFrame(data)

        
        if host is True:
            host_index = halo_load['host.index'][0]
            return halo.loc[halo.index == host_index]
        else:
            return halo
    
    
def mask_sim_to_halo(pathtofolder, nfiles, snapshot, star_snapdict=None, gas_snapdict=None, host_halo=None, lim = True ):
    if star_snapdict is None:
        star_snapdict, gas_snapdict = load_sim(pathtofolder, nfiles, snapshot)
    if host_halo is None:
        host_halo = load_halo(pathtofolder, nfiles, host=True)

    host_halo['x'] = host_halo['x'] * star_snapdict['a']
    host_halo['y'] = host_halo['y'] * star_snapdict['a'] 
    host_halo['z'] = host_halo['z'] * star_snapdict['a'] 
    host_halo['Halfmass_Radius'] = host_halo['Halfmass_Radius'] * star_snapdict['a'] 
    host_halo['rvir'] = host_halo['rvir'] * star_snapdict['a'] 
    
    star_snapdict['x'] = star_snapdict['x'] - host_halo['x'].values[0]
    star_snapdict['y'] = star_snapdict['y'] - host_halo['y'].values[0]
    star_snapdict['z'] = star_snapdict['z'] - host_halo['z'].values[0]
    star_snapdict['r'] = (star_snapdict['x']**2 + star_snapdict['y']**2 + star_snapdict['z']**2) ** 0.5
    star_snapdict['r_xy'] = (star_snapdict['x']**2 + star_snapdict['y']**2) ** 0.5
    star_snapdict['r_yz'] = (star_snapdict['y']**2 + star_snapdict['z']**2) ** 0.5
    star_snapdict['r_zx'] = (star_snapdict['x']**2 + star_snapdict['z']**2) ** 0.5

    gas_snapdict['x'] = gas_snapdict['x'] - host_halo['x'].values[0]
    gas_snapdict['y'] = gas_snapdict['y'] - host_halo['y'].values[0]
    gas_snapdict['z'] = gas_snapdict['z'] - host_halo['z'].values[0]
    gas_snapdict['r'] = (gas_snapdict['x']**2 + gas_snapdict['y']**2 + gas_snapdict['z']**2) ** 0.5
    gas_snapdict['r_xy'] = (gas_snapdict['x']**2 + gas_snapdict['y']**2) ** 0.5
    gas_snapdict['r_yz'] = (gas_snapdict['y']**2 + gas_snapdict['z']**2) ** 0.5
    gas_snapdict['r_zx'] = (gas_snapdict['x']**2 + gas_snapdict['z']**2) ** 0.5
    
    if lim is True:
        mask_star = star_snapdict['r'] < host_halo['Halfmass_Radius'].values[0]
        mask_gas = gas_snapdict['r'] < host_halo['Halfmass_Radius'].values[0]
        
        for key in ['x','y','z','r','r_xy','r_yz','r_zx','Masses','Metallicity','ParticleIDs','StellarFormationTime', 'StellarAge']:
            star_snapdict[key] = star_snapdict[key][mask_star]
        
        for key in ['x','y','z','r','r_xy','r_yz','r_zx','Masses','Metallicity','ParticleIDs','hsml']:
            gas_snapdict[key] = gas_snapdict[key][mask_gas]

        return star_snapdict, gas_snapdict, host_halo
    
    else:
        return star_snapdict, gas_snapdict, host_halo
    



def radius_mass_in_limit(star_snapdict,limits,rtype='r',fraction=None):
    
    sort_index = np.argsort(star_snapdict[rtype])
    radius = star_snapdict[rtype][sort_index]
    mass = star_snapdict['Masses'][sort_index]
    
    mass_cum = np.cumsum(mass)
    mass_tot = mass_cum[-1]
    rad_tot = radius[-1]

    mass_measure = []
    radius_measure = []
    
    for i in limits:
        if fraction is 'mass':
            mask = mass_cum <= mass_tot * i
        elif fraction is 'radius':
            mask = radius <= rad_tot * i
        else:
            mask = radius <= i
        
        m = mass_cum[mask][-1]
        mass_measure.append(m)
        
        r = radius[mask][-1]
        radius_measure.append(r)
        
    return radius_measure, mass_measure


def get_mock_observation(star_snapdict, gas_snapdict, bands=[1,2,3], 
                         FOV=25, pixels=500, view='xy', center = 'mass',
                         minden=57650, dynrange=52622824.0/57650, mass_scaler = 1 / 1e10, return_type='mock_image'):  
    '''
    mass_scaler: the code expects the masses to be in the simulation units of 1e10 Msun.
                    if the mass is in units Msun, then use mass_scaler = 1 / 1e10. If
                    it is in simulation units is mass_scaler = 1
    '''
        
    if len(star_snapdict['hsml']) == 0:
        star_hsml = get_particle_hsml(star_snapdict['x'],star_snapdict['y'],star_snapdict['z'])
        star_snapdict['hsml'] = np.append(star_snapdict['hsml'],star_hsml)
    
    # Make a copy so that it does not change the function outside of the function
    star_snapdict = star_snapdict.copy()
    gas_snapdict = gas_snapdict.copy()
    
    
    mask_star = star_snapdict['r'] < FOV
    mask_gas = gas_snapdict['r'] < FOV
        
    for key in ['x','y','z','Masses','Metallicity','StellarAge','hsml']:
        star_snapdict[key] = star_snapdict[key][mask_star]

    for key in ['x','y','z','Masses','Metallicity','hsml']:
        gas_snapdict[key] = gas_snapdict[key][mask_gas]

    
    
    kappas,lums = raytrace_projection.read_band_lums_from_tables(bands,
                                                                 star_snapdict['Masses']*mass_scaler,
                                                                 star_snapdict['StellarAge'],
                                                                 star_snapdict['Metallicity'])

    if view == 'xy':
        coords_stars = [star_snapdict['x'],star_snapdict['y'],star_snapdict['z']]  
        coords_gas = [gas_snapdict['x'],gas_snapdict['y'],gas_snapdict['z']]

    if view == 'yz':
        coords_stars = [star_snapdict['y'],star_snapdict['z'],star_snapdict['x']]  
        coords_gas = [gas_snapdict['y'],gas_snapdict['z'],gas_snapdict['x']]
        
    if view == 'zx':
        coords_stars = [star_snapdict['z'],star_snapdict['x'],star_snapdict['y']]  
        coords_gas = [gas_snapdict['z'],gas_snapdict['x'],gas_snapdict['y']]
    
    if center == 'mass':
        cm = center_mass(np.transpose([coords_stars[0],coords_stars[1],coords_stars[2]]),
                         star_snapdict['Masses'])
    if center == 'light':
        cm = center_mass(np.transpose([coords_stars[0],coords_stars[1],coords_stars[2]]),
                         lums[2,:])
        
    gas_out,out_band0,out_band1,out_band2 = raytrace_ugr_attenuation(coords_stars[0]-cm[0],
                                                                     coords_stars[1]-cm[1],
                                                                     coords_stars[2]-cm[2],
                                                                     star_snapdict['Masses']*mass_scaler,
                                                                     star_snapdict['StellarAge'],
                                                                     star_snapdict['Metallicity'],
                                                                     star_snapdict['hsml'],
                                                                     coords_gas[0]-cm[0],
                                                                     coords_gas[1]-cm[1],
                                                                     coords_gas[2]-cm[2],
                                                                     gas_snapdict['Masses']*mass_scaler,
                                                                     gas_snapdict['Metallicity'],
                                                                     gas_snapdict['hsml'],     
                                                                     kappas,
                                                                     lums,
                                                                     pixels=pixels,
                                                                     xlim = (-FOV,FOV),
                                                                     ylim = (-FOV,FOV),
                                                                     zlim = (-FOV,FOV)
                                                                    )
    
    
    if return_type is 'lum':
        unit_factor = 1e10 # gives units in Lsun
        return lums[0]*unit_factor, lums[1]*unit_factor, lums[2]*unit_factor

    
    elif return_type is 'lum_proj':
        unit_factor = 1e10 # gives units in Lsun        
        return out_band0*unit_factor, out_band1*unit_factor, out_band2*unit_factor
    
    
    elif return_type is 'SB_lum':
        unit_factor = 1e10/(2*FOV/pixels)**2 # gives units in Lsun kpc^-2
        return out_band0*unit_factor, out_band1*unit_factor, out_band2*unit_factor
    
    
    elif return_type is 'mag':
        unit_factor = 1e10 # gives units in Lsun
        mag0 = mag_sun_ab(bands[0]) - 2.5 * np.log10(lums[0] * unit_factor)
        mag1 = mag_sun_ab(bands[1]) - 2.5 * np.log10(lums[1] * unit_factor)
        mag2 = mag_sun_ab(bands[2]) - 2.5 * np.log10(lums[2] * unit_factor)
        return mag0, mag1, mag2
    
    
    elif return_type is 'mag_proj':
        unit_factor = 1e10 # gives units in Lsun
        mag0 = mag_sun_ab[bands[0]] - 2.5 * np.log10(out_band0 * unit_factor)
        mag1 = mag_sun_ab[bands[1]] - 2.5 * np.log10(out_band1 * unit_factor)
        mag2 = mag_sun_ab[bands[2]] - 2.5 * np.log10(out_band2 * unit_factor)
        return mag0, mag1, mag2
    
    
    elif return_type is 'SB_mag':
        unit_factor = 1e10/(2*FOV/pixels)**2 # gives units in Lsun kpc^-2        
        mag0 = lum_to_mag_SB(out_band0 * unit_factor)
        mag1 = lum_to_mag_SB(out_band1 * unit_factor)
        mag2 = lum_to_mag_SB(out_band2 * unit_factor)
        return mag0, mag1, mag2
    
    
    elif return_type is 'mock_image':
        unit_factor = 1e10/(2*FOV/pixels)**2
        out_band0,out_band1,out_band2 = out_band0*unit_factor, out_band1*unit_factor, out_band2*unit_factor
        image24, massmap = makethreepic.make_threeband_image_process_bandmaps(
                            copy.copy(out_band0),copy.copy(out_band1),copy.copy(out_band2),
                            maxden=dynrange * minden, dynrange=dynrange,
                            pixels=pixels)

        return np.rot90(image24,k=1,axes=(0,1)),np.rot90(out_band0,k=1,axes=(0,1)),np.rot90(out_band1,k=1,axes=(0,1)),np.rot90(out_band2,k=1,axes=(0,1))


def convert_kpc_to_arcsec(size_kpc,z):
    kpc_to_arcsec_converstion = astropy.cosmology.Planck13.arcsec_per_kpc_proper(z)
    conversion = size_kpc * kpc_to_arcsec_converstion
    return conversion.value
