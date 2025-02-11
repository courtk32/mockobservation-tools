
import numpy as np
import pandas as pd
import h5py 
import astropy
import copy
import glob

#import sys
#sys.path.insert(0,"/export/nfs0home/kleinca/fire_tools/FIRE_studio")
#sys.path.append("/nfspool-0/home/kleinca/myenviro/lib/python3.6/site-packages")
#sys.path.append("/export/nfs0home/kleinca/my_tools/mockobservation_tools")


from firestudio.utils.stellar_utils.load_stellar_hsml import get_particle_hsml
from firestudio.utils.stellar_utils import raytrace_projection
from firestudio.studios.star_studio import raytrace_ugr_attenuation
import firestudio.utils.stellar_utils.make_threeband_image as makethreepic
from abg_python.galaxy.cosmoExtractor import orientDiskFromSnapdicts,offsetRotateSnapshot
from abg_python.physics_utils import getTemperature

from astropy.cosmology import Planck13
from general_tools import center_mass, lum_to_mag_SB, sun_abs_mag


def load_sim(
    pathtofolder, 
    snapshot,
    dark_matter=False,
    gas_temp=False):
    
    '''
    Loads one snap shot of the FIRE Simulations
    It assumes all units are in simulation units
    
    Parameters
    ----------
    pathtofolder: str, path to the snapshot simulation directory, ex: snapdir_600/
    snapshot:     int, snapshot number you are loading
    
    Returns
    -------
    gas_snapdict: h,a,Coordinates,Velocities,Masses,Metallicity,hsml,ParticleIDs
    star_snapdict:h,a,Coordinates,Velocities,Masses,Metallicity,hsml,ParticleIDs,StellarFormationTime[scalefactor],StellarAge[Gyr] 
    
    units:  positions [physical kpc], masses [M_sun], 
            Metallicity 'total' metal mass (everything not H, He) [mass fraction – so “solar” would be ~0.02] 
    
    Example
    -------
    pathtofolder = '/data17/grenache/aalazar/FIRE/GVB/m12b_res7100/output/hdf5/snapdir_600/'
    star_snapdict, gas_snapdict = load_sim(pathtofolder,600)

    '''

    gas_snapdict  = {'h':np.empty(0),'a':np.empty(0),
                     'Coordinates':np.empty((0,3)),'Velocities':np.empty((0,3)),
                     'Masses':np.empty(0),'Metallicity':np.empty(0),
                     'hsml':np.empty(0), 'ParticleIDs':np.empty(0)}
    
    if gas_temp is True:
        gas_snapdict['Temperature'] = np.empty(0)
        gas_snapdict['InternalEnergy'] = np.empty(0)
        gas_snapdict['ElectronAbundance'] = np.empty(0)
        gas_snapdict['Metallicity_He'] = np.empty(0)
        
    
    star_snapdict = {'h':np.empty(0),'a':np.empty(0),
                     'Coordinates':np.empty((0,3)),'Velocities':np.empty((0,3)), 
                     'Masses':np.empty(0),'Metallicity':np.empty(0),
                     'hsml':np.empty(0), 'ParticleIDs':np.empty(0),
                     'StellarFormationTime':np.empty(0), 'StellarAge':np.empty(0)}
    
    if dark_matter is True:
        dark_snapdict = {'h':np.empty(0),'a':np.empty(0),
                        'Coordinates':np.empty((0,3)),'Velocities':np.empty((0,3)), 
                        'Masses':np.empty(0), 'ParticleIDs':np.empty(0)}
    

    snap_files = glob.glob(pathtofolder +'*.hdf5')

    #...Loop through each snap file
    for i in snap_files:
        f = h5py.File(i, 'r')

        h = f['Header'].attrs['HubbleParam'] 
        z_snap = f['Header'].attrs['Redshift']
        a_snap = 1/(1+z_snap)
        
        gas_snapdict['Coordinates'] = np.append(gas_snapdict['Coordinates'],
                                                np.array([f['PartType0']['Coordinates'][:,0]*a_snap/h,
                                                          f['PartType0']['Coordinates'][:,1]*a_snap/h,
                                                          f['PartType0']['Coordinates'][:,2]*a_snap/h
                                                         ]).T, axis=0)
        gas_snapdict['Velocities'] = np.append(gas_snapdict['Velocities'],
                                                 np.array([f['PartType0']['Velocities'][:,0],
                                                           f['PartType0']['Velocities'][:,1],
                                                           f['PartType0']['Velocities'][:,2]
                                                          ]).T, axis=0)
        
        gas_snapdict['Masses']      = np.append(gas_snapdict['Masses'],f['PartType0']['Masses'][:]*(10**10)/h)
        gas_snapdict['Metallicity'] = np.append(gas_snapdict['Metallicity'],f['PartType0']['Metallicity'][:,0])
        gas_snapdict['hsml']        = np.append(gas_snapdict['hsml'],f['PartType0']['SmoothingLength'][:])
        gas_snapdict['ParticleIDs'] = np.append(gas_snapdict['ParticleIDs'],f['PartType0']['ParticleIDs'][:])
        
        if gas_temp is True:
            gas_snapdict['InternalEnergy']   = np.append(gas_snapdict['InternalEnergy'],f['PartType0']['InternalEnergy'][:])
            gas_snapdict['ElectronAbundance']= np.append(gas_snapdict['ElectronAbundance'],f['PartType0']['ElectronAbundance'][:])
            gas_snapdict['Metallicity_He']   = np.append(gas_snapdict['Metallicity_He'],f['PartType0']['Metallicity'][:,1])
        
        
        star_snapdict['Coordinates'] = np.append(star_snapdict['Coordinates'],
                                                 np.array([f['PartType4']['Coordinates'][:,0]*a_snap/h,
                                                           f['PartType4']['Coordinates'][:,1]*a_snap/h,
                                                           f['PartType4']['Coordinates'][:,2]*a_snap/h
                                                          ]).T, axis=0)
        star_snapdict['Velocities'] = np.append(star_snapdict['Velocities'],
                                                 np.array([f['PartType4']['Velocities'][:,0],
                                                           f['PartType4']['Velocities'][:,1],
                                                           f['PartType4']['Velocities'][:,2]
                                                          ]).T, axis=0)
        
        star_snapdict['Masses'] =      np.append(star_snapdict['Masses'],f['PartType4']['Masses'][:]*(10**10)/h)
        star_snapdict['Metallicity'] = np.append(star_snapdict['Metallicity'],f['PartType4']['Metallicity'][:,0])
        star_snapdict['StellarFormationTime'] = np.append(star_snapdict['StellarFormationTime'],f['PartType4']['StellarFormationTime'][:])
        star_snapdict['ParticleIDs'] = np.append(star_snapdict['ParticleIDs'],f['PartType4']['ParticleIDs'][:])
        
        
        if dark_matter is True:
            dark_snapdict['Coordinates'] = np.append(dark_snapdict['Coordinates'],
                                                     np.array([f['PartType1']['Coordinates'][:,0]*a_snap/h,
                                                          f['PartType1']['Coordinates'][:,1]*a_snap/h,
                                                          f['PartType1']['Coordinates'][:,2]*a_snap/h
                                                         ]).T, axis=0)
            dark_snapdict['Velocities'] = np.append(dark_snapdict['Velocities'],
                                                    np.array([f['PartType1']['Velocities'][:,0],
                                                           f['PartType1']['Velocities'][:,1],
                                                           f['PartType1']['Velocities'][:,2]
                                                          ]).T, axis=0)
        
            dark_snapdict['Masses']      = np.append(dark_snapdict['Masses'],f['PartType1']['Masses'][:]*(10**10)/h)
            dark_snapdict['ParticleIDs'] = np.append(dark_snapdict['ParticleIDs'],f['PartType1']['ParticleIDs'][:])
            dark_snapdict['h'] = h
            dark_snapdict['a'] = a_snap
            
            

        
        

        f.close()
    
    z_form = 1/star_snapdict['StellarFormationTime'] - 1
    star_snapdict['StellarAge'] = np.array( ( Planck13.lookback_time( z_form ) ) ) 
        
    star_snapdict['h'] = h
    star_snapdict['a'] = a_snap
    
    gas_snapdict['h'] = h
    gas_snapdict['a'] = a_snap
    
    if gas_temp is True:
        gas_snapdict['Temperature'] = getTemperature(U_code = gas_snapdict['InternalEnergy'],
                                                     helium_mass_fraction = gas_snapdict['Metallicity_He'], 
                                                     ElectronAbundance = gas_snapdict['ElectronAbundance'])
    if dark_matter is True:
        return dark_snapdict, star_snapdict, gas_snapdict
    else:
        return star_snapdict, gas_snapdict



def load_halo(
    pathtofolder,
    snapshot, 
    host=True,
    filetype='ascii', 
    hostnumber=1):
    
    '''
    Load the halo files, there are several different configurations of file set up
    
    Parameters
    ----------
    pathtofolder: str, path to the snapshot halo directory, ex: /halo/rockstar_dm/hdf5/
    snapshot:     int, snapshot number you are loading
    host:         bool, If true only returns the host values, otherwise it will return all halo data
    filetype:     str, 'ascii' or 'hdf5'
    hostnumber:   int, use for the MW elvis pairs. hostnumber=2 to get the second galaxy info
    
    Returns
    -------
    halo: id,x,y,z,mvir,Halfmass_Radius,mbound_vir,rvir
    
    units are:  positions/distance [comoving kpc], masses [M_sun], 
                
    Example
    -------
    pathtofolder = '/data17/grenache/aalazar/FIRE/GVB/m10c_res500/halo/rockstar_dm/catalog/'
    halo = load_halo(pathtofolder, 184, filetype='ascii')
    
    pathtofolder = '/data17/grenache/aalazar/FIRE/GVB/m12_elvis_ThelmaLouise_res4000/halo/rockstar_dm/hdf5/'
    Thelma_halo = gal.load_halo(pathtofolder, 600, host=True, filetype='hdf5', halo=1)
    Louise_halo = gal.load_halo(pathtofolder, 600, host=True, filetype='hdf5', halo=2)
    
    '''
    
    if filetype=='ascii':
        # sizes are in sim code, need the 1/h factor
        h = 0.710000
        #halo_files = glob.glob(pathtofolder +'halos_'+str(snapshot)+'*.ascii')
        halo_files = glob.glob(pathtofolder +'halos_*.ascii')
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
            if hostnumber == 2:
                host_index = halo_load['host2.index'][0]
            else:
                host_index = halo_load['host.index'][0]
            return halo.loc[halo.index == host_index]
        
        else:
            return halo
    
    
def mask_sim_to_halo(
    star_snapdict, 
    gas_snapdict, 
    host_halo, 
    orient=True, 
    lim = True, 
    limvalue=None):
    
    '''
    This function will center the cordinates on the host halo and return the star and gas dictionaries
    If limits are given, then the function will mask out particles outside of the limit
    If orient is True then it will orient the particle toward the axis of angular momentum (xy plane is the disk)
    
    Parameters
    ----------
    star_snapdict: dict, The star particle dictionary from load_sim
    gas_snapdict:  dict, The gas particle dictionary from load_sim    
    host_halo:     dataframe, Halo info from load_halo with only host info
    orient:        bool, If true it will orient the coordinates based on the axis of net angular momentum
    lim:           bool, If true it will mask the star and gas dict within limvalue radius
    limvalue:      float, The physical radius in kpc that for which the particles will be masked out
                    if limvalue=None, then it will mask using a limit of the host Halfmass_Radius
    
    Returns
    -------
    star_snapdict, gas_snapdict, host_halo
    
    '''
    if limvalue is None:
        limvalue = host_halo['Halfmass_Radius'].values[0]
                                                
    host_halo['x'] = host_halo['x'] * star_snapdict['a']
    host_halo['y'] = host_halo['y'] * star_snapdict['a'] 
    host_halo['z'] = host_halo['z'] * star_snapdict['a'] 
    host_halo['Halfmass_Radius'] = host_halo['Halfmass_Radius'] * star_snapdict['a'] 
    host_halo['rvir'] = host_halo['rvir'] * star_snapdict['a'] 
    
    star_snapdict['Coordinates'] = star_snapdict['Coordinates'] - [host_halo['x'].values[0],host_halo['y'].values[0],host_halo['z'].values[0]]
    star_snapdict['r'] = (star_snapdict['Coordinates'][:,0]**2 + 
                          star_snapdict['Coordinates'][:,1]**2 + 
                          star_snapdict['Coordinates'][:,2]**2) ** 0.5
    star_snapdict['r_xy'] = (star_snapdict['Coordinates'][:,0]**2 + star_snapdict['Coordinates'][:,1]**2) ** 0.5
    star_snapdict['r_yz'] = (star_snapdict['Coordinates'][:,1]**2 + star_snapdict['Coordinates'][:,2]**2) ** 0.5
    star_snapdict['r_zx'] = (star_snapdict['Coordinates'][:,2]**2 + star_snapdict['Coordinates'][:,0]**2) ** 0.5
     
    if len(star_snapdict['hsml']) == 0:
        star_hsml = get_particle_hsml(star_snapdict['Coordinates'][:,0],
                                      star_snapdict['Coordinates'][:,1],
                                      star_snapdict['Coordinates'][:,2])
        star_snapdict['hsml'] = np.append(star_snapdict['hsml'],star_hsml)
    
    gas_snapdict['Coordinates'] = gas_snapdict['Coordinates'] - [host_halo['x'].values[0], host_halo['y'].values[0], host_halo['z'].values[0]]     
    gas_snapdict['r'] = (gas_snapdict['Coordinates'][:,0]**2 + 
                         gas_snapdict['Coordinates'][:,1]**2 + 
                         gas_snapdict['Coordinates'][:,2]**2) ** 0.5
    gas_snapdict['r_xy'] = (gas_snapdict['Coordinates'][:,0]**2 + gas_snapdict['Coordinates'][:,1]**2) ** 0.5
    gas_snapdict['r_yz'] = (gas_snapdict['Coordinates'][:,1]**2 + gas_snapdict['Coordinates'][:,2]**2) ** 0.5
    gas_snapdict['r_zx'] = (gas_snapdict['Coordinates'][:,2]**2 + gas_snapdict['Coordinates'][:,0]**2) ** 0.5

    if lim is True:
        mask_star = star_snapdict['r'] < limvalue
        mask_gas = gas_snapdict['r'] < limvalue
            
        for key in list(star_snapdict.keys())[2:]:
            star_snapdict[key] = star_snapdict[key][mask_star]
        
        for key in list(gas_snapdict.keys())[2:]: # no mask for a, h keys
            gas_snapdict[key] = gas_snapdict[key][mask_gas]
            
        if orient is True:    
            theta_TB,phi_TB,vscom = orientDiskFromSnapdicts(star_snapdict,gas_snapdict,limvalue,[0,0,0],orient_stars=True)
            star_snapdict = offsetRotateSnapshot(star_snapdict,[0,0,0],vscom,theta_TB,phi_TB,0)
            gas_snapdict = offsetRotateSnapshot(gas_snapdict,[0,0,0],vscom,theta_TB,phi_TB,0)
        
        return star_snapdict, gas_snapdict, host_halo
    
    else:
        if orient is True:
            theta_TB,phi_TB,vscom = orientDiskFromSnapdicts(star_snapdict,gas_snapdict,limvalue,[0,0,0],orient_stars=True)
            star_snapdict = offsetRotateSnapshot(star_snapdict,[0,0,0],vscom,theta_TB,phi_TB,0)
            gas_snapdict = offsetRotateSnapshot(gas_snapdict,[0,0,0],vscom,theta_TB,phi_TB,0)
        
        return star_snapdict, gas_snapdict, host_halo

    
    
def load_sim_FIREBox(
    obj_path,  
    ahf_path=None,
    mass_unit = 'simulation' ,
    length_unit = 'simulation', 
    star_keys=['stellar_x','stellar_y','stellar_z',
               'stellar_vx','stellar_vy','stellar_vz',
               'stellar_mass','stellar_metal_00','stellar_id','stellar_tform'],
    gas_keys=['stellar_x','stellar_y','stellar_z',
               'stellar_vx','stellar_vy','stellar_vz',
               'stellar_mass','stellar_metal_00','stellar_id','stellar_tform',''],
    gen_keys=['redshift','Mvir','Rvir'],
    orient=False):
    
    '''
    Loads FIREBox galaxies preped by Prof. Jorge Moreno
    Assumes file is in h5py
    Sets up data to be used for mock images
    Converts all units to physical units
    ahf_path contains particle ids. This will mask to only bound particles 
    There is star_keys and gas_keys which allowes you to input the keys used in the simulation data
        Different versions of the sims are saved with different keys. 
    
    Parameters
    ----------
    obj_path:    str, path to the galaxy object file
    ahf_path:    str, path to file that contains bound particle
    mass_unit:   str, physical/simulation units of the input particle masses, converts to physical
    length_unit: str, physical/simulation units of the input particle length, converts to physical 
    key_values:  Allows to adjust if there are different key values for the data
                        gas  coords, velocity, mass, metalicity, id, hsml
                        star coords, velocity, mass, metalicity, id, tform
                        Must be input exactly like the example.
    
                    default (consistent with Jorge initial run):  
                    gas_keys =  ['gas_x','gas_y','gas_z',
                                 'gas_vx','gas_vy','gas_vz',
                                 'gas_mass','gas_metal_00','gas_id', 'gas_hsml']
                    
                    star_keys = ['stellar_x','stellar_y','stellar_z',
                                 'stellar_vx','stellar_vy','stellar_vz',
                                 'stellar_mass','stellar_metal_00','stellar_id','stellar_tform']
                                 
                    gen_keys =  ['redshift','Mvir','Rvir']
                                 
                                 
                    Other version (consistent with Jorge 2nd run):   
                    gas_keys = ['gas_x','gas_y','gas_z',
                                'gas_vx','gas_vy','gas_vz',
                                'gas_mass','gas_total_metallicity','gas_id', 'gas_hsml']
                    
                    star_keys = ['stars_x','stars_y','stars_z',
                                 'stars_vx','stars_vy','stars_vz',
                                 'stars_mass','gas_total_metallicity','stars_id','stars_formation_time']
                                                     
                    gen_keys =  ['redshift','object_Mvir','object_Rvir']

    Returns
    -------
    star_snapdict, gas_snapdict
    
    Example
    -------
    #using objects_1200_original data
    obj_path = '/DFS-L/DATA/cosmo/jgmoren1/FIREBox/FB15N1024/objects_1200_original/particles_within_Rvir_object_0.hdf5'    
    ahf_path = '/DFS-L/DATA/cosmo/jgmoren1/FIREBox/FB15N1024/objects_1200_original/bound_particle_filters_object_0.hdf5'
    star_snapdict, gas_snapdict = load_sim_FIREBox(obj_path, ahf_path)
    
    
    #using old_objects_1200 data
    FB15N1024/old_objects_1200/object
    obj_path = '/DFS-L/DATA/cosmo/jgmoren1/FIREBox/FB15N1024/old_objects_1200/object_0.hdf5'    
    ahf_path = '/DFS-L/DATA/cosmo/jgmoren1/FIREBox/FB15N1024/old_objects_1200/object_0.hdf5'
    star_snapdict, gas_snapdict = load_sim_FIREBox(obj_path, ahf_path)
    
    
    '''
    
    #define input keys, easier to keep track
    gaskey_x,  gaskey_y,  gaskey_z  = [gas_keys[0],gas_keys[1],gas_keys[2]]
    gaskey_vx, gaskey_vy, gaskey_vz = [gas_keys[3],gas_keys[4],gas_keys[5]]
    gaskey_mass  = gas_keys[6]
    gaskey_metal = gas_keys[7]
    gaskey_id    = gas_keys[8]
    gaskey_hsml  = gas_keys[9]
        
    starkey_x,  starkey_y,  starkey_z  = [star_keys[0],star_keys[1],star_keys[2]]
    starkey_vx, starkey_vy, starkey_vz = [star_keys[3],star_keys[4],star_keys[5]]
    starkey_mass  = star_keys[6]
    starkey_metal = star_keys[7]
    starkey_id    = star_keys[8]
    starkey_tform = star_keys[9]
    
    genkey_redshift = gen_keys[0]
    genkey_Mvir = gen_keys[1]
    genkey_Rvir = gen_keys[2]
    
    
    f = h5py.File(obj_path, 'r')
        
        
    if ahf_path is None:
        
        # if no ahf then no mask, all elements are true
        gas_mask  = np.full(np.shape(f[gaskey_mass]), True)
        star_mask  = np.full(np.shape(f[starkey_mass]), True)

    else:
        ahf = h5py.File(ahf_path, 'r')

        #Gas
        gas_ids =  ahf['particleIDs'][ahf['partTypes'][:]==0]
        gas_mask = np.in1d(f[gaskey_id],gas_ids)

        #Stars
        star_ids = ahf['particleIDs'][ahf['partTypes'][:]==4]
        star_mask = np.in1d(f[starkey_id],star_ids)
    

    h = 0.6774
    z_snap = f[genkey_redshift][()]
    a_snap = 1/(1+z_snap)
    Mvir, Rvir = f[genkey_Mvir][()], f[genkey_Rvir][()]

        
    #Unit conversion 
    if mass_unit == 'physical':
        mass_conversion = (10**10)
    elif mass_unit == 'simulation':
        mass_conversion = (10**10)/h
  
    
    if length_unit == 'physical':
        length_conversion = 1
    elif length_unit == 'simulation':
        length_conversion = a_snap/h
        

    

    gas_snapdict  = {'h':np.empty(0),'a':np.empty(0), 'Rvir':np.empty(0), 'Mvir':np.empty(0),
                     'Coordinates':np.empty((0,3)),'Velocities':np.empty((0,3)),
                     'Masses':np.empty(0),'Metallicity':np.empty(0),
                     'hsml':np.empty(0), 'ParticleIDs':np.empty(0),
                     'r':np.empty(0),'r_xy':np.empty(0),'r_yz':np.empty(0),'r_zx':np.empty(0)}

    star_snapdict = {'h':np.empty(0),'a':np.empty(0), 'Rvir':np.empty(0), 'Mvir':np.empty(0),
                     'Coordinates':np.empty((0,3)),'Velocities':np.empty((0,3)), 
                     'Masses':np.empty(0),'Metallicity':np.empty(0),
                     'hsml':np.empty(0), 'ParticleIDs':np.empty(0),
                     'StellarFormationTime':np.empty(0), 'StellarAge':np.empty(0),
                     'r':np.empty(0),'r_xy':np.empty(0),'r_yz':np.empty(0),'r_zx':np.empty(0)}
    
   
    
    
    #Load Gas Particles
    gas_snapdict['h'] = h
    gas_snapdict['a'] = a_snap
    gas_snapdict['Rvir'] = Rvir
    gas_snapdict['Mvir'] = Mvir

    gas_snapdict['Coordinates'] = np.array([f[gaskey_x][gas_mask]*length_conversion,
                                            f[gaskey_y][gas_mask]*length_conversion,
                                            f[gaskey_z][gas_mask]*length_conversion
                                           ]).T
    
    gas_snapdict['r'] = (gas_snapdict['Coordinates'][:,0]**2 + 
                         gas_snapdict['Coordinates'][:,1]**2 + 
                         gas_snapdict['Coordinates'][:,2]**2) ** 0.5
    gas_snapdict['r_xy'] = (gas_snapdict['Coordinates'][:,0]**2 + gas_snapdict['Coordinates'][:,1]**2) ** 0.5
    gas_snapdict['r_yz'] = (gas_snapdict['Coordinates'][:,1]**2 + gas_snapdict['Coordinates'][:,2]**2) ** 0.5
    gas_snapdict['r_zx'] = (gas_snapdict['Coordinates'][:,2]**2 + gas_snapdict['Coordinates'][:,0]**2) ** 0.5
    
    
    gas_snapdict['Velocities'] = np.array([f[gaskey_vx][gas_mask],
                                           f[gaskey_vy][gas_mask],
                                           f[gaskey_vz][gas_mask]
                                          ]).T  

    gas_snapdict['Masses']      = f[gaskey_mass][gas_mask]*mass_conversion
    gas_snapdict['Metallicity'] = f[gaskey_metal][gas_mask]
    gas_snapdict['hsml']        = f[gaskey_hsml][gas_mask]
    gas_snapdict['ParticleIDs'] = f[gaskey_id][gas_mask]
  
    
    #Load Star Particles
    star_snapdict['h'] = h
    star_snapdict['a'] = a_snap
    star_snapdict['Rvir'] = Rvir
    star_snapdict['Mvir'] = Mvir

    star_snapdict['Coordinates'] = np.array([f[starkey_x][star_mask]*length_conversion,
                                             f[starkey_y][star_mask]*length_conversion,
                                             f[starkey_z][star_mask]*length_conversion
                                            ]).T
    star_snapdict['Velocities'] = np.array([f[starkey_vx][star_mask],
                                            f[starkey_vy][star_mask],
                                            f[starkey_vz][star_mask]
                                           ]).T  

    star_snapdict['Masses']      = f[starkey_mass][star_mask]*mass_conversion
    star_snapdict['Metallicity'] = f[gaskey_metal][star_mask]
    star_snapdict['ParticleIDs'] = f[starkey_id][star_mask]
    star_snapdict['StellarFormationTime'] = f[starkey_tform][star_mask]    
    
    star_snapdict['r'] = (star_snapdict['Coordinates'][:,0]**2 + 
                          star_snapdict['Coordinates'][:,1]**2 + 
                          star_snapdict['Coordinates'][:,2]**2) ** 0.5
    star_snapdict['r_xy'] = (star_snapdict['Coordinates'][:,0]**2 + star_snapdict['Coordinates'][:,1]**2) ** 0.5
    star_snapdict['r_yz'] = (star_snapdict['Coordinates'][:,1]**2 + star_snapdict['Coordinates'][:,2]**2) ** 0.5
    star_snapdict['r_zx'] = (star_snapdict['Coordinates'][:,2]**2 + star_snapdict['Coordinates'][:,0]**2) ** 0.5
    
   
    z_form = 1/star_snapdict['StellarFormationTime'] - 1
    star_snapdict['StellarAge'] = np.array( ( Planck13.lookback_time( z_form ) ) ) 
    

    if orient is True:    
        youngstarsMax = star_snapdict['StellarAge'] < .5
        youngstars = {'Coordinates': star_snapdict['Coordinates'][youngstarsMax], 
                      'Velocities': star_snapdict['Velocities'][youngstarsMax], 
                      'Masses': star_snapdict['Masses'][youngstarsMax]}
        
        theta_TB,phi_TB,vscom = orientDiskFromSnapdicts(youngstars,gas_snapdict,Rvir,[0,0,0],orient_stars=True)
        star_snapdict = offsetRotateSnapshot(star_snapdict,[0,0,0],vscom,theta_TB,phi_TB,0)
        gas_snapdict = offsetRotateSnapshot(gas_snapdict,[0,0,0],vscom,theta_TB,phi_TB,0)
    
    
    f.close()
    
    return star_snapdict, gas_snapdict

    
    
def load_sim_FIREBox_old(
    obj_path,    
    ahf_path = None,
    dark_matter=False,
    mass_unit = 'physical' ,
    length_unit = 'physical'):
    
    '''
    Loads FIREBox galaxies preped by Prof. Jorge Moreno
    Sets up data to be used for mock images
    Converts all units to physical units
    Assume physical units from Moreno Data
    ahf_path contains particle ids. This will mask to only bound particles 
    
    Parameters
    ----------
    obj_path:    str, path to the galaxy object file
    ahf_path:    str, path to file that contains bound particle
    mass_unit:   str, physical/simulation what units the particle masses are in
    length_unit: str, physical/simulation what units the particle length are in
    
    Returns
    -------
    star_snapdict, gas_snapdict
    
    Example
    -------
    obj_path = '/DFS-L/DATA/cosmo/jgmoren1/FIREBox/FB15N1024/objects_1200/object_3094.hdf5',    
    ahf_path = '/DFS-L/DATA/cosmo/jgmoren1/FIREBox/FB15N1024/ahf_objects_1200/ahf_object_3094.hdf5',
    
    star_snapdict, gas_snapdict = load_sim_FIREBox(obj_path, ahf_path)
    
    '''
    f = h5py.File(obj_path, 'r')
        
        
    if ahf_path is None:
        # if no ahf then no mask, all elements are true
        gas_mask  = np.full(np.shape(f['gas_x']), True)
        star_mask  = np.full(np.shape(f['stars_x']), True)
    else:
        ahf = h5py.File(ahf_path, 'r')

        #Gas
        gas_ids =  ahf['particleIDs'][ahf['partTypes'][:]==0]
        gas_mask = np.in1d(f['gas_id'],gas_ids)

        #Stars
        star_ids = ahf['particleIDs'][ahf['partTypes'][:]==4]
        star_mask = np.in1d(f['stars_id'],star_ids)
    

    h = 0.6774
    z_snap = f['redshift'][()]
    a_snap = 1/(1+z_snap)
        
    #Unit conversion 
    if mass_unit == 'physical':
        mass_conversion = (10**10)
    elif mass_unit == 'simulation':
        mass_conversion = (10**10)/h
    
    
    if length_unit == 'physical':
        length_conversion = 1
    elif length_unit == 'simulation':
        length_conversion = a_snap/h
        
        
    Mvir, Rvir = f['object_Mvir'][()]*mass_conversion, f['object_Rvir'][()]*length_conversion

    gas_snapdict  = {'h':np.empty(0),'a':np.empty(0), 'Rvir':np.empty(0), 'Mvir':np.empty(0),
                     'Coordinates':np.empty((0,3)),'Velocities':np.empty((0,3)),
                     'Masses':np.empty(0),'Metallicity':np.empty(0),
                     'hsml':np.empty(0), 'ParticleIDs':np.empty(0),
                     'r':np.empty(0),'r_xy':np.empty(0),'r_yz':np.empty(0),'r_zx':np.empty(0)}

    star_snapdict = {'h':np.empty(0),'a':np.empty(0), 'Rvir':np.empty(0), 'Mvir':np.empty(0),
                     'Coordinates':np.empty((0,3)),'Velocities':np.empty((0,3)), 
                     'Masses':np.empty(0),'Metallicity':np.empty(0),
                     'hsml':np.empty(0), 'ParticleIDs':np.empty(0),
                     'StellarFormationTime':np.empty(0), 'StellarAge':np.empty(0),
                     'r':np.empty(0),'r_xy':np.empty(0),'r_yz':np.empty(0),'r_zx':np.empty(0)}
    
    
    if dark_matter is True:
        dark_snapdict = {'h':np.empty(0),'a':np.empty(0),
                        'Coordinates':np.empty((0,3)),'Velocities':np.empty((0,3)), 
                        'Masses':np.empty(0), 'ParticleIDs':np.empty(0)}
        
        dark_snapdict['Coordinates'] = np.append(dark_snapdict['Coordinates'],
                                                 np.array([f['PartType1']['Coordinates'][:,0]*a_snap/h,
                                                      f['PartType1']['Coordinates'][:,1]*a_snap/h,
                                                      f['PartType1']['Coordinates'][:,2]*a_snap/h
                                                     ]).T, axis=0)
        dark_snapdict['Velocities'] = np.append(dark_snapdict['Velocities'],
                                                np.array([f['PartType1']['Velocities'][:,0],
                                                       f['PartType1']['Velocities'][:,1],
                                                       f['PartType1']['Velocities'][:,2]
                                                      ]).T, axis=0)

        dark_snapdict['Masses']      = np.append(dark_snapdict['Masses'],f['PartType1']['Masses'][:]*(10**10)/h)
        dark_snapdict['ParticleIDs'] = np.append(dark_snapdict['ParticleIDs'],f['PartType1']['ParticleIDs'][:])
        dark_snapdict['h'] = h
        dark_snapdict['a'] = a_snap


    
    
    
    #Load Gas Particles
    gas_snapdict['h'] = h
    gas_snapdict['a'] = a_snap
    gas_snapdict['Rvir'] = Rvir
    gas_snapdict['Mvir'] = Mvir
    
    gas_snapdict['Coordinates'] = np.array([f['gas_x'][gas_mask]*length_conversion,
                                            f['gas_y'][gas_mask]*length_conversion,
                                            f['gas_z'][gas_mask]*length_conversion
                                           ]).T
    
    gas_snapdict['r'] = (gas_snapdict['Coordinates'][:,0]**2 + 
                         gas_snapdict['Coordinates'][:,1]**2 + 
                         gas_snapdict['Coordinates'][:,2]**2) ** 0.5
    gas_snapdict['r_xy'] = (gas_snapdict['Coordinates'][:,0]**2 + gas_snapdict['Coordinates'][:,1]**2) ** 0.5
    gas_snapdict['r_yz'] = (gas_snapdict['Coordinates'][:,1]**2 + gas_snapdict['Coordinates'][:,2]**2) ** 0.5
    gas_snapdict['r_zx'] = (gas_snapdict['Coordinates'][:,2]**2 + gas_snapdict['Coordinates'][:,0]**2) ** 0.5

    
    
    
    gas_snapdict['Velocities'] = np.array([f['gas_vx'][gas_mask],
                                           f['gas_vy'][gas_mask],
                                           f['gas_vz'][gas_mask]
                                          ]).T  
    
    gas_snapdict['Masses']      = f['gas_mass'][gas_mask]*mass_conversion
    gas_snapdict['Metallicity'] = f['gas_total_metallicity'][gas_mask]
    gas_snapdict['hsml']        = f['gas_hsml'][gas_mask]
    gas_snapdict['ParticleIDs'] = f['gas_id'][gas_mask]

    
    
    #Load Star Particles
    star_snapdict['h'] = h
    star_snapdict['a'] = a_snap
    star_snapdict['Rvir'] = Rvir
    star_snapdict['Mvir'] = Mvir

    star_snapdict['Coordinates'] = np.array([f['stars_x'][star_mask]*length_conversion,
                                             f['stars_y'][star_mask]*length_conversion,
                                             f['stars_z'][star_mask]*length_conversion
                                            ]).T
    
    star_snapdict['r'] = (star_snapdict['Coordinates'][:,0]**2 + 
                          star_snapdict['Coordinates'][:,1]**2 + 
                          star_snapdict['Coordinates'][:,2]**2) ** 0.5
    star_snapdict['r_xy'] = (star_snapdict['Coordinates'][:,0]**2 + star_snapdict['Coordinates'][:,1]**2) ** 0.5
    star_snapdict['r_yz'] = (star_snapdict['Coordinates'][:,1]**2 + star_snapdict['Coordinates'][:,2]**2) ** 0.5
    star_snapdict['r_zx'] = (star_snapdict['Coordinates'][:,2]**2 + star_snapdict['Coordinates'][:,0]**2) ** 0.5
    
    
    star_snapdict['Velocities'] = np.array([f['stars_vx'][star_mask],
                                            f['stars_vy'][star_mask],
                                            f['stars_vz'][star_mask]
                                           ]).T   
    
    star_snapdict['Masses']      = f['stars_mass'][star_mask]*mass_conversion
    star_snapdict['Metallicity'] = f['stars_total_metallicity'][star_mask]
    star_snapdict['ParticleIDs'] = f['stars_id'][star_mask]
    star_snapdict['StellarFormationTime'] = f['stars_formation_time'][star_mask]
    
    z_form = 1/star_snapdict['StellarFormationTime'] - 1
    star_snapdict['StellarAge'] = np.array( ( Planck13.lookback_time( z_form ) ) ) 

     
    if dark_matter is True:
        return star_snapdict, gas_snapdict, dark_snapdict
    else:
        return star_snapdict, gas_snapdict



def get_mock_observation(
    star_snapdict, 
    gas_snapdict, 
    bands=[1,2,3], 
    FOV=25, 
    pixels=500, 
    view='xy', 
    center = 'none',
    minden=57650, 
    dynrange=52622824.0/57650, 
    mass_scalar = 1 / 1e10, 
    hsml_scalar = 1.4, 
    return_type='mock_image',
    QUIET=False):  
    
    '''
    
    Parameters
    ----------
    star_snapdict: dict, The star particle dictionary from load_sim
    gas_snapdict:  dict, The gas particle dictionary from load_sim   
    FOV:           float, Field of view for the image in kpc, physical distance from the center of the galaxy to the image edge
    pixels:        int, The number of pixel across the image
    view:          str, The plane that the galaxy is projected, options: 'xy','yz','zx'
    mass_scalar:   The code expects the masses to be in the simulation units of 1e10 Msun.
                    if the mass is in units Msun, then use mass_scalar = 1e-10. If
                    it is in simulation units is mass_scalar = 1
    return_type:   What values types are returned, Options:
                    'lum':      Luminosity of each particle in each band (Lsun), length = nparticles
                    'lum_proj': Luminosity per pixel (Lsun/1pixel^2) in each band, shape is (pixel, pixel)
                    'SB_lum':   Luminosity per square distance ((Lsun/kpc^2), shape is (pixel, pixel)
                    'mag':      Magnitude of each particle in each band (Lsun), length = nparticles
                    'mag_proj': Magnitude per pixel (mag/1pixel^2) in each band, shape is (pixel, pixel)
                    'SB_mag':   Magnitude per square angle ((mag/acrsec^2), shape is (pixel, pixel)
                    'mock_image': Mock color image, and the Luminosity per square distance for each band
                    
    Returns
    -------     
    Refer to return_type to see the different values returned by the function
        
    '''
        
    if len(star_snapdict['hsml']) == 0:
        star_hsml = get_particle_hsml(star_snapdict['Coordinates'][:,0],
                                      star_snapdict['Coordinates'][:,1],
                                      star_snapdict['Coordinates'][:,2])
        star_snapdict['hsml'] = np.append(star_snapdict['hsml'],star_hsml)
    
    # Make a copy so that it does not change the function outside of the function
    star_snapdict = star_snapdict.copy()
    gas_snapdict = gas_snapdict.copy()
    
    
    mask_star = star_snapdict['r'] < FOV
    mask_gas = gas_snapdict['r'] < FOV
        
    for key in ['Coordinates','Masses','Metallicity','StellarAge','hsml']:
        star_snapdict[key] = star_snapdict[key][mask_star]

    for key in ['Coordinates','Masses','Metallicity','hsml']:
        gas_snapdict[key] = gas_snapdict[key][mask_gas]

    
    
    kappas,lums = raytrace_projection.read_band_lums_from_tables(bands,
                                                                 star_snapdict['Masses']*mass_scalar,
                                                                 star_snapdict['StellarAge'],
                                                                 star_snapdict['Metallicity'])

    if view == 'xy':
        coords_stars = [star_snapdict['Coordinates'][:,0],star_snapdict['Coordinates'][:,1],star_snapdict['Coordinates'][:,2]]  
        coords_gas = [gas_snapdict['Coordinates'][:,0],gas_snapdict['Coordinates'][:,1],gas_snapdict['Coordinates'][:,2]]

    if view == 'yz':
        coords_stars = [star_snapdict['Coordinates'][:,1],star_snapdict['Coordinates'][:,2],star_snapdict['Coordinates'][:,0]]  
        coords_gas = [gas_snapdict['Coordinates'][:,1],gas_snapdict['Coordinates'][:,2],gas_snapdict['Coordinates'][:,0]]
        
    if view == 'zx':
        coords_stars = [star_snapdict['Coordinates'][:,2],star_snapdict['Coordinates'][:,0],star_snapdict['Coordinates'][:,1]]  
        coords_gas = [gas_snapdict['Coordinates'][:,2],gas_snapdict['Coordinates'][:,0],gas_snapdict['Coordinates'][:,1]]
    
    if center == 'mass':
        cm = center_mass(np.transpose([coords_stars[0],coords_stars[1],coords_stars[2]]),
                         star_snapdict['Masses'])
    if center == 'light':
        cm = center_mass(np.transpose([coords_stars[0],coords_stars[1],coords_stars[2]]),
                         lums[2,:])
    if center == 'none':
        cm = [0,0,0]

        
    gas_out,out_band0,out_band1,out_band2 = raytrace_ugr_attenuation(coords_stars[0]-cm[0],
                                                                     coords_stars[1]-cm[1],
                                                                     coords_stars[2]-cm[2],
                                                                     star_snapdict['Masses']*mass_scalar,
                                                                     star_snapdict['StellarAge'],
                                                                     star_snapdict['Metallicity'],
                                                                     star_snapdict['hsml']*hsml_scalar,
                                                                     coords_gas[0]-cm[0],
                                                                     coords_gas[1]-cm[1],
                                                                     coords_gas[2]-cm[2],
                                                                     gas_snapdict['Masses']*mass_scalar,
                                                                     gas_snapdict['Metallicity'],
                                                                     gas_snapdict['hsml']*hsml_scalar,     
                                                                     kappas,
                                                                     lums,
                                                                     pixels=pixels,
                                                                     xlim = (-FOV,FOV),
                                                                     ylim = (-FOV,FOV),
                                                                     zlim = (-FOV,FOV),
                                                                     QUIET=QUIET)
    
    # rotate image array
    # enable the columns are x, row are y, 
    # first values is the upper left value of the image
    gas_out = np.rot90(gas_out,k=1,axes=(0,1))
    out_band0 = np.rot90(out_band0,k=1,axes=(0,1))
    out_band1 = np.rot90(out_band1,k=1,axes=(0,1))
    out_band2 = np.rot90(out_band2,k=1,axes=(0,1))
    
        
    
    if return_type == 'lum':
        unit_factor = 1e10 # gives units in Lsun
        return lums[0]*unit_factor, lums[1]*unit_factor, lums[2]*unit_factor

    
    elif return_type == 'lum_proj':
        unit_factor = 1e10 # gives units in Lsun        
        return out_band0*unit_factor, out_band1*unit_factor, out_band2*unit_factor
    
    
    elif return_type == 'SB_lum':
        unit_factor = 1e10/(2*FOV/pixels)**2 # gives units in Lsun kpc^-2
        return out_band0*unit_factor, out_band1*unit_factor, out_band2*unit_factor
    
    
    elif return_type == 'mag':
        unit_factor = 1e10 # gives units in Lsun
        mag0 = sun_abs_mag(bands[0]) - 2.5 * np.log10(lums[0] * unit_factor)
        mag1 = sun_abs_mag(bands[1]) - 2.5 * np.log10(lums[1] * unit_factor)
        mag2 = sun_abs_mag(bands[2]) - 2.5 * np.log10(lums[2] * unit_factor)
        return mag0, mag1, mag2
    
    
    elif return_type == 'mag_proj':
        unit_factor = 1e10 # gives units in Lsun
        mag0 = sun_abs_mag(bands[0]) - 2.5 * np.log10(out_band0 * unit_factor)
        mag1 = sun_abs_mag(bands[1]) - 2.5 * np.log10(out_band1 * unit_factor)
        mag2 = sun_abs_mag(bands[2]) - 2.5 * np.log10(out_band2 * unit_factor)
        return mag0, mag1, mag2
    
    
    elif return_type == 'SB_mag':
        unit_factor = 1e10/(2*FOV/pixels)**2 # gives units in Lsun kpc^-2        
        mag0 = lum_to_mag_SB(out_band0 * unit_factor)
        mag1 = lum_to_mag_SB(out_band1 * unit_factor)
        mag2 = lum_to_mag_SB(out_band2 * unit_factor)
        return mag0, mag1, mag2
    
    
    elif return_type == 'mock_image':
        unit_factor = 1e10/(2*FOV/pixels)**2
        out_band0,out_band1,out_band2 = out_band0*unit_factor, out_band1*unit_factor, out_band2*unit_factor
        image24, massmap = makethreepic.make_threeband_image_process_bandmaps(
                            copy.copy(out_band0),copy.copy(out_band1),copy.copy(out_band2),
                            maxden=dynrange * minden, dynrange=dynrange,
                            pixels=pixels)

        return image24, out_band0, out_band1, out_band2
    
    
    elif return_type == 'gas_mass':
        # Returns the total gas mass in each pixel
        return gas_out

    
    
def get_mock_massimage(
    star_snapdict, 
    gas_snapdict, 
    FOV=25, 
    pixels=500, 
    view='xy', 
    center = 'none',
    mass_scalar = 1 / 1e10,
    hsml_scalar=1.4,
    QUIET=False):  
    
    '''
    
    Parameters
    ----------
    star_snapdict: dict, The star particle dictionary from load_sim
    gas_snapdict:  dict, The gas particle dictionary from load_sim   
    FOV:           float, Field of view for the image in kpc, physical distance from the center of the galaxy to the image edge
    pixels:        int, The number of pixel across the image
    view:          str, The plane that the galaxy is projected, options: 'xy','yz','zx'
    mass_scalar:   The code expects the masses to be in the simulation units of 1e10 Msun.
                    if the mass is in units Msun, then use mass_scalar = 1e-10. If
                    it is in simulation units is mass_scalar = 1
                    
    Returns
    -------     
    Refer to return_type to see the different values returned by the function
        
    '''
        
    if len(star_snapdict['hsml']) == 0:
        star_hsml = get_particle_hsml(star_snapdict['Coordinates'][:,0],
                                      star_snapdict['Coordinates'][:,1],
                                      star_snapdict['Coordinates'][:,2])
        star_snapdict['hsml'] = np.append(star_snapdict['hsml'],star_hsml)
    
    # Make a copy so that it does not change the function outside of the function
    star_snapdict = star_snapdict.copy()
    gas_snapdict = gas_snapdict.copy()
    
    mask_star = star_snapdict['r'] < FOV
    mask_gas = gas_snapdict['r'] < FOV
        
    for key in ['Coordinates','Masses','Metallicity','StellarAge','hsml']:
        star_snapdict[key] = star_snapdict[key][mask_star]

    for key in ['Coordinates','Masses','Metallicity','hsml']:
        gas_snapdict[key] = gas_snapdict[key][mask_gas]


    if view == 'xy':
        coords_stars = [star_snapdict['Coordinates'][:,0],star_snapdict['Coordinates'][:,1],star_snapdict['Coordinates'][:,2]]  
        coords_gas = [gas_snapdict['Coordinates'][:,0],gas_snapdict['Coordinates'][:,1],gas_snapdict['Coordinates'][:,2]]

    if view == 'yz':
        coords_stars = [star_snapdict['Coordinates'][:,1],star_snapdict['Coordinates'][:,2],star_snapdict['Coordinates'][:,0]]  
        coords_gas = [gas_snapdict['Coordinates'][:,1],gas_snapdict['Coordinates'][:,2],gas_snapdict['Coordinates'][:,0]]
        
    if view == 'zx':
        coords_stars = [star_snapdict['Coordinates'][:,2],star_snapdict['Coordinates'][:,0],star_snapdict['Coordinates'][:,1]]  
        coords_gas = [gas_snapdict['Coordinates'][:,2],gas_snapdict['Coordinates'][:,0],gas_snapdict['Coordinates'][:,1]]
    
    if center == 'mass':
        cm = center_mass(np.transpose([coords_stars[0],coords_stars[1],coords_stars[2]]),
                         star_snapdict['Masses'])

    if center == 'none':
        cm = [0,0,0]
        
    kappas = np.array([0.,0.,0.])
    
    massWeight = np.array([star_snapdict['Masses']*mass_scalar,
                           star_snapdict['Masses']*mass_scalar,
                           star_snapdict['Masses']*mass_scalar])

    # It is created to 
    gas_out,out_mass0,out_mass1,out_mass2 = raytrace_ugr_attenuation(coords_stars[0]-cm[0],
                                                                     coords_stars[1]-cm[1],
                                                                     coords_stars[2]-cm[2],
                                                                     star_snapdict['Masses']*mass_scalar,
                                                                     star_snapdict['StellarAge'],
                                                                     star_snapdict['Metallicity'],
                                                                     star_snapdict['hsml']*hsml_scalar,
                                                                     coords_gas[0]-cm[0],
                                                                     coords_gas[1]-cm[1],
                                                                     coords_gas[2]-cm[2],
                                                                     gas_snapdict['Masses']*mass_scalar,
                                                                     gas_snapdict['Metallicity'],
                                                                     gas_snapdict['hsml']*hsml_scalar,     
                                                                     kappas,
                                                                     massWeight,
                                                                     pixels=pixels,
                                                                     xlim = (-FOV,FOV),
                                                                     ylim = (-FOV,FOV),
                                                                     zlim = (-FOV,FOV),
                                                                     QUIET=QUIET
                                                                    )
    
    # all out_mass are the same
    
    
    # rotate image array
    # enable the columns are x, row are y, 
    # first values is the upper left value of the image
    out_mass0 = np.rot90(out_mass0,k=1,axes=(0,1))
    
    return out_mass0, len(star_snapdict['Masses'])
     
    
    
def ParamsFromPath(path,typeint=True):
    file = path.split('/')[-1]
    galaxyID = file.split('_')[1]
    FOV =  file.split('_')[-2][3:]
    pixel = file.split('_')[-1][1:-5]
    
    if typeint is True:
        return int(galaxyID), int(FOV), int(pixel)
    else:
        return galaxyID, FOV, pixel
    
    
    
def ImagePathFromGalaxyID(directory,galaxyID):
    for file in directory:
        galaxyfileID = ParamsFromPath(file)[0]
        if galaxyfileID == galaxyID:
            return file
    
    

def convert_kpc_to_arcsec(
    physical_size,
    z):
    '''
    Given a physical size and redshift, it will calculate the angular size
    
    Parameters
    ----------
    physical_size: float, the physical size of the object [kpc]
    z:             float, the redshift the object is located  
    
    Returns
    -------
    angular_size [arcsec]
    '''
    
    kpc_to_arcsec_converstion = astropy.cosmology.Planck13.arcsec_per_kpc_proper(z)
    conversion = physical_size * kpc_to_arcsec_converstion
    return conversion.value

