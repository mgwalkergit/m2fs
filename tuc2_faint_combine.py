import numpy as np
import astropy
import sqlutil
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import StdDevUncertainty
from astropy.nddata import CCDData
from astropy.coordinates import SkyCoord
import dustmaps.sfd
import astropy.units as u
from astropy.modeling import models
import os
import csv
from os import path
import scipy
import m2fs_process as m2fs
import mycode
import crossmatcher
matplotlib.use('TkAgg')
from matplotlib.patches import Ellipse
import dill as pickle
from isochrones.mist import MIST_Isochrone
from isochrones.mist import MIST_EvolutionTrack
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from isochrones import get_ichrone
from pymultinest.solve import solve
from pymultinest import Analyzer

m2fshires_fits_calibrated_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_HiRes_catalog.fits'
m2fshires=fits.open(m2fshires_fits_calibrated_filename)[1].data

#obs1=np.where((m2fshires.target_system=='tuc2')&(m2fshires.obs==1))[0]
obs1=np.where((m2fshires.target_system=='tuc2')&(m2fshires.n_good_obs==0))&(m2fshires.n_obs>1)&(m2fshires.obs==1))[0]

for j in range(0,len(obs1)):

    to_combine=np.where(m2fshires.gaia_source_id==m2fshires.gaia_source_id[obs1][j])[0]

    #vlos_raw=m2fshires.vlos_raw-m2fshires.vhelio_correction
    vlos_raw=0.-m2fshires.vhelio_correction#make sure not to re-apply heliocentric correction after fitting spectrum

    npix=10000
    wav_min=512
    wav2=np.linspace(5130,5187,10000)
    nspec=np.zeros((len(to_combine),len(wav2)))
    nvar=np.zeros((len(to_combine),len(wav2)))
    
    for i in range(0,len(to_combine)):
        print(i,len(to_combine))
        to_combine_spec=fits.open('/hildafs/projects/phy200028p/mgwalker/fits_files/'+m2fshires['fits_filename'][to_combine[i]])
        to_combine_wav=to_combine_spec[1].data[m2fshires['fits_index'][to_combine[i]]]
        to_combine_wav2=to_combine_wav/(1+vlos_raw[to_combine[i]]/2.99792e5)
        to_combine_skysub=to_combine_spec[2].data[m2fshires['fits_index'][to_combine[i]]]
        to_combine_mask=to_combine_spec[4].data[m2fshires['fits_index'][to_combine[i]]]
        to_combine_bestfit=to_combine_spec[5].data[m2fshires['fits_index'][to_combine[i]]]
        to_combine_varspec=to_combine_spec[3].data[m2fshires['fits_index'][to_combine[i]]]
        to_combine_varspec2=(10.**m2fshires['logs1_raw'][to_combine[i]])*to_combine_varspec+(10.**m2fshires['logs2_raw'][to_combine[i]])**2
        
        rastring,decstring=mycode.coordstring(m2fshires['ra'][to_combine[i]],m2fshires['dec'][to_combine[i]])
        coordstring=rastring+decstring
        ymin=np.min(to_combine_skysub[to_combine_mask==0])-0.3*(np.max(to_combine_skysub[to_combine_mask==0])-np.min(to_combine_skysub[to_combine_mask==0]))
        ymax=1.25*np.max(to_combine_skysub[to_combine_mask==0])
        magstring='G='+str.format('{0:.2f}',round(m2fshires['gaia_gmag'][to_combine[i]],2))
        vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fshires['vlos'][to_combine[i]],2))+'\pm'+str.format('{0:.1f}',round(m2fshires['vlos_error'][to_combine[i]],2))+'$ km/s'
        teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fshires['teff'][to_combine[i]],0))+'\pm'+str.format('{0:.0f}',round(m2fshires['teff_error'][to_combine[i]],2))+'$ K'
        loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fshires['logg'][to_combine[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fshires['logg_error'][to_combine[i]],2))+'$'
        zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fshires['feh'][to_combine[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fshires['feh_error'][to_combine[i]],2))+'$'
        alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fshires['mgfe'][to_combine[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fshires['mgfe_error'][to_combine[i]],2))+'$'
        nspec[i]=np.interp(wav2,to_combine_wav2,to_combine_skysub)
        nvar[i]=np.interp(wav2,to_combine_wav2,to_combine_varspec)
    
    to_combine_sum=nspec.sum(axis=0)
    wav0=np.linspace(5130,5187,1000)
    interp2=np.interp(wav0,wav2,to_combine_sum)

    gs=plt.GridSpec(15,15)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:10,0:15])

    ax1.plot(wav0,interp2,color='r',lw=0.5)
    ax1.set_xlabel(r'$\lambda$ (Angs.)')
    ax1.set_ylabel('sum of sky-subtracted spectra')
    ax1.legend(loc=3)
    plt.savefig('to_combine_sum.pdf',dpi=200)
    plt.show()
