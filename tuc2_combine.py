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

plot_spectra=True

m2fshires_fits_calibrated_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_HiRes_catalog.fits'
m2fsmedres_fits_calibrated_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_MedRes_catalog.fits'

m2fshires=fits.open(m2fshires_fits_calibrated_filename)[1].data

if plot_spectra:

    
    tuc2binary_coords=SkyCoord('22:50:41.07','-58:31:08.3',unit=(u.hourangle,u.deg))
    dist=np.sqrt((m2fshires.ra-tuc2binary_coords.ra.deg)**2+(m2fshires.dec-tuc2binary_coords.dec.deg)**2)*3600.
    tuc2binary=np.where((dist<1.)&(m2fshires.good_obs>0))[0]
    vlos_raw=m2fshires.vlos_raw-m2fshires.vhelio_correction

    m2fshires=fits.open(m2fshires_fits_calibrated_filename)[1].data
    m2fsmedres=fits.open(m2fsmedres_fits_calibrated_filename)[1].data

    npix=10000
    wav_min=512
    wav2=np.linspace(5130,5187,10000)
    nspec=np.zeros((len(tuc2binary),len(wav2)))
    nvar=np.zeros((len(tuc2binary),len(wav2)))
    
    for i in range(0,len(tuc2binary)):
        gs=plt.GridSpec(15,15)
        gs.update(wspace=0,hspace=0)
        fig=plt.figure(figsize=(6,6))
        ax1=fig.add_subplot(gs[0:10,0:15])

        print(i)
        tuc2binary_spec=fits.open('/hildafs/projects/phy200028p/mgwalker/fits_files/'+m2fshires['fits_filename'][tuc2binary[i]])
        tuc2binary_wav=tuc2binary_spec[1].data[m2fshires['fits_index'][tuc2binary[i]]]
        tuc2binary_wav2=tuc2binary_wav/(1+vlos_raw[tuc2binary[i]]/2.99792e5)
        tuc2binary_skysub=tuc2binary_spec[2].data[m2fshires['fits_index'][tuc2binary[i]]]
        tuc2binary_mask=tuc2binary_spec[4].data[m2fshires['fits_index'][tuc2binary[i]]]
        tuc2binary_bestfit=tuc2binary_spec[5].data[m2fshires['fits_index'][tuc2binary[i]]]
        tuc2binary_varspec=tuc2binary_spec[3].data[m2fshires['fits_index'][tuc2binary[i]]]
        tuc2binary_varspec2=(10.**m2fshires['logs1_raw'][tuc2binary[i]])*tuc2binary_varspec+(10.**m2fshires['logs2_raw'][tuc2binary[i]])**2
        
        rastring,decstring=mycode.coordstring(m2fshires['ra'][tuc2binary[i]],m2fshires['dec'][tuc2binary[i]])
        coordstring=rastring+decstring
        ymin=np.min(tuc2binary_skysub[tuc2binary_mask==0])-0.3*(np.max(tuc2binary_skysub[tuc2binary_mask==0])-np.min(tuc2binary_skysub[tuc2binary_mask==0]))
        ymax=1.25*np.max(tuc2binary_skysub[tuc2binary_mask==0])
        magstring='G='+str.format('{0:.2f}',round(m2fshires['gaia_gmag'][tuc2binary[i]],2))
        vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fshires['vlos'][tuc2binary[i]],2))+'\pm'+str.format('{0:.1f}',round(m2fshires['vlos_error'][tuc2binary[i]],2))+'$ km/s'
        teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fshires['teff'][tuc2binary[i]],0))+'\pm'+str.format('{0:.0f}',round(m2fshires['teff_error'][tuc2binary[i]],2))+'$ K'
        loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fshires['logg'][tuc2binary[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fshires['logg_error'][tuc2binary[i]],2))+'$'
        zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fshires['feh'][tuc2binary[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fshires['feh_error'][tuc2binary[i]],2))+'$'
        alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fshires['mgfe'][tuc2binary[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fshires['mgfe_error'][tuc2binary[i]],2))+'$'
        ax1.tick_params(right=False,top=False,left=True,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=2,labelsize=6)
        ax1.plot(tuc2binary_wav[tuc2binary_mask==0],tuc2binary_skysub[tuc2binary_mask==0],color='k',lw=0.5)
        ax1.plot(tuc2binary_wav[tuc2binary_mask==0],tuc2binary_bestfit[tuc2binary_mask==0],color='r',lw=0.5)
        ax1.set_xlim([5130,5190])
        ax1.set_ylim([ymin,ymax])
        ax1.set_ylabel('counts',fontsize=8)
        ax1.set_yticks([0,50,100])

        ax1.text(0.03,0.95,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
        ax1.text(0.99,0.95,magstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
        plt.savefig('tuc2binary_epoch'+str(i+1)+'.pdf',dpi=200)
        #plt.show()
        plt.close()
        nspec[i]=np.interp(wav2,tuc2binary_wav2,tuc2binary_skysub)
        nvar[i]=np.interp(wav2,tuc2binary_wav2,tuc2binary_varspec)

    
    tuc2binary_sum=nspec.sum(axis=0)
    wav0=np.linspace(5130,5187,1000)
    interp2=np.interp(wav0,wav2,tuc2binary_sum)
    #ax1.plot(wav0,interp2,color='k',lw=0.5)
    #ax1.set_xlabel(r'$\lambda$ (Angs.)')
    #ax1.set_ylabel('sum of sky-subtracted spectra')
    #plt.savefig('tuc2binary_sum.pdf',dpi=200)
    #plt.show()
        
