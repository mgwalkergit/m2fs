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
m2fshires=fits.open(m2fshires_fits_calibrated_filename)[1].data

if plot_spectra:

    to_combine=np.where((m2fshires.target_system=='sex')&(m2fshires.hjd>2460424.)&(m2fshires.vlos_raw_error>5.)&(m2fshires.gaia_gmag<20.0))[0]
    to_combine2=np.where((m2fshires.target_system=='sex')&(m2fshires.hjd>2460424.)&(m2fshires.vlos_raw_error<5.)&(m2fshires.gaia_gmag<20.0)&(m2fshires.gaia_gmag>19.))[0]
    vlos_raw=m2fshires.vlos_raw-m2fshires.vhelio_correction

    npix=10000
    wav_min=512
    wav2=np.linspace(5130,5187,10000)
    nspec=np.zeros((len(to_combine),len(wav2)))
    nvar=np.zeros((len(to_combine),len(wav2)))
    
    for i in range(0,len(to_combine)):
        gs=plt.GridSpec(15,15)
        gs.update(wspace=0,hspace=0)
        fig=plt.figure(figsize=(6,6))
        ax1=fig.add_subplot(gs[0:10,0:15])

        print(i)
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
        ax1.tick_params(right=False,top=False,left=True,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=2,labelsize=6)
        ax1.plot(to_combine_wav[to_combine_mask==0],to_combine_skysub[to_combine_mask==0],color='k',lw=0.5)
        ax1.plot(to_combine_wav[to_combine_mask==0],to_combine_bestfit[to_combine_mask==0],color='r',lw=0.5)
        ax1.set_xlim([5130,5190])
        ax1.set_ylim([ymin,ymax])
        ax1.set_ylabel('counts',fontsize=8)
        ax1.set_yticks([0,50,100])

        ax1.text(0.03,0.95,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
        ax1.text(0.99,0.95,magstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
        plt.savefig('to_combine_epoch'+str(i+1)+'.pdf',dpi=200)
        #plt.show()
        plt.close()
        nspec[i]=np.interp(wav2,to_combine_wav2,to_combine_skysub)
        nvar[i]=np.interp(wav2,to_combine_wav2,to_combine_varspec)
    
    to_combine_sum=nspec.sum(axis=0)
    wav0=np.linspace(5130,5187,1000)
    interp2=np.interp(wav0,wav2,to_combine_sum)



    wav2=np.linspace(5130,5187,10000)
    nspec=np.zeros((len(to_combine2),len(wav2)))
    nvar=np.zeros((len(to_combine2),len(wav2)))

    for i in range(0,len(to_combine2)):
        gs=plt.GridSpec(15,15)
        gs.update(wspace=0,hspace=0)
        fig=plt.figure(figsize=(6,6))
        ax1=fig.add_subplot(gs[0:10,0:15])

        print(i)
        to_combine2_spec=fits.open('/hildafs/projects/phy200028p/mgwalker/fits_files/'+m2fshires['fits_filename'][to_combine2[i]])
        to_combine2_wav=to_combine2_spec[1].data[m2fshires['fits_index'][to_combine2[i]]]
        to_combine2_wav2=to_combine2_wav/(1+vlos_raw[to_combine2[i]]/2.99792e5)
        to_combine2_skysub=to_combine2_spec[2].data[m2fshires['fits_index'][to_combine2[i]]]
        to_combine2_mask=to_combine2_spec[4].data[m2fshires['fits_index'][to_combine2[i]]]
        to_combine2_bestfit=to_combine2_spec[5].data[m2fshires['fits_index'][to_combine2[i]]]
        to_combine2_varspec=to_combine2_spec[3].data[m2fshires['fits_index'][to_combine2[i]]]
        to_combine2_varspec2=(10.**m2fshires['logs1_raw'][to_combine2[i]])*to_combine2_varspec+(10.**m2fshires['logs2_raw'][to_combine2[i]])**2
        
        rastring,decstring=mycode.coordstring(m2fshires['ra'][to_combine2[i]],m2fshires['dec'][to_combine2[i]])
        coordstring=rastring+decstring
        ymin=np.min(to_combine2_skysub[to_combine2_mask==0])-0.3*(np.max(to_combine2_skysub[to_combine2_mask==0])-np.min(to_combine2_skysub[to_combine2_mask==0]))
        ymax=1.25*np.max(to_combine2_skysub[to_combine2_mask==0])
        magstring='G='+str.format('{0:.2f}',round(m2fshires['gaia_gmag'][to_combine2[i]],2))
        vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fshires['vlos'][to_combine2[i]],2))+'\pm'+str.format('{0:.1f}',round(m2fshires['vlos_error'][to_combine2[i]],2))+'$ km/s'
        teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fshires['teff'][to_combine2[i]],0))+'\pm'+str.format('{0:.0f}',round(m2fshires['teff_error'][to_combine2[i]],2))+'$ K'
        loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fshires['logg'][to_combine2[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fshires['logg_error'][to_combine2[i]],2))+'$'
        zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fshires['feh'][to_combine2[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fshires['feh_error'][to_combine2[i]],2))+'$'
        alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fshires['mgfe'][to_combine2[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fshires['mgfe_error'][to_combine2[i]],2))+'$'
        ax1.tick_params(right=False,top=False,left=True,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=2,labelsize=6)
        ax1.plot(to_combine2_wav[to_combine2_mask==0],to_combine2_skysub[to_combine2_mask==0],color='k',lw=0.5)
        ax1.plot(to_combine2_wav[to_combine2_mask==0],to_combine2_bestfit[to_combine2_mask==0],color='r',lw=0.5)
        ax1.set_xlim([5130,5190])
        ax1.set_ylim([ymin,ymax])
        ax1.set_ylabel('counts',fontsize=8)
        ax1.set_yticks([0,50,100])

        ax1.text(0.03,0.95,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
        ax1.text(0.99,0.95,magstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
        plt.savefig('to_combine2_epoch'+str(i+1)+'.pdf',dpi=200)
        #plt.show()
        plt.close()
        nspec[i]=np.interp(wav2,to_combine2_wav2,to_combine2_skysub)
        nvar[i]=np.interp(wav2,to_combine2_wav2,to_combine2_varspec)
    
    to_combine2_sum=nspec.sum(axis=0)
    wav0=np.linspace(5130,5187,1000)
    interp3=np.interp(wav0,wav2,to_combine2_sum)

                         



    
    ax1.plot(wav0,interp2,color='r',lw=0.5,label=r'bad, G$<$20')
    ax1.plot(wav0,interp3,color='k',lw=0.5,label=r'good, 19 $< G <$ 20')
    ax1.set_xlabel(r'$\lambda$ (Angs.)')
    ax1.set_ylabel('sum of sky-subtracted spectra')
    ax1.legend(loc=3)
    plt.savefig('to_combine_sum.pdf',dpi=200)
    plt.show()
