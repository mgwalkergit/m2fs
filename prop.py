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

plot_cmdmap=True

m2fshires_fits_table_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_HiRes_table.fits'
m2fsmedres_fits_table_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_MedRes_table.fits'
hecto_fits_table_filename='/hildafs/projects/phy200028p/mgwalker/nelson/hecto_fits_table.fits'
hecto_gcs_fits_table_filename='/hildafs/projects/phy200028p/mgwalker/nelson/hecto_gcs_fits_table.fits'
m2fshires_fits_calibrated_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_HiRes_catalog.fits'
m2fsmedres_fits_calibrated_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_MedRes_catalog.fits'
hecto_fits_calibrated_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/hecto_catalog.fits'
hecto_gcs_fits_calibrated_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/hecto_gcs_catalog.fits'

m2fshires_catalog=fits.open(m2fshires_fits_table_filename)[1].data
m2fsmedres_catalog=fits.open(m2fsmedres_fits_table_filename)[1].data
hecto_catalog=fits.open(hecto_fits_table_filename)[1].data
mike_catalog=fits.open('/hildafs/projects/phy200028p/mgwalker/mike/mike_data.fits')[1].data

m2fshires_ra=m2fshires_catalog['ra']
m2fshires_dec=m2fshires_catalog['dec']
m2fshires_v=m2fshires_catalog['vlos_raw']
m2fshires_sigv=m2fshires_catalog['vlos_error']
m2fshires_teff=m2fshires_catalog['teff_raw']
m2fshires_sigteff=m2fshires_catalog['teff_error']
m2fshires_logg=m2fshires_catalog['logg_raw']
m2fshires_siglogg=m2fshires_catalog['logg_error']
m2fshires_z=m2fshires_catalog['feh_raw']
m2fshires_sigz=m2fshires_catalog['feh_error']
m2fshires_alpha=m2fshires_catalog['mgfe_raw']
m2fshires_sigalpha=m2fshires_catalog['mgfe_error']
m2fshires_v_mean=m2fshires_catalog['vlos_raw_mean']
m2fshires_sigv_mean=m2fshires_catalog['vlos_mean_error']
m2fshires_teff_mean=m2fshires_catalog['teff_raw_mean']
m2fshires_sigteff_mean=m2fshires_catalog['teff_mean_error']
m2fshires_logg_mean=m2fshires_catalog['logg_raw_mean']
m2fshires_siglogg_mean=m2fshires_catalog['logg_mean_error']
m2fshires_z_mean=m2fshires_catalog['feh_raw_mean']
m2fshires_sigz_mean=m2fshires_catalog['feh_mean_error']
m2fshires_alpha_mean=m2fshires_catalog['mgfe_raw_mean']
m2fshires_sigalpha_mean=m2fshires_catalog['mgfe_mean_error']
m2fshires_teffprior_v=m2fshires_catalog['teffprior_vlos_raw']
m2fshires_teffprior_sigv=m2fshires_catalog['teffprior_vlos_error']
m2fshires_teffprior_teff=m2fshires_catalog['teffprior_teff_raw']
m2fshires_teffprior_sigteff=m2fshires_catalog['teffprior_teff_error']
m2fshires_teffprior_logg=m2fshires_catalog['teffprior_logg_raw']
m2fshires_teffprior_siglogg=m2fshires_catalog['teffprior_logg_error']
m2fshires_teffprior_z=m2fshires_catalog['teffprior_feh_raw']
m2fshires_teffprior_sigz=m2fshires_catalog['teffprior_feh_error']
m2fshires_teffprior_alpha=m2fshires_catalog['teffprior_mgfe_raw']
m2fshires_teffprior_sigalpha=m2fshires_catalog['teffprior_mgfe_error']
m2fshires_teffprior_v_mean=m2fshires_catalog['teffprior_vlos_raw_mean']
m2fshires_teffprior_sigv_mean=m2fshires_catalog['teffprior_vlos_mean_error']
m2fshires_teffprior_teff_mean=m2fshires_catalog['teffprior_teff_raw_mean']
m2fshires_teffprior_sigteff_mean=m2fshires_catalog['teffprior_teff_mean_error']
m2fshires_teffprior_logg_mean=m2fshires_catalog['teffprior_logg_raw_mean']
m2fshires_teffprior_siglogg_mean=m2fshires_catalog['teffprior_logg_mean_error']
m2fshires_teffprior_z_mean=m2fshires_catalog['teffprior_feh_raw_mean']
m2fshires_teffprior_sigz_mean=m2fshires_catalog['teffprior_feh_mean_error']
m2fshires_teffprior_alpha_mean=m2fshires_catalog['teffprior_mgfe_raw_mean']
m2fshires_teffprior_sigalpha_mean=m2fshires_catalog['teffprior_mgfe_mean_error']
m2fshires_obs=m2fshires_catalog['obs']
m2fshires_nobs=m2fshires_catalog['n_obs']
m2fshires_goodobs=m2fshires_catalog['good_obs']
m2fshires_goodnobs=m2fshires_catalog['good_n_obs']

m2fsmedres_ra=m2fsmedres_catalog['ra']
m2fsmedres_dec=m2fsmedres_catalog['dec']
m2fsmedres_v=m2fsmedres_catalog['vlos_raw']
m2fsmedres_sigv=m2fsmedres_catalog['vlos_error']
m2fsmedres_teff=m2fsmedres_catalog['teff_raw']
m2fsmedres_sigteff=m2fsmedres_catalog['teff_error']
m2fsmedres_logg=m2fsmedres_catalog['logg_raw']
m2fsmedres_siglogg=m2fsmedres_catalog['logg_error']
m2fsmedres_z=m2fsmedres_catalog['feh_raw']
m2fsmedres_sigz=m2fsmedres_catalog['feh_error']
m2fsmedres_alpha=m2fsmedres_catalog['mgfe_raw']
m2fsmedres_sigalpha=m2fsmedres_catalog['mgfe_error']
m2fsmedres_v_mean=m2fsmedres_catalog['vlos_raw_mean']
m2fsmedres_sigv_mean=m2fsmedres_catalog['vlos_mean_error']
m2fsmedres_teff_mean=m2fsmedres_catalog['teff_raw_mean']
m2fsmedres_sigteff_mean=m2fsmedres_catalog['teff_mean_error']
m2fsmedres_logg_mean=m2fsmedres_catalog['logg_raw_mean']
m2fsmedres_siglogg_mean=m2fsmedres_catalog['logg_mean_error']
m2fsmedres_z_mean=m2fsmedres_catalog['feh_raw_mean']
m2fsmedres_sigz_mean=m2fsmedres_catalog['feh_mean_error']
m2fsmedres_alpha_mean=m2fsmedres_catalog['mgfe_raw_mean']
m2fsmedres_sigalpha_mean=m2fsmedres_catalog['mgfe_mean_error']
m2fsmedres_teffprior_v=m2fsmedres_catalog['teffprior_vlos_raw']
m2fsmedres_teffprior_sigv=m2fsmedres_catalog['teffprior_vlos_error']
m2fsmedres_teffprior_teff=m2fsmedres_catalog['teffprior_teff_raw']
m2fsmedres_teffprior_sigteff=m2fsmedres_catalog['teffprior_teff_error']
m2fsmedres_teffprior_logg=m2fsmedres_catalog['teffprior_logg_raw']
m2fsmedres_teffprior_siglogg=m2fsmedres_catalog['teffprior_logg_error']
m2fsmedres_teffprior_z=m2fsmedres_catalog['teffprior_feh_raw']
m2fsmedres_teffprior_sigz=m2fsmedres_catalog['teffprior_feh_error']
m2fsmedres_teffprior_alpha=m2fsmedres_catalog['teffprior_mgfe_raw']
m2fsmedres_teffprior_sigalpha=m2fsmedres_catalog['teffprior_mgfe_error']
m2fsmedres_teffprior_v_mean=m2fsmedres_catalog['teffprior_vlos_raw_mean']
m2fsmedres_teffprior_sigv_mean=m2fsmedres_catalog['teffprior_vlos_mean_error']
m2fsmedres_teffprior_teff_mean=m2fsmedres_catalog['teffprior_teff_raw_mean']
m2fsmedres_teffprior_sigteff_mean=m2fsmedres_catalog['teffprior_teff_mean_error']
m2fsmedres_teffprior_logg_mean=m2fsmedres_catalog['teffprior_logg_raw_mean']
m2fsmedres_teffprior_siglogg_mean=m2fsmedres_catalog['teffprior_logg_mean_error']
m2fsmedres_teffprior_z_mean=m2fsmedres_catalog['teffprior_feh_raw_mean']
m2fsmedres_teffprior_sigz_mean=m2fsmedres_catalog['teffprior_feh_mean_error']
m2fsmedres_teffprior_alpha_mean=m2fsmedres_catalog['teffprior_mgfe_raw_mean']
m2fsmedres_teffprior_sigalpha_mean=m2fsmedres_catalog['teffprior_mgfe_mean_error']
m2fsmedres_obs=m2fsmedres_catalog['obs']
m2fsmedres_nobs=m2fsmedres_catalog['n_obs']
m2fsmedres_goodobs=m2fsmedres_catalog['good_obs']
m2fsmedres_goodnobs=m2fsmedres_catalog['good_n_obs']

hecto_ra=hecto_catalog['ra']
hecto_dec=hecto_catalog['dec']
hecto_v=hecto_catalog['vlos_raw']
hecto_sigv=hecto_catalog['vlos_error']
hecto_teff=hecto_catalog['teff_raw']
hecto_sigteff=hecto_catalog['teff_error']
hecto_logg=hecto_catalog['logg_raw']
hecto_siglogg=hecto_catalog['logg_error']
hecto_z=hecto_catalog['feh_raw']
hecto_sigz=hecto_catalog['feh_error']
hecto_alpha=hecto_catalog['mgfe_raw']
hecto_sigalpha=hecto_catalog['mgfe_error']
hecto_v_mean=hecto_catalog['vlos_raw_mean']
hecto_sigv_mean=hecto_catalog['vlos_mean_error']
hecto_teff_mean=hecto_catalog['teff_raw_mean']
hecto_sigteff_mean=hecto_catalog['teff_mean_error']
hecto_logg_mean=hecto_catalog['logg_raw_mean']
hecto_siglogg_mean=hecto_catalog['logg_mean_error']
hecto_z_mean=hecto_catalog['feh_raw_mean']
hecto_sigz_mean=hecto_catalog['feh_mean_error']
hecto_alpha_mean=hecto_catalog['mgfe_raw_mean']
hecto_sigalpha_mean=hecto_catalog['mgfe_mean_error']
hecto_teffprior_v=hecto_catalog['teffprior_vlos_raw']
hecto_teffprior_sigv=hecto_catalog['teffprior_vlos_error']
hecto_teffprior_teff=hecto_catalog['teffprior_teff_raw']
hecto_teffprior_sigteff=hecto_catalog['teffprior_teff_error']
hecto_teffprior_logg=hecto_catalog['teffprior_logg_raw']
hecto_teffprior_siglogg=hecto_catalog['teffprior_logg_error']
hecto_teffprior_z=hecto_catalog['teffprior_feh_raw']
hecto_teffprior_sigz=hecto_catalog['teffprior_feh_error']
hecto_teffprior_alpha=hecto_catalog['teffprior_mgfe_raw']
hecto_teffprior_sigalpha=hecto_catalog['teffprior_mgfe_error']
hecto_teffprior_v_mean=hecto_catalog['teffprior_vlos_raw_mean']
hecto_teffprior_sigv_mean=hecto_catalog['teffprior_vlos_mean_error']
hecto_teffprior_teff_mean=hecto_catalog['teffprior_teff_raw_mean']
hecto_teffprior_sigteff_mean=hecto_catalog['teffprior_teff_mean_error']
hecto_teffprior_logg_mean=hecto_catalog['teffprior_logg_raw_mean']
hecto_teffprior_siglogg_mean=hecto_catalog['teffprior_logg_mean_error']
hecto_teffprior_z_mean=hecto_catalog['teffprior_feh_raw_mean']
hecto_teffprior_sigz_mean=hecto_catalog['teffprior_feh_mean_error']
hecto_teffprior_alpha_mean=hecto_catalog['teffprior_mgfe_raw_mean']
hecto_teffprior_sigalpha_mean=hecto_catalog['teffprior_mgfe_mean_error']
hecto_obs=hecto_catalog['obs']
hecto_nobs=hecto_catalog['n_obs']
hecto_goodobs=hecto_catalog['good_obs']
hecto_goodnobs=hecto_catalog['good_n_obs']

m2fs_coords=SkyCoord(m2fshires_ra,m2fshires_dec,unit=(u.deg,u.deg))
hecto_coords=SkyCoord(hecto_ra,hecto_dec,unit=(u.deg,u.deg))

gs=plt.GridSpec(19,19)
gs.update(wspace=0,hspace=0)
fig=plt.figure(figsize=(6,6))
ax11=fig.add_subplot(gs[8:13,0:5])
ax21=fig.add_subplot(gs[8:13,7:12])
ax31=fig.add_subplot(gs[8:13,14:19])

ax12=fig.add_subplot(gs[0:5,0:5])
ax22=fig.add_subplot(gs[0:5,7:12])
ax32=fig.add_subplot(gs[0:5,14:19])

gal='draco_1'
dsph=fits.open('dsph_parameters.fits')
this_dsph=np.where(dsph[1].data['name']==gal)[0]
center=SkyCoord(dsph[1].data['ra'][this_dsph],dsph[1].data['dec'][this_dsph],unit=(u.deg,u.deg))
keep=np.where(hecto_catalog['good_obs']==1)[0]
xi,eta=mycode.etaxiarr(hecto_coords.ra.rad,hecto_coords.dec.rad,center.ra.rad,center.dec.rad)
dra_r=np.sqrt(xi**2+eta**2)
mem=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['logg_raw_mean']<3.)&(hecto_catalog['gaia_gmag']<19.5))[0]
mem2=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['logg_raw_mean']<3.)&(hecto_catalog['gaia_gmag']<19.5)&(r<30.))[0]
keep2=np.where((hecto_catalog['good_obs']==1)&(r<30.))[0]

ell=Ellipse(xy=[0.,0.],height=1,width=1,angle=0,fc='0.7',ec=None,fill=True,alpha=1)
ax11.add_artist(ell)

ax11.scatter(xi[keep]/60,eta[keep]/60,s=1,color='0.3',alpha=0.3,rasterized=True)
ax11.scatter(xi[mem]/60,eta[mem]/60,s=1,color='r',alpha=0.3,rasterized=True)
ax11.set_xlim([0.55,-0.55])
ax11.set_ylim([-0.55,0.55])
ax11.set_xlabel(r'$\Delta$R.A. [deg]')
ax11.set_ylabel(r'$\Delta$Dec. [deg]')
ax12.set_title('Draco')

ax12.scatter(hecto_catalog['gaia_bpmag'][keep2]-hecto_catalog['gaia_rpmag'][keep2],hecto_catalog['gaia_gmag'][keep2],s=1,color='0.3',alpha=0.3,rasterized=True)
ax12.scatter(hecto_catalog['gaia_bpmag'][mem2]-hecto_catalog['gaia_rpmag'][mem2],hecto_catalog['gaia_gmag'][mem2],s=1,color='r',alpha=0.3,rasterized=True)
ax12.set_xlim([0.5,2])
ax12.set_ylim([21,14])
ax12.set_xlabel('Gaia BP-RP')
ax12.set_ylabel('Gaia G')

gal='ursa_minor_1'
dsph=fits.open('dsph_parameters.fits')
this_dsph=np.where(dsph[1].data['name']==gal)[0]
center=SkyCoord(dsph[1].data['ra'][this_dsph],dsph[1].data['dec'][this_dsph],unit=(u.deg,u.deg))
keep=np.where(hecto_catalog['good_obs']==1)[0]
xi,eta=mycode.etaxiarr(hecto_coords.ra.rad,hecto_coords.dec.rad,center.ra.rad,center.dec.rad)
umi_r=np.sqrt(xi**2+eta**2)
mem=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['logg_raw_mean']<3.)&(hecto_catalog['gaia_gmag']<19.5))[0]
mem2=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['logg_raw_mean']<3.)&(hecto_catalog['gaia_gmag']<19.5)&(r<30.))[0]
keep2=np.where((hecto_catalog['good_obs']==1)&(r<30.))[0]

ell=Ellipse(xy=[0.,0.],height=1,width=1,angle=0,fc='0.7',ec=None,fill=True,alpha=1)
ax21.add_artist(ell)

ax22.scatter(hecto_catalog['gaia_bpmag'][keep2]-hecto_catalog['gaia_rpmag'][keep2],hecto_catalog['gaia_gmag'][keep2],s=1,color='0.3',alpha=0.3,rasterized=True)
ax22.scatter(hecto_catalog['gaia_bpmag'][mem2]-hecto_catalog['gaia_rpmag'][mem2],hecto_catalog['gaia_gmag'][mem2],s=1,color='r',alpha=0.3,rasterized=True)
ax22.set_xlim([0.5,2])
ax22.set_ylim([21,14])
ax22.set_xlabel('Gaia BP-RP')

ax21.scatter(xi[keep]/60,eta[keep]/60,s=1,color='0.3',alpha=0.3,rasterized=True)
ax21.scatter(xi[mem]/60,eta[mem]/60,s=1,color='r',alpha=0.3,rasterized=True)
ax21.set_xlim([0.55,-0.55])
ax21.set_ylim([-0.55,0.55])
ax21.set_xlabel(r'$\Delta$R.A. [deg]')
ax22.set_title('Ursa Minor')
#ax21.set_ylabel(r'$\Delta$Dec. [deg]')

gal='sextans_1'
dsph=fits.open('dsph_parameters.fits')
this_dsph=np.where(dsph[1].data['name']==gal)[0]
center=SkyCoord(dsph[1].data['ra'][this_dsph],dsph[1].data['dec'][this_dsph],unit=(u.deg,u.deg))
keep=np.where(hecto_catalog['good_obs']==1)[0]
xi,eta=mycode.etaxiarr(hecto_coords.ra.rad,hecto_coords.dec.rad,center.ra.rad,center.dec.rad)
sex_r=np.sqrt(xi**2+eta**2)
mem=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['logg_raw_mean']<3.)&(r<30)&(hecto_catalog['gaia_gmag']<19.5))[0]
mem2=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['logg_raw_mean']<3.)&(hecto_catalog['gaia_gmag']<19.5)&(r<30.))[0]
keep2=np.where((hecto_catalog['good_obs']==1)&(r<30.))[0]

ell=Ellipse(xy=[0.,0.],height=1,width=1,angle=0,fc='0.7',ec=None,fill=True,alpha=1)
ax31.add_artist(ell)

ax32.scatter(hecto_catalog['gaia_bpmag'][keep2]-hecto_catalog['gaia_rpmag'][keep2],hecto_catalog['gaia_gmag'][keep2],s=1,color='0.3',alpha=0.3,rasterized=True)
ax32.scatter(hecto_catalog['gaia_bpmag'][mem2]-hecto_catalog['gaia_rpmag'][mem2],hecto_catalog['gaia_gmag'][mem2],s=1,color='r',alpha=0.3,rasterized=True)
ax32.set_xlim([0.5,2])
ax32.set_ylim([21,14])
ax32.set_xlabel('Gaia BP-RP')

ax31.scatter(xi[keep]/60,eta[keep]/60,s=1,color='0.3',alpha=0.3,rasterized=True)
ax31.scatter(xi[mem]/60,eta[mem]/60,s=1,color='r',alpha=0.3,rasterized=True)
ax31.set_xlim([0.55,-0.55])
ax31.set_ylim([-0.55,0.55])
ax31.set_xlabel(r'$\Delta$R.A. [deg]')
ax32.set_title('Sextans')
#ax31.set_ylabel(r'$\Delta$Dec. [deg]')

plt.savefig('prop.pdf',dpi=300)
plt.show()
plt.close()

dra_keep=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['target_system']=='dra')&(hecto_catalog['logg_raw_mean']<3.))[0]
umi_keep=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['target_system']=='umi')&(hecto_catalog['logg_raw_mean']<3.))[0]
sex_keep=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['target_system']=='sex')&(hecto_catalog['logg_raw_mean']<3.))[0]

dra_goodnobs=hecto_catalog['good_n_obs'][dra_keep]
umi_goodnobs=hecto_catalog['good_n_obs'][umi_keep]
sex_goodnobs=hecto_catalog['good_n_obs'][sex_keep]

dra_goodnobs2=np.copy(dra_goodnobs)
umi_goodnobs2=np.copy(umi_goodnobs)
sex_goodnobs2=np.copy(sex_goodnobs)

#dra_obs=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['target_system']=='dra')&(hecto_catalog['logg_raw_mean']<3.)&(dra_r<30.))[0]
#umi_obs=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['target_system']=='umi')&(hecto_catalog['logg_raw_mean']<3.)&(umi_r<30.))[0]
#sex_obs=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['target_system']=='sex')&(hecto_catalog['logg_raw_mean']<3.)&(sex_r<30.))[0]

dra_order=np.flip(np.argsort(dra_goodnobs))
i=0
while i<240:
    if dra_r[dra_keep][dra_order[i]]<30:
        dra_goodnobs2[dra_order[i]]+=2
    print(i)
    i+=1

umi_order=np.flip(np.argsort(umi_goodnobs))
i=0
while i<240:
    if umi_r[umi_keep][umi_order[i]]<30:
        umi_goodnobs2[umi_order[i]]+=2
    print(i)
    i+=1

sex_order=np.flip(np.argsort(sex_goodnobs))
i=0
while i<240:
    if sex_r[sex_keep][sex_order[i]]<30:
        sex_goodnobs2[sex_order[i]]+=2
    print(i)
    i+=1
    
plt.hist(dra_goodnobs,range=[0,15],histtype='step',color='k',bins=15,label='Draco (current)')
plt.hist(dra_goodnobs2,range=[0,15],histtype='step',color='k',bins=15,label='Draco (proposed)',linestyle=':')
#plt.hist(umi_goodnobs,range=[0,15],histtype='step',color='b',bins=15,label='Ursa Minor')
#plt.hist(umi_goodnobs2,range=[0,15],histtype='step',color='b',bins=15,label='Ursa Minor',linestyle=':')
#plt.hist(sex_goodnobs,range=[0,15],histtype='step',color='r',bins=15,label='Sextans')
#plt.hist(sex_goodnobs2,range=[0,15],histtype='step',color='r',bins=15,label='Sextans',linestyle=':')
plt.yscale('log')
plt.legend(loc=1)
plt.xlim([0,16])
plt.ylim([1,1000])
plt.xlabel('Number of epochs')
plt.ylabel('Number of member stars')

plt.show()
