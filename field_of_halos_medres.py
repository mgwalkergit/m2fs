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

plot_nobs=False
plot_error_adjust=False
previous_work=True
apply_zeropoint=False
compare_sspp=False
compare_lowmetallicity=False
plot_field_of_halos=False
plot_spectra=False
plot_weird_spectra=False
plot_cmdmap=False

m2fs_fits_table_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_medres_fits_table.fits'
hecto_fits_table_filename='/hildafs/projects/phy200028p/mgwalker/nelson/hecto_fits_table.fits'
m2fs_fits_calibrated_filename=m2fs_fits_table_filename.split('m2fs_fits_table.fits')[0]+'m2fs_medres_calibrated.fits'
hecto_fits_calibrated_filename=m2fs_fits_table_filename.split('m2fs_fits_table.fits')[0]+'hecto_calibrated.fits'

frac_v=np.array([1.,1.])
offset_v=np.array([-5.,5.])
sigmaout_v=np.array([-1.,2.])
scale_v=np.array([-0.,0.])
floor_v=np.array([-10.,-10.])
dmax_v=100.

frac_teff=np.array([1.,1.])
offset_teff=np.array([-500.,500.])
sigmaout_teff=np.array([-1.,3.])
scale_teff=np.array([-0.,0.])
floor_teff=np.array([-10.,-10.])
dmax_teff=2000.

frac_logg=np.array([1.,1.])
offset_logg=np.array([-1.,1.])
sigmaout_logg=np.array([-2.,0.7])
scale_logg=np.array([-0.,0.])
floor_logg=np.array([-10.,-10.])
dmax_logg=2.5

frac_z=np.array([1.,1.])
offset_z=np.array([-1.,1.])
sigmaout_z=np.array([-2.,0.7])
scale_z=np.array([-0.,0.])
floor_z=np.array([-10.,-10.])
dmax_z=2.5

frac_alpha=np.array([1.,1.])
offset_alpha=np.array([-1.,1.])
sigmaout_alpha=np.array([-2.,0.3])
scale_alpha=np.array([-0.,0.])
floor_alpha=np.array([-10.,-10.])
dmax_alpha=1.

m2fs_catalog=fits.open(m2fs_fits_table_filename)[1].data
hecto_catalog=fits.open(hecto_fits_table_filename)[1].data

m2fs_ra=m2fs_catalog['ra_deg']
m2fs_dec=m2fs_catalog['dec_deg']
m2fs_v=m2fs_catalog['vlos_raw']
m2fs_sigv=m2fs_catalog['vlos_error']
m2fs_teff=m2fs_catalog['teff_raw']
m2fs_sigteff=m2fs_catalog['teff_error']
m2fs_logg=m2fs_catalog['logg_raw']
m2fs_siglogg=m2fs_catalog['logg_error']
m2fs_z=m2fs_catalog['feh_raw']
m2fs_sigz=m2fs_catalog['feh_error']
m2fs_alpha=m2fs_catalog['mgfe_raw']
m2fs_sigalpha=m2fs_catalog['mgfe_error']
m2fs_v_mean=m2fs_catalog['vlos_raw_mean']
m2fs_sigv_mean=m2fs_catalog['vlos_mean_error']
m2fs_teff_mean=m2fs_catalog['teff_raw_mean']
m2fs_sigteff_mean=m2fs_catalog['teff_mean_error']
m2fs_logg_mean=m2fs_catalog['logg_raw_mean']
m2fs_siglogg_mean=m2fs_catalog['logg_mean_error']
m2fs_z_mean=m2fs_catalog['feh_raw_mean']
m2fs_sigz_mean=m2fs_catalog['feh_mean_error']
m2fs_alpha_mean=m2fs_catalog['mgfe_raw_mean']
m2fs_sigalpha_mean=m2fs_catalog['mgfe_mean_error']
m2fs_obs=m2fs_catalog['obs']
m2fs_nobs=m2fs_catalog['n_obs']
m2fs_goodobs=m2fs_catalog['good_obs']
m2fs_goodnobs=m2fs_catalog['good_n_obs']

hecto_ra=hecto_catalog['ra_deg']
hecto_dec=hecto_catalog['dec_deg']
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
hecto_obs=hecto_catalog['obs']
hecto_nobs=hecto_catalog['n_obs']
hecto_goodobs=hecto_catalog['good_obs']
hecto_goodnobs=hecto_catalog['good_n_obs']

if plot_nobs:
    gs=plt.GridSpec(9,9)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:9,0:9])
    ax1.hist(m2fs_nobs,density=False,histtype='step',color='navy',lw=1,alpha=0.99,label='M2FS',bins=19,range=[1,20])
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$N_{\rm stars}$',fontsize=16)
    ax1.set_xlabel(r'$N_{\rm obs}$',fontsize=16)
    ax1.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    ax1.set_xticklabels(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17'])
    ax1.set_xlim([0,17])
    ax1.legend(loc=1)
    plt.savefig('nobs.pdf',dpi=200)
    plt.show()
    plt.close()

#coords=SkyCoord('22:50:41.07','−58:31:08.3',unit=(u.hourangle,u.degree))
#dist=np.sqrt((m2fs_catalog['ra_deg']-coords.ra.deg)**2+(m2fs_catalog['dec_deg']-coords.dec.deg)**2)*3600
#mike_dist=np.sqrt((mike_catalog['ra_deg']-coords.ra.deg)**2+(mike_catalog['dec_deg']-coords.dec.deg)**2)*3600
#binary=np.where((m2fs_catalog['object']=='tuc2')&(dist<3.)&(m2fs_catalog['vlos_error']<5.))[0]
#mike_binary=np.where((mike_dist<20.)&(mike_catalog['vlos_error']<5.))[0]
#g0=open('tuc2_binary.dat','w')
#for i in range(0,len(binary)):
#    string=str(m2fs_catalog['hjd'][binary[i]])+' '+str(round(m2fs_catalog['vlos'][binary[i]],2))+' '+str(round(m2fs_catalog['vlos_error'][binary[i]],2))+' \n'
#    print(string)
#    g0.write(string)
#for i in range(0,len(mike_binary)):
#    string=str(mike_catalog['hjd'][mike_binary[i]])+' '+str(round(mike_catalog['vlos'][mike_binary[i]],2))+' '+str(round(mike_catalog['vlos_error'][mike_binary[i]],2))+' \n'
#    print(string)
#    g0.write(string)
#g0.close()
#plt.errorbar(m2fs_catalog['hjd'][binary],m2fs_catalog['vlos'][binary],yerr=m2fs_catalog['vlos_error'][binary],color='k',label='M2FS',fmt='.')
#plt.errorbar(mike_catalog['hjd'][mike_binary],mike_catalog['vlos'][mike_binary],yerr=mike_catalog['vlos_error'][mike_binary],color='r',label='MIKE',fmt='.')
#plt.xlabel('Heliocentric Julian Date [days]')
#plt.ylabel('line-of-sight velocity [km/s]')
#plt.legend(loc=4)
#plt.savefig('tuc2binary.pdf',dpi=200)
#plt.show()
#plt.close()
#np.pause()

def bit_solve(n,k):
    temp=n>>(k-1)
    if temp&1:
        return True
    return False

mist=MIST_Isochrone()
mist_track=MIST_EvolutionTrack()
bc_grid=MISTBolometricCorrectionGrid(['G','BP','RP','DECam_g','DECam_r','DECam_i','DECam_z','DECam_Y','SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z','PS_g','PS_r','PS_i','PS_z'])

if plot_error_adjust:
    gs=plt.GridSpec(10,10)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:2,0:2])
    ax2=fig.add_subplot(gs[0:2,2:4])
    ax3=fig.add_subplot(gs[0:2,4:6])
    ax4=fig.add_subplot(gs[0:2,6:8])
    ax5=fig.add_subplot(gs[0:2,8:10])

    ax6=fig.add_subplot(gs[2:4,0:2])
    ax7=fig.add_subplot(gs[2:4,2:4])
    ax8=fig.add_subplot(gs[2:4,4:6])
    ax9=fig.add_subplot(gs[2:4,6:8])
    ax10=fig.add_subplot(gs[2:4,8:10])
    
    x=np.linspace(-10,10,1000)
    gaussian=scipy.stats.norm(loc=0,scale=1)
    y=gaussian.pdf(x)

    m2fs_x,m2fs_sigx,m2fs_repeat_ind1,m2fs_repeat_ind2,m2fs_thing,m2fs_thing_symbol,m2fs_thing_dmax,m2fs_result,m2fs_bestfit=pickle.load(open('/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_ian_vlos_adjust.pkl','rb'))
    m2fs_sigx2=np.sqrt((10.**np.array(m2fs_bestfit['parameters'])[0])**2+(m2fs_sigx*10.**(np.array(m2fs_bestfit['parameters'])[1]))**2)

    ax1.plot(x,y,color='k',lw=0.5)
    ax6.plot(x,y,color='k',lw=0.5)
    
    ax1.hist((m2fs_x[m2fs_repeat_ind1]-m2fs_x[m2fs_repeat_ind2])/np.sqrt(m2fs_sigx[m2fs_repeat_ind1]**2+m2fs_sigx[m2fs_repeat_ind2]**2),range=[-10,10],bins=100,density=True,histtype='step',color='navy',lw=0.5,alpha=0.99,label='M2FS original')
    
    ax1.hist((m2fs_x[m2fs_repeat_ind1]-m2fs_x[m2fs_repeat_ind2])/np.sqrt(m2fs_sigx2[m2fs_repeat_ind1]**2+m2fs_sigx2[m2fs_repeat_ind2]**2),range=[-10,10],bins=100,density=True,histtype='step',color='r',lw=0.5,alpha=0.99,label='M2FS adjusted')
    
    ax1.text(0.05,0.95,r'M2FS',horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=5)
    
    ax1.set_xlim([-4,4])
    #ax1.set_xlabel(r'$\Delta V_{\rm LOS}/\sigma_{\Delta V_{\rm LOS}}$',fontsize=7)
    #ax1.set_ylabel('normalized count',fontsize=7)
    ax6.set_xlim([-4,4])
    ax6.set_xlabel(r'$\Delta V_{\rm LOS}/\sigma_{\Delta V_{\rm LOS}}$',fontsize=7)
    ax6.set_ylabel('                          normalized count',fontsize=7)
    #ax1.legend(loc=1,fontsize=5)
    #ax1.text(0.05,0.93,r'$X=V_{\rm LOS}$',horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=7)
    ax1.set_xticks([-2,0,2])
    #ax1.set_xticklabels(['-2','0','2'],fontsize=7)
    ax6.set_xticks([-2,0,2])
    ax6.set_xticklabels(['-2','0','2'],fontsize=7)
    ax1.set_yticks([])
    ax6.set_yticks([])
    ax1.set_yticklabels([])
    ax6.set_yticklabels([])
    
    m2fs_x,m2fs_sigx,m2fs_repeat_ind1,m2fs_repeat_ind2,m2fs_thing,m2fs_thing_symbol,m2fs_thing_dmax,m2fs_result,m2fs_bestfit=pickle.load(open('/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_ian_teff_adjust.pkl','rb'))
    m2fs_sigx2=np.sqrt((10.**np.array(m2fs_bestfit['parameters'])[0])**2+(m2fs_sigx*10.**(np.array(m2fs_bestfit['parameters'])[1]))**2)

    ax2.plot(x,y,color='k',lw=0.5)
    ax7.plot(x,y,color='k',lw=0.5)
    
    ax2.hist((m2fs_x[m2fs_repeat_ind1]-m2fs_x[m2fs_repeat_ind2])/np.sqrt(m2fs_sigx[m2fs_repeat_ind1]**2+m2fs_sigx[m2fs_repeat_ind2]**2),range=[-10,10],bins=100,density=True,histtype='step',color='navy',lw=0.5,alpha=0.99,label='M2FS original')

    ax2.hist((m2fs_x[m2fs_repeat_ind1]-m2fs_x[m2fs_repeat_ind2])/np.sqrt(m2fs_sigx2[m2fs_repeat_ind1]**2+m2fs_sigx2[m2fs_repeat_ind2]**2),range=[-10,10],bins=100,density=True,histtype='step',color='r',lw=0.5,alpha=0.99,label='M2FS adjusted')
    
    ax2.set_xlim([-4,4])
    #ax2.set_xlabel(r'$\Delta T_{\rm eff}/\sigma_{\Delta T_{\rm eff}}$',fontsize=7)
    ax7.set_xlim([-4,4])
    ax7.set_xlabel(r'$\Delta T_{\rm eff}/\sigma_{\Delta T_{\rm eff}}$',fontsize=7)
    #ax2.legend(loc=1,fontsize=5)
    #ax2.text(0.05,0.93,r'$X=T_{\rm eff}$',horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=7)
    ax2.set_xticks([-2,0,2])
    ax2.set_xticklabels([])
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax7.set_xticks([-2,0,2])
    ax7.set_xticklabels(['-2','0','2'],fontsize=7)
    ax7.set_yticks([])
    ax7.set_yticklabels([])
    
    m2fs_x,m2fs_sigx,m2fs_repeat_ind1,m2fs_repeat_ind2,m2fs_thing,m2fs_thing_symbol,m2fs_thing_dmax,m2fs_result,m2fs_bestfit=pickle.load(open('/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_ian_logg_adjust.pkl','rb'))
    m2fs_sigx2=np.sqrt((10.**np.array(m2fs_bestfit['parameters'])[0])**2+(m2fs_sigx*10.**(np.array(m2fs_bestfit['parameters'])[1]))**2)
    
    ax3.plot(x,y,color='k',lw=0.5)
    ax8.plot(x,y,color='k',lw=0.5)
    
    ax3.hist((m2fs_x[m2fs_repeat_ind1]-m2fs_x[m2fs_repeat_ind2])/np.sqrt(m2fs_sigx[m2fs_repeat_ind1]**2+m2fs_sigx[m2fs_repeat_ind2]**2),range=[-10,10],bins=100,density=True,histtype='step',color='navy',lw=0.5,alpha=0.99,label='M2FS original')
    
    ax3.hist((m2fs_x[m2fs_repeat_ind1]-m2fs_x[m2fs_repeat_ind2])/np.sqrt(m2fs_sigx2[m2fs_repeat_ind1]**2+m2fs_sigx2[m2fs_repeat_ind2]**2),range=[-10,10],bins=100,density=True,histtype='step',color='r',lw=0.5,alpha=0.99,label='M2FS adjusted')
    
    ax3.set_xlim([-4,4])
    #ax3.set_xlabel(r'$\Delta \log g/\sigma_{\Delta \log g}$',fontsize=7)
    ax8.set_xlim([-4,4])
    ax8.set_xlabel(r'$\Delta \log g/\sigma_{\Delta \log g}$',fontsize=7)
    #ax3.legend(loc=1,fontsize=5)
    #ax3.text(0.05,0.93,r'$X=\log g$',horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=7)
    ax3.set_xticks([-2,0,2])
    ax3.set_xticklabels([])
    ax3.set_yticks([])
    ax3.set_yticklabels([])
    ax8.set_xticks([-2,0,2])
    ax8.set_xticklabels(['-2','0','2'],fontsize=7)
    ax8.set_yticks([])
    ax8.set_yticklabels([])
    
    m2fs_x,m2fs_sigx,m2fs_repeat_ind1,m2fs_repeat_ind2,m2fs_thing,m2fs_thing_symbol,m2fs_thing_dmax,m2fs_result,m2fs_bestfit=pickle.load(open('/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_ian_z_adjust.pkl','rb'))
    m2fs_sigx2=np.sqrt((10.**np.array(m2fs_bestfit['parameters'])[0])**2+(m2fs_sigx*10.**(np.array(m2fs_bestfit['parameters'])[1]))**2)
    
    ax4.plot(x,y,color='k',lw=0.5)
    ax9.plot(x,y,color='k',lw=0.5)
    
    ax4.hist((m2fs_x[m2fs_repeat_ind1]-m2fs_x[m2fs_repeat_ind2])/np.sqrt(m2fs_sigx[m2fs_repeat_ind1]**2+m2fs_sigx[m2fs_repeat_ind2]**2),range=[-10,10],bins=100,density=True,histtype='step',color='navy',lw=0.5,alpha=0.99,label='M2FS original')
    
    ax4.hist((m2fs_x[m2fs_repeat_ind1]-m2fs_x[m2fs_repeat_ind2])/np.sqrt(m2fs_sigx2[m2fs_repeat_ind1]**2+m2fs_sigx2[m2fs_repeat_ind2]**2),range=[-10,10],bins=100,density=True,histtype='step',color='r',lw=0.5,alpha=0.99,label='M2FS adjusted')
    
    ax4.set_xlim([-4,4])
    #ax4.set_xlabel(r'$\Delta [\mathrm{Fe/H}]/\sigma_{\Delta [\mathrm{Fe/H}]}$',fontsize=7)
    ax9.set_xlim([-4,4])
    ax9.set_xlabel(r'$\Delta [\mathrm{Fe/H}]/\sigma_{\Delta [\mathrm{Fe/H}]}$',fontsize=7)
    #ax4.legend(loc=1,fontsize=5)
    #ax4.text(0.05,0.93,r'$X=$[Fe/H]',horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes,fontsize=7)
    ax4.set_xticks([-2,0,2])
    ax4.set_xticklabels([])
    ax4.set_yticks([])
    ax4.set_yticklabels([])
    ax9.set_xticks([-2,0,2])
    ax9.set_xticklabels(['-2','0','2'],fontsize=7)
    ax9.set_yticks([])
    ax9.set_yticklabels([])
    
    m2fs_x,m2fs_sigx,m2fs_repeat_ind1,m2fs_repeat_ind2,m2fs_thing,m2fs_thing_symbol,m2fs_thing_dmax,m2fs_result,m2fs_bestfit=pickle.load(open('/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_ian_alpha_adjust.pkl','rb'))
    m2fs_sigx2=np.sqrt((10.**np.array(m2fs_bestfit['parameters'])[0])**2+(m2fs_sigx*10.**(np.array(m2fs_bestfit['parameters'])[1]))**2)
    
    ax5.plot(x,y,color='k',lw=0.5)
    ax10.plot(x,y,color='k',lw=0.5)
    
    ax5.hist((m2fs_x[m2fs_repeat_ind1]-m2fs_x[m2fs_repeat_ind2])/np.sqrt(m2fs_sigx[m2fs_repeat_ind1]**2+m2fs_sigx[m2fs_repeat_ind2]**2),range=[-10,10],bins=100,density=True,histtype='step',color='navy',lw=0.5,alpha=0.99,label='before')

    ax5.hist((m2fs_x[m2fs_repeat_ind1]-m2fs_x[m2fs_repeat_ind2])/np.sqrt(m2fs_sigx2[m2fs_repeat_ind1]**2+m2fs_sigx2[m2fs_repeat_ind2]**2),range=[-10,10],bins=100,density=True,histtype='step',color='r',lw=0.5,alpha=0.99,label='after')

    ax5.set_xlim([-4,4])
#ax5.set_xlabel(r'$\Delta [\mathrm{Mg/Fe}]/\sigma_{\Delta [\mathrm{Mg/Fe}]}$',fontsize=7)
    ax5.legend(loc=1,fontsize=5,borderaxespad=0,handlelength=1)
    ax10.set_xlim([-4,4])
    ax10.set_xlabel(r'$\Delta [\mathrm{Mg/Fe}]/\sigma_{\Delta [\mathrm{Mg/Fe}]}$',fontsize=7)
    #ax10.legend(loc=1,fontsize=5,borderaxespad=0)
    #ax5.text(0.05,0.93,r'$X=$[Mg/Fe]',horizontalalignment='left',verticalalignment='top',transform=ax5.transAxes,fontsize=7)
    ax5.set_xticks([-2,0,2])
    ax5.set_xticklabels(['-2','0','2'],fontsize=7)
    ax5.set_yticks([])
    ax5.set_yticklabels([])
    ax10.set_xticks([-2,0,2])
    ax10.set_xticklabels(['-2','0','2'],fontsize=7)
    ax10.set_yticks([])
    ax10.set_yticklabels([])
    
    plt.savefig('medres_error_adjust.pdf',dpi=200)
    plt.show()
    plt.close()

    gs=plt.GridSpec(12,12)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:2,0:4])
    ax2=fig.add_subplot(gs[2:4,0:4])
    ax3=fig.add_subplot(gs[4:6,0:4])
    ax4=fig.add_subplot(gs[6:8,0:4])
    ax5=fig.add_subplot(gs[8:10,0:4])
    ax6=fig.add_subplot(gs[10:12,0:4])

    x=m2fs_catalog['gaia_gmag']
    y1=m2fs_catalog['sn_ratio']
    y2=m2fs_catalog['vlos_raw_error']
    y3=m2fs_catalog['teff_raw_error']
    y4=m2fs_catalog['logg_raw_error']
    y5=m2fs_catalog['feh_raw_error']
    y6=m2fs_catalog['mgfe_raw_error']
    z2=np.full(len(m2fs_catalog),'navy')
    z=z2
    ind=np.arange(len(x))
    ind2=np.random.permutation(ind)
    
    ax1.scatter(x[ind2],y1[ind2],color=z[ind2],s=0.25,alpha=0.3,rasterized=True)
    ax1.set_yscale('log')
    ax1.set_xticks([])
    ax1.set_ylim([0.05,200])
    ax1.set_yticks([0.1,1,10,100])
    ax1.set_ylabel('S/N/pixel',fontsize=8)
    ax1.set_yticklabels(['0.1','1','10','100'],fontsize=8)
    ax1.set_xlim([21,12])
    ax1.scatter([-999],[-999],s=1,color='navy',alpha=0.5,label='M2FS')
    ax1.scatter([-999],[-999],s=1,color='r',alpha=0.5,label='Hecto')
    ax1.legend(loc=4,fontsize=6,borderaxespad=0)

    ax2.scatter(x[ind2],y2[ind2],color=z[ind2],s=0.25,alpha=0.3,rasterized=True)
    ax2.set_yscale('log')
    ax2.set_xscale('linear')
    ax2.set_xticks([])
    ax2.set_yticks([0.1,1,10])
    ax2.set_yticklabels(['0.1','1','10'],fontsize=8)
    ax2.set_ylabel(r'$\sigma_{V_{\rm LOS}}$ [km/s]',fontsize=8)
    ax2.set_ylim([0.05,90])
    ax2.set_xlim([21,12])

    ax3.scatter(x[ind2],y3[ind2],color=z[ind2],s=0.25,alpha=0.3,rasterized=True)
    ax3.set_yscale('log')
    ax3.set_xscale('linear')
    ax3.set_xticks([])
    ax3.set_ylabel(r'$\sigma_{T_{\rm eff}}$ [K]',fontsize=8)
    ax3.set_ylim([5,2000])
    ax3.set_yticks([10,100,1000])
    ax3.set_yticklabels(['10','100','1000'],fontsize=8)
    ax3.set_xlim([21,12])

    ax4.scatter(x[ind2],y4[ind2],color=z[ind2],s=0.25,alpha=0.3,rasterized=True)
    ax4.set_yscale('log')
    ax4.set_xscale('linear')
    ax4.set_xticks([])
    ax4.set_yticks([0.1,1])
    ax4.set_yticklabels([0.1,1],fontsize=8)
    ax4.set_ylabel(r'$\sigma_{\log g}$',fontsize=8)
    ax4.set_ylim([0.05,2])
    ax4.set_xlim([21,12])

    ax5.scatter(x[ind2],y5[ind2],color=z[ind2],s=0.25,alpha=0.3,rasterized=True)
    ax5.set_yscale('log')
    ax5.set_xscale('linear')
    ax5.set_xticks([])
    ax5.set_yticks([0.01,0.1,1])
    ax5.set_yticklabels(['0.01','0.1','1'],fontsize=8)
    ax5.set_ylabel(r'$\sigma_{\rm [Fe/H]}$',fontsize=8)
    ax5.set_ylim([0.005,2])
    ax5.set_xlim([21,12])
    
    ax6.scatter(x[ind2],y6[ind2],color=z[ind2],s=0.25,alpha=0.3,rasterized=True)
    ax6.set_yscale('log')
    ax6.set_xscale('linear')
    ax6.set_xlim([21,12])
    ax6.set_yticks([0.01,0.1])
    ax6.set_yticklabels(['0.01','0.1'],fontsize=8)
    ax6.set_ylabel(r'$\sigma_{\rm [Mg/Fe]}$',fontsize=8)
    ax6.set_ylim([0.01,0.9])
    ax6.set_xticks([20,18,16,14])
    ax6.set_xticklabels(['20','18','16','14'],fontsize=8)
    ax6.set_xlabel('G mag',fontsize=8)
    plt.savefig('sn_ratio.pdf',dpi=200)
    plt.show()
    plt.close()

if previous_work:

    apogee_objid,apogee_ra,apogee_dec,apogee_vlos,apogee_sigvlos,apogee_stdvlos,apogee_teff,apogee_sigteff,apogee_logg,apogee_siglogg,apogee_z,apogee_sigz,apogee_flagz,apogee_mg,apogee_sigmg,apogee_flagmg,apogee_alpha,apogee_sigalpha,apogee_param,apogee_sigparam,apogee_flag,apogee_nvisits,apogee_rv_flag,apogee_aspcap_flag=sqlutil.get('''select apogee_id,ra,dec,vhelio_avg,verr,vscatter,teff,teff_err,logg,logg_err,fe_h,fe_h_err,fe_h_flag,mg_fe,mg_fe_err,mg_fe_flag,alpha_m,alpha_m_err,param,param_cov,rv_flag,aspcapflag,nvisits,aspcapflag from apogee_dr17.allstar where teff!= \'nan\' and logg!=\'nan\' and fe_h!=\'nan\' and m_h!=\'nan\' limit 1000000''',db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')
    binary_repr_v=np.vectorize(np.binary_repr)
    bit_solve_v=np.vectorize(bit_solve)

    kirby=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/kirby10.fits')[1].data
    kirby_ra=[]
    kirby_dec=[]
    kirby_teff=[]
    kirby_sigteff=[]
    kirby_stdteff=[]
    kirby_logg=[]
    kirby_siglogg=[]
    kirby_stdlogg=[]
    kirby_z=[]
    kirby_sigz=[]
    kirby_stdz=[]
    kirby_alpha=[]
    kirby_sigalpha=[]
    kirby_stdalpha=[]
    kirby_used=np.zeros(len(kirby),dtype='int')
    for i in range(0,len(kirby)):
        if ((kirby_used[i]==0)&(kirby['Teff'][i]==kirby['Teff'][i])&(kirby['logg'][i]==kirby['logg'][i])&(kirby['__Fe_H_'][i]==kirby['__Fe_H_'][i])&(kirby['__Mg_Fe_'][i]==kirby['__Mg_Fe_'][i])):
            kirby_ra.append(kirby['RAJ2000'][i])
            kirby_dec.append(kirby['DEJ2000'][i])
            this=np.where((kirby['dSph']==kirby['dSph'][i])&(kirby['Name']==kirby['Name'][i])&(kirby['Teff']==kirby['Teff'])&(kirby['logg']==kirby['logg'])&(kirby['__Fe_H_']==kirby['__Fe_H_'])&(kirby['__Mg_Fe_']==kirby['__Mg_Fe_']))[0][0]
            kirby_used[this]=1

            xxx=np.sum(kirby['Teff'][this]/kirby['e_Teff'][this]**2)/np.sum(1./kirby['e_Teff'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./kirby['e_Teff'][this]**2))
            if sigxxx!=sigxxx:
                np.pause()
            stdxxx=np.sqrt(np.sum((kirby['Teff'][this]-xxx)**2)/np.sum(1./kirby['e_Teff'][this]**2))
            kirby_teff.append(xxx)
            kirby_sigteff.append(sigxxx)
            kirby_stdteff.append(stdxxx)

            xxx=np.sum(kirby['logg'][this]/kirby['e_logg'][this]**2)/np.sum(1./kirby['e_logg'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./kirby['e_logg'][this]**2))
            stdxxx=np.sqrt(np.sum((kirby['logg'][this]-xxx)**2)/np.sum(1./kirby['e_logg'][this]**2))
            kirby_logg.append(xxx)
            kirby_siglogg.append(sigxxx)
            kirby_stdlogg.append(stdxxx)

            xxx=np.sum(kirby['__Fe_H_'][this]/kirby['e__Fe_H_'][this]**2)/np.sum(1./kirby['e__Fe_H_'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./kirby['e__Fe_H_'][this]**2))
            stdxxx=np.sqrt(np.sum((kirby['__Fe_H_'][this]-xxx)**2)/np.sum(1./kirby['e__Fe_H_'][this]**2))
            kirby_z.append(xxx)
            kirby_sigz.append(sigxxx)
            kirby_stdz.append(stdxxx)

            xxx=np.sum(kirby['__Mg_Fe_'][this]/kirby['e__Mg_Fe_'][this]**2)/np.sum(1./kirby['e__Mg_Fe_'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./kirby['e__Mg_Fe_'][this]**2))
            stdxxx=np.sqrt(np.sum((kirby['__Mg_Fe_'][this]-xxx)**2)/np.sum(1./kirby['e__Mg_Fe_'][this]**2))
            kirby_alpha.append(xxx)
            kirby_sigalpha.append(sigxxx)
            kirby_stdalpha.append(stdxxx)

    kirby_ra=np.array(kirby_ra)
    kirby_dec=np.array(kirby_dec)
    
    kirby_teff=np.array(kirby_teff)
    kirby_sigteff=np.array(kirby_sigteff)
    kirby_stdteff=np.array(kirby_stdteff)

    kirby_logg=np.array(kirby_logg)
    kirby_siglogg=np.array(kirby_siglogg)
    kirby_stdlogg=np.array(kirby_stdlogg)

    kirby_z=np.array(kirby_z)
    kirby_sigz=np.array(kirby_sigz)
    kirby_stdz=np.array(kirby_stdz)

    kirby_alpha=np.array(kirby_alpha)
    kirby_sigalpha=np.array(kirby_sigalpha)
    kirby_stdalpha=np.array(kirby_stdalpha)


    with open('/hildafs/projects/phy200028p/mgwalker/m2fs/pace21_fornax.dat') as f:
        data=f.readlines()[34:]
    pace21_fornax_id=[]
    pace21_fornax_ra=[]
    pace21_fornax_dec=[]
    pace21_fornax_hjd=[]
    pace21_fornax_snratio=[]
    pace21_fornax_vlos=[]
    pace21_fornax_e_vlos=[]
    pace21_fornax_teff=[]
    pace21_fornax_e_teff=[]
    pace21_fornax_logg=[]
    pace21_fornax_e_logg=[]
    pace21_fornax_feh=[]
    pace21_fornax_e_feh=[]    
    for line in data:
        p=line.split()
        pace21_fornax_id.append(p[0])
        pace21_fornax_ra.append(float(p[1]))
        pace21_fornax_dec.append(float(p[2]))
        pace21_fornax_hjd.append(float(p[3]))
        pace21_fornax_snratio.append(float(p[4]))
        pace21_fornax_vlos.append(float(p[5]))
        pace21_fornax_e_vlos.append(float(p[6]))
        pace21_fornax_teff.append(float(p[9]))
        pace21_fornax_e_teff.append(float(p[10]))
        pace21_fornax_logg.append(float(p[13]))
        pace21_fornax_e_logg.append(float(p[14]))
        pace21_fornax_feh.append(float(p[17]))
        pace21_fornax_e_feh.append(float(p[18]))
    pace21_fornax_ra=np.array(pace21_fornax_ra)
    pace21_fornax_dec=np.array(pace21_fornax_dec)
    pace21_fornax_hjd=np.array(pace21_fornax_hjd)
    pace21_fornax_snratio=np.array(pace21_fornax_snratio)
    pace21_fornax_vlos=np.array(pace21_fornax_vlos)
    pace21_fornax_e_vlos=np.array(pace21_fornax_e_vlos)
    pace21_fornax_teff=np.array(pace21_fornax_teff)
    pace21_fornax_e_teff=np.array(pace21_fornax_e_teff)
    pace21_fornax_logg=np.array(pace21_fornax_logg)
    pace21_fornax_e_logg=np.array(pace21_fornax_e_logg)
    pace21_fornax_feh=np.array(pace21_fornax_feh)
    pace21_fornax_e_feh=np.array(pace21_fornax_e_feh)

    with open('/hildafs/projects/phy200028p/mgwalker/m2fs/caldwell17_cra2.dat') as f:
        data=f.readlines()[31:]
    caldwell17_cra2_id=[]
    caldwell17_cra2_rah=[]
    caldwell17_cra2_ram=[]
    caldwell17_cra2_ras=[]
    caldwell17_cra2_decd=[]
    caldwell17_cra2_decm=[]
    caldwell17_cra2_decs=[]
    caldwell17_cra2_hjd=[]
    caldwell17_cra2_snratio=[]
    caldwell17_cra2_vlos=[]
    caldwell17_cra2_e_vlos=[]
    caldwell17_cra2_teff=[]
    caldwell17_cra2_e_teff=[]
    caldwell17_cra2_logg=[]
    caldwell17_cra2_e_logg=[]
    caldwell17_cra2_feh=[]
    caldwell17_cra2_e_feh=[]    
    for line in data:
        p=line.split()
        caldwell17_cra2_id.append(p[0])
        caldwell17_cra2_rah.append(float(p[1]))
        caldwell17_cra2_ram.append(float(p[2]))
        caldwell17_cra2_ras.append(float(p[3]))
        caldwell17_cra2_decd.append(float(p[4]))
        caldwell17_cra2_decm.append(float(p[5]))
        caldwell17_cra2_decs.append(float(p[6]))
        caldwell17_cra2_hjd.append(float(p[9]))
        caldwell17_cra2_snratio.append(float(p[10]))
        caldwell17_cra2_vlos.append(float(p[11]))
        caldwell17_cra2_e_vlos.append(float(p[12]))
        caldwell17_cra2_teff.append(float(p[13]))
        caldwell17_cra2_e_teff.append(float(p[14]))
        caldwell17_cra2_logg.append(float(p[15]))
        caldwell17_cra2_e_logg.append(float(p[16]))
        caldwell17_cra2_feh.append(float(p[17]))
        caldwell17_cra2_e_feh.append(float(p[18]))
    caldwell17_cra2_rah=np.array(caldwell17_cra2_rah)
    caldwell17_cra2_ram=np.array(caldwell17_cra2_ram)
    caldwell17_cra2_ras=np.array(caldwell17_cra2_ras)
    caldwell17_cra2_decd=np.array(caldwell17_cra2_decd)
    caldwell17_cra2_decm=np.array(caldwell17_cra2_decm)
    caldwell17_cra2_decs=np.array(caldwell17_cra2_decs)
    caldwell17_cra2_hjd=np.array(caldwell17_cra2_hjd)
    caldwell17_cra2_snratio=np.array(caldwell17_cra2_snratio)
    caldwell17_cra2_vlos=np.array(caldwell17_cra2_vlos)
    caldwell17_cra2_e_vlos=np.array(caldwell17_cra2_e_vlos)
    caldwell17_cra2_teff=np.array(caldwell17_cra2_teff)
    caldwell17_cra2_e_teff=np.array(caldwell17_cra2_e_teff)
    caldwell17_cra2_logg=np.array(caldwell17_cra2_logg)
    caldwell17_cra2_e_logg=np.array(caldwell17_cra2_e_logg)
    caldwell17_cra2_feh=np.array(caldwell17_cra2_feh)
    caldwell17_cra2_e_feh=np.array(caldwell17_cra2_e_feh)
    rah=[]
    decd=[]
    for i in range(0,len(caldwell17_cra2_rah)):
        rah.append(str(int(caldwell17_cra2_rah[i]))+':'+str(int(caldwell17_cra2_ram[i]))+':'+str(caldwell17_cra2_ras[i]))
        decd.append(str(int(caldwell17_cra2_decd[i]))+':'+str(int(caldwell17_cra2_decm[i]))+':'+str(caldwell17_cra2_decs[i]))
    coords=SkyCoord(rah,decd,unit=(u.hourangle,u.deg))
    caldwell17_cra2_ra=coords.ra.deg
    caldwell17_cra2_dec=coords.dec.deg
    
    sspp_walker15_draco=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/walker15_draco.fits')[1].data
    sspp_walker15_ret2=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/walker15_ret2.fits')[1].data
    sspp_walker16_tuc2gru1=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/walker16_tuc2gru1.fits')[1].data
    sspp_spencer17_leo2=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/spencer17_leo2.fits')[1].data
    sspp_spencer18_umi=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/spencer18_umi.fits')[1].data
    sspp_koposov18_hyd1=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/koposov18_hyd1.fits')[1].data
    
    col1=fits.Column(name='ra',format='D',array=np.concatenate([sspp_walker15_draco['RAJ2000'],sspp_walker15_ret2['RAJ2000'],sspp_walker16_tuc2gru1['RAJ2000'],sspp_spencer17_leo2['RAJ2000'],sspp_spencer18_umi['RAJ2000'],sspp_koposov18_hyd1['RAJ2000'],pace21_fornax_ra,caldwell17_cra2_ra]))
    col2=fits.Column(name='dec',format='D',array=np.concatenate([sspp_walker15_draco['DEJ2000'],sspp_walker15_ret2['DEJ2000'],sspp_walker16_tuc2gru1['DEJ2000'],sspp_spencer17_leo2['DEJ2000'],sspp_spencer18_umi['DEJ2000'],sspp_koposov18_hyd1['DEJ2000'],pace21_fornax_dec,caldwell17_cra2_dec]))
    col3=fits.Column(name='vlos',format='D',array=np.concatenate([sspp_walker15_draco['vlos'],sspp_walker15_ret2['vlos'],sspp_walker16_tuc2gru1['vlos'],sspp_spencer17_leo2['RVel'],sspp_spencer18_umi['vlosmean'],sspp_koposov18_hyd1['HRV'],pace21_fornax_vlos,caldwell17_cra2_vlos]))
    col4=fits.Column(name='vlos_err',format='D',array=np.concatenate([sspp_walker15_draco['e_vlos'],sspp_walker15_ret2['e_vlos'],sspp_walker16_tuc2gru1['e_vlos'],sspp_spencer17_leo2['e_RVel'],sspp_spencer18_umi['e_vlosmean'],sspp_koposov18_hyd1['e_HRV'],pace21_fornax_e_vlos,caldwell17_cra2_e_vlos]))
    col5=fits.Column(name='teff',format='D',array=np.concatenate([sspp_walker15_draco['teff'],sspp_walker15_ret2['teff'],sspp_walker16_tuc2gru1['teff'],sspp_spencer17_leo2['Teff'],sspp_spencer18_umi['Teffmean'],sspp_koposov18_hyd1['Teff'],pace21_fornax_teff,caldwell17_cra2_teff]))
    col6=fits.Column(name='teff_err',format='D',array=np.concatenate([sspp_walker15_draco['e_teff'],sspp_walker15_ret2['e_teff'],sspp_walker16_tuc2gru1['e_teff'],sspp_spencer17_leo2['e_Teff'],sspp_spencer18_umi['e_Teffmean'],sspp_koposov18_hyd1['e_Teff'],pace21_fornax_e_teff,caldwell17_cra2_e_teff]))
    col7=fits.Column(name='logg',format='D',array=np.concatenate([sspp_walker15_draco['logg'],sspp_walker15_ret2['logg'],sspp_walker16_tuc2gru1['logg'],sspp_spencer17_leo2['log_g_'],sspp_spencer18_umi['log_g_mean'],sspp_koposov18_hyd1['loggmean'],pace21_fornax_logg,caldwell17_cra2_logg]))
    col8=fits.Column(name='logg_err',format='D',array=np.concatenate([sspp_walker15_draco['e_logg'],sspp_walker15_ret2['e_logg'],sspp_walker16_tuc2gru1['e_logg'],sspp_spencer17_leo2['e_log_g_'],sspp_spencer18_umi['e_log_g_mean'],sspp_koposov18_hyd1['e_loggmean'],pace21_fornax_e_logg,caldwell17_cra2_e_logg]))
    col9=fits.Column(name='feh',format='D',array=np.concatenate([sspp_walker15_draco['__Fe_H_'],sspp_walker15_ret2['__Fe_H_'],sspp_walker16_tuc2gru1['__Fe_H_'],sspp_spencer17_leo2['__Fe_H_'],sspp_spencer18_umi['__Fe_H_mean'],sspp_koposov18_hyd1['__Fe_H_'],pace21_fornax_feh,caldwell17_cra2_feh]))
    col10=fits.Column(name='feh_err',format='D',array=np.concatenate([sspp_walker15_draco['e__Fe_H_'],sspp_walker15_ret2['e__Fe_H_'],sspp_walker16_tuc2gru1['e__Fe_H_'],sspp_spencer17_leo2['e__Fe_H_'],sspp_spencer18_umi['e__Fe_H_mean'],sspp_koposov18_hyd1['e__Fe_H_'],pace21_fornax_e_feh,caldwell17_cra2_e_feh]))
    cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10])
    sspp=fits.FITS_rec.from_columns(cols)
    sspp_table=fits.BinTableHDU.from_columns(cols)
    sspp_table.writeto('sspp_published.fits',overwrite=True)
    
    sspp_ra=[]
    sspp_dec=[]
    sspp_vlos=[]
    sspp_sigvlos=[]
    sspp_stdvlos=[]
    sspp_teff=[]
    sspp_sigteff=[]
    sspp_stdteff=[]
    sspp_logg=[]
    sspp_siglogg=[]
    sspp_stdlogg=[]
    sspp_z=[]
    sspp_sigz=[]
    sspp_stdz=[]
    sspp_used=np.zeros(len(sspp),dtype='int')
    for i in range(0,len(sspp)):
        if ((sspp_used[i]==0)&(sspp['vlos'][i]==sspp['vlos'][i])&(sspp['teff'][i]==sspp['teff'][i])&(sspp['logg'][i]==sspp['logg'][i])&(sspp['feh'][i]==sspp['feh'][i])):
            sspp_ra.append(sspp['ra'][i])
            sspp_dec.append(sspp['dec'][i])
            dist=np.sqrt((sspp['ra']-sspp['ra'][i])**2+(sspp['dec']-sspp['dec'][i])**2)*3600.
            this=np.where((dist<1.)&(sspp['teff']==sspp['teff'])&(sspp['logg']==sspp['logg'])&(sspp['feh']==sspp['feh']))[0]
            sspp_used[this]=1

            xxx=np.sum(sspp['vlos'][this]/sspp['vlos_err'][this]**2)/np.sum(1./sspp['vlos_err'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./sspp['vlos_err'][this]**2))
            if sigxxx!=sigxxx:
                np.pause()
            stdxxx=np.sqrt(np.sum((sspp['vlos'][this]-xxx)**2)/np.sum(1./sspp['vlos_err'][this]**2))
            sspp_vlos.append(xxx)
            sspp_sigvlos.append(sigxxx)
            sspp_stdvlos.append(stdxxx)

            xxx=np.sum(sspp['teff'][this]/sspp['teff_err'][this]**2)/np.sum(1./sspp['teff_err'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./sspp['teff_err'][this]**2))
            if sigxxx!=sigxxx:
                np.pause()
            stdxxx=np.sqrt(np.sum((sspp['teff'][this]-xxx)**2)/np.sum(1./sspp['teff_err'][this]**2))
            sspp_teff.append(xxx)
            sspp_sigteff.append(sigxxx)
            sspp_stdteff.append(stdxxx)

            xxx=np.sum(sspp['logg'][this]/sspp['logg_err'][this]**2)/np.sum(1./sspp['logg_err'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./sspp['logg_err'][this]**2))
            stdxxx=np.sqrt(np.sum((sspp['logg'][this]-xxx)**2)/np.sum(1./sspp['logg_err'][this]**2))
            sspp_logg.append(xxx)
            sspp_siglogg.append(sigxxx)
            sspp_stdlogg.append(stdxxx)

            xxx=np.sum(sspp['feh'][this]/sspp['feh_err'][this]**2)/np.sum(1./sspp['feh_err'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./sspp['feh_err'][this]**2))
            stdxxx=np.sqrt(np.sum((sspp['feh'][this]-xxx)**2)/np.sum(1./sspp['feh_err'][this]**2))
            sspp_z.append(xxx)
            sspp_sigz.append(sigxxx)
            sspp_stdz.append(stdxxx)

    sspp_ra=np.array(sspp_ra)
    sspp_dec=np.array(sspp_dec)
    sspp_vlos=np.array(sspp_vlos)
    sspp_sigvlos=np.array(sspp_sigvlos)
    sspp_stdvlos=np.array(sspp_stdvlos)
    sspp_teff=np.array(sspp_teff)
    sspp_sigteff=np.array(sspp_sigteff)
    sspp_stdteff=np.array(sspp_stdteff)
    sspp_logg=np.array(sspp_logg)
    sspp_siglogg=np.array(sspp_siglogg)
    sspp_stdlogg=np.array(sspp_stdlogg)
    sspp_z=np.array(sspp_z)
    sspp_sigz=np.array(sspp_sigz)
    sspp_stdz=np.array(sspp_stdz)

    h3_sextans=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/h3_sextans_rcat.fits')[1].data
    h3_draco=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/h3_draco_rcat.fits')[1].data
    h3_ursaminor=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/h3_umin_rcat.fits')[1].data

    col1=fits.Column(name='GAIAEDR3_RA',format='D',array=np.concatenate([h3_sextans['GAIAEDR3_RA'],h3_draco['GAIAEDR3_RA'],h3_ursaminor['GAIAEDR3_RA']]))
    col2=fits.Column(name='GAIAEDR3_DEC',format='D',array=np.concatenate([h3_sextans['GAIAEDR3_DEC'],h3_draco['GAIAEDR3_DEC'],h3_ursaminor['GAIAEDR3_DEC']]))
    col3=fits.Column(name='Vrad',format='D',array=np.concatenate([h3_sextans['Vrad'],h3_draco['Vrad'],h3_ursaminor['Vrad']]))
    col4=fits.Column(name='Vrad_err',format='D',array=np.concatenate([h3_sextans['Vrad_err'],h3_draco['Vrad_err'],h3_ursaminor['Vrad_err']]))
    col5=fits.Column(name='Teff',format='D',array=np.concatenate([h3_sextans['Teff'],h3_draco['Teff'],h3_ursaminor['Teff']]))
    col6=fits.Column(name='Teff_err',format='D',array=np.concatenate([h3_sextans['Teff_err'],h3_draco['Teff_err'],h3_ursaminor['Teff_err']]))
    col7=fits.Column(name='log(g)',format='D',array=np.concatenate([h3_sextans['log(g)'],h3_draco['log(g)'],h3_ursaminor['log(g)']]))
    col8=fits.Column(name='log(g)_err',format='D',array=np.concatenate([h3_sextans['log(g)_err'],h3_draco['log(g)_err'],h3_ursaminor['log(g)_err']]))
    col9=fits.Column(name='[Fe/H]',format='D',array=np.concatenate([h3_sextans['[Fe/H]'],h3_draco['[Fe/H]'],h3_ursaminor['[Fe/H]']]))
    col10=fits.Column(name='[Fe/H]_err',format='D',array=np.concatenate([h3_sextans['[Fe/H]_err'],h3_draco['[Fe/H]_err'],h3_ursaminor['[Fe/H]_err']]))
    col11=fits.Column(name='[a/Fe]',format='D',array=np.concatenate([h3_sextans['[a/Fe]'],h3_draco['[a/Fe]'],h3_ursaminor['[a/Fe]']]))
    col12=fits.Column(name='[a/Fe]_err',format='D',array=np.concatenate([h3_sextans['[a/Fe]_err'],h3_draco['[a/Fe]_err'],h3_ursaminor['[a/Fe]_err']]))
    cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12])
    h3=fits.FITS_rec.from_columns(cols)

    h3_ra=[]
    h3_dec=[]
    h3_vlos=[]
    h3_sigvlos=[]
    h3_stdvlos=[]
    h3_teff=[]
    h3_sigteff=[]
    h3_stdteff=[]
    h3_logg=[]
    h3_siglogg=[]
    h3_stdlogg=[]
    h3_z=[]
    h3_sigz=[]
    h3_stdz=[]
    h3_alpha=[]
    h3_sigalpha=[]
    h3_stdalpha=[]
    h3_used=np.zeros(len(h3),dtype='int')
    for i in range(0,len(h3)):
        if ((h3_used[i]==0)&(h3['Vrad'][i]==h3['Vrad'][i])&(h3['Teff'][i]==h3['Teff'][i])&(h3['log(g)'][i]==h3['log(g)'][i])&(h3['[Fe/H]'][i]==h3['[Fe/H]'][i])&(h3['[a/Fe]'][i]==h3['[a/Fe]'][i])):
            h3_ra.append(h3['GAIAEDR3_RA'][i])
            h3_dec.append(h3['GAIAEDR3_DEC'][i])
            this=np.where((h3['GAIAEDR3_RA']==h3['GAIAEDR3_RA'][i])&(h3['GAIAEDR3_DEC']==h3['GAIAEDR3_DEC'][i])&(h3['Vrad']==h3['Vrad'])&(h3['Teff']==h3['Teff'])&(h3['log(g)']==h3['log(g)'])&(h3['[Fe/H]']==h3['[Fe/H]'])&(h3['[a/Fe]']==h3['[a/Fe]']))[0]
            h3_used[this]=1

            xxx=np.sum(h3['Vrad'][this]/h3['Vrad_err'][this]**2)/np.sum(1./h3['Vrad_err'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./h3['Vrad_err'][this]**2))
            if sigxxx!=sigxxx:
                np.pause()
            stdxxx=np.sqrt(np.sum((h3['Vrad'][this]-xxx)**2)/np.sum(1./h3['Vrad_err'][this]**2))
            h3_vlos.append(xxx)
            h3_sigvlos.append(sigxxx)
            h3_stdvlos.append(stdxxx)

            xxx=np.sum(h3['Teff'][this]/h3['Teff_err'][this]**2)/np.sum(1./h3['Teff_err'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./h3['Teff_err'][this]**2))
            if sigxxx!=sigxxx:
                np.pause()
            stdxxx=np.sqrt(np.sum((h3['Teff'][this]-xxx)**2)/np.sum(1./h3['Teff_err'][this]**2))
            h3_teff.append(xxx)
            h3_sigteff.append(sigxxx)
            h3_stdteff.append(stdxxx)

            xxx=np.sum(h3['log(g)'][this]/h3['log(g)_err'][this]**2)/np.sum(1./h3['log(g)_err'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./h3['log(g)_err'][this]**2))
            stdxxx=np.sqrt(np.sum((h3['log(g)'][this]-xxx)**2)/np.sum(1./h3['log(g)_err'][this]**2))
            h3_logg.append(xxx)
            h3_siglogg.append(sigxxx)
            h3_stdlogg.append(stdxxx)

            xxx=np.sum(h3['[Fe/H]'][this]/h3['[Fe/H]_err'][this]**2)/np.sum(1./h3['[Fe/H]_err'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./h3['[Fe/H]_err'][this]**2))
            stdxxx=np.sqrt(np.sum((h3['[Fe/H]'][this]-xxx)**2)/np.sum(1./h3['[Fe/H]_err'][this]**2))
            h3_z.append(xxx)
            h3_sigz.append(sigxxx)
            h3_stdz.append(stdxxx)

            xxx=np.sum(h3['[a/Fe]'][this]/h3['[a/Fe]_err'][this]**2)/np.sum(1./h3['[a/Fe]_err'][this]**2)
            sigxxx=1./np.sqrt(np.sum(1./h3['[a/Fe]_err'][this]**2))
            stdxxx=np.sqrt(np.sum((h3['[a/Fe]'][this]-xxx)**2)/np.sum(1./h3['[a/Fe]_err'][this]**2))
            h3_alpha.append(xxx)
            h3_sigalpha.append(sigxxx)
            h3_stdalpha.append(stdxxx)

    h3_ra=np.array(h3_ra)
    h3_dec=np.array(h3_dec)
    
    h3_vlos=np.array(h3_vlos)
    h3_sigvlos=np.array(h3_sigvlos)
    h3_stdvlos=np.array(h3_stdvlos)
    
    h3_teff=np.array(h3_teff)
    h3_sigteff=np.array(h3_sigteff)
    h3_stdteff=np.array(h3_stdteff)

    h3_logg=np.array(h3_logg)
    h3_siglogg=np.array(h3_siglogg)
    h3_stdlogg=np.array(h3_stdlogg)

    h3_z=np.array(h3_z)
    h3_sigz=np.array(h3_sigz)
    h3_stdz=np.array(h3_stdz)

    h3_alpha=np.array(h3_alpha)
    h3_sigalpha=np.array(h3_sigalpha)
    h3_stdalpha=np.array(h3_stdalpha)

    walker1=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/walker09a.fits')[1].data
    walker2=fits.open('/hildafs/projects/phy200028p/mgwalker/m2fs/walker09.fits')[1].data
    walker_vlos=[]
    walker_sigvlos=[]
    walker_stdvlos=[]
    for i in range(0,len(walker1)):
        this=np.where(walker2['target']==walker1['target'][i])[0][0]
        walker_v=np.sum(walker2['hv'][this]/walker2['e_hv'][this]**2)/np.sum(1./walker2['e_hv'][this]**2)
        walker_sigv=1./np.sqrt(np.sum(1./walker2['e_hv'][this]**2))
        walker_stdv=np.sqrt(np.sum((walker2['hv'][this]-walker_v)**2)/np.sum(1./walker2['e_hv'][this]**2))
        walker_vlos.append(walker_v)
        walker_sigvlos.append(walker_sigv)
        walker_stdvlos.append(walker_stdv)
    walker_vlos=np.array(walker_vlos)
    walker_sigvlos=np.array(walker_sigvlos)
    walker_stdvlos=np.array(walker_stdvlos)
    walker_ra=walker1['RAJ2000']
    walker_dec=walker1['DEJ2000']

    m2fs_hecto_compare_v=[]
    m2fs_hecto_compare_teff=[]
    m2fs_hecto_compare_logg=[]
    m2fs_hecto_compare_z=[]
    m2fs_hecto_compare_alpha=[]

    m2fs_kirby_compare_v=[]
    m2fs_kirby_compare_teff=[]
    m2fs_kirby_compare_logg=[]
    m2fs_kirby_compare_z=[]
    m2fs_kirby_compare_alpha=[]

    m2fs_h3_compare_v=[]
    m2fs_h3_compare_teff=[]
    m2fs_h3_compare_logg=[]
    m2fs_h3_compare_z=[]
    m2fs_h3_compare_alpha=[]

    m2fs_walker_compare_v=[]
    m2fs_walker_compare_teff=[]
    m2fs_walker_compare_logg=[]
    m2fs_walker_compare_z=[]
    m2fs_walker_compare_alpha=[]
    
    m2fs_apogee_compare_v=[]
    m2fs_apogee_compare_teff=[]
    m2fs_apogee_compare_logg=[]
    m2fs_apogee_compare_z=[]
    m2fs_apogee_compare_alpha=[]
    m2fs_apogee_compare_aspcap_flag=[]
    m2fs_apogee_compare_rv_flag=[]

    m2fs_sspp_compare_v=[]
    m2fs_sspp_compare_teff=[]
    m2fs_sspp_compare_logg=[]
    m2fs_sspp_compare_z=[]
    m2fs_sspp_compare_alpha=[]

    hecto_kirby_compare_v=[]
    hecto_kirby_compare_teff=[]
    hecto_kirby_compare_logg=[]
    hecto_kirby_compare_z=[]
    hecto_kirby_compare_alpha=[]

    hecto_h3_compare_v=[]
    hecto_h3_compare_teff=[]
    hecto_h3_compare_logg=[]
    hecto_h3_compare_z=[]
    hecto_h3_compare_alpha=[]

    hecto_walker_compare_v=[]
    hecto_walker_compare_teff=[]
    hecto_walker_compare_logg=[]
    hecto_walker_compare_z=[]
    hecto_walker_compare_alpha=[]
    
    hecto_apogee_compare_v=[]
    hecto_apogee_compare_teff=[]
    hecto_apogee_compare_logg=[]
    hecto_apogee_compare_z=[]
    hecto_apogee_compare_alpha=[]
    hecto_apogee_compare_aspcap_flag=[]
    hecto_apogee_compare_rv_flag=[]

    hecto_sspp_compare_v=[]
    hecto_sspp_compare_teff=[]
    hecto_sspp_compare_logg=[]
    hecto_sspp_compare_z=[]
    hecto_sspp_compare_alpha=[]

    kirby_h3_compare_v=[]
    kirby_h3_compare_teff=[]
    kirby_h3_compare_logg=[]
    kirby_h3_compare_z=[]
    kirby_h3_compare_alpha=[]

    kirby_walker_compare_v=[]
    kirby_walker_compare_teff=[]
    kirby_walker_compare_logg=[]
    kirby_walker_compare_z=[]
    kirby_walker_compare_alpha=[]
    
    kirby_apogee_compare_v=[]
    kirby_apogee_compare_teff=[]
    kirby_apogee_compare_logg=[]
    kirby_apogee_compare_z=[]
    kirby_apogee_compare_alpha=[]
    kirby_apogee_compare_aspcap_flag=[]
    kirby_apogee_compare_rv_flag=[]

    kirby_sspp_compare_v=[]
    kirby_sspp_compare_teff=[]
    kirby_sspp_compare_logg=[]
    kirby_sspp_compare_z=[]
    kirby_sspp_compare_alpha=[]

    h3_walker_compare_v=[]
    h3_walker_compare_teff=[]
    h3_walker_compare_logg=[]
    h3_walker_compare_z=[]
    h3_walker_compare_alpha=[]
    
    h3_apogee_compare_v=[]
    h3_apogee_compare_teff=[]
    h3_apogee_compare_logg=[]
    h3_apogee_compare_z=[]
    h3_apogee_compare_alpha=[]
    h3_apogee_compare_aspcap_flag=[]
    h3_apogee_compare_rv_flag=[]

    h3_sspp_compare_v=[]
    h3_sspp_compare_teff=[]
    h3_sspp_compare_logg=[]
    h3_sspp_compare_z=[]
    h3_sspp_compare_alpha=[]

    walker_apogee_compare_v=[]
    walker_apogee_compare_teff=[]
    walker_apogee_compare_logg=[]
    walker_apogee_compare_z=[]
    walker_apogee_compare_alpha=[]
    walker_apogee_compare_aspcap_flag=[]
    walker_apogee_compare_rv_flag=[]

    walker_sspp_compare_v=[]
    walker_sspp_compare_teff=[]
    walker_sspp_compare_logg=[]
    walker_sspp_compare_z=[]
    walker_sspp_compare_alpha=[]

    sspp_apogee_compare_v=[]
    sspp_apogee_compare_teff=[]
    sspp_apogee_compare_logg=[]
    sspp_apogee_compare_z=[]
    sspp_apogee_compare_alpha=[]
    sspp_apogee_compare_aspcap_flag=[]
    sspp_apogee_compare_rv_flag=[]
    
    for i in range(0,len(m2fs_ra)):
        if ((m2fs_goodobs[i]==1)):
            dist=np.sqrt((1./np.cos(m2fs_dec[i]*np.pi/180.)*(m2fs_ra[i]-hecto_ra))**2+(m2fs_dec[i]-hecto_dec)**2)*3600.
            this=np.where((dist<1.)&(hecto_goodobs==1))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    m2fs_hecto_compare_v.append((m2fs_v_mean[i],m2fs_sigv_mean[i],hecto_v[this[j]],hecto_sigv[this[j]],0,1))
                    m2fs_hecto_compare_teff.append((m2fs_teff_mean[i],m2fs_sigteff_mean[i],hecto_teff[this[j]],hecto_sigteff[this[j]],0,1))
                    m2fs_hecto_compare_logg.append((m2fs_logg_mean[i],m2fs_siglogg_mean[i],hecto_logg[this[j]],hecto_siglogg[this[j]],0,1))
                    m2fs_hecto_compare_z.append((m2fs_z_mean[i],m2fs_sigz_mean[i],hecto_z[this[j]],hecto_sigz[this[j]],0,1))
                    m2fs_hecto_compare_alpha.append((m2fs_alpha_mean[i],m2fs_sigalpha_mean[i],hecto_alpha[this[j]],hecto_sigalpha[this[j]],0,1))

            dist=np.sqrt((1./np.cos(m2fs_dec[i]*np.pi/180.)*(m2fs_ra[i]-apogee_ra))**2+(m2fs_dec[i]-apogee_dec)**2)*3600.
            this=np.where((dist<1.)&(apogee_stdvlos<10.)&(apogee_teff==apogee_teff)&(apogee_logg==apogee_logg)&(apogee_z==apogee_z)&(apogee_alpha==apogee_alpha))[0]
            if ((len(this)>0)&(m2fs_z_mean[i]>-2.5)):#apogee can't measure fe/h < -2.5
                for j in range(0,len(this)):
                    m2fs_apogee_compare_v.append((m2fs_v_mean[i],m2fs_sigv_mean[i],apogee_vlos[this[j]],apogee_sigvlos[this[j]],0,5))
                    m2fs_apogee_compare_teff.append((m2fs_teff_mean[i],m2fs_sigteff_mean[i],apogee_teff[this[j]],apogee_sigteff[this[j]],0,5))
                    m2fs_apogee_compare_logg.append((m2fs_logg_mean[i],m2fs_siglogg_mean[i],apogee_logg[this[j]],apogee_siglogg[this[j]],0,5))
                    m2fs_apogee_compare_z.append((m2fs_z_mean[i],m2fs_sigz_mean[i],apogee_z[this[j]],apogee_sigz[this[j]],0,5))
                    m2fs_apogee_compare_alpha.append((m2fs_alpha_mean[i],m2fs_sigalpha_mean[i],apogee_mg[this[j]],apogee_sigmg[this[j]],0,5))
                    m2fs_apogee_compare_aspcap_flag.append(apogee_aspcap_flag[this[j]])
                    m2fs_apogee_compare_rv_flag.append(apogee_rv_flag[this[j]])

            dist=np.sqrt((1./np.cos(m2fs_dec[i]*np.pi/180.)*(m2fs_ra[i]-kirby_ra))**2+(m2fs_dec[i]-kirby_dec)**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
#                    m2fs_kirby_compare_v.append((m2fs_v_mean[i],m2fs_sigv_mean[i],kirby[''][this[j]],kirby[''][this[j]],0,2))
                    m2fs_kirby_compare_teff.append((m2fs_teff_mean[i],m2fs_sigteff_mean[i],kirby_teff[this[j]],kirby_sigteff[this[j]],0,2))
                    m2fs_kirby_compare_logg.append((m2fs_logg_mean[i],m2fs_siglogg_mean[i],kirby_logg[this[j]],kirby_siglogg[this[j]],0,2))
                    m2fs_kirby_compare_z.append((m2fs_z_mean[i],m2fs_sigz_mean[i],kirby_z[this[j]],kirby_sigz[this[j]],0,2))
                    m2fs_kirby_compare_alpha.append((m2fs_alpha_mean[i],m2fs_sigalpha_mean[i],kirby_alpha[this[j]],kirby_sigalpha[this[j]],0,2))

            dist=np.sqrt((1./np.cos(m2fs_dec[i]*np.pi/180.)*(m2fs_ra[i]-h3_ra))**2+(m2fs_dec[i]-h3_dec)**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    m2fs_h3_compare_v.append((m2fs_v_mean[i],m2fs_sigv_mean[i],h3_vlos[this[j]],h3_sigvlos[this[j]],0,3))
                    m2fs_h3_compare_teff.append((m2fs_teff_mean[i],m2fs_sigteff_mean[i],h3_teff[this[j]],h3_sigteff[this[j]],0,3))
                    m2fs_h3_compare_logg.append((m2fs_logg_mean[i],m2fs_siglogg_mean[i],h3_logg[this[j]],h3_siglogg[this[j]],0,3))
                    m2fs_h3_compare_z.append((m2fs_z_mean[i],m2fs_sigz_mean[i],h3_z[this[j]],h3_sigz[this[j]],0,3))
                    m2fs_h3_compare_alpha.append((m2fs_alpha_mean[i],m2fs_sigalpha_mean[i],h3_alpha[this[j]],h3_sigalpha[this[j]],0,3))

            dist=np.sqrt((1./np.cos(m2fs_dec[i]*np.pi/180.)*(m2fs_ra[i]-sspp_ra))**2+(m2fs_dec[i]-sspp_dec)**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    m2fs_sspp_compare_v.append((m2fs_v_mean[i],m2fs_sigv_mean[i],sspp_vlos[this[j]],sspp_sigvlos[this[j]],0,6))
                    m2fs_sspp_compare_teff.append((m2fs_teff_mean[i],m2fs_sigteff_mean[i],sspp_teff[this[j]],sspp_sigteff[this[j]],0,6))
                    m2fs_sspp_compare_logg.append((m2fs_logg_mean[i],m2fs_siglogg_mean[i],sspp_logg[this[j]],sspp_siglogg[this[j]],0,6))
                    m2fs_sspp_compare_z.append((m2fs_z_mean[i],m2fs_sigz_mean[i],sspp_z[this[j]],sspp_sigz[this[j]],0,6))

            dist=np.sqrt((1./np.cos(m2fs_dec[i]*np.pi/180.)*(m2fs_ra[i]-walker1['RAJ2000']))**2+(m2fs_dec[i]-walker1['DEJ2000'])**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    m2fs_walker_compare_v.append((m2fs_v_mean[i],m2fs_sigv_mean[i],walker_vlos[this[j]],walker_sigvlos[this[j]],0,4))

    for i in range(0,len(hecto_ra)):
        if ((hecto_goodobs[i]==1)):
            dist=np.sqrt((1./np.cos(hecto_dec[i]*np.pi/180.)*(hecto_ra[i]-apogee_ra))**2+(hecto_dec[i]-apogee_dec)**2)*3600.
            this=np.where((dist<1.)&(apogee_stdvlos<10.)&(apogee_teff==apogee_teff)&(apogee_logg==apogee_logg)&(apogee_z==apogee_z)&(apogee_alpha==apogee_alpha))[0]
            if ((len(this)>0)&(hecto_z_mean[i]>-2.5)):#apogee can't measure fe/h < -2.5
                for j in range(0,len(this)):
                    hecto_apogee_compare_v.append((hecto_v_mean[i],hecto_sigv_mean[i],apogee_vlos[this[j]],apogee_sigvlos[this[j]],1,5))
                    hecto_apogee_compare_teff.append((hecto_teff_mean[i],hecto_sigteff_mean[i],apogee_teff[this[j]],apogee_sigteff[this[j]],1,5))
                    hecto_apogee_compare_logg.append((hecto_logg_mean[i],hecto_siglogg_mean[i],apogee_logg[this[j]],apogee_siglogg[this[j]],1,5))
                    hecto_apogee_compare_z.append((hecto_z_mean[i],hecto_sigz_mean[i],apogee_z[this[j]],apogee_sigz[this[j]],1,5))
                    hecto_apogee_compare_alpha.append((hecto_alpha_mean[i],hecto_sigalpha_mean[i],apogee_mg[this[j]],apogee_sigmg[this[j]],1,5))
                    hecto_apogee_compare_aspcap_flag.append(apogee_aspcap_flag[this[j]])
                    hecto_apogee_compare_rv_flag.append(apogee_rv_flag[this[j]])

            dist=np.sqrt((1./np.cos(hecto_dec[i]*np.pi/180.)*(hecto_ra[i]-kirby_ra))**2+(hecto_dec[i]-kirby_dec)**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
#                    hecto_kirby_compare_v.append((hecto_v_mean[i],hecto_sigv_mean[i],kirby[''][this[j]],kirby[''][this[j]]))
                    hecto_kirby_compare_teff.append((hecto_teff_mean[i],hecto_sigteff_mean[i],kirby_teff[this[j]],kirby_sigteff[this[j]],1,2))
                    hecto_kirby_compare_logg.append((hecto_logg_mean[i],hecto_siglogg_mean[i],kirby_logg[this[j]],kirby_siglogg[this[j]],1,2))
                    hecto_kirby_compare_z.append((hecto_z_mean[i],hecto_sigz_mean[i],kirby_z[this[j]],kirby_sigz[this[j]],1,2))
                    hecto_kirby_compare_alpha.append((hecto_alpha_mean[i],hecto_sigalpha_mean[i],kirby_alpha[this[j]],kirby_sigalpha[this[j]],1,2))

            dist=np.sqrt((1./np.cos(hecto_dec[i]*np.pi/180.)*(hecto_ra[i]-h3_ra))**2+(hecto_dec[i]-h3_dec)**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    hecto_h3_compare_v.append((hecto_v_mean[i],hecto_sigv_mean[i],h3_vlos[this[j]],h3_sigvlos[this[j]],1,3))
                    hecto_h3_compare_teff.append((hecto_teff_mean[i],hecto_sigteff_mean[i],h3_teff[this[j]],h3_sigteff[this[j]],1,3))
                    hecto_h3_compare_logg.append((hecto_logg_mean[i],hecto_siglogg_mean[i],h3_logg[this[j]],h3_siglogg[this[j]],1,3))
                    hecto_h3_compare_z.append((hecto_z_mean[i],hecto_sigz_mean[i],h3_z[this[j]],h3_sigz[this[j]],1,3))
                    hecto_h3_compare_alpha.append((hecto_alpha_mean[i],hecto_sigalpha_mean[i],h3_alpha[this[j]],h3_sigalpha[this[j]],1,3))

            dist=np.sqrt((1./np.cos(hecto_dec[i]*np.pi/180.)*(hecto_ra[i]-sspp_ra))**2+(hecto_dec[i]-sspp_dec)**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    hecto_sspp_compare_v.append((hecto_v_mean[i],hecto_sigv_mean[i],sspp_vlos[this[j]],sspp_sigvlos[this[j]],1,6))
                    hecto_sspp_compare_teff.append((hecto_teff_mean[i],hecto_sigteff_mean[i],sspp_teff[this[j]],sspp_sigteff[this[j]],1,6))
                    hecto_sspp_compare_logg.append((hecto_logg_mean[i],hecto_siglogg_mean[i],sspp_logg[this[j]],sspp_siglogg[this[j]],1,6))
                    hecto_sspp_compare_z.append((hecto_z_mean[i],hecto_sigz_mean[i],sspp_z[this[j]],sspp_sigz[this[j]],1,6))

            dist=np.sqrt((1./np.cos(hecto_dec[i]*np.pi/180.)*(hecto_ra[i]-walker1['RAJ2000']))**2+(hecto_dec[i]-walker1['DEJ2000'])**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    hecto_walker_compare_v.append((hecto_v_mean[i],hecto_sigv_mean[i],walker_vlos[this[j]],walker_sigvlos[this[j]],1,4))

    for i in range(0,len(kirby_ra)):
        if (i==i):
            dist=np.sqrt((1./np.cos(kirby_dec[i]*np.pi/180.)*(kirby_ra[i]-apogee_ra))**2+(kirby_dec[i]-apogee_dec)**2)*3600.
            this=np.where((dist<1.)&(apogee_stdvlos<10.)&(apogee_teff==apogee_teff)&(apogee_logg==apogee_logg)&(apogee_z==apogee_z)&(apogee_alpha==apogee_alpha))[0]
            if ((len(this)>0)&(kirby_z[i]>-2.5)):
                for j in range(0,len(this)):
#                    kirby_apogee_compare_v.append((kirby_v_mean[i],kirby_sigv_mean[i],apogee_vlos[this[j]],apogee_sigvlos[this[j]],2,5))
                    kirby_apogee_compare_teff.append((kirby_teff[i],kirby_sigteff[i],apogee_teff[this[j]],apogee_sigteff[this[j]],2,5))
                    kirby_apogee_compare_logg.append((kirby_logg[i],kirby_siglogg[i],apogee_logg[this[j]],apogee_siglogg[this[j]],2,5))
                    kirby_apogee_compare_z.append((kirby_z[i],kirby_sigz[i],apogee_z[this[j]],apogee_sigz[this[j]],2,5))
                    kirby_apogee_compare_alpha.append((kirby_alpha[i],kirby_sigalpha[i],apogee_mg[this[j]],apogee_sigmg[this[j]],2,5))
                    kirby_apogee_compare_aspcap_flag.append(apogee_aspcap_flag[this[j]])
                    kirby_apogee_compare_rv_flag.append(apogee_rv_flag[this[j]])

            dist=np.sqrt((1./np.cos(kirby_dec[i]*np.pi/180.)*(kirby_ra[i]-h3_ra))**2+(kirby_dec[i]-h3_dec)**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
#                    kirby_h3_compare_v.append((kirby_v_mean[i],kirby_sigv_mean[i],h3_vlos[this[j]],h3_sigvlos[this[j]],2,3))
                    kirby_h3_compare_teff.append((kirby_teff[i],kirby_sigteff[i],h3_teff[this[j]],h3_sigteff[this[j]],2,3))
                    kirby_h3_compare_logg.append((kirby_logg[i],kirby_siglogg[i],h3_logg[this[j]],h3_siglogg[this[j]],2,3))
                    kirby_h3_compare_z.append((kirby_z[i],kirby_sigz[i],h3_z[this[j]],h3_sigz[this[j]],2,3))
                    kirby_h3_compare_alpha.append((kirby_alpha[i],kirby_sigalpha[i],h3_alpha[this[j]],h3_sigalpha[this[j]],2,3))

            dist=np.sqrt((1./np.cos(kirby_dec[i]*np.pi/180.)*(kirby_ra[i]-sspp_ra))**2+(kirby_dec[i]-sspp_dec)**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
#                    kirby_sspp_compare_v.append((kirby_v_mean[i],kirby_sigv_mean[i],sspp_vlos[this[j]],sspp_sigvlos[this[j]],2,6))
                    kirby_sspp_compare_teff.append((kirby_teff[i],kirby_sigteff[i],sspp_teff[this[j]],sspp_sigteff[this[j]],2,6))
                    kirby_sspp_compare_logg.append((kirby_logg[i],kirby_siglogg[i],sspp_logg[this[j]],sspp_siglogg[this[j]],2,6))
                    kirby_sspp_compare_z.append((kirby_z[i],kirby_sigz[i],sspp_z[this[j]],sspp_sigz[this[j]],2,6))

    for i in range(0,len(h3_ra)):
        if (i==i):
            dist=np.sqrt((1./np.cos(h3_dec[i]*np.pi/180.)*(h3_ra[i]-apogee_ra))**2+(h3_dec[i]-apogee_dec)**2)*3600.
            this=np.where((dist<1.)&(apogee_stdvlos<10.)&(apogee_teff==apogee_teff)&(apogee_logg==apogee_logg)&(apogee_z==apogee_z)&(apogee_alpha==apogee_alpha))[0]
            if ((len(this)>0)&(h3_z[i]>-2.5)):
                for j in range(0,len(this)):
                    h3_apogee_compare_v.append((h3_vlos[i],h3_sigvlos[i],apogee_vlos[this[j]],apogee_sigvlos[this[j]],3,5))
                    h3_apogee_compare_teff.append((h3_teff[i],h3_sigteff[i],apogee_teff[this[j]],apogee_sigteff[this[j]],3,5))
                    h3_apogee_compare_logg.append((h3_logg[i],h3_siglogg[i],apogee_logg[this[j]],apogee_siglogg[this[j]],3,5))
                    h3_apogee_compare_z.append((h3_z[i],h3_sigz[i],apogee_z[this[j]],apogee_sigz[this[j]],3,5))
                    h3_apogee_compare_alpha.append((h3_alpha[i],h3_sigalpha[i],apogee_mg[this[j]],apogee_sigmg[this[j]],3,5))
                    h3_apogee_compare_aspcap_flag.append(apogee_aspcap_flag[this[j]])
                    h3_apogee_compare_rv_flag.append(apogee_rv_flag[this[j]])

            dist=np.sqrt((1./np.cos(h3_dec[i]*np.pi/180.)*(h3_ra[i]-sspp_ra))**2+(h3_dec[i]-sspp_dec)**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    h3_sspp_compare_v.append((h3_vlos[i],h3_sigvlos[i],sspp_vlos[this[j]],sspp_sigvlos[this[j]],3,6))
                    h3_sspp_compare_teff.append((h3_teff[i],h3_sigteff[i],sspp_teff[this[j]],sspp_sigteff[this[j]],3,6))
                    h3_sspp_compare_logg.append((h3_logg[i],h3_siglogg[i],sspp_logg[this[j]],sspp_siglogg[this[j]],3,6))
                    h3_sspp_compare_z.append((h3_z[i],h3_sigz[i],sspp_z[this[j]],sspp_sigz[this[j]],3,6))

            dist=np.sqrt((1./np.cos(h3_dec[i]*np.pi/180.)*(h3_ra[i]-walker1['RAJ2000']))**2+(h3_dec[i]-walker1['DEJ2000'])**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    h3_walker_compare_v.append((h3_vlos[i],h3_sigvlos[i],walker_vlos[this[j]],walker_sigvlos[this[j]],3,4))

    for i in range(0,len(walker_ra)):
        if (i==i):
            dist=np.sqrt((1./np.cos(walker_dec[i]*np.pi/180.)*(walker_ra[i]-apogee_ra))**2+(walker_dec[i]-apogee_dec)**2)*3600.
            this=np.where((dist<1.)&(apogee_stdvlos<10.)&(apogee_teff==apogee_teff)&(apogee_logg==apogee_logg)&(apogee_z==apogee_z)&(apogee_alpha==apogee_alpha))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    walker_apogee_compare_v.append((walker_vlos[i],walker_sigvlos[i],apogee_vlos[this[j]],apogee_sigvlos[this[j]],4,5))
                    walker_apogee_compare_aspcap_flag.append(apogee_aspcap_flag[this[j]])
                    walker_apogee_compare_rv_flag.append(apogee_rv_flag[this[j]])

            dist=np.sqrt((1./np.cos(walker_dec[i]*np.pi/180.)*(walker_ra[i]-sspp_ra))**2+(walker_dec[i]-sspp_dec)**2)*3600.
            this=np.where((dist<1.))[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    walker_sspp_compare_v.append((walker_vlos[i],walker_sigvlos[i],sspp_vlos[this[j]],sspp_sigvlos[this[j]],4,6))

    for i in range(0,len(sspp_ra)):
        if (i==i):
            dist=np.sqrt((1./np.cos(sspp_dec[i]*np.pi/180.)*(sspp_ra[i]-apogee_ra))**2+(sspp_dec[i]-apogee_dec)**2)*3600.
            this=np.where((dist<1.)&(apogee_stdvlos<10.)&(apogee_teff==apogee_teff)&(apogee_logg==apogee_logg)&(apogee_z==apogee_z)&(apogee_alpha==apogee_alpha))[0]
            if ((len(this)>0)&(sspp_z[i]>-2.5)):
                for j in range(0,len(this)):
                    sspp_apogee_compare_v.append((sspp_vlos[i],sspp_sigvlos[i],apogee_vlos[this[j]],apogee_sigvlos[this[j]],6,5))
                    sspp_apogee_compare_teff.append((sspp_teff[i],sspp_sigteff[i],apogee_teff[this[j]],apogee_sigteff[this[j]],6,5))
                    sspp_apogee_compare_logg.append((sspp_logg[i],sspp_siglogg[i],apogee_logg[this[j]],apogee_siglogg[this[j]],6,5))
                    sspp_apogee_compare_z.append((sspp_z[i],sspp_sigz[i],apogee_z[this[j]],apogee_sigz[this[j]],6,5))
                    #sspp_apogee_compare_alpha.append((sspp_alpha[i],sspp_sigalpha[i],apogee_mg[this[j]],apogee_sigmg[this[j]],6,5))
                    sspp_apogee_compare_aspcap_flag.append(apogee_aspcap_flag[this[j]])
                    sspp_apogee_compare_rv_flag.append(apogee_rv_flag[this[j]])

    m2fs_hecto_compare_v=np.array(m2fs_hecto_compare_v)
    m2fs_hecto_compare_teff=np.array(m2fs_hecto_compare_teff)
    m2fs_hecto_compare_logg=np.array(m2fs_hecto_compare_logg)
    m2fs_hecto_compare_z=np.array(m2fs_hecto_compare_z)
    m2fs_hecto_compare_alpha=np.array(m2fs_hecto_compare_alpha)

    m2fs_kirby_compare_v=np.array(m2fs_kirby_compare_v)
    m2fs_kirby_compare_teff=np.array(m2fs_kirby_compare_teff)
    m2fs_kirby_compare_logg=np.array(m2fs_kirby_compare_logg)
    m2fs_kirby_compare_z=np.array(m2fs_kirby_compare_z)
    m2fs_kirby_compare_alpha=np.array(m2fs_kirby_compare_alpha)
    m2fs_kirby_keep2=np.where((m2fs_kirby_compare_teff.T[2]==m2fs_kirby_compare_teff.T[2])&(m2fs_kirby_compare_logg.T[2]==m2fs_kirby_compare_logg.T[2])&(m2fs_kirby_compare_z.T[2]==m2fs_kirby_compare_z.T[2])&(m2fs_kirby_compare_alpha.T[2]==m2fs_kirby_compare_alpha.T[2]))[0]
    
    m2fs_h3_compare_v=np.array(m2fs_h3_compare_v)
    m2fs_h3_compare_teff=np.array(m2fs_h3_compare_teff)
    m2fs_h3_compare_logg=np.array(m2fs_h3_compare_logg)
    m2fs_h3_compare_z=np.array(m2fs_h3_compare_z)
    m2fs_h3_compare_alpha=np.array(m2fs_h3_compare_alpha)

    m2fs_walker_compare_v=np.array(m2fs_walker_compare_v)
    m2fs_walker_compare_teff=np.array(m2fs_walker_compare_teff)
    m2fs_walker_compare_logg=np.array(m2fs_walker_compare_logg)
    m2fs_walker_compare_z=np.array(m2fs_walker_compare_z)
    m2fs_walker_compare_alpha=np.array(m2fs_walker_compare_alpha)
    
    m2fs_apogee_compare_v=np.array(m2fs_apogee_compare_v)
    m2fs_apogee_compare_teff=np.array(m2fs_apogee_compare_teff)
    m2fs_apogee_compare_logg=np.array(m2fs_apogee_compare_logg)
    m2fs_apogee_compare_z=np.array(m2fs_apogee_compare_z)
    m2fs_apogee_compare_alpha=np.array(m2fs_apogee_compare_alpha)
    m2fs_apogee_compare_aspcap_flag=np.array(m2fs_apogee_compare_aspcap_flag)
    m2fs_apogee_compare_rv_flag=np.array(m2fs_apogee_compare_rv_flag)

    m2fs_sspp_compare_v=np.array(m2fs_sspp_compare_v)
    m2fs_sspp_compare_teff=np.array(m2fs_sspp_compare_teff)
    m2fs_sspp_compare_logg=np.array(m2fs_sspp_compare_logg)
    m2fs_sspp_compare_z=np.array(m2fs_sspp_compare_z)
    m2fs_sspp_compare_alpha=np.array(m2fs_sspp_compare_alpha)

    hecto_kirby_compare_v=np.array(hecto_kirby_compare_v)
    hecto_kirby_compare_teff=np.array(hecto_kirby_compare_teff)
    hecto_kirby_compare_logg=np.array(hecto_kirby_compare_logg)
    hecto_kirby_compare_z=np.array(hecto_kirby_compare_z)
    hecto_kirby_compare_alpha=np.array(hecto_kirby_compare_alpha)

    hecto_h3_compare_v=np.array(hecto_h3_compare_v)
    hecto_h3_compare_teff=np.array(hecto_h3_compare_teff)
    hecto_h3_compare_logg=np.array(hecto_h3_compare_logg)
    hecto_h3_compare_z=np.array(hecto_h3_compare_z)
    hecto_h3_compare_alpha=np.array(hecto_h3_compare_alpha)

    hecto_walker_compare_v=np.array(hecto_walker_compare_v)
    hecto_walker_compare_teff=np.array(hecto_walker_compare_teff)
    hecto_walker_compare_logg=np.array(hecto_walker_compare_logg)
    hecto_walker_compare_z=np.array(hecto_walker_compare_z)
    hecto_walker_compare_alpha=np.array(hecto_walker_compare_alpha)
    
    hecto_apogee_compare_v=np.array(hecto_apogee_compare_v)
    hecto_apogee_compare_teff=np.array(hecto_apogee_compare_teff)
    hecto_apogee_compare_logg=np.array(hecto_apogee_compare_logg)
    hecto_apogee_compare_z=np.array(hecto_apogee_compare_z)
    hecto_apogee_compare_alpha=np.array(hecto_apogee_compare_alpha)
    hecto_apogee_compare_aspcap_flag=np.array(hecto_apogee_compare_aspcap_flag)
    hecto_apogee_compare_rv_flag=np.array(hecto_apogee_compare_rv_flag)

    hecto_sspp_compare_v=np.array(hecto_sspp_compare_v)
    hecto_sspp_compare_teff=np.array(hecto_sspp_compare_teff)
    hecto_sspp_compare_logg=np.array(hecto_sspp_compare_logg)
    hecto_sspp_compare_z=np.array(hecto_sspp_compare_z)
    hecto_sspp_compare_alpha=np.array(hecto_sspp_compare_alpha)

    kirby_h3_compare_v=np.array(kirby_h3_compare_v)
    kirby_h3_compare_teff=np.array(kirby_h3_compare_teff)
    kirby_h3_compare_logg=np.array(kirby_h3_compare_logg)
    kirby_h3_compare_z=np.array(kirby_h3_compare_z)
    kirby_h3_compare_alpha=np.array(kirby_h3_compare_alpha)

    kirby_walker_compare_v=np.array(kirby_walker_compare_v)
    kirby_walker_compare_teff=np.array(kirby_walker_compare_teff)
    kirby_walker_compare_logg=np.array(kirby_walker_compare_logg)
    kirby_walker_compare_z=np.array(kirby_walker_compare_z)
    kirby_walker_compare_alpha=np.array(kirby_walker_compare_alpha)
    
    kirby_apogee_compare_v=np.array(kirby_apogee_compare_v)
    kirby_apogee_compare_teff=np.array(kirby_apogee_compare_teff)
    kirby_apogee_compare_logg=np.array(kirby_apogee_compare_logg)
    kirby_apogee_compare_z=np.array(kirby_apogee_compare_z)
    kirby_apogee_compare_alpha=np.array(kirby_apogee_compare_alpha)
    kirby_apogee_compare_aspcap_flag=np.array(kirby_apogee_compare_aspcap_flag)
    kirby_apogee_compare_rv_flag=np.array(kirby_apogee_compare_rv_flag)
    kirby_apogee_keep1=np.where(kirby_apogee_compare_rv_flag==False)[0]
    kirby_apogee_keep2=np.where(kirby_apogee_compare_aspcap_flag==False)[0]

    kirby_sspp_compare_v=np.array(kirby_sspp_compare_v)
    kirby_sspp_compare_teff=np.array(kirby_sspp_compare_teff)
    kirby_sspp_compare_logg=np.array(kirby_sspp_compare_logg)
    kirby_sspp_compare_z=np.array(kirby_sspp_compare_z)
    kirby_sspp_compare_alpha=np.array(kirby_sspp_compare_alpha)

    h3_walker_compare_v=np.array(h3_walker_compare_v)
    h3_walker_compare_teff=np.array(h3_walker_compare_teff)
    h3_walker_compare_logg=np.array(h3_walker_compare_logg)
    h3_walker_compare_z=np.array(h3_walker_compare_z)
    h3_walker_compare_alpha=np.array(h3_walker_compare_alpha)
    
    h3_apogee_compare_v=np.array(h3_apogee_compare_v)
    h3_apogee_compare_teff=np.array(h3_apogee_compare_teff)
    h3_apogee_compare_logg=np.array(h3_apogee_compare_logg)
    h3_apogee_compare_z=np.array(h3_apogee_compare_z)
    h3_apogee_compare_alpha=np.array(h3_apogee_compare_alpha)
    h3_apogee_compare_aspcap_flag=np.array(h3_apogee_compare_aspcap_flag)
    h3_apogee_compare_rv_flag=np.array(h3_apogee_compare_rv_flag)

    h3_sspp_compare_v=np.array(h3_sspp_compare_v)
    h3_sspp_compare_teff=np.array(h3_sspp_compare_teff)
    h3_sspp_compare_logg=np.array(h3_sspp_compare_logg)
    h3_sspp_compare_z=np.array(h3_sspp_compare_z)
    h3_sspp_compare_alpha=np.array(h3_sspp_compare_alpha)

    walker_apogee_compare_v=np.array(walker_apogee_compare_v)
    walker_apogee_compare_teff=np.array(walker_apogee_compare_teff)
    walker_apogee_compare_logg=np.array(walker_apogee_compare_logg)
    walker_apogee_compare_z=np.array(walker_apogee_compare_z)
    walker_apogee_compare_alpha=np.array(walker_apogee_compare_alpha)
    walker_apogee_compare_aspcap_flag=np.array(walker_apogee_compare_aspcap_flag)
    walker_apogee_compare_rv_flag=np.array(walker_apogee_compare_rv_flag)

    walker_sspp_compare_v=np.array(walker_sspp_compare_v)
    walker_sspp_compare_teff=np.array(walker_sspp_compare_teff)
    walker_sspp_compare_logg=np.array(walker_sspp_compare_logg)
    walker_sspp_compare_z=np.array(walker_sspp_compare_z)
    walker_sspp_compare_alpha=np.array(walker_sspp_compare_alpha)

    sspp_apogee_compare_v=np.array(sspp_apogee_compare_v)
    sspp_apogee_compare_teff=np.array(sspp_apogee_compare_teff)
    sspp_apogee_compare_logg=np.array(sspp_apogee_compare_logg)
    sspp_apogee_compare_z=np.array(sspp_apogee_compare_z)
    sspp_apogee_compare_alpha=np.array(sspp_apogee_compare_alpha)
    sspp_apogee_compare_aspcap_flag=np.array(sspp_apogee_compare_aspcap_flag)
    sspp_apogee_compare_rv_flag=np.array(sspp_apogee_compare_rv_flag)

    m2fs_apogee_mask1=bit_solve_v(m2fs_apogee_compare_rv_flag,np.full(len(m2fs_apogee_compare_rv_flag),12))
    m2fs_apogee_mask2=bit_solve_v(m2fs_apogee_compare_aspcap_flag,np.full(len(m2fs_apogee_compare_aspcap_flag),8))
    m2fs_apogee_keep1=np.where(m2fs_apogee_mask1==False)[0]
    m2fs_apogee_keep2=np.where(m2fs_apogee_mask2==False)[0]

    hecto_apogee_mask1=bit_solve_v(hecto_apogee_compare_rv_flag,np.full(len(hecto_apogee_compare_rv_flag),12))
    hecto_apogee_mask2=bit_solve_v(hecto_apogee_compare_aspcap_flag,np.full(len(hecto_apogee_compare_aspcap_flag),8))
    hecto_apogee_keep1=np.where(hecto_apogee_mask1==False)[0]
    hecto_apogee_keep2=np.where(hecto_apogee_mask2==False)[0]

    h3_apogee_mask1=bit_solve_v(h3_apogee_compare_rv_flag,np.full(len(h3_apogee_compare_rv_flag),12))
    h3_apogee_mask2=bit_solve_v(h3_apogee_compare_aspcap_flag,np.full(len(h3_apogee_compare_aspcap_flag),8))
    h3_apogee_keep1=np.where(h3_apogee_mask1==False)[0]
    h3_apogee_keep2=np.where(h3_apogee_mask2==False)[0]

    kirby_apogee_mask1=bit_solve_v(kirby_apogee_compare_rv_flag,np.full(len(kirby_apogee_compare_rv_flag),12))
    kirby_apogee_mask2=bit_solve_v(kirby_apogee_compare_aspcap_flag,np.full(len(kirby_apogee_compare_aspcap_flag),8))
    kirby_apogee_keep1=np.where(kirby_apogee_mask1==False)[0]
    kirby_apogee_keep2=np.where(kirby_apogee_mask2==False)[0]

    walker_apogee_mask1=bit_solve_v(walker_apogee_compare_rv_flag,np.full(len(walker_apogee_compare_rv_flag),12))
    walker_apogee_mask2=bit_solve_v(walker_apogee_compare_aspcap_flag,np.full(len(walker_apogee_compare_aspcap_flag),8))
    walker_apogee_keep1=np.where(walker_apogee_mask1==False)[0]
    walker_apogee_keep2=np.where(walker_apogee_mask2==False)[0]

    sspp_apogee_mask1=bit_solve_v(sspp_apogee_compare_rv_flag,np.full(len(sspp_apogee_compare_rv_flag),12))
    sspp_apogee_mask2=bit_solve_v(sspp_apogee_compare_aspcap_flag,np.full(len(sspp_apogee_compare_aspcap_flag),8))
    sspp_apogee_keep1=np.where(sspp_apogee_mask1==False)[0]
    sspp_apogee_keep2=np.where(sspp_apogee_mask2==False)[0]
    
    x1=np.concatenate([m2fs_hecto_compare_v.T[0],m2fs_h3_compare_v.T[0],m2fs_walker_compare_v.T[0],m2fs_apogee_compare_v.T[0][m2fs_apogee_keep1],m2fs_sspp_compare_v.T[0],hecto_h3_compare_v.T[0],hecto_walker_compare_v.T[0],hecto_apogee_compare_v.T[0][hecto_apogee_keep1],hecto_sspp_compare_v.T[0],h3_walker_compare_v.T[0],h3_apogee_compare_v.T[0][h3_apogee_keep1],h3_sspp_compare_v.T[0],walker_apogee_compare_v.T[0][walker_apogee_keep1],walker_sspp_compare_v.T[0],sspp_apogee_compare_v.T[0][sspp_apogee_keep1]])
    sigx1=np.concatenate([m2fs_hecto_compare_v.T[1],m2fs_h3_compare_v.T[1],m2fs_walker_compare_v.T[1],m2fs_apogee_compare_v.T[1][m2fs_apogee_keep1],m2fs_sspp_compare_v.T[1],hecto_h3_compare_v.T[1],hecto_walker_compare_v.T[1],hecto_apogee_compare_v.T[1][hecto_apogee_keep1],hecto_sspp_compare_v.T[1],h3_walker_compare_v.T[1],h3_apogee_compare_v.T[1][h3_apogee_keep1],h3_sspp_compare_v.T[1],walker_apogee_compare_v.T[1][walker_apogee_keep1],walker_sspp_compare_v.T[1],sspp_apogee_compare_v.T[1][sspp_apogee_keep1]])
    x2=np.concatenate([m2fs_hecto_compare_v.T[2],m2fs_h3_compare_v.T[2],m2fs_walker_compare_v.T[2],m2fs_apogee_compare_v.T[2][m2fs_apogee_keep1],m2fs_sspp_compare_v.T[2],hecto_h3_compare_v.T[2],hecto_walker_compare_v.T[2],hecto_apogee_compare_v.T[2][hecto_apogee_keep1],hecto_sspp_compare_v.T[2],h3_walker_compare_v.T[2],h3_apogee_compare_v.T[2][h3_apogee_keep1],h3_sspp_compare_v.T[2],walker_apogee_compare_v.T[2][walker_apogee_keep1],walker_sspp_compare_v.T[2],sspp_apogee_compare_v.T[2][sspp_apogee_keep1]])
    sigx2=np.concatenate([m2fs_hecto_compare_v.T[3],m2fs_h3_compare_v.T[3],m2fs_walker_compare_v.T[3],m2fs_apogee_compare_v.T[3][m2fs_apogee_keep1],m2fs_sspp_compare_v.T[3],hecto_h3_compare_v.T[3],hecto_walker_compare_v.T[3],hecto_apogee_compare_v.T[3][hecto_apogee_keep1],hecto_sspp_compare_v.T[3],h3_walker_compare_v.T[3],h3_apogee_compare_v.T[3][h3_apogee_keep1],h3_sspp_compare_v.T[3],walker_apogee_compare_v.T[3][walker_apogee_keep1],walker_sspp_compare_v.T[3],sspp_apogee_compare_v.T[3][sspp_apogee_keep1]])
    survey1=np.concatenate([m2fs_hecto_compare_v.T[4],m2fs_h3_compare_v.T[4],m2fs_walker_compare_v.T[4],m2fs_apogee_compare_v.T[4][m2fs_apogee_keep1],m2fs_sspp_compare_v.T[4],hecto_h3_compare_v.T[4],hecto_walker_compare_v.T[4],hecto_apogee_compare_v.T[4][hecto_apogee_keep1],hecto_sspp_compare_v.T[4],h3_walker_compare_v.T[4],h3_apogee_compare_v.T[4][h3_apogee_keep1],h3_sspp_compare_v.T[4],walker_apogee_compare_v.T[4][walker_apogee_keep1],walker_sspp_compare_v.T[4],sspp_apogee_compare_v.T[4][sspp_apogee_keep1]])
    survey2=np.concatenate([m2fs_hecto_compare_v.T[5],m2fs_h3_compare_v.T[5],m2fs_walker_compare_v.T[5],m2fs_apogee_compare_v.T[5][m2fs_apogee_keep1],m2fs_sspp_compare_v.T[5],hecto_h3_compare_v.T[5],hecto_walker_compare_v.T[5],hecto_apogee_compare_v.T[5][hecto_apogee_keep1],hecto_sspp_compare_v.T[5],h3_walker_compare_v.T[5],h3_apogee_compare_v.T[5][h3_apogee_keep1],h3_sspp_compare_v.T[5],walker_apogee_compare_v.T[5][walker_apogee_keep1],walker_sspp_compare_v.T[5],sspp_apogee_compare_v.T[5][sspp_apogee_keep1]])

    pickle.dump((x1,sigx1,x2,sigx2,survey1,survey2),open('crap_medres_vlos.pkl','wb'))
    offset=offset_v
    dmax=dmax_v
    keep=np.where((survey1!=6)&(survey2!=6))[0]
    vlos_result,vlos_bestfit=m2fs.get_globaloffset(x1[keep],sigx1[keep],x2[keep],sigx2[keep],survey1[keep],survey2[keep],offset,dmax)
    pickle.dump((vlos_result,vlos_bestfit),open('vlos_medres_zeropointshift.pkl','wb'))
    
    x1=np.concatenate([m2fs_hecto_compare_teff.T[0],m2fs_h3_compare_teff.T[0],m2fs_kirby_compare_teff.T[0],m2fs_apogee_compare_teff.T[0][m2fs_apogee_keep2],m2fs_sspp_compare_teff.T[0],hecto_h3_compare_teff.T[0],hecto_kirby_compare_teff.T[0],hecto_apogee_compare_teff.T[0][hecto_apogee_keep2],hecto_sspp_compare_teff.T[0],kirby_h3_compare_teff.T[0],kirby_apogee_compare_teff.T[0][kirby_apogee_keep2],kirby_sspp_compare_teff.T[0],h3_apogee_compare_teff.T[0][h3_apogee_keep2],h3_sspp_compare_teff.T[0],sspp_apogee_compare_teff.T[0][sspp_apogee_keep2]])
    sigx1=np.concatenate([m2fs_hecto_compare_teff.T[1],m2fs_h3_compare_teff.T[1],m2fs_kirby_compare_teff.T[1],m2fs_apogee_compare_teff.T[1][m2fs_apogee_keep2],m2fs_sspp_compare_teff.T[1],hecto_h3_compare_teff.T[1],hecto_kirby_compare_teff.T[1],hecto_apogee_compare_teff.T[1][hecto_apogee_keep2],hecto_sspp_compare_teff.T[1],kirby_h3_compare_teff.T[1],kirby_apogee_compare_teff.T[1][kirby_apogee_keep2],kirby_sspp_compare_teff.T[1],h3_apogee_compare_teff.T[1][h3_apogee_keep2],h3_sspp_compare_teff.T[1],sspp_apogee_compare_teff.T[1][sspp_apogee_keep2]])
    x2=np.concatenate([m2fs_hecto_compare_teff.T[2],m2fs_h3_compare_teff.T[2],m2fs_kirby_compare_teff.T[2],m2fs_apogee_compare_teff.T[2][m2fs_apogee_keep2],m2fs_sspp_compare_teff.T[2],hecto_h3_compare_teff.T[2],hecto_kirby_compare_teff.T[2],hecto_apogee_compare_teff.T[2][hecto_apogee_keep2],hecto_sspp_compare_teff.T[2],kirby_h3_compare_teff.T[2],kirby_apogee_compare_teff.T[2][kirby_apogee_keep2],kirby_sspp_compare_teff.T[2],h3_apogee_compare_teff.T[2][h3_apogee_keep2],h3_sspp_compare_teff.T[2],sspp_apogee_compare_teff.T[2][sspp_apogee_keep2]])
    sigx2=np.concatenate([m2fs_hecto_compare_teff.T[3],m2fs_h3_compare_teff.T[3],m2fs_kirby_compare_teff.T[3],m2fs_apogee_compare_teff.T[3][m2fs_apogee_keep2],m2fs_sspp_compare_teff.T[3],hecto_h3_compare_teff.T[3],hecto_kirby_compare_teff.T[3],hecto_apogee_compare_teff.T[3][hecto_apogee_keep2],hecto_sspp_compare_teff.T[3],kirby_h3_compare_teff.T[3],kirby_apogee_compare_teff.T[3][kirby_apogee_keep2],kirby_sspp_compare_teff.T[3],h3_apogee_compare_teff.T[3][h3_apogee_keep2],h3_sspp_compare_teff.T[3],sspp_apogee_compare_teff.T[3][sspp_apogee_keep2]])
    survey1=np.concatenate([m2fs_hecto_compare_teff.T[4],m2fs_h3_compare_teff.T[4],m2fs_kirby_compare_teff.T[4],m2fs_apogee_compare_teff.T[4][m2fs_apogee_keep2],m2fs_sspp_compare_teff.T[4],hecto_h3_compare_teff.T[4],hecto_kirby_compare_teff.T[4],hecto_apogee_compare_teff.T[4][hecto_apogee_keep2],hecto_sspp_compare_teff.T[4],kirby_h3_compare_teff.T[4],kirby_apogee_compare_teff.T[4][kirby_apogee_keep2],kirby_sspp_compare_teff.T[4],h3_apogee_compare_teff.T[4][h3_apogee_keep2],h3_sspp_compare_teff.T[4],sspp_apogee_compare_teff.T[4][sspp_apogee_keep2]])
    survey2=np.concatenate([m2fs_hecto_compare_teff.T[5],m2fs_h3_compare_teff.T[5],m2fs_kirby_compare_teff.T[5],m2fs_apogee_compare_teff.T[5][m2fs_apogee_keep2],m2fs_sspp_compare_teff.T[5],hecto_h3_compare_teff.T[5],hecto_kirby_compare_teff.T[5],hecto_apogee_compare_teff.T[5][hecto_apogee_keep2],hecto_sspp_compare_teff.T[5],kirby_h3_compare_teff.T[5],kirby_apogee_compare_teff.T[5][kirby_apogee_keep2],kirby_sspp_compare_teff.T[5],h3_apogee_compare_teff.T[5][h3_apogee_keep2],h3_sspp_compare_teff.T[5],sspp_apogee_compare_teff.T[5][sspp_apogee_keep2]])
    
    pickle.dump((x1,sigx1,x2,sigx2,survey1,survey2),open('crap_medres_teff.pkl','wb'))
    offset=offset_teff
    dmax=dmax_teff
    keep=np.where((survey1!=6)&(survey2!=6))[0]
    teff_result,teff_bestfit=m2fs.get_globaloffset(x1[keep],sigx1[keep],x2[keep],sigx2[keep],survey1[keep],survey2[keep],offset,dmax)
    pickle.dump((teff_result,teff_bestfit),open('teff_medres_zeropointshift.pkl','wb'))

    x1=np.concatenate([m2fs_hecto_compare_logg.T[0],m2fs_h3_compare_logg.T[0],m2fs_kirby_compare_logg.T[0],m2fs_apogee_compare_logg.T[0][m2fs_apogee_keep2],m2fs_sspp_compare_logg.T[0],hecto_h3_compare_logg.T[0],hecto_kirby_compare_logg.T[0],hecto_apogee_compare_logg.T[0][hecto_apogee_keep2],hecto_sspp_compare_logg.T[0],kirby_h3_compare_logg.T[0],kirby_apogee_compare_logg.T[0][kirby_apogee_keep2],kirby_sspp_compare_logg.T[0],h3_apogee_compare_logg.T[0][h3_apogee_keep2],h3_sspp_compare_logg.T[0],sspp_apogee_compare_logg.T[0][sspp_apogee_keep2]])
    sigx1=np.concatenate([m2fs_hecto_compare_logg.T[1],m2fs_h3_compare_logg.T[1],m2fs_kirby_compare_logg.T[1],m2fs_apogee_compare_logg.T[1][m2fs_apogee_keep2],m2fs_sspp_compare_logg.T[1],hecto_h3_compare_logg.T[1],hecto_kirby_compare_logg.T[1],hecto_apogee_compare_logg.T[1][hecto_apogee_keep2],hecto_sspp_compare_logg.T[1],kirby_h3_compare_logg.T[1],kirby_apogee_compare_logg.T[1][kirby_apogee_keep2],kirby_sspp_compare_logg.T[1],h3_apogee_compare_logg.T[1][h3_apogee_keep2],h3_sspp_compare_logg.T[1],sspp_apogee_compare_logg.T[1][sspp_apogee_keep2]])
    x2=np.concatenate([m2fs_hecto_compare_logg.T[2],m2fs_h3_compare_logg.T[2],m2fs_kirby_compare_logg.T[2],m2fs_apogee_compare_logg.T[2][m2fs_apogee_keep2],m2fs_sspp_compare_logg.T[2],hecto_h3_compare_logg.T[2],hecto_kirby_compare_logg.T[2],hecto_apogee_compare_logg.T[2][hecto_apogee_keep2],hecto_sspp_compare_logg.T[2],kirby_h3_compare_logg.T[2],kirby_apogee_compare_logg.T[2][kirby_apogee_keep2],kirby_sspp_compare_logg.T[2],h3_apogee_compare_logg.T[2][h3_apogee_keep2],h3_sspp_compare_logg.T[2],sspp_apogee_compare_logg.T[2][sspp_apogee_keep2]])
    sigx2=np.concatenate([m2fs_hecto_compare_logg.T[3],m2fs_h3_compare_logg.T[3],m2fs_kirby_compare_logg.T[3],m2fs_apogee_compare_logg.T[3][m2fs_apogee_keep2],m2fs_sspp_compare_logg.T[3],hecto_h3_compare_logg.T[3],hecto_kirby_compare_logg.T[3],hecto_apogee_compare_logg.T[3][hecto_apogee_keep2],hecto_sspp_compare_logg.T[3],kirby_h3_compare_logg.T[3],kirby_apogee_compare_logg.T[3][kirby_apogee_keep2],kirby_sspp_compare_logg.T[3],h3_apogee_compare_logg.T[3][h3_apogee_keep2],h3_sspp_compare_logg.T[3],sspp_apogee_compare_logg.T[3][sspp_apogee_keep2]])
    survey1=np.concatenate([m2fs_hecto_compare_logg.T[4],m2fs_h3_compare_logg.T[4],m2fs_kirby_compare_logg.T[4],m2fs_apogee_compare_logg.T[4][m2fs_apogee_keep2],m2fs_sspp_compare_logg.T[4],hecto_h3_compare_logg.T[4],hecto_kirby_compare_logg.T[4],hecto_apogee_compare_logg.T[4][hecto_apogee_keep2],hecto_sspp_compare_logg.T[4],kirby_h3_compare_logg.T[4],kirby_apogee_compare_logg.T[4][kirby_apogee_keep2],kirby_sspp_compare_logg.T[4],h3_apogee_compare_logg.T[4][h3_apogee_keep2],h3_sspp_compare_logg.T[4],sspp_apogee_compare_logg.T[4][sspp_apogee_keep2]])
    survey2=np.concatenate([m2fs_hecto_compare_logg.T[5],m2fs_h3_compare_logg.T[5],m2fs_kirby_compare_logg.T[5],m2fs_apogee_compare_logg.T[5][m2fs_apogee_keep2],m2fs_sspp_compare_logg.T[5],hecto_h3_compare_logg.T[5],hecto_kirby_compare_logg.T[5],hecto_apogee_compare_logg.T[5][hecto_apogee_keep2],hecto_sspp_compare_logg.T[5],kirby_h3_compare_logg.T[5],kirby_apogee_compare_logg.T[5][kirby_apogee_keep2],kirby_sspp_compare_logg.T[5],h3_apogee_compare_logg.T[5][h3_apogee_keep2],h3_sspp_compare_logg.T[5],sspp_apogee_compare_logg.T[5][sspp_apogee_keep2]])
    
    pickle.dump((x1,sigx1,x2,sigx2,survey1,survey2),open('crap_medres_logg.pkl','wb'))
    offset=offset_logg
    dmax=dmax_logg
    keep=np.where((survey1!=6)&(survey2!=6))[0]
    logg_result,logg_bestfit=m2fs.get_globaloffset(x1[keep],sigx1[keep],x2[keep],sigx2[keep],survey1[keep],survey2[keep],offset,dmax)
    pickle.dump((logg_result,logg_bestfit),open('logg_medres_zeropointshift.pkl','wb'))

    x1=np.concatenate([m2fs_hecto_compare_z.T[0],m2fs_h3_compare_z.T[0],m2fs_kirby_compare_z.T[0],m2fs_apogee_compare_z.T[0][m2fs_apogee_keep2],m2fs_sspp_compare_z.T[0],hecto_h3_compare_z.T[0],hecto_kirby_compare_z.T[0],hecto_apogee_compare_z.T[0][hecto_apogee_keep2],hecto_sspp_compare_z.T[0],kirby_h3_compare_z.T[0],kirby_apogee_compare_z.T[0][kirby_apogee_keep2],kirby_sspp_compare_z.T[0],h3_apogee_compare_z.T[0][h3_apogee_keep2],h3_sspp_compare_z.T[0],sspp_apogee_compare_z.T[0][sspp_apogee_keep2]])
    sigx1=np.concatenate([m2fs_hecto_compare_z.T[1],m2fs_h3_compare_z.T[1],m2fs_kirby_compare_z.T[1],m2fs_apogee_compare_z.T[1][m2fs_apogee_keep2],m2fs_sspp_compare_z.T[1],hecto_h3_compare_z.T[1],hecto_kirby_compare_z.T[1],hecto_apogee_compare_z.T[1][hecto_apogee_keep2],hecto_sspp_compare_z.T[1],kirby_h3_compare_z.T[1],kirby_apogee_compare_z.T[1][kirby_apogee_keep2],kirby_sspp_compare_z.T[1],h3_apogee_compare_z.T[1][h3_apogee_keep2],h3_sspp_compare_z.T[1],sspp_apogee_compare_z.T[1][sspp_apogee_keep2]])
    x2=np.concatenate([m2fs_hecto_compare_z.T[2],m2fs_h3_compare_z.T[2],m2fs_kirby_compare_z.T[2],m2fs_apogee_compare_z.T[2][m2fs_apogee_keep2],m2fs_sspp_compare_z.T[2],hecto_h3_compare_z.T[2],hecto_kirby_compare_z.T[2],hecto_apogee_compare_z.T[2][hecto_apogee_keep2],hecto_sspp_compare_z.T[2],kirby_h3_compare_z.T[2],kirby_apogee_compare_z.T[2][kirby_apogee_keep2],kirby_sspp_compare_z.T[2],h3_apogee_compare_z.T[2][h3_apogee_keep2],h3_sspp_compare_z.T[2],sspp_apogee_compare_z.T[2][sspp_apogee_keep2]])
    sigx2=np.concatenate([m2fs_hecto_compare_z.T[3],m2fs_h3_compare_z.T[3],m2fs_kirby_compare_z.T[3],m2fs_apogee_compare_z.T[3][m2fs_apogee_keep2],m2fs_sspp_compare_z.T[3],hecto_h3_compare_z.T[3],hecto_kirby_compare_z.T[3],hecto_apogee_compare_z.T[3][hecto_apogee_keep2],hecto_sspp_compare_z.T[3],kirby_h3_compare_z.T[3],kirby_apogee_compare_z.T[3][kirby_apogee_keep2],kirby_sspp_compare_z.T[3],h3_apogee_compare_z.T[3][h3_apogee_keep2],h3_sspp_compare_z.T[3],sspp_apogee_compare_z.T[3][sspp_apogee_keep2]])
    survey1=np.concatenate([m2fs_hecto_compare_z.T[4],m2fs_h3_compare_z.T[4],m2fs_kirby_compare_z.T[4],m2fs_apogee_compare_z.T[4][m2fs_apogee_keep2],m2fs_sspp_compare_z.T[4],hecto_h3_compare_z.T[4],hecto_kirby_compare_z.T[4],hecto_apogee_compare_z.T[4][hecto_apogee_keep2],hecto_sspp_compare_z.T[4],kirby_h3_compare_z.T[4],kirby_apogee_compare_z.T[4][kirby_apogee_keep2],kirby_sspp_compare_z.T[4],h3_apogee_compare_z.T[4][h3_apogee_keep2],h3_sspp_compare_z.T[4],sspp_apogee_compare_z.T[4][sspp_apogee_keep2]])
    survey2=np.concatenate([m2fs_hecto_compare_z.T[5],m2fs_h3_compare_z.T[5],m2fs_kirby_compare_z.T[5],m2fs_apogee_compare_z.T[5][m2fs_apogee_keep2],m2fs_sspp_compare_z.T[5],hecto_h3_compare_z.T[5],hecto_kirby_compare_z.T[5],hecto_apogee_compare_z.T[5][hecto_apogee_keep2],hecto_sspp_compare_z.T[5],kirby_h3_compare_z.T[5],kirby_apogee_compare_z.T[5][kirby_apogee_keep2],kirby_sspp_compare_z.T[5],h3_apogee_compare_z.T[5][h3_apogee_keep2],h3_sspp_compare_z.T[5],sspp_apogee_compare_z.T[5][sspp_apogee_keep2]])
    
    pickle.dump((x1,sigx1,x2,sigx2,survey1,survey2),open('crap_medres_z.pkl','wb'))
    offset=offset_z
    dmax=dmax_z
    keep=np.where((survey1!=6)&(survey2!=6))[0]
    z_result,z_bestfit=m2fs.get_globaloffset(x1[keep],sigx1[keep],x2[keep],sigx2[keep],survey1[keep],survey2[keep],offset,dmax)
    pickle.dump((z_result,z_bestfit),open('z_medres_zeropointshift.pkl','wb'))

    x1=np.concatenate([m2fs_hecto_compare_alpha.T[0],m2fs_h3_compare_alpha.T[0],m2fs_kirby_compare_alpha.T[0],m2fs_apogee_compare_alpha.T[0][m2fs_apogee_keep2],hecto_h3_compare_alpha.T[0],hecto_kirby_compare_alpha.T[0],hecto_apogee_compare_alpha.T[0][hecto_apogee_keep2],kirby_h3_compare_alpha.T[0],kirby_apogee_compare_alpha.T[0][kirby_apogee_keep2],h3_apogee_compare_alpha.T[0][h3_apogee_keep2]])
    sigx1=np.concatenate([m2fs_hecto_compare_alpha.T[1],m2fs_h3_compare_alpha.T[1],m2fs_kirby_compare_alpha.T[1],m2fs_apogee_compare_alpha.T[1][m2fs_apogee_keep2],hecto_h3_compare_alpha.T[1],hecto_kirby_compare_alpha.T[1],hecto_apogee_compare_alpha.T[1][hecto_apogee_keep2],kirby_h3_compare_alpha.T[1],kirby_apogee_compare_alpha.T[1][kirby_apogee_keep2],h3_apogee_compare_alpha.T[1][h3_apogee_keep2]])
    x2=np.concatenate([m2fs_hecto_compare_alpha.T[2],m2fs_h3_compare_alpha.T[2],m2fs_kirby_compare_alpha.T[2],m2fs_apogee_compare_alpha.T[2][m2fs_apogee_keep2],hecto_h3_compare_alpha.T[2],hecto_kirby_compare_alpha.T[2],hecto_apogee_compare_alpha.T[2][hecto_apogee_keep2],kirby_h3_compare_alpha.T[2],kirby_apogee_compare_alpha.T[2][kirby_apogee_keep2],h3_apogee_compare_alpha.T[2][h3_apogee_keep2]])
    sigx2=np.concatenate([m2fs_hecto_compare_alpha.T[3],m2fs_h3_compare_alpha.T[3],m2fs_kirby_compare_alpha.T[3],m2fs_apogee_compare_alpha.T[3][m2fs_apogee_keep2],hecto_h3_compare_alpha.T[3],hecto_kirby_compare_alpha.T[3],hecto_apogee_compare_alpha.T[3][hecto_apogee_keep2],kirby_h3_compare_alpha.T[3],kirby_apogee_compare_alpha.T[3][kirby_apogee_keep2],h3_apogee_compare_alpha.T[3][h3_apogee_keep2]])
    survey1=np.concatenate([m2fs_hecto_compare_alpha.T[4],m2fs_h3_compare_alpha.T[4],m2fs_kirby_compare_alpha.T[4],m2fs_apogee_compare_alpha.T[4][m2fs_apogee_keep2],hecto_h3_compare_alpha.T[4],hecto_kirby_compare_alpha.T[4],hecto_apogee_compare_alpha.T[4][hecto_apogee_keep2],kirby_h3_compare_alpha.T[4],kirby_apogee_compare_alpha.T[4][kirby_apogee_keep2],h3_apogee_compare_alpha.T[4][h3_apogee_keep2]])
    survey2=np.concatenate([m2fs_hecto_compare_alpha.T[5],m2fs_h3_compare_alpha.T[5],m2fs_kirby_compare_alpha.T[5],m2fs_apogee_compare_alpha.T[5][m2fs_apogee_keep2],hecto_h3_compare_alpha.T[5],hecto_kirby_compare_alpha.T[5],hecto_apogee_compare_alpha.T[5][hecto_apogee_keep2],kirby_h3_compare_alpha.T[5],kirby_apogee_compare_alpha.T[5][kirby_apogee_keep2],h3_apogee_compare_alpha.T[5][h3_apogee_keep2]])
    
    pickle.dump((x1,sigx1,x2,sigx2,survey1,survey2),open('crap_medres_alpha.pkl','wb'))
    offset=offset_alpha
    dmax=dmax_alpha
    keep=np.where((survey1!=6)&(survey2!=6))[0]
    alpha_result,alpha_bestfit=m2fs.get_globaloffset(x1[keep],sigx1[keep],x2[keep],sigx2[keep],survey1[keep],survey2[keep],offset,dmax)
    pickle.dump((alpha_result,alpha_bestfit),open('alpha_medres_zeropointshift.pkl','wb'))

    g1=open('offsets_medres.tex','w')

    string='\\newcommand{\mtwofsmedreshectovn}{'+str(len(m2fs_hecto_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedreshthreevn}{'+str(len(m2fs_h3_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedreswalkervn}{'+str(len(m2fs_walker_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresapogeevn}{'+str(len(m2fs_apogee_compare_v.T[0][m2fs_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresssppvn}{'+str(len(m2fs_sspp_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectohthreevn}{'+str(len(hecto_h3_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectowalkervn}{'+str(len(hecto_walker_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectoapogeevn}{'+str(len(hecto_apogee_compare_v.T[0][hecto_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectossppvn}{'+str(len(hecto_sspp_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreewalkervn}{'+str(len(h3_walker_compare_v.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreeapogeevn}{'+str(len(h3_apogee_compare_v.T[0][h3_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\walkerapogeevn}{'+str(len(walker_apogee_compare_v.T[0][walker_apogee_keep1]))+'} \n'
    g1.write(string)

    string='\\newcommand{\mtwofsmedreshectoteffn}{'+str(len(m2fs_hecto_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedreskirbyteffn}{'+str(len(m2fs_kirby_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedreshthreeteffn}{'+str(len(m2fs_h3_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresapogeeteffn}{'+str(len(m2fs_apogee_compare_teff.T[0][m2fs_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresssppteffn}{'+str(len(m2fs_sspp_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectokirbyteffn}{'+str(len(hecto_kirby_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectohthreeteffn}{'+str(len(hecto_h3_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectoapogeeteffn}{'+str(len(hecto_apogee_compare_teff.T[0][hecto_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectossppteffn}{'+str(len(hecto_sspp_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyhthreeteffn}{'+str(len(kirby_h3_compare_teff.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyapogeeteffn}{'+str(len(kirby_apogee_compare_teff.T[0][kirby_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreeapogeeteffn}{'+str(len(h3_apogee_compare_teff.T[0][h3_apogee_keep1]))+'} \n'
    g1.write(string)

    string='\\newcommand{\mtwofsmedreshectologgn}{'+str(len(m2fs_hecto_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedreskirbyloggn}{'+str(len(m2fs_kirby_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedreshthreeloggn}{'+str(len(m2fs_h3_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresapogeeloggn}{'+str(len(m2fs_apogee_compare_logg.T[0][m2fs_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedressspploggn}{'+str(len(m2fs_sspp_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectokirbyloggn}{'+str(len(hecto_kirby_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectohthreeloggn}{'+str(len(hecto_h3_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectoapogeeloggn}{'+str(len(hecto_apogee_compare_logg.T[0][hecto_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectosspploggn}{'+str(len(hecto_sspp_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyhthreeloggn}{'+str(len(kirby_h3_compare_logg.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyapogeeloggn}{'+str(len(kirby_apogee_compare_logg.T[0][kirby_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreeapogeeloggn}{'+str(len(h3_apogee_compare_logg.T[0][h3_apogee_keep1]))+'} \n'
    g1.write(string)

    string='\\newcommand{\mtwofsmedreshectofehn}{'+str(len(m2fs_hecto_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedreskirbyfehn}{'+str(len(m2fs_kirby_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedreshthreefehn}{'+str(len(m2fs_h3_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresapogeefehn}{'+str(len(m2fs_apogee_compare_z.T[0][m2fs_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresssppfehn}{'+str(len(m2fs_sspp_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectokirbyfehn}{'+str(len(hecto_kirby_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectohthreefehn}{'+str(len(hecto_h3_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectoapogeefehn}{'+str(len(hecto_apogee_compare_z.T[0][hecto_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectossppfehn}{'+str(len(hecto_sspp_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyhthreefehn}{'+str(len(kirby_h3_compare_z.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyapogeefehn}{'+str(len(kirby_apogee_compare_z.T[0][kirby_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreeapogeefehn}{'+str(len(h3_apogee_compare_z.T[0][h3_apogee_keep1]))+'} \n'
    g1.write(string)

    string='\\newcommand{\mtwofsmedreshectoalphan}{'+str(len(m2fs_hecto_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedreskirbyalphan}{'+str(len(m2fs_kirby_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedreshthreealphan}{'+str(len(m2fs_h3_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresapogeealphan}{'+str(len(m2fs_apogee_compare_alpha.T[0][m2fs_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectokirbyalphan}{'+str(len(hecto_kirby_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectohthreealphan}{'+str(len(hecto_h3_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hectoapogeealphan}{'+str(len(hecto_apogee_compare_alpha.T[0][hecto_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyhthreealphan}{'+str(len(kirby_h3_compare_alpha.T[0]))+'} \n'
    g1.write(string)
    string='\\newcommand{\kirbyapogeealphan}{'+str(len(kirby_apogee_compare_alpha.T[0][kirby_apogee_keep1]))+'} \n'
    g1.write(string)
    string='\\newcommand{\hthreeapogeealphan}{'+str(len(h3_apogee_compare_alpha.T[0][h3_apogee_keep1]))+'} \n'
    g1.write(string)
    
    string='\\newcommand{\mtwofsmedresvoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectovoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[1])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[1])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\kirbyvoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[2])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[2])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hthreevoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[3])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[3])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\walkervoffset}{$'+str('{0:.2f}'.format(np.mean(vlos_result['samples'].T[4])))+'\pm '+str('{0:.2f}'.format(np.std(vlos_result['samples'].T[4])))+'$} \n'
    g1.write(string)

    string='\\newcommand{\mtwofsmedresteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[0])))+'\pm '+str(int(np.std(teff_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectoteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[1])))+'\pm '+str(int(np.std(teff_result['samples'].T[1])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\kirbyteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[2])))+'\pm '+str(int(np.std(teff_result['samples'].T[2])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hthreeteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[3])))+'\pm '+str(int(np.std(teff_result['samples'].T[3])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\walkerteffoffset}{$'+str(int(np.mean(teff_result['samples'].T[4])))+'\pm '+str(int(np.std(teff_result['samples'].T[4])))+'$} \n'
    g1.write(string)

    string='\\newcommand{\mtwofsmedresloggoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectologgoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[1])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[1])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\kirbyloggoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[2])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[2])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hthreeloggoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[3])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[3])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\walkerloggoffset}{$'+str('{0:.2f}'.format(np.mean(logg_result['samples'].T[4])))+'\pm '+str('{0:.2f}'.format(np.std(logg_result['samples'].T[4])))+'$} \n'
    g1.write(string)

    string='\\newcommand{\mtwofsmedresfehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectofehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[1])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[1])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\kirbyfehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[2])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[2])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hthreefehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[3])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[3])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\walkerfehoffset}{$'+str('{0:.2f}'.format(np.mean(z_result['samples'].T[4])))+'\pm '+str('{0:.2f}'.format(np.std(z_result['samples'].T[4])))+'$} \n'
    g1.write(string)

    string='\\newcommand{\mtwofsmedresalphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectoalphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[1])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[1])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\kirbyalphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[2])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[2])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hthreealphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[3])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[3])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\walkeralphaoffset}{$'+str('{0:.2f}'.format(np.mean(alpha_result['samples'].T[4])))+'\pm '+str('{0:.2f}'.format(np.std(alpha_result['samples'].T[4])))+'$} \n'
    g1.write(string)

    g1.close()

    gs=plt.GridSpec(17,15)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax11=fig.add_subplot(gs[0:3,0:6])
    ax21=fig.add_subplot(gs[5:8,0:6])
    ax31=fig.add_subplot(gs[8:11,0:6])
    ax41=fig.add_subplot(gs[11:14,0:6])

#    m2fs_keep=np.where((m2fs_walker_compare_v.T[1]<5.)&(m2fs_walker_compare_v.T[3]<5.)&(m2fs_walker_compare_logg.T[1]<0.5)&(m2fs_walker_compare_logg.T[3]<0.5)&(m2fs_walker_compare_z.T[1]<0.5)&(m2fs_walker_compare_z.T[3]<0.5)&(m2fs_walker_compare_alpha.T[1]<0.5)&(m2fs_walker_compare_alpha.T[3]<0.5))[0]
#    hecto_keep=np.where((hecto_walker_compare_v.T[1]<5.)&(hecto_walker_compare_v.T[3]<5.)&(hecto_walker_compare_logg.T[1]<0.5)&(hecto_walker_compare_logg.T[3]<0.5)&(hecto_walker_compare_z.T[1]<0.5)&(hecto_walker_compare_z.T[3]<0.5)&(hecto_walker_compare_alpha.T[1]<0.5)&(hecto_walker_compare_alpha.T[3]<0.5))[0]

    m2fs_keep=np.where((m2fs_hecto_compare_v.T[1]<5.)&(m2fs_hecto_compare_v.T[3]<5.))[0]
    ax11.errorbar(m2fs_hecto_compare_v.T[0][m2fs_keep],m2fs_hecto_compare_v.T[0][m2fs_keep]-m2fs_hecto_compare_v.T[2][m2fs_keep],xerr=m2fs_hecto_compare_v.T[1][m2fs_keep],yerr=np.sqrt((m2fs_hecto_compare_v.T[1][m2fs_keep])**2+(m2fs_hecto_compare_v.T[3][m2fs_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='k',ms=1,rasterized=True)
    m2fs_offset=np.mean(vlos_result['samples'].T[1])-np.mean(vlos_result['samples'].T[0])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-x0-m2fs_offset
    ax11.plot([-10000,10000],[0,0],lw=0.5,linestyle='--',color='k')
    ax11.set_xlim([-400,400])
    ax11.set_ylim([-9.9,10])
    ax11.set_yticks([-5,0,5,10])
    ax11.set_yticklabels([-5,0,5,10],fontsize=7)
    ax11.set_xlabel(r'$V_{\rm LOS,M2FS}$ [km/s]',fontsize=7)
    ax11.set_ylabel(r'$V_{\rm LOS,Hecto}$ [km/s]',fontsize=7)

    m2fs_keep=np.where((m2fs_walker_compare_v.T[1]<5.)&(m2fs_walker_compare_v.T[3]<5.))[0]
    hecto_keep=np.where((hecto_walker_compare_v.T[1]<5.)&(hecto_walker_compare_v.T[3]<5.))[0]
    ax21.errorbar(m2fs_walker_compare_v.T[0][m2fs_keep],m2fs_walker_compare_v.T[0][m2fs_keep]-m2fs_walker_compare_v.T[2][m2fs_keep],xerr=m2fs_walker_compare_v.T[1][m2fs_keep],yerr=np.sqrt((m2fs_walker_compare_v.T[1][m2fs_keep])**2+(m2fs_walker_compare_v.T[3][m2fs_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax21.errorbar(hecto_walker_compare_v.T[0][hecto_keep],hecto_walker_compare_v.T[0][hecto_keep]-hecto_walker_compare_v.T[2][hecto_keep],xerr=hecto_walker_compare_v.T[1][hecto_keep],yerr=np.sqrt((hecto_walker_compare_v.T[1][hecto_keep])**2+(hecto_walker_compare_v.T[3][hecto_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(vlos_result['samples'].T[4])-np.mean(vlos_result['samples'].T[0])
    hecto_offset=np.mean(vlos_result['samples'].T[4])-np.mean(vlos_result['samples'].T[1])
    print(offset)
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-x0-m2fs_offset
    hecto_y0=x0-x0-hecto_offset
    ax21.plot([-10000,10000],[0,0],lw=0.5,linestyle='--',color='k')
    ax21.set_xlim([-400,400])
    ax21.set_ylim([-9.9,10])
    ax21.set_yticks([-5,0,5,10])
    ax21.set_yticklabels([-5,0,5,10],fontsize=7)
    ax21.set_xticklabels([])
    ax21.set_ylabel(r'$V_{\rm LOS}-V_{\rm LOS,W09}$ [km/s]',fontsize=7,labelpad=-3)
    ax21.legend(loc=2,fontsize=6,borderaxespad=0)
    
    m2fs_keep=np.where((m2fs_apogee_compare_v.T[1]<5.)&(m2fs_apogee_compare_v.T[3]<5.))[0]
    hecto_keep=np.where((hecto_apogee_compare_v.T[1]<5.)&(hecto_apogee_compare_v.T[3]<5.))[0]
    ax31.errorbar(m2fs_apogee_compare_v.T[0][m2fs_keep],m2fs_apogee_compare_v.T[0][m2fs_keep]-m2fs_apogee_compare_v.T[2][m2fs_keep],xerr=m2fs_apogee_compare_v.T[1][m2fs_keep],yerr=np.sqrt((m2fs_apogee_compare_v.T[1][m2fs_keep])**2+(m2fs_apogee_compare_v.T[3][m2fs_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='r',ms=1,rasterized=True)
    ax31.errorbar(hecto_apogee_compare_v.T[0][hecto_keep],hecto_apogee_compare_v.T[0][hecto_keep]-hecto_apogee_compare_v.T[2][hecto_keep],xerr=hecto_apogee_compare_v.T[1][hecto_keep],yerr=np.sqrt((hecto_apogee_compare_v.T[1][hecto_keep])**2+(hecto_apogee_compare_v.T[3][hecto_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    m2fs_offset=0.-np.mean(vlos_result['samples'].T[0])
    hecto_offset=0.-np.mean(vlos_result['samples'].T[1])
    print(offset)
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-x0-m2fs_offset
    hecto_y0=x0-x0-hecto_offset
    ax31.plot([-10000,10000],[0,0],lw=0.5,linestyle='--',color='k')
    ax31.set_xlim([-400,400])
    ax31.set_ylim([-9.9,9.9])
    ax31.set_yticks([-5,0,5])
    ax31.set_yticklabels([-5,0,5],fontsize=7)
    ax31.set_xticklabels([])
    ax31.set_ylabel(r'$V_{\rm LOS}-V_{\rm LOS,Apogee}$ [km/s]',fontsize=7,labelpad=10)

    m2fs_keep=np.where((m2fs_h3_compare_v.T[1]<5.)&(m2fs_h3_compare_v.T[3]<5.))[0]
    hecto_keep=np.where((hecto_h3_compare_v.T[1]<5.)&(hecto_h3_compare_v.T[3]<5.))[0]
    ax41.errorbar(m2fs_h3_compare_v.T[0][m2fs_keep],m2fs_h3_compare_v.T[0][m2fs_keep]-m2fs_h3_compare_v.T[2][m2fs_keep],xerr=m2fs_h3_compare_v.T[1][m2fs_keep],yerr=np.sqrt((m2fs_h3_compare_v.T[1][m2fs_keep])**2+(m2fs_h3_compare_v.T[3][m2fs_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='r',ms=1,rasterized=True)
    ax41.errorbar(hecto_h3_compare_v.T[0][hecto_keep],hecto_h3_compare_v.T[0][hecto_keep]-hecto_h3_compare_v.T[2][hecto_keep],xerr=hecto_h3_compare_v.T[1][hecto_keep],yerr=np.sqrt((hecto_h3_compare_v.T[1][hecto_keep])**2+(hecto_h3_compare_v.T[3][hecto_keep])**2),alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    m2fs_offset=np.mean(vlos_result['samples'].T[3])-np.mean(vlos_result['samples'].T[0])
    hecto_offset=np.mean(vlos_result['samples'].T[3])-np.mean(vlos_result['samples'].T[1])
    print(offset)
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-x0-m2fs_offset
    hecto_y0=x0-x0-hecto_offset
    ax41.plot([-10000,10000],[0,0],lw=0.5,linestyle='--',color='k')
    ax41.set_ylabel(r'$V_{\rm LOS}-V_{\rm LOS,H3}$ [km/s]',fontsize=7,labelpad=-3)
    ax41.set_xlim([-400,400])
    ax41.set_ylim([-10,9.9])
    ax41.set_yticks([-10,-5,0,5])
    ax41.set_yticklabels([-10,-5,0,5],fontsize=7)
    ax41.set_xticks([-400,-200,0,200,400])
    ax41.set_xticklabels([-400,-200,0,200,400],fontsize=7)
    ax41.set_xlabel(r'$V_{\rm LOS}$ [km/s]',fontsize=7)    

    plt.savefig('v_offset.pdf',dpi=270)
    plt.show()
    plt.close()
    
    gs=plt.GridSpec(18,18)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax11=fig.add_subplot(gs[0:3,0:3])
    ax21=fig.add_subplot(gs[5:8,0:3])
    ax31=fig.add_subplot(gs[8:11,0:3])
    ax41=fig.add_subplot(gs[11:14,0:3])

    ax12=fig.add_subplot(gs[0:3,5:8])
    ax22=fig.add_subplot(gs[5:8,5:8])
    ax32=fig.add_subplot(gs[8:11,5:8])
    ax42=fig.add_subplot(gs[11:14,5:8])

    ax13=fig.add_subplot(gs[0:3,10:13])
    ax23=fig.add_subplot(gs[5:8,10:13])
    ax33=fig.add_subplot(gs[8:11,10:13])
    ax43=fig.add_subplot(gs[11:14,10:13])
    
    ax14=fig.add_subplot(gs[0:3,15:18])
    ax24=fig.add_subplot(gs[5:8,15:18])
    ax34=fig.add_subplot(gs[8:11,15:18])
    ax44=fig.add_subplot(gs[11:14,15:18])
    
    m2fs_keep=np.where((m2fs_hecto_compare_v.T[1]<5.)&(m2fs_hecto_compare_v.T[3]<5.)&(m2fs_hecto_compare_logg.T[1]<0.5)&(m2fs_hecto_compare_logg.T[3]<0.5)&(m2fs_hecto_compare_z.T[1]<0.5)&(m2fs_hecto_compare_z.T[3]<0.5)&(m2fs_hecto_compare_alpha.T[1]<0.5)&(m2fs_hecto_compare_alpha.T[3]<0.5))[0]

    ax11.errorbar(m2fs_hecto_compare_teff.T[0][m2fs_keep]/1000,m2fs_hecto_compare_teff.T[2][m2fs_keep]/1000,xerr=m2fs_hecto_compare_teff.T[1][m2fs_keep]/1000,yerr=m2fs_hecto_compare_teff.T[3][m2fs_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='k',ms=1,rasterized=True)
    m2fs_offset=np.mean(teff_result['samples'].T[1])-np.mean(teff_result['samples'].T[0])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset/1000
    ax11.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax11.set_xlim([3.5,8])
    ax11.set_ylim([3.5,8])
    ax11.set_xticks([4,5,6,7,8])
    ax11.set_yticks([4,5,6,7,8])
    ax11.set_xticklabels([4,5,6,7,8],fontsize=6)
    ax11.set_yticklabels([4,5,6,7,8],fontsize=6)
    ax11.set_xlabel(r'$T_{\rm eff,M2FS}$ [$10^3$ K]',fontsize=6)
    ax11.set_ylabel(r'$T_{\rm eff,Hecto}$ [$10^3$ K]',fontsize=6)

    m2fs_keep=np.where((m2fs_kirby_compare_logg.T[1]<0.5)&(m2fs_kirby_compare_logg.T[3]<0.5)&(m2fs_kirby_compare_z.T[1]<0.5)&(m2fs_kirby_compare_z.T[3]<0.5)&(m2fs_kirby_compare_alpha.T[1]<0.5)&(m2fs_kirby_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_kirby_compare_logg.T[1]<0.5)&(hecto_kirby_compare_logg.T[3]<0.5)&(hecto_kirby_compare_z.T[1]<0.5)&(hecto_kirby_compare_z.T[3]<0.5)&(hecto_kirby_compare_alpha.T[1]<0.5)&(hecto_kirby_compare_alpha.T[3]<0.5))[0]
    ax21.errorbar(m2fs_kirby_compare_teff.T[0][m2fs_keep]/1000,m2fs_kirby_compare_teff.T[2][m2fs_keep]/1000,xerr=m2fs_kirby_compare_teff.T[1][m2fs_keep]/1000,yerr=m2fs_kirby_compare_teff.T[3][m2fs_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax21.errorbar(hecto_kirby_compare_teff.T[0][hecto_keep]/1000,hecto_kirby_compare_teff.T[2][hecto_keep]/1000,xerr=hecto_kirby_compare_teff.T[1][hecto_keep]/1000,yerr=hecto_kirby_compare_teff.T[3][hecto_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(teff_result['samples'].T[2])-np.mean(teff_result['samples'].T[0])
    hecto_offset=np.mean(teff_result['samples'].T[2])-np.mean(teff_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset/1000
    hecto_y0=x0-hecto_offset/1000
    ax21.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax21.set_xlim([3.5,8])
    ax21.set_ylim([3.5,7.99])
    ax21.set_yticks([4,5,6,7])
    ax21.set_yticklabels([4,5,6,7],fontsize=6)
    ax21.set_xticklabels([])
    ax21.set_ylabel(r'$T_{\rm eff,K10}$ [K]',fontsize=6)
    ax21.legend(loc=2,fontsize=5,borderaxespad=0)


    m2fs_keep=np.where((m2fs_apogee_compare_logg.T[1]<0.5)&(m2fs_apogee_compare_logg.T[3]<0.5)&(m2fs_apogee_compare_z.T[1]<0.5)&(m2fs_apogee_compare_z.T[3]<0.5)&(m2fs_apogee_compare_alpha.T[1]<0.5)&(m2fs_apogee_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_apogee_compare_logg.T[1]<0.5)&(hecto_apogee_compare_logg.T[3]<0.5)&(hecto_apogee_compare_z.T[1]<0.5)&(hecto_apogee_compare_z.T[3]<0.5)&(hecto_apogee_compare_alpha.T[1]<0.5)&(hecto_apogee_compare_alpha.T[3]<0.5))[0]
    ax31.errorbar(m2fs_apogee_compare_teff.T[0][m2fs_keep]/1000,m2fs_apogee_compare_teff.T[2][m2fs_keep]/1000,xerr=m2fs_apogee_compare_teff.T[1][m2fs_keep]/1000,yerr=m2fs_apogee_compare_teff.T[3][m2fs_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax31.errorbar(hecto_apogee_compare_teff.T[0][hecto_keep]/1000,hecto_apogee_compare_teff.T[2][hecto_keep]/1000,xerr=hecto_apogee_compare_teff.T[1][hecto_keep]/1000,yerr=hecto_apogee_compare_teff.T[3][hecto_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=0.-np.mean(teff_result['samples'].T[0])
    hecto_offset=0.-np.mean(teff_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset/1000
    hecto_y0=x0-hecto_offset/1000
    ax31.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax31.set_xlim([3.5,8])
    ax31.set_ylim([3.5,7.99])
    ax31.set_yticks([4,5,6,7])
    ax31.set_yticklabels([4,5,6,7],fontsize=6)
    ax31.set_xticklabels([])
    ax31.set_ylabel(r'$T_{\rm eff,Apogee}$ [K]',fontsize=6)
    
    m2fs_keep=np.where((m2fs_h3_compare_logg.T[1]<0.5)&(m2fs_h3_compare_logg.T[3]<0.5)&(m2fs_h3_compare_z.T[1]<0.5)&(m2fs_h3_compare_z.T[3]<0.5)&(m2fs_h3_compare_alpha.T[1]<0.5)&(m2fs_h3_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_h3_compare_logg.T[1]<0.5)&(hecto_h3_compare_logg.T[3]<0.5)&(hecto_h3_compare_z.T[1]<0.5)&(hecto_h3_compare_z.T[3]<0.5)&(hecto_h3_compare_alpha.T[1]<0.5)&(hecto_h3_compare_alpha.T[3]<0.5))[0]
    ax41.errorbar(m2fs_h3_compare_teff.T[0][m2fs_keep]/1000,m2fs_h3_compare_teff.T[2][m2fs_keep]/1000,xerr=m2fs_h3_compare_teff.T[1][m2fs_keep]/1000,yerr=m2fs_h3_compare_teff.T[3][m2fs_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax41.errorbar(hecto_h3_compare_teff.T[0][hecto_keep]/1000,hecto_h3_compare_teff.T[2][hecto_keep]/1000,xerr=hecto_h3_compare_teff.T[1][hecto_keep]/1000,yerr=hecto_h3_compare_teff.T[3][hecto_keep]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(teff_result['samples'].T[3])-np.mean(teff_result['samples'].T[0])
    hecto_offset=np.mean(teff_result['samples'].T[3])-np.mean(teff_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset/1000
    hecto_y0=x0-hecto_offset/1000
    ax41.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax41.set_xlim([3.5,8])
    ax41.set_ylim([3.5,7.99])
    ax41.set_xticks([4,5,6,7,8])
    ax41.set_yticks([4,5,6,7])
    ax41.set_yticklabels([4,5,6,7],fontsize=6)
    ax41.set_xticklabels([4,5,6,7,8])
    ax41.set_ylabel(r'$T_{\rm eff,H3}$ [K]',fontsize=6)
    ax41.set_xlabel(r'$T_{\rm eff}$ [K]',fontsize=6)

    m2fs_keep=np.where((m2fs_hecto_compare_v.T[1]<5.)&(m2fs_hecto_compare_v.T[3]<5.)&(m2fs_hecto_compare_logg.T[1]<0.5)&(m2fs_hecto_compare_logg.T[3]<0.5)&(m2fs_hecto_compare_z.T[1]<0.5)&(m2fs_hecto_compare_z.T[3]<0.5)&(m2fs_hecto_compare_alpha.T[1]<0.5)&(m2fs_hecto_compare_alpha.T[3]<0.5))[0]

    ax12.errorbar(m2fs_hecto_compare_logg.T[0][m2fs_keep],m2fs_hecto_compare_logg.T[2][m2fs_keep],xerr=m2fs_hecto_compare_logg.T[1][m2fs_keep],yerr=m2fs_hecto_compare_logg.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='k',ms=1,rasterized=True)
    m2fs_offset=np.mean(logg_result['samples'].T[1])-np.mean(logg_result['samples'].T[0])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    ax12.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax12.set_xlim([0,5])
    ax12.set_ylim([0.01,5])
    ax12.set_xticks([0,1,2,3,4,5])
    ax12.set_yticks([0,1,2,3,4,5])
    ax12.set_xticklabels([0,1,2,3,4,5],fontsize=6)
    ax12.set_yticklabels([0,1,2,3,4,5],fontsize=6)
    ax12.set_xlabel(r'$\log g_{\rm M2FS}$',fontsize=6)
    ax12.set_ylabel(r'$\log g_{\rm Hecto}$',fontsize=6)

    m2fs_keep=np.where((m2fs_kirby_compare_logg.T[1]<0.5)&(m2fs_kirby_compare_logg.T[3]<0.5)&(m2fs_kirby_compare_z.T[1]<0.5)&(m2fs_kirby_compare_z.T[3]<0.5)&(m2fs_kirby_compare_alpha.T[1]<0.5)&(m2fs_kirby_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_kirby_compare_logg.T[1]<0.5)&(hecto_kirby_compare_logg.T[3]<0.5)&(hecto_kirby_compare_z.T[1]<0.5)&(hecto_kirby_compare_z.T[3]<0.5)&(hecto_kirby_compare_alpha.T[1]<0.5)&(hecto_kirby_compare_alpha.T[3]<0.5))[0]
    ax22.errorbar(m2fs_kirby_compare_logg.T[0][m2fs_keep],m2fs_kirby_compare_logg.T[2][m2fs_keep],xerr=m2fs_kirby_compare_logg.T[1][m2fs_keep],yerr=m2fs_kirby_compare_logg.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax22.errorbar(hecto_kirby_compare_logg.T[0][hecto_keep],hecto_kirby_compare_logg.T[2][hecto_keep],xerr=hecto_kirby_compare_logg.T[1][hecto_keep],yerr=hecto_kirby_compare_logg.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(logg_result['samples'].T[2])-np.mean(logg_result['samples'].T[0])
    hecto_offset=np.mean(logg_result['samples'].T[2])-np.mean(logg_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    hecto_y0=x0-hecto_offset
    ax22.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax22.set_xlim([0,5])
    ax22.set_ylim([0.01,4.99])
    ax22.set_yticks([1,2,3,4])
    ax22.set_yticklabels([1,2,3,4],fontsize=6)
    ax22.set_xticklabels([])
    ax22.set_ylabel(r'$\log g_{\rm K10}$',fontsize=6)

    m2fs_keep=np.where((m2fs_apogee_compare_logg.T[1]<0.5)&(m2fs_apogee_compare_logg.T[3]<0.5)&(m2fs_apogee_compare_z.T[1]<0.5)&(m2fs_apogee_compare_z.T[3]<0.5)&(m2fs_apogee_compare_alpha.T[1]<0.5)&(m2fs_apogee_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_apogee_compare_logg.T[1]<0.5)&(hecto_apogee_compare_logg.T[3]<0.5)&(hecto_apogee_compare_z.T[1]<0.5)&(hecto_apogee_compare_z.T[3]<0.5)&(hecto_apogee_compare_alpha.T[1]<0.5)&(hecto_apogee_compare_alpha.T[3]<0.5))[0]
    ax32.errorbar(m2fs_apogee_compare_logg.T[0][m2fs_keep],m2fs_apogee_compare_logg.T[2][m2fs_keep],xerr=m2fs_apogee_compare_logg.T[1][m2fs_keep],yerr=m2fs_apogee_compare_logg.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax32.errorbar(hecto_apogee_compare_logg.T[0][hecto_keep],hecto_apogee_compare_logg.T[2][hecto_keep],xerr=hecto_apogee_compare_logg.T[1][hecto_keep],yerr=hecto_apogee_compare_logg.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(logg_result['samples'].T[3])-np.mean(logg_result['samples'].T[0])
    hecto_offset=np.mean(logg_result['samples'].T[3])-np.mean(logg_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    hecto_y0=x0-hecto_offset
    ax32.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax32.set_xlim([0,5])
    ax32.set_ylim([0.01,4.99])
    ax32.set_yticks([1,2,3,4])
    ax32.set_yticklabels([1,2,3,4],fontsize=6)
    ax32.set_xticklabels([])
    ax32.set_ylabel(r'$\log g_{\rm Apo}$',fontsize=6)
    
    m2fs_keep=np.where((m2fs_h3_compare_logg.T[1]<0.5)&(m2fs_h3_compare_logg.T[3]<0.5)&(m2fs_h3_compare_z.T[1]<0.5)&(m2fs_h3_compare_z.T[3]<0.5)&(m2fs_h3_compare_alpha.T[1]<0.5)&(m2fs_h3_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_h3_compare_logg.T[1]<0.5)&(hecto_h3_compare_logg.T[3]<0.5)&(hecto_h3_compare_z.T[1]<0.5)&(hecto_h3_compare_z.T[3]<0.5)&(hecto_h3_compare_alpha.T[1]<0.5)&(hecto_h3_compare_alpha.T[3]<0.5))[0]
    ax42.errorbar(m2fs_h3_compare_logg.T[0][m2fs_keep],m2fs_h3_compare_logg.T[2][m2fs_keep],xerr=m2fs_h3_compare_logg.T[1][m2fs_keep],yerr=m2fs_h3_compare_logg.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax42.errorbar(hecto_h3_compare_logg.T[0][hecto_keep],hecto_h3_compare_logg.T[2][hecto_keep],xerr=hecto_h3_compare_logg.T[1][hecto_keep],yerr=hecto_h3_compare_logg.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(logg_result['samples'].T[3])-np.mean(logg_result['samples'].T[0])
    hecto_offset=np.mean(logg_result['samples'].T[3])-np.mean(logg_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    hecto_y0=x0-hecto_offset
    ax42.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax42.set_xlim([0,5])
    ax42.set_ylim([0,4.99])
    ax42.set_xticks([0,1,2,3,4,5])
    ax42.set_yticks([0,1,2,3,4])
    ax42.set_yticklabels([0,1,2,3,4],fontsize=6)
    ax42.set_xticklabels([0,1,2,3,4,5])
    ax42.set_ylabel(r'$\log g_{\rm H3}$',fontsize=6)
    ax42.set_xlabel(r'$\log g$',fontsize=6)
    
    m2fs_keep=np.where((m2fs_hecto_compare_v.T[1]<5.)&(m2fs_hecto_compare_v.T[3]<5.)&(m2fs_hecto_compare_logg.T[1]<0.5)&(m2fs_hecto_compare_logg.T[3]<0.5)&(m2fs_hecto_compare_z.T[1]<0.5)&(m2fs_hecto_compare_z.T[3]<0.5)&(m2fs_hecto_compare_alpha.T[1]<0.5)&(m2fs_hecto_compare_alpha.T[3]<0.5))[0]

    ax13.errorbar(m2fs_hecto_compare_z.T[0][m2fs_keep],m2fs_hecto_compare_z.T[2][m2fs_keep],xerr=m2fs_hecto_compare_z.T[1][m2fs_keep],yerr=m2fs_hecto_compare_z.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='k',ms=1,rasterized=True)
    m2fs_offset=np.mean(z_result['samples'].T[1])-np.mean(z_result['samples'].T[0])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    ax13.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax13.set_xlim([-4,1])
    ax13.set_ylim([-4,1])
    ax13.set_xticks([-4,-3,-2,-1,0,1])
    ax13.set_yticks([-4,-3,-2,-1,0,1])
    ax13.set_xticklabels([-4,-3,-2,-1,0,1],fontsize=6)
    ax13.set_yticklabels([-4,-3,-2,-1,0,1],fontsize=6)
    ax13.set_xlabel(r'[Fe/H]$_{\rm M2FS}$',fontsize=6)
    ax13.set_ylabel(r'[Fe/H]$_{\rm Hecto}$',fontsize=6)

    m2fs_keep=np.where((m2fs_kirby_compare_logg.T[1]<0.5)&(m2fs_kirby_compare_logg.T[3]<0.5)&(m2fs_kirby_compare_z.T[1]<0.5)&(m2fs_kirby_compare_z.T[3]<0.5)&(m2fs_kirby_compare_alpha.T[1]<0.5)&(m2fs_kirby_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_kirby_compare_logg.T[1]<0.5)&(hecto_kirby_compare_logg.T[3]<0.5)&(hecto_kirby_compare_z.T[1]<0.5)&(hecto_kirby_compare_z.T[3]<0.5)&(hecto_kirby_compare_alpha.T[1]<0.5)&(hecto_kirby_compare_alpha.T[3]<0.5))[0]
    ax23.errorbar(m2fs_kirby_compare_z.T[0][m2fs_keep],m2fs_kirby_compare_z.T[2][m2fs_keep],xerr=m2fs_kirby_compare_z.T[1][m2fs_keep],yerr=m2fs_kirby_compare_z.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax23.errorbar(hecto_kirby_compare_z.T[0][hecto_keep],hecto_kirby_compare_z.T[2][hecto_keep],xerr=hecto_kirby_compare_z.T[1][hecto_keep],yerr=hecto_kirby_compare_z.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(z_result['samples'].T[2])-np.mean(z_result['samples'].T[0])
    hecto_offset=np.mean(z_result['samples'].T[2])-np.mean(z_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    hecto_y0=x0-hecto_offset
    ax23.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax23.set_xlim([-4,1])
    ax23.set_ylim([-4,1])
    ax23.set_yticks([-3,-2,-1,0,1])
    ax23.set_yticklabels([-3,-2,-1,0,1],fontsize=6)
    ax23.set_xticklabels([])
    ax23.set_ylabel(r'[Fe/H]$_{\rm K10}$',fontsize=6)

    m2fs_keep=np.where((m2fs_apogee_compare_logg.T[1]<0.5)&(m2fs_apogee_compare_logg.T[3]<0.5)&(m2fs_apogee_compare_z.T[1]<0.5)&(m2fs_apogee_compare_z.T[3]<0.5)&(m2fs_apogee_compare_alpha.T[1]<0.5)&(m2fs_apogee_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_apogee_compare_logg.T[1]<0.5)&(hecto_apogee_compare_logg.T[3]<0.5)&(hecto_apogee_compare_z.T[1]<0.5)&(hecto_apogee_compare_z.T[3]<0.5)&(hecto_apogee_compare_alpha.T[1]<0.5)&(hecto_apogee_compare_alpha.T[3]<0.5))[0]
    ax33.errorbar(m2fs_apogee_compare_z.T[0][m2fs_keep],m2fs_apogee_compare_z.T[2][m2fs_keep],xerr=m2fs_apogee_compare_z.T[1][m2fs_keep],yerr=m2fs_apogee_compare_z.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax33.errorbar(hecto_apogee_compare_z.T[0][hecto_keep],hecto_apogee_compare_z.T[2][hecto_keep],xerr=hecto_apogee_compare_z.T[1][hecto_keep],yerr=hecto_apogee_compare_z.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(z_result['samples'].T[3])-np.mean(z_result['samples'].T[0])
    hecto_offset=np.mean(z_result['samples'].T[3])-np.mean(z_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    hecto_y0=x0-hecto_offset
    ax33.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax33.set_xlim([-4,1])
    ax33.set_ylim([-4,1])
    ax33.set_yticks([-3,-2,-1,0])
    ax33.set_yticklabels([-3,-2,-1,0],fontsize=6)
    ax33.set_xticklabels([])
    ax33.set_ylabel(r'[Fe/H]$_{\rm Apo}$',fontsize=6)
    
    m2fs_keep=np.where((m2fs_h3_compare_logg.T[1]<0.5)&(m2fs_h3_compare_logg.T[3]<0.5)&(m2fs_h3_compare_z.T[1]<0.5)&(m2fs_h3_compare_z.T[3]<0.5)&(m2fs_h3_compare_alpha.T[1]<0.5)&(m2fs_h3_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_h3_compare_logg.T[1]<0.5)&(hecto_h3_compare_logg.T[3]<0.5)&(hecto_h3_compare_z.T[1]<0.5)&(hecto_h3_compare_z.T[3]<0.5)&(hecto_h3_compare_alpha.T[1]<0.5)&(hecto_h3_compare_alpha.T[3]<0.5))[0]
    ax43.errorbar(m2fs_h3_compare_z.T[0][m2fs_keep],m2fs_h3_compare_z.T[2][m2fs_keep],xerr=m2fs_h3_compare_z.T[1][m2fs_keep],yerr=m2fs_h3_compare_z.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax43.errorbar(hecto_h3_compare_z.T[0][hecto_keep],hecto_h3_compare_z.T[2][hecto_keep],xerr=hecto_h3_compare_z.T[1][hecto_keep],yerr=hecto_h3_compare_z.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(z_result['samples'].T[3])-np.mean(z_result['samples'].T[0])
    hecto_offset=np.mean(z_result['samples'].T[3])-np.mean(z_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    hecto_y0=x0-hecto_offset
    ax43.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax43.set_xlim([-4,1])
    ax43.set_ylim([-4,1])
    ax43.set_xticks([-4,-3,-2,-1,0,1])
    ax43.set_yticks([-4,-3,-2,-1,0])
    ax43.set_yticklabels([-4,-3,-2,-1,0],fontsize=6)
    ax43.set_xticklabels([-4,-3,-2,-1,0,1],fontsize=6)
    ax43.set_ylabel(r'[Fe/H]$_{\rm H3}$',fontsize=6)
    ax43.set_xlabel(r'[Fe/H]',fontsize=6)
    
    m2fs_keep=np.where((m2fs_hecto_compare_v.T[1]<5.)&(m2fs_hecto_compare_v.T[3]<5.)&(m2fs_hecto_compare_logg.T[1]<0.5)&(m2fs_hecto_compare_logg.T[3]<0.5)&(m2fs_hecto_compare_z.T[1]<0.5)&(m2fs_hecto_compare_z.T[3]<0.5)&(m2fs_hecto_compare_alpha.T[1]<0.5)&(m2fs_hecto_compare_alpha.T[3]<0.5))[0]

    ax14.errorbar(m2fs_hecto_compare_alpha.T[0][m2fs_keep],m2fs_hecto_compare_alpha.T[2][m2fs_keep],xerr=m2fs_hecto_compare_alpha.T[1][m2fs_keep],yerr=m2fs_hecto_compare_alpha.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='k',ms=1,rasterized=True)
    m2fs_offset=np.mean(alpha_result['samples'].T[1])-np.mean(alpha_result['samples'].T[0])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    ax14.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax14.set_xlim([-1,1])
    ax14.set_ylim([-1,1])
    ax14.set_xticks([-1,-0.5,0,0.5,1])
    ax14.set_yticks([-1,-0.5,0,0.5,1])
    ax14.set_xticklabels([-1,-0.5,0,0.5,1],fontsize=6)
    ax14.set_yticklabels([-1,-0.5,0,0.5,1],fontsize=6)
    ax14.set_xlabel(r'[Mg/Fe]$_{\rm M2FS}$',fontsize=6)
    ax14.set_ylabel(r'[Mg/Fe]$_{\rm Hecto}$',fontsize=6)

    m2fs_keep=np.where((m2fs_kirby_compare_logg.T[1]<0.5)&(m2fs_kirby_compare_logg.T[3]<0.5)&(m2fs_kirby_compare_z.T[1]<0.5)&(m2fs_kirby_compare_z.T[3]<0.5)&(m2fs_kirby_compare_alpha.T[1]<0.5)&(m2fs_kirby_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_kirby_compare_logg.T[1]<0.5)&(hecto_kirby_compare_logg.T[3]<0.5)&(hecto_kirby_compare_z.T[1]<0.5)&(hecto_kirby_compare_z.T[3]<0.5)&(hecto_kirby_compare_alpha.T[1]<0.5)&(hecto_kirby_compare_alpha.T[3]<0.5))[0]
    ax24.errorbar(m2fs_kirby_compare_alpha.T[0][m2fs_keep],m2fs_kirby_compare_alpha.T[2][m2fs_keep],xerr=m2fs_kirby_compare_alpha.T[1][m2fs_keep],yerr=m2fs_kirby_compare_alpha.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax24.errorbar(hecto_kirby_compare_alpha.T[0][hecto_keep],hecto_kirby_compare_alpha.T[2][hecto_keep],xerr=hecto_kirby_compare_alpha.T[1][hecto_keep],yerr=hecto_kirby_compare_alpha.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(alpha_result['samples'].T[2])-np.mean(alpha_result['samples'].T[0])
    hecto_offset=np.mean(alpha_result['samples'].T[2])-np.mean(alpha_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    hecto_y0=x0-hecto_offset
    ax24.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax24.set_xlim([-1,1])
    ax24.set_ylim([-1,1])
    ax24.set_yticks([-0.5,0,0.5])
    ax24.set_yticklabels([-0.5,0,0.5],fontsize=6)
    ax24.set_xticklabels([])
    ax24.set_ylabel(r'[Mg/Fe]$_{\rm K10}$',fontsize=6)

    m2fs_keep=np.where((m2fs_apogee_compare_logg.T[1]<0.5)&(m2fs_apogee_compare_logg.T[3]<0.5)&(m2fs_apogee_compare_z.T[1]<0.5)&(m2fs_apogee_compare_z.T[3]<0.5)&(m2fs_apogee_compare_alpha.T[1]<0.5)&(m2fs_apogee_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_apogee_compare_logg.T[1]<0.5)&(hecto_apogee_compare_logg.T[3]<0.5)&(hecto_apogee_compare_z.T[1]<0.5)&(hecto_apogee_compare_z.T[3]<0.5)&(hecto_apogee_compare_alpha.T[1]<0.5)&(hecto_apogee_compare_alpha.T[3]<0.5))[0]
    ax34.errorbar(m2fs_apogee_compare_alpha.T[0][m2fs_keep],m2fs_apogee_compare_alpha.T[2][m2fs_keep],xerr=m2fs_apogee_compare_alpha.T[1][m2fs_keep],yerr=m2fs_apogee_compare_alpha.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax34.errorbar(hecto_apogee_compare_alpha.T[0][hecto_keep],hecto_apogee_compare_alpha.T[2][hecto_keep],xerr=hecto_apogee_compare_alpha.T[1][hecto_keep],yerr=hecto_apogee_compare_alpha.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(alpha_result['samples'].T[3])-np.mean(alpha_result['samples'].T[0])
    hecto_offset=np.mean(alpha_result['samples'].T[3])-np.mean(alpha_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    hecto_y0=x0-hecto_offset
    ax34.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax34.set_xlim([-1,1])
    ax34.set_ylim([-1,1])
    ax34.set_yticks([-0.5,0,0.5])
    ax34.set_yticklabels([-0.5,0,0.5],fontsize=6)
    ax34.set_xticklabels([])
    ax34.set_ylabel(r'[Mg/Fe]$_{\rm Apo}$',fontsize=6)
    
    m2fs_keep=np.where((m2fs_h3_compare_logg.T[1]<0.5)&(m2fs_h3_compare_logg.T[3]<0.5)&(m2fs_h3_compare_z.T[1]<0.5)&(m2fs_h3_compare_z.T[3]<0.5)&(m2fs_h3_compare_alpha.T[1]<0.5)&(m2fs_h3_compare_alpha.T[3]<0.5))[0]
    hecto_keep=np.where((hecto_h3_compare_logg.T[1]<0.5)&(hecto_h3_compare_logg.T[3]<0.5)&(hecto_h3_compare_z.T[1]<0.5)&(hecto_h3_compare_z.T[3]<0.5)&(hecto_h3_compare_alpha.T[1]<0.5)&(hecto_h3_compare_alpha.T[3]<0.5))[0]
    ax44.errorbar(m2fs_h3_compare_alpha.T[0][m2fs_keep],m2fs_h3_compare_alpha.T[2][m2fs_keep],xerr=m2fs_h3_compare_alpha.T[1][m2fs_keep],yerr=m2fs_h3_compare_alpha.T[3][m2fs_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='r',label='M2FS',ms=1,rasterized=True)
    ax44.errorbar(hecto_h3_compare_alpha.T[0][hecto_keep],hecto_h3_compare_alpha.T[2][hecto_keep],xerr=hecto_h3_compare_alpha.T[1][hecto_keep],yerr=hecto_h3_compare_alpha.T[3][hecto_keep],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',label='Hecto',ms=1,rasterized=True)
    m2fs_offset=np.mean(alpha_result['samples'].T[3])-np.mean(alpha_result['samples'].T[0])
    hecto_offset=np.mean(alpha_result['samples'].T[3])-np.mean(alpha_result['samples'].T[1])
    x0=np.linspace(-500,10000,100)
    m2fs_y0=x0-m2fs_offset
    hecto_y0=x0-hecto_offset
    ax44.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax44.set_xlim([-1,1])
    ax44.set_ylim([-1,1])
    ax44.set_xticks([-1,-0.5,0,0.5,1])
    ax44.set_yticks([-1,-0.5,0,0.5])
    ax44.set_yticklabels([-1,-0.5,0,0.5],fontsize=6)
    ax44.set_xticklabels([-1,-0.5,0,0.5,1],fontsize=6)
    ax44.set_ylabel(r'[Mg/Fe]$_{\rm H3}$',fontsize=6)
    ax44.set_xlabel(r'[Mg/Fe]',fontsize=6)
    
    plt.savefig('atm_offset.pdf',dpi=270)
    plt.show()
    plt.close()    

if apply_zeropoint:

    m2fs_catalog0=fits.open(m2fs_fits_table_filename)[1].data
    hecto_catalog0=fits.open(hecto_fits_table_filename)[1].data
    vlos_result,vlos_bestfit=pickle.load(open('vlos_zeropointshift.pkl','rb'))
    teff_result,teff_bestfit=pickle.load(open('teff_zeropointshift.pkl','rb'))
    logg_result,logg_bestfit=pickle.load(open('logg_zeropointshift.pkl','rb'))
    z_result,z_bestfit=pickle.load(open('z_zeropointshift.pkl','rb'))
    alpha_result,alpha_bestfit=pickle.load(open('alpha_zeropointshift.pkl','rb'))

    col1=fits.Column(name='vlos',format='D',array=m2fs_catalog0['vlos_raw']-np.mean(vlos_result['samples'].T[0]))
#    col2=fits.Column(name='vlos_error',format='D',array=m2fs_catalog0['vlos_raw_error_rescaled'])
    col2=fits.Column(name='vlos_mean',format='D',array=m2fs_catalog0['vlos_raw_mean']-np.mean(vlos_result['samples'].T[0]))
#    col4=fits.Column(name='vlos_mean_error',format='D',array=m2fs_catalog0['vlos_raw_mean_error'])
#    col5=fits.Column(name='vlos_mean_scatter',format='D',array=m2fs_catalog0['vlos_raw_mean_scatter'])
    col3=fits.Column(name='teff',format='D',array=m2fs_catalog0['teff_raw']-np.mean(teff_result['samples'].T[0]))
#    col7=fits.Column(name='teff_error',format='D',array=m2fs_catalog0['teff_raw_error_rescaled'])
    col4=fits.Column(name='teff_mean',format='D',array=m2fs_catalog0['teff_raw_mean']-np.mean(teff_result['samples'].T[0]))
#    col9=fits.Column(name='teff_mean_error',format='D',array=m2fs_catalog0['teff_raw_mean_error'])
#    col10=fits.Column(name='teff_mean_scatter',format='D',array=m2fs_catalog0['teff_raw_mean_scatter'])
    col5=fits.Column(name='logg',format='D',array=m2fs_catalog0['logg_raw']-np.mean(logg_result['samples'].T[0]))
#    col12=fits.Column(name='logg_error',format='D',array=m2fs_catalog0['logg_raw_error_rescaled'])
    col6=fits.Column(name='logg_mean',format='D',array=m2fs_catalog0['logg_raw_mean']-np.mean(logg_result['samples'].T[0]))
#    col14=fits.Column(name='logg_mean_error',format='D',array=m2fs_catalog0['logg_raw_mean_error'])
#    col15=fits.Column(name='logg_mean_scatter',format='D',array=m2fs_catalog0['logg_raw_mean_scatter'])
    col7=fits.Column(name='feh',format='D',array=m2fs_catalog0['feh_raw']-np.mean(z_result['samples'].T[0]))
#    col17=fits.Column(name='feh_error',format='D',array=m2fs_catalog0['feh_raw_error_rescaled'])
    col8=fits.Column(name='feh_mean',format='D',array=m2fs_catalog0['feh_raw_mean']-np.mean(z_result['samples'].T[0]))
#    col19=fits.Column(name='feh_mean_error',format='D',array=m2fs_catalog0['feh_raw_mean_error'])
#    col20=fits.Column(name='feh_mean_scatter',format='D',array=m2fs_catalog0['feh_raw_mean_scatter'])
    col9=fits.Column(name='mgfe',format='D',array=m2fs_catalog0['mgfe_raw']-np.mean(alpha_result['samples'].T[0]))
#    col22=fits.Column(name='mgfe_error',format='D',array=m2fs_catalog0['mgfe_raw_error_rescaled'])
    col10=fits.Column(name='mgfe_mean',format='D',array=m2fs_catalog0['mgfe_raw_mean']-np.mean(alpha_result['samples'].T[0]))
#    col24=fits.Column(name='mgfe_mean_error',format='D',array=m2fs_catalog0['mgfe_raw_mean_error'])
#    col25=fits.Column(name='mgfe_mean_scatter',format='D',array=m2fs_catalog0['mgfe_raw_mean_scatter'])

    raw_cols=m2fs_catalog0.columns
    new_cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10])
    cols=raw_cols+new_cols
    table_hdu=fits.BinTableHDU.from_columns(cols)
    table_hdu.writeto(m2fs_fits_calibrated_filename,overwrite=True)

    col1=fits.Column(name='vlos',format='D',array=hecto_catalog0['vlos_raw']-np.mean(vlos_result['samples'].T[1]))
#    col2=fits.Column(name='vlos_error',format='D',array=hecto_catalog0['vlos_raw_error_rescaled'])
    col2=fits.Column(name='vlos_mean',format='D',array=hecto_catalog0['vlos_raw_mean']-np.mean(vlos_result['samples'].T[1]))
#    col4=fits.Column(name='vlos_mean_error',format='D',array=hecto_catalog0['vlos_raw_mean_error'])
#    col5=fits.Column(name='vlos_mean_scatter',format='D',array=hecto_catalog0['vlos_raw_mean_scatter'])
    col3=fits.Column(name='teff',format='D',array=hecto_catalog0['teff_raw']-np.mean(teff_result['samples'].T[1]))
#    col7=fits.Column(name='teff_error',format='D',array=hecto_catalog0['teff_raw_error_rescaled'])
    col4=fits.Column(name='teff_mean',format='D',array=hecto_catalog0['teff_raw_mean']-np.mean(teff_result['samples'].T[1]))
#    col9=fits.Column(name='teff_mean_error',format='D',array=hecto_catalog0['teff_raw_mean_error'])
#    col10=fits.Column(name='teff_mean_scatter',format='D',array=hecto_catalog0['teff_raw_mean_scatter'])
    col5=fits.Column(name='logg',format='D',array=hecto_catalog0['logg_raw']-np.mean(logg_result['samples'].T[1]))
#    col12=fits.Column(name='logg_error',format='D',array=hecto_catalog0['logg_raw_error_rescaled'])
    col6=fits.Column(name='logg_mean',format='D',array=hecto_catalog0['logg_raw_mean']-np.mean(logg_result['samples'].T[1]))
#    col14=fits.Column(name='logg_mean_error',format='D',array=hecto_catalog0['logg_raw_mean_error'])
#    col15=fits.Column(name='logg_mean_scatter',format='D',array=hecto_catalog0['logg_raw_mean_scatter'])
    col7=fits.Column(name='feh',format='D',array=hecto_catalog0['feh_raw']-np.mean(z_result['samples'].T[1]))
#    col17=fits.Column(name='feh_error',format='D',array=hecto_catalog0['feh_raw_error_rescaled'])
    col8=fits.Column(name='feh_mean',format='D',array=hecto_catalog0['feh_raw_mean']-np.mean(z_result['samples'].T[1]))
#    col19=fits.Column(name='feh_mean_error',format='D',array=hecto_catalog0['feh_raw_mean_error'])
#    col20=fits.Column(name='feh_mean_scatter',format='D',array=hecto_catalog0['feh_raw_mean_scatter'])
    col9=fits.Column(name='mgfe',format='D',array=hecto_catalog0['mgfe_raw']-np.mean(alpha_result['samples'].T[1]))
#    col22=fits.Column(name='mgfe_error',format='D',array=hecto_catalog0['mgfe_raw_error_rescaled'])
    col10=fits.Column(name='mgfe_mean',format='D',array=hecto_catalog0['mgfe_raw_mean']-np.mean(alpha_result['samples'].T[1]))
#    col24=fits.Column(name='mgfe_mean_error',format='D',array=hecto_catalog0['mgfe_raw_mean_error'])
#    col25=fits.Column(name='mgfe_mean_scatter',format='D',array=hecto_catalog0['mgfe_raw_mean_scatter'])

    raw_cols=hecto_catalog0.columns
    new_cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10])
    cols=raw_cols+new_cols
    table_hdu=fits.BinTableHDU.from_columns(cols)
    table_hdu.writeto(hecto_fits_calibrated_filename,overwrite=True)

if compare_sspp:

    m2fs_catalog=fits.open(m2fs_fits_calibrated_filename)[1].data
    hecto_catalog=fits.open(hecto_fits_calibrated_filename)[1].data
    sspp=fits.open('sspp_published.fits')[1].data

    m2fs_sspp_compare_v=[]
    m2fs_sspp_compare_teff=[]
    m2fs_sspp_compare_logg=[]
    m2fs_sspp_compare_z=[]
    hecto_sspp_compare_v=[]
    hecto_sspp_compare_teff=[]
    hecto_sspp_compare_logg=[]
    hecto_sspp_compare_z=[]
    for i in range(0,len(m2fs_catalog)):
        if ((m2fs_catalog['good_obs'][i]==1)):
            dist=np.sqrt((1./np.cos(m2fs_catalog['dec_deg'][i]*np.pi/180.)*(m2fs_catalog['ra_deg'][i]-sspp['ra']))**2+(m2fs_catalog['dec_deg'][i]-sspp['dec'])**2)*3600.
            this=np.where(dist<1.)[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    m2fs_sspp_compare_v.append((m2fs_catalog['vlos_mean'][i],m2fs_catalog['vlos_mean_error'][i],sspp['vlos'][this[j]],sspp['vlos_err'][this[j]]))
                    m2fs_sspp_compare_teff.append((m2fs_catalog['teff_mean'][i],m2fs_catalog['teff_mean_error'][i],sspp['teff'][this[j]],sspp['teff_err'][this[j]]))
                    m2fs_sspp_compare_logg.append((m2fs_catalog['logg_mean'][i],m2fs_catalog['logg_mean_error'][i],sspp['logg'][this[j]],sspp['logg_err'][this[j]]))
                    m2fs_sspp_compare_z.append((m2fs_catalog['feh_mean'][i],m2fs_catalog['feh_mean_error'][i],sspp['feh'][this[j]],sspp['feh_err'][this[j]]))
    for i in range(0,len(hecto_catalog)):
        if ((hecto_catalog['good_obs'][i]==1)):
            dist=np.sqrt((1./np.cos(hecto_catalog['dec_deg'][i]*np.pi/180.)*(hecto_catalog['ra_deg'][i]-sspp['ra']))**2+(hecto_catalog['dec_deg'][i]-sspp['dec'])**2)*3600.
            this=np.where(dist<1.)[0]
            if len(this)>0:
                for j in range(0,len(this)):
                    hecto_sspp_compare_v.append((hecto_catalog['vlos_mean'][i],hecto_catalog['vlos_mean_error'][i],sspp['vlos'][this[j]],sspp['vlos_err'][this[j]]))
                    hecto_sspp_compare_teff.append((hecto_catalog['teff_mean'][i],hecto_catalog['teff_mean_error'][i],sspp['teff'][this[j]],sspp['teff_err'][this[j]]))
                    hecto_sspp_compare_logg.append((hecto_catalog['logg_mean'][i],hecto_catalog['logg_mean_error'][i],sspp['logg'][this[j]],sspp['logg_err'][this[j]]))
                    hecto_sspp_compare_z.append((hecto_catalog['feh_mean'][i],hecto_catalog['feh_mean_error'][i],sspp['feh'][this[j]],sspp['feh_err'][this[j]]))
    m2fs_sspp_compare_v=np.array(m2fs_sspp_compare_v)
    m2fs_sspp_compare_teff=np.array(m2fs_sspp_compare_teff)
    m2fs_sspp_compare_logg=np.array(m2fs_sspp_compare_logg)
    m2fs_sspp_compare_z=np.array(m2fs_sspp_compare_z)
    hecto_sspp_compare_v=np.array(hecto_sspp_compare_v)
    hecto_sspp_compare_teff=np.array(hecto_sspp_compare_teff)
    hecto_sspp_compare_logg=np.array(hecto_sspp_compare_logg)
    hecto_sspp_compare_z=np.array(hecto_sspp_compare_z)

    g1=open('sspp_offsets.tex','w')
    string='\\newcommand{\mtwofsmedresssppvn}{$'+str(len(m2fs_sspp_compare_v.T[0]))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectossppvn}{$'+str(len(hecto_sspp_compare_v.T[0]))+'$} \n'
    g1.write(string)
    
    offset=offset_v
    dmax=dmax_v
    m2fs_sspp_vlos_result,m2fs_sspp_vlos_bestfit=m2fs.get_ssppoffset(m2fs_sspp_compare_v.T[0],m2fs_sspp_compare_v.T[1],m2fs_sspp_compare_v.T[2],m2fs_sspp_compare_v.T[3],np.full(len(m2fs_sspp_compare_v.T),0,dtype='int'),np.full(len(m2fs_sspp_compare_v.T),6,dtype='int'),offset,dmax)
    hecto_sspp_vlos_result,hecto_sspp_vlos_bestfit=m2fs.get_ssppoffset(hecto_sspp_compare_v.T[0],hecto_sspp_compare_v.T[1],hecto_sspp_compare_v.T[2],hecto_sspp_compare_v.T[3],np.full(len(hecto_sspp_compare_v.T),1,dtype='int'),np.full(len(hecto_sspp_compare_v.T),6,dtype='int'),offset,dmax)
    string='\\newcommand{\mtwofsmedresssppvoffset}{$'+str('{0:.2f}'.format(np.mean(m2fs_sspp_vlos_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(m2fs_sspp_vlos_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectossppvoffset}{$'+str('{0:.2f}'.format(np.mean(hecto_sspp_vlos_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(hecto_sspp_vlos_result['samples'].T[0])))+'$} \n'
    g1.write(string)

    offset=offset_teff
    dmax=dmax_teff
    m2fs_sspp_teff_result,m2fs_sspp_teff_bestfit=m2fs.get_ssppoffset(m2fs_sspp_compare_teff.T[0],m2fs_sspp_compare_teff.T[1],m2fs_sspp_compare_teff.T[2],m2fs_sspp_compare_teff.T[3],np.full(len(m2fs_sspp_compare_teff.T),0,dtype='int'),np.full(len(m2fs_sspp_compare_teff.T),6,dtype='int'),offset,dmax)
    hecto_sspp_teff_result,hecto_sspp_teff_bestfit=m2fs.get_ssppoffset(hecto_sspp_compare_teff.T[0],hecto_sspp_compare_teff.T[1],hecto_sspp_compare_teff.T[2],hecto_sspp_compare_teff.T[3],np.full(len(hecto_sspp_compare_teff.T),1,dtype='int'),np.full(len(hecto_sspp_compare_teff.T),6,dtype='int'),offset,dmax)
    string='\\newcommand{\mtwofsmedresssppteffoffset}{$'+str(int(np.mean(m2fs_sspp_teff_result['samples'].T[0])))+'\pm '+str(int(np.std(m2fs_sspp_teff_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectossppteffoffset}{$'+str(int(np.mean(hecto_sspp_teff_result['samples'].T[0])))+'\pm '+str(int(np.std(hecto_sspp_teff_result['samples'].T[0])))+'$} \n'
    g1.write(string)

    offset=offset_logg
    dmax=dmax_logg
    m2fs_sspp_logg_result,m2fs_sspp_logg_bestfit=m2fs.get_ssppoffset(m2fs_sspp_compare_logg.T[0],m2fs_sspp_compare_logg.T[1],m2fs_sspp_compare_logg.T[2],m2fs_sspp_compare_logg.T[3],np.full(len(m2fs_sspp_compare_logg.T),0,dtype='int'),np.full(len(m2fs_sspp_compare_logg.T),6,dtype='int'),offset,dmax)
    hecto_sspp_logg_result,hecto_sspp_logg_bestfit=m2fs.get_ssppoffset(hecto_sspp_compare_logg.T[0],hecto_sspp_compare_logg.T[1],hecto_sspp_compare_logg.T[2],hecto_sspp_compare_logg.T[3],np.full(len(hecto_sspp_compare_logg.T),1,dtype='int'),np.full(len(hecto_sspp_compare_logg.T),6,dtype='int'),offset,dmax)
    string='\\newcommand{\mtwofsmedressspploggoffset}{$'+str('{0:.2f}'.format(np.mean(m2fs_sspp_logg_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(m2fs_sspp_logg_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectosspploggoffset}{$'+str('{0:.2f}'.format(np.mean(hecto_sspp_logg_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(hecto_sspp_logg_result['samples'].T[0])))+'$} \n'
    g1.write(string)

    offset=offset_z
    dmax=dmax_z
    m2fs_sspp_feh_result,m2fs_sspp_feh_bestfit=m2fs.get_ssppoffset(m2fs_sspp_compare_z.T[0],m2fs_sspp_compare_z.T[1],m2fs_sspp_compare_z.T[2],m2fs_sspp_compare_z.T[3],np.full(len(m2fs_sspp_compare_z.T),0,dtype='int'),np.full(len(m2fs_sspp_compare_z.T),6,dtype='int'),offset,dmax)
    hecto_sspp_feh_result,hecto_sspp_feh_bestfit=m2fs.get_ssppoffset(hecto_sspp_compare_z.T[0],hecto_sspp_compare_z.T[1],hecto_sspp_compare_z.T[2],hecto_sspp_compare_z.T[3],np.full(len(hecto_sspp_compare_z.T),1,dtype='int'),np.full(len(hecto_sspp_compare_z.T),6,dtype='int'),offset,dmax)
    string='\\newcommand{\mtwofsmedresssppfehoffset}{$'+str('{0:.2f}'.format(np.mean(m2fs_sspp_feh_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(m2fs_sspp_feh_result['samples'].T[0])))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectossppfehoffset}{$'+str('{0:.2f}'.format(np.mean(hecto_sspp_feh_result['samples'].T[0])))+'\pm '+str('{0:.2f}'.format(np.std(hecto_sspp_feh_result['samples'].T[0])))+'$} \n'
    g1.write(string)

    g1.close()
    
    gs=plt.GridSpec(17,15)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax11=fig.add_subplot(gs[0:3,0:6])
    ax21=fig.add_subplot(gs[5:8,0:6])
    ax31=fig.add_subplot(gs[8:11,0:6])
    ax41=fig.add_subplot(gs[11:14,0:6])
    
    ax11.errorbar(m2fs_sspp_compare_v.T[0],m2fs_sspp_compare_v.T[0]-m2fs_sspp_compare_v.T[2],xerr=m2fs_sspp_compare_v.T[1],yerr=np.sqrt(m2fs_sspp_compare_v.T[1]**2+m2fs_sspp_compare_v.T[3]**2),alpha=0.3,fmt='.',elinewidth=0.5,color='r',ms=1,rasterized=True)
    ax11.errorbar(hecto_sspp_compare_v.T[0],hecto_sspp_compare_v.T[0]-hecto_sspp_compare_v.T[2],xerr=hecto_sspp_compare_v.T[1],yerr=np.sqrt(hecto_sspp_compare_v.T[1]**2+hecto_sspp_compare_v.T[3]**2),alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    ax11.plot([-10000,10000],[0,0],lw=0.5,linestyle='--',color='k')
    ax11.set_xlim([-400,400])
    ax11.set_ylim([-9.9,10])
    ax11.set_yticks([-5,0,5,10])
    ax11.set_yticklabels([-5,0,5,10],fontsize=7)
    ax11.set_xlabel(r'$V_{\rm LOS,M2FS}$ [km/s]',fontsize=7)
    ax11.set_ylabel(r'$V_{\rm LOS,M2FS}-V_{\rm LOS,Hecto}$ [km/s]',fontsize=7)

    ax21.errorbar(m2fs_sspp_compare_teff.T[0]/1000,m2fs_sspp_compare_teff.T[2]/1000,xerr=m2fs_sspp_compare_teff.T[1]/1000,yerr=m2fs_sspp_compare_teff.T[3]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='r',ms=1,rasterized=True)
    ax21.errorbar(hecto_sspp_compare_teff.T[0]/1000,hecto_sspp_compare_teff.T[2]/1000,xerr=hecto_sspp_compare_teff.T[1]/1000,yerr=hecto_sspp_compare_teff.T[3]/1000,alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    ax21.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax21.set_xlim([3.5,8])
    ax21.set_ylim([3.5,7.99])
    ax21.set_yticks([4,5,6,7])
    ax21.set_yticklabels([4,5,6,7],fontsize=6)
    ax21.set_xticklabels([])
    ax21.set_ylabel(r'$T_{\rm eff,K10}$ [K]',fontsize=6)
    ax21.legend(loc=2,fontsize=5,borderaxespad=0)    

    ax31.errorbar(m2fs_sspp_compare_logg.T[0],m2fs_sspp_compare_logg.T[2],xerr=m2fs_sspp_compare_logg.T[1],yerr=m2fs_sspp_compare_logg.T[3],alpha=0.3,fmt='.',elinewidth=0.5,color='r',ms=1,rasterized=True)
    ax31.errorbar(hecto_sspp_compare_logg.T[0],hecto_sspp_compare_logg.T[2],xerr=hecto_sspp_compare_logg.T[1],yerr=hecto_sspp_compare_logg.T[3],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    ax31.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax31.set_xlim([0,5])
    ax31.set_ylim([0,5])
    ax31.set_yticks([0,1,2,3,4,5])
    ax31.set_yticklabels([0,1,2,3,4,5],fontsize=6)
    ax31.set_xticklabels([])
    ax31.set_ylabel(r'logg',fontsize=6)
    ax31.legend(loc=2,fontsize=5,borderaxespad=0)    
    
    ax41.errorbar(m2fs_sspp_compare_z.T[0],m2fs_sspp_compare_z.T[2],xerr=m2fs_sspp_compare_z.T[1],yerr=m2fs_sspp_compare_z.T[3],alpha=0.3,fmt='.',elinewidth=0.5,color='r',ms=1,rasterized=True)
    ax41.errorbar(hecto_sspp_compare_z.T[0],hecto_sspp_compare_z.T[2],xerr=hecto_sspp_compare_z.T[1],yerr=hecto_sspp_compare_z.T[3],alpha=0.3,fmt='.',elinewidth=0.5,color='navy',ms=1,rasterized=True)
    ax41.plot([-10000,10000],[-10000,10000],lw=0.5,linestyle='--',color='k')
    ax41.set_xlim([-5,1])
    ax41.set_ylim([-5,1])
    ax41.set_yticks([-5,-4,-3,-2,-1,0])
    ax41.set_yticklabels([-5,-4,-3,-2,-1,0],fontsize=6)
    ax41.set_xticklabels([])
    ax41.set_ylabel(r'[Fe/H]',fontsize=6)
    ax41.legend(loc=2,fontsize=5,borderaxespad=0)    
    
    plt.show()
    plt.close()

    
if compare_lowmetallicity:
    
    m2fs=fits.open(m2fs_fits_calibrated_filename)[1].data
    hecto=fits.open(hecto_fits_calibrated_filename)[1].data
    with open('/hildafs/projects/phy200028p/mgwalker/m2fs/lowmetallicity.dat') as f:
        data=f.readlines()
    lowz_id=[]
    lowz_rah=[]
    lowz_ram=[]
    lowz_ras=[]
    lowz_decd=[]
    lowz_decm=[]
    lowz_decs=[]
    lowz_feh=[]
    lowz_sigfeh=[]
    lowz_mgfe=[]
    lowz_sigmgfe=[]
    lowz_author=[]

    for line in data:
        p=line.split()
        lowz_id.append(p[0])
        lowz_rah.append(float(p[1]))
        lowz_ram.append(float(p[2]))
        lowz_ras.append(float(p[3]))
        lowz_decd.append(float(p[4]))
        lowz_decm.append(float(p[5]))
        lowz_decs.append(float(p[6]))
        lowz_feh.append(float(p[7]))
        lowz_sigfeh.append(float(p[8]))
        lowz_mgfe.append(float(p[9]))
        lowz_sigmgfe.append(float(p[10]))
        lowz_author.append(p[11])
    lowz_id=np.array(lowz_id)
    lowz_rah=np.array(lowz_rah)
    lowz_ram=np.array(lowz_ram)
    lowz_ras=np.array(lowz_ras)
    lowz_decd=np.array(lowz_decd)
    lowz_decm=np.array(lowz_decm)
    lowz_decs=np.array(lowz_decs)
    lowz_feh=np.array(lowz_feh)
    lowz_sigfeh=np.array(lowz_sigfeh)
    lowz_mgfe=np.array(lowz_mgfe)
    lowz_sigmgfe=np.array(lowz_sigmgfe)
    lowz_author=np.array(lowz_author)

    lowz_coords=SkyCoord((lowz_rah,lowz_ram,lowz_ras),(lowz_decd,lowz_decm,lowz_decs),unit=(u.hourangle,u.deg))
    simon15=fits.open('simon15.fits')[1].data

    lowz_id=np.concatenate((lowz_id,simon15['name']))
    lowz_ra=np.concatenate((lowz_coords.ra.deg,simon15['raj2000']))
    lowz_dec=np.concatenate((lowz_coords.dec.deg,simon15['dej2000']))
    lowz_feh=np.concatenate((lowz_feh,simon15['__Fe_H_']))
    lowz_sigfeh=np.concatenate((lowz_sigfeh,np.zeros(len(simon15))))
    lowz_mgfe=np.concatenate((lowz_mgfe,np.zeros(len(simon15))-999))
    lowz_sigmgfe=np.concatenate((lowz_sigmgfe,np.zeros(len(simon15))))
    lowz_author=np.concatenate((lowz_author,np.full(len(simon15),'simon15',dtype='str')))

    lowz_objid=[]
    lowz_x=[]
    lowz_y=[]
    lowz_sigx=[]
    lowz_sigy=[]
    lowz_x2=[]
    lowz_y2=[]
    lowz_sigx2=[]
    lowz_sigy2=[]
    lowz_name=[]
    lowz_filename=[]
    lowz_index=[]
    lowz_instr=[]

    for i in range(0,len(lowz_ra)):
        dist=np.sqrt((1./np.cos(lowz_dec[i]*np.pi/180.)*(lowz_ra[i]-m2fs['ra_deg']))**2+(lowz_dec[i]-m2fs['dec_deg'])**2)*3600.
        this=np.where((dist<1.)&(m2fs['good_obs']==1))[0]
        if len(this)>0:
            for j in range(0,len(this)):
                lowz_objid.append(lowz_id[i])
                lowz_x.append(m2fs['feh_mean'][this[j]])
                lowz_sigx.append(m2fs['feh_mean_error'][this[j]])
                lowz_y.append(lowz_feh[i])
                lowz_sigy.append(lowz_sigfeh[i])
                lowz_x2.append(m2fs['mgfe_mean'][this[j]])
                lowz_sigx2.append(m2fs['mgfe_mean_error'][this[j]])
                lowz_y2.append(lowz_mgfe[i])
                lowz_sigy2.append(lowz_sigmgfe[i])
                lowz_name.append(lowz_author[i])
                lowz_filename.append(m2fs['fits_filename'][this[j]])
                lowz_index.append(m2fs['fits_index'][this[j]])
                lowz_instr.append('m2fs')

        dist=np.sqrt((1./np.cos(lowz_dec[i]*np.pi/180.)*(lowz_ra[i]-hecto['ra_deg']))**2+(lowz_dec[i]-hecto['dec_deg'])**2)*3600.
        this=np.where((dist<1.)&(hecto['good_obs']==1))[0]
        if len(this)>0:
            for j in range(0,len(this)):
                lowz_objid.append(lowz_id[i])
                lowz_x.append(hecto['feh_mean'][this[j]])
                lowz_sigx.append(hecto['feh_mean_error'][this[j]])
                lowz_y.append(lowz_feh[i])
                lowz_sigy.append(lowz_sigfeh[i])
                lowz_x2.append(hecto['mgfe_mean'][this[j]])
                lowz_sigx2.append(hecto['mgfe_mean_error'][this[j]])
                lowz_y2.append(lowz_mgfe[i])
                lowz_sigy2.append(lowz_sigmgfe[i])
                lowz_name.append(lowz_author[i])
                lowz_filename.append(hecto['fits_filename'][this[j]])
                lowz_index.append(hecto['fits_index'][this[j]])
                lowz_instr.append('hecto')                    
    lowz_objid=np.array(lowz_objid)
    lowz_x=np.array(lowz_x)
    lowz_y=np.array(lowz_y)
    lowz_sigx=np.array(lowz_sigx)
    lowz_sigy=np.array(lowz_sigy)
    lowz_x2=np.array(lowz_x2)
    lowz_y2=np.array(lowz_y2)
    lowz_sigx2=np.array(lowz_sigx2)
    lowz_sigy2=np.array(lowz_sigy2)
    lowz_name=np.array(lowz_name)
    lowz_filename=np.array(lowz_filename)
    lowz_index=np.array(lowz_index)
    lowz_instr=np.array(lowz_instr)
    
    stark=np.where(lowz_name=='starkenburg13')[0]
    simon15=np.where(lowz_name=='s')[0]
    aoki=np.where(lowz_name=='aoki09')[0]
    norris=np.where(lowz_name=='norris10')[0]
    taf=np.where(lowz_name=='tafelmeyer10')[0]
    lucc=np.where(lowz_name=='lucchesi20')[0]
    geisler=np.where(lowz_name=='geisler05')[0]
    shetrone03=np.where(lowz_name=='shetrone03')[0]
    frebel10=np.where((lowz_name=='frebel10a')|(lowz_name=='frebel10b'))[0]
    letarte=np.where(lowz_name=='letarte10')[0]
    shetrone01=np.where(lowz_name=='shetrone01')[0]
    simon10=np.where(lowz_name=='simon10')[0]
    sadakane=np.where(lowz_name=='sadakane04')[0]
    cohen=np.where(lowz_name=='cohen10')[0]
    shetrone98=np.where(lowz_name=='shetrone98')[0]
    fulbright=np.where(lowz_name=='fulbright04')[0]
    koch=np.where(lowz_name=='koch08')[0]
    
    gs=plt.GridSpec(10,10)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:4,0:4])
    ax2=fig.add_subplot(gs[0:4,6:10])

    cols=[]
    for i in range(0,100):
        cols.append((np.random.random(),np.random.random(),np.random.random()))
    ax1.plot([-5,1],[-5,1],linestyle='--',color='k',lw=1)
    i=0
    ax1.errorbar(lowz_x[shetrone01],lowz_y[shetrone01],xerr=lowz_sigx[shetrone01],yerr=lowz_sigy[shetrone01],fmt='.',elinewidth=1,label='Shetrone+2001',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[shetrone03],lowz_y[shetrone03],xerr=lowz_sigx[shetrone03],yerr=lowz_sigy[shetrone03],fmt='.',elinewidth=1,label='Shetrone+2003',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[fulbright],lowz_y[fulbright],xerr=lowz_sigx[fulbright],yerr=lowz_sigy[fulbright],fmt='.',elinewidth=1,label='Fulbright+2004',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[sadakane],lowz_y[sadakane],xerr=lowz_sigx[sadakane],yerr=lowz_sigy[sadakane],fmt='.',elinewidth=1,label='Sadakane+2004',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[aoki],lowz_y[aoki],xerr=lowz_sigx[aoki],yerr=lowz_sigy[aoki],fmt='.',elinewidth=1,label='Aoki+2009',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[cohen],lowz_y[cohen],xerr=lowz_sigx[cohen],yerr=lowz_sigy[cohen],fmt='.',elinewidth=1,label='Cohen+2009',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[taf],lowz_y[taf],xerr=lowz_sigx[taf],yerr=lowz_sigy[taf],fmt='.',elinewidth=1,label='Tafelmeyer+2010',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[frebel10],lowz_y[frebel10],xerr=lowz_sigx[frebel10],yerr=lowz_sigy[frebel10],fmt='.',elinewidth=1,label='Frebel+2010',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[norris],lowz_y[norris],xerr=lowz_sigx[norris],yerr=lowz_sigy[norris],fmt='.',elinewidth=1,label='Norris+2010',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[stark],lowz_y[stark],xerr=lowz_sigx[stark],yerr=lowz_sigy[stark],fmt='.',elinewidth=1,label='Starkenburg+2013',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[simon15],lowz_y[simon15],xerr=lowz_sigx[simon15],yerr=lowz_sigy[simon15],fmt='.',elinewidth=1,label='Simon+2015',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax1.errorbar(lowz_x[lucc],lowz_y[lucc],xerr=lowz_sigx[lucc],yerr=lowz_sigy[lucc],fmt='.',elinewidth=1,label='Lucchesi+2020',rasterized=True,markersize=3,color=cols[i])
#    ax1.errorbar(lowz_x[geisler],lowz_y[geisler],xerr=lowz_sigx[geisler],yerr=lowz_sigy[geisler],fmt='.',elinewidth=1,label='Geislerhesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[shetrone03],lowz_y[shetrone03],xerr=lowz_sigx[shetrone03],yerr=lowz_sigy[shetrone03],fmt='.',elinewidth=1,label='Shetrone03hesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[letarte],lowz_y[letarte],xerr=lowz_sigx[letarte],yerr=lowz_sigy[letarte],fmt='.',elinewidth=1,label='Letartehesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[shetrone01],lowz_y[shetrone01],xerr=lowz_sigx[shetrone01],yerr=lowz_sigy[shetrone01],fmt='.',elinewidth=1,label='Shetrone01hesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[simon10],lowz_y[simon10],xerr=lowz_sigx[simon10],yerr=lowz_sigy[simon10],fmt='.',elinewidth=1,label='Simon10hesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[shetrone98],lowz_y[shetrone98],xerr=lowz_sigx[shetrone98],yerr=lowz_sigy[shetrone98],fmt='.',elinewidth=1,label='Shetrone98hesi+2020',rasterized=True)
#    ax1.errorbar(lowz_x[koch],lowz_y[koch],xerr=lowz_sigx[koch],yerr=lowz_sigy[koch],fmt='.',elinewidth=1,label='Kochhesi+2020',rasterized=True)

    ax1.set_xlim([-4.5,1])
    ax1.set_ylim([-4.5,1])
    ax1.set_xticks([-4,-3,-2,-1,0,1])
    ax1.set_yticks([-4,-3,-2,-1,0,1])
    ax1.set_xlabel('[Fe/H], this work')
    ax1.set_ylabel('[Fe/H], previous')
    ax1.legend(loc=2,fontsize=4)

    ax2.plot([-5,1],[-5,1],linestyle='--',color='k',lw=1)
    i=0
    ax2.errorbar(lowz_x2[shetrone01],lowz_y2[shetrone01],xerr=lowz_sigx2[shetrone01],yerr=lowz_sigy2[shetrone01],fmt='.',elinewidth=1,label='Shetrone+2001',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax2.errorbar(lowz_x2[shetrone03],lowz_y2[shetrone03],xerr=lowz_sigx2[shetrone03],yerr=lowz_sigy2[shetrone03],fmt='.',elinewidth=1,label='Shetrone+2003',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax2.errorbar(lowz_x2[fulbright],lowz_y2[fulbright],xerr=lowz_sigx2[fulbright],yerr=lowz_sigy2[fulbright],fmt='.',elinewidth=1,label='Fulbright+2004',rasterized=True,markersize=3,color=cols[i])
    i+=1
#    ax2.errorbar(lowz_x2[sadakane],lowz_y2[sadakane],xerr=lowz_sigx2[sadakane],yerr=lowz_sigy2[sadakane],fmt='.',elinewidth=1,label='Sadakane+2004',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax2.errorbar(lowz_x2[aoki],lowz_y2[aoki],xerr=lowz_sigx2[aoki],yerr=lowz_sigy2[aoki],fmt='.',elinewidth=1,label='Aoki+2009',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax2.errorbar(lowz_x2[cohen],lowz_y2[cohen],xerr=lowz_sigx2[cohen],yerr=lowz_sigy2[cohen],fmt='.',elinewidth=1,label='Cohen+2009',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax2.errorbar(lowz_x2[taf],lowz_y2[taf],xerr=lowz_sigx2[taf],yerr=lowz_sigy2[taf],fmt='.',elinewidth=1,label='Tafelmeyer+2010',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax2.errorbar(lowz_x2[frebel10],lowz_y2[frebel10],xerr=lowz_sigx2[frebel10],yerr=lowz_sigy2[frebel10],fmt='.',elinewidth=1,label='Frebel+2010',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax2.errorbar(lowz_x2[norris],lowz_y2[norris],xerr=lowz_sigx2[norris],yerr=lowz_sigy2[norris],fmt='.',elinewidth=1,label='Norris+2010',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax2.errorbar(lowz_x2[stark],lowz_y2[stark],xerr=lowz_sigx2[stark],yerr=lowz_sigy2[stark],fmt='.',elinewidth=1,label='Starkenburg+2013',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax2.errorbar(lowz_x2[simon15],lowz_y2[simon15],xerr=lowz_sigx2[simon15],yerr=lowz_sigy2[simon15],fmt='.',elinewidth=1,label='Simon+2015',rasterized=True,markersize=3,color=cols[i])
    i+=1
    ax2.errorbar(lowz_x2[lucc],lowz_y2[lucc],xerr=lowz_sigx2[lucc],yerr=lowz_sigy2[lucc],fmt='.',elinewidth=1,label='Lucchesi+2020',rasterized=True,markersize=3,color=cols[i])

    ax2.set_xlim([-1,1])
    ax2.set_ylim([-1,1])
    ax2.set_xlabel('[Mg/Fe], this work')
    ax2.set_ylabel('[Mg/Fe], previous')
#    ax2.legend(loc=2,fontsize=5)

    plt.savefig('compare_lowmetallicity.pdf',dpi=300)
    plt.show()
    plt.close()

    for i in range(0,len(lowz_x)):
        shite=fits.open(lowz_filename[i])
        if lowz_instr[i]=='m2fs':
            wav=shite[1].data[lowz_index[i]]
            skysub=shite[2].data[lowz_index[i]]
            bestfit=shite[5].data[lowz_index[i]]
            mask=shite[4].data[lowz_index[i]]
            keep=np.where(mask==0)[0]
            print(i,lowz_x[i],lowz_y[i],lowz_name[i])
            plt.plot(wav[keep],skysub[keep],color='k')
            plt.plot(wav[keep],bestfit[keep],color='r')
            plt.xlim([5130,5190])
            plt.savefig('aoki_spec.pdf',dpi=300)
            plt.show()
        if lowz_instr[i]=='hecto':
            wav=shite[0].data[lowz_index[i]]
            skysub=shite[1].data[lowz_index[i]]
            bestfit=shite[4].data[lowz_index[i]]
            mask=shite[3].data[lowz_index[i]]
            keep=np.where(mask==0)[0]
            print(i,lowz_x[i],lowz_y[i],lowz_name[i])
            plt.plot(wav[keep],skysub[keep],color='k')
            plt.plot(wav[keep],bestfit[keep],color='r')
            plt.xlim([5150,5300])
            plt.show()
            
        plt.close()

        
if plot_field_of_halos:

    m2fs=fits.open(m2fs_fits_calibrated_filename)[1].data
    hecto=fits.open(hecto_fits_calibrated_filename)[1].data
    dsph=fits.open('dsph_parameters.fits')

    obj=np.concatenate([m2fs['target_system'],hecto['target_system']])
    ra=np.concatenate([m2fs['ra_deg'],hecto['ra_deg']])
    dec=np.concatenate([m2fs['dec_deg'],hecto['dec_deg']])
    v=np.concatenate([m2fs['vlos'],hecto['vlos']])
    sigv=np.concatenate([m2fs['vlos_error'],hecto['vlos_error']])
    teff=np.concatenate([m2fs['teff'],hecto['teff']])
    sigteff=np.concatenate([m2fs['teff_error'],hecto['teff_error']])
    logg=np.concatenate([m2fs['logg'],hecto['logg']])
    siglogg=np.concatenate([m2fs['logg_error'],hecto['logg_error']])
    z=np.concatenate([m2fs['feh'],hecto['feh']])
    sigz=np.concatenate([m2fs['feh_error'],hecto['feh_error']])
    alpha=np.concatenate([m2fs['mgfe'],hecto['mgfe']])
    sigalpha=np.concatenate([m2fs['mgfe_error'],hecto['mgfe_error']])
    vlos_mean=np.concatenate([m2fs['vlos_mean'],hecto['vlos_mean']])
    sigvlos_mean=np.concatenate([m2fs['vlos_mean_error'],hecto['vlos_mean_error']])
    teff_mean=np.concatenate([m2fs['teff_mean'],hecto['teff_mean']])
    sigteff_mean=np.concatenate([m2fs['teff_mean_error'],hecto['teff_mean_error']])
    logg_mean=np.concatenate([m2fs['logg_mean'],hecto['logg_mean']])
    siglogg_mean=np.concatenate([m2fs['logg_mean_error'],hecto['logg_mean_error']])
    z_mean=np.concatenate([m2fs['feh_mean'],hecto['feh_mean']])
    sigz_mean=np.concatenate([m2fs['feh_mean_error'],hecto['feh_mean_error']])
    alpha_mean=np.concatenate([m2fs['mgfe_mean'],hecto['mgfe_mean']])
    sigalpha_mean=np.concatenate([m2fs['mgfe_mean_error'],hecto['mgfe_mean_error']])
    pmra=np.concatenate([m2fs['gaia_pmra'],hecto['gaia_pmra']])
    pmdec=np.concatenate([m2fs['gaia_pmdec'],hecto['gaia_pmdec']])
    pmra_error=np.concatenate([m2fs['gaia_sigpmra'],hecto['gaia_sigpmra']])
    pmdec_error=np.concatenate([m2fs['gaia_sigpmdec'],hecto['gaia_sigpmdec']])
    obs=np.concatenate([m2fs['obs'],hecto['obs']])
    nobs=np.concatenate([m2fs['n_obs'],hecto['n_obs']])
    goodobs=np.concatenate([m2fs['good_obs'],hecto['good_obs']])
    goodnobs=np.concatenate([m2fs['good_n_obs'],hecto['good_n_obs']])
    carbon_flag=np.concatenate([m2fs['carbon_flag'],hecto['carbon_flag']])
    chi2_flag=np.concatenate([m2fs['chi2_flag'],hecto['chi2_flag']])
    agn_flag=np.concatenate([m2fs['gaia_agn'],hecto['gaia_agn']])

    g000=open('for_mario.dat','w')
    g000.write('# target # RA_deg # Dec_deg # Gaia G # pmRA # pmDec # Vlos # eVlos # Vlos_mean # eVlos_mean # Teff # eTeff # Teff_mean # eTeff_mean # logg # elogg # logg_mean # elogg_mean # Fe/H # eFe/H # Fe/H_mean # eFe/H_mean # Mg/Fe # eMg/Fe # Mg/Fe_mean # eMg/Fe_mean # obs # n_obs # good_obs # good_n_obs # \n')
    for i in range(0,len(m2fs)):
        if m2fs['vlos_mean'][i]==m2fs['vlos_mean'][i]:
            obj_string=m2fs['target_system'][i]+' '
            ra_string=str(round(m2fs['ra_deg'][i],7))+' '
            dec_string=str(round(m2fs['dec_deg'][i],7))+' '
            gaia_gmag_string=str(round(m2fs['gaia_gmag'][i],3))+' '
            gaia_pmra_string=str(round(m2fs['gaia_pmra'][i],3))+' '
            gaia_pmdec_string=str(round(m2fs['gaia_pmdec'][i],3))+' '
            vlos_string=str(round(m2fs['vlos'][i],2))+' '
            vlos_error_string=str(round(m2fs['vlos_error'][i],2))+' '
            vlos_mean_string=str(round(m2fs['vlos_mean'][i],2))+' '
            vlos_mean_error_string=str(round(m2fs['vlos_mean_error'][i],2))+' '
            teff_string=str(int(m2fs['teff'][i]))+' '
            teff_error_string=str(int(m2fs['teff_error'][i]))+' '
            teff_mean_string=str(int(m2fs['teff_mean'][i]))+' '
            teff_mean_error_string=str(int(m2fs['teff_mean_error'][i]))+' '
            logg_string=str(round(m2fs['logg'][i],2))+' '
            logg_error_string=str(round(m2fs['logg_error'][i],2))+' '
            logg_mean_string=str(round(m2fs['logg_mean'][i],2))+' '
            logg_mean_error_string=str(round(m2fs['logg_mean_error'][i],2))+' '
            feh_string=str(round(m2fs['feh'][i],2))+' '
            feh_error_string=str(round(m2fs['feh_error'][i],2))+' '
            feh_mean_string=str(round(m2fs['feh_mean'][i],2))+' '
            feh_mean_error_string=str(round(m2fs['feh_mean_error'][i],2))+' '
            mgfe_string=str(round(m2fs['mgfe'][i],2))+' '
            mgfe_error_string=str(round(m2fs['mgfe_error'][i],2))+' '
            mgfe_mean_string=str(round(m2fs['mgfe_mean'][i],2))+' '
            mgfe_mean_error_string=str(round(m2fs['mgfe_mean_error'][i],2))+' '
            obs_string=str(m2fs['obs'][i])+' '
            nobs_string=str(m2fs['n_obs'][i])+' '
            goodobs_string=str(m2fs['good_obs'][i])+' '
            goodnobs_string=str(m2fs['good_n_obs'][i])+' '

            g000.write(obj_string+ra_string+dec_string+gaia_gmag_string+gaia_pmra_string+gaia_pmdec_string+vlos_string+vlos_error_string+vlos_mean_string+vlos_mean_error_string+teff_string+teff_error_string+teff_mean_string+teff_mean_error_string+logg_string+logg_error_string+logg_mean_string+logg_mean_error_string+feh_string+feh_error_string+feh_mean_string+feh_mean_error_string+mgfe_string+mgfe_error_string+mgfe_mean_string+mgfe_mean_error_string+obs_string+nobs_string+goodobs_string+goodnobs_string+' \n')
    g000.close()
        
    field_object_andrew=np.empty(len(obj),dtype='object')
    field_object_formal=np.empty(len(obj),dtype='object')
    this=np.where(obj=='ant2')[0]
    field_object_andrew[this]='antlia_2'
    field_object_formal[this]='Antlia 2'
    this=np.where(obj=='boo1')[0]
    field_object_andrew[this]='bootes_1'
    field_object_formal[this]='Bootes 1'
    this=np.where(obj=='boo2')[0]
    field_object_andrew[this]='bootes_2'
    field_object_formal[this]='Bootes 2'
    this=np.where(obj=='boo3')[0]
    field_object_andrew[this]='bootes_3'
    field_object_formal[this]='Bootes 3'
    this=np.where(obj=='car')[0]
    field_object_andrew[this]='carina_1'
    field_object_formal[this]='Carina'
    this=np.where(obj=='cra')[0]
    field_object_andrew[this]='crater'
    field_object_formal[this]='Crater'
    this=np.where(obj=='cra2')[0]
    field_object_andrew[this]='crater_2'
    field_object_formal[this]='Crater 2'
    this=np.where(obj=='cvn1')[0]
    field_object_andrew[this]='canes_venatici_1'
    field_object_formal[this]='CVnI'
    this=np.where(obj=='dra')[0]
    field_object_andrew[this]='draco_1'
    field_object_formal[this]='Draco'
    this=np.where(obj=='for')[0]
    field_object_andrew[this]='fornax_1'
    field_object_formal[this]='Fornax'
    this=np.where(obj=='gru1')[0]
    field_object_andrew[this]='grus_1'
    field_object_formal[this]='Grus 1'
    this=np.where(obj=='gru2')[0]
    field_object_andrew[this]='grus_2'
    field_object_formal[this]='Grus 2'
    this=np.where(obj=='hor2')[0]
    field_object_andrew[this]='horologium_2'
    field_object_formal[this]='Horologium 2'
    this=np.where(obj=='hyd1')[0]
    field_object_andrew[this]='hydrus_1'
    field_object_formal[this]='Hydrus 1'
    this=np.where(obj=='ind1')[0]
    field_object_andrew[this]='indus_1'
    field_object_formal[this]='Indus 1'
    this=np.where(obj=='ind2')[0]
    field_object_andrew[this]='indus_2'
    field_object_formal[this]='Indus 2'
    this=np.where(obj=='kgo2')[0]
    field_object_andrew[this]='kgo2'
    field_object_formal[this]='KGO 2'
    this=np.where(obj=='kgo4')[0]
    field_object_andrew[this]='kgo4'
    field_object_formal[this]='KGO 4'
    this=np.where(obj=='kgo7')[0]
    field_object_andrew[this]='kgo7'
    field_object_formal[this]='KGO 7'
    this=np.where(obj=='kgo8')[0]
    field_object_andrew[this]='kgo8'
    field_object_formal[this]='KGO 8'
    this=np.where(obj=='kgo10')[0]
    field_object_andrew[this]='kgo10'
    field_object_formal[this]='KGO 10'
    this=np.where(obj=='kgo13')[0]
    field_object_andrew[this]='kgo13'
    field_object_formal[this]='KGO 13'
    this=np.where(obj=='kgo22')[0]
    field_object_andrew[this]='kgo22'
    field_object_formal[this]='KGO 22'
    this=np.where(obj=='kop2')[0]
    field_object_andrew[this]='koposov_2'
    field_object_formal[this]='Koposov 2'
    this=np.where(obj=='leo1')[0]
    field_object_andrew[this]='leo_1'
    field_object_formal[this]='Leo I'
    this=np.where(obj=='leo2')[0]
    field_object_andrew[this]='leo_2'
    field_object_formal[this]='Leo II'
    this=np.where(obj=='leo4')[0]
    field_object_andrew[this]='leo_4'
    field_object_formal[this]='Leo IV'
    this=np.where(obj=='leo5')[0]
    field_object_andrew[this]='leo_5'
    field_object_formal[this]='Leo V'
    this=np.where(obj=='pal1')[0]
    field_object_andrew[this]='palomar_1'
    field_object_formal[this]='Palomar 1'
    this=np.where(obj=='pal5')[0]
    field_object_andrew[this]='palomar_5'
    field_object_formal[this]='Palomar 5'
    this=np.where(obj=='pho2')[0]
    field_object_andrew[this]='phoenix_2'
    field_object_formal[this]='Phoenix 2'
    this=np.where(obj=='psc')[0]
    field_object_andrew[this]='pisces_1'
    field_object_formal[this]='Pisces 1'
    this=np.where(obj=='psc2')[0]
    field_object_andrew[this]='pisces_2'
    field_object_formal[this]='Pisces 2'
    this=np.where(obj=='ret2')[0]
    field_object_andrew[this]='reticulum_2'
    field_object_formal[this]='Reticulum 2'
    this=np.where(obj=='sgr2')[0]
    field_object_andrew[this]='sagittarius_2'
    field_object_formal[this]='Sagittarius II'
    this=np.where(obj=='scl')[0]
    field_object_andrew[this]='sculptor_1'
    field_object_formal[this]='Sculptor'
    this=np.where(obj=='seg1')[0]
    field_object_andrew[this]='segue_1'
    field_object_formal[this]='Segue 1'
    this=np.where(obj=='seg2')[0]
    field_object_andrew[this]='segue_2'
    field_object_formal[this]='Segue 2'
    this=np.where(obj=='seg3')[0]
    field_object_andrew[this]='segue_3'
    field_object_formal[this]='Segue 3'
    this=np.where(obj=='sex')[0]
    field_object_andrew[this]='sextans_1'
    field_object_formal[this]='Sextans'
    this=np.where(obj=='tri2')[0]
    field_object_andrew[this]='triangulum_2'
    field_object_formal[this]='Triangulum 2'
    this=np.where(obj=='tuc2')[0]
    field_object_andrew[this]='tucana_2'
    field_object_formal[this]='Tucana 2'
    this=np.where(obj=='tuc3')[0]
    field_object_andrew[this]='tucana_3'
    field_object_formal[this]='Tucana 3'
    this=np.where(obj=='tuc4')[0]
    field_object_andrew[this]='tucana_4'
    field_object_formal[this]='Tucana 4'
    this=np.where(obj=='tuc5')[0]
    field_object_andrew[this]='tucana_5'
    field_object_formal[this]='Tucana 5'
    this=np.where(obj=='uma1')[0]
    field_object_andrew[this]='ursa_major_1'
    field_object_formal[this]='UMa I'
    this=np.where(obj=='uma2')[0]
    field_object_andrew[this]='ursa_major_2'
    field_object_formal[this]='UMa II'
    this=np.where(obj=='umi')[0]
    field_object_andrew[this]='ursa_minor_1'
    field_object_formal[this]='Ursa Minor'

    hecto_keep=np.where((hecto['good_obs']==1)&(hecto['logg_mean_error']<0.5)&(hecto['mgfe_mean_error']<0.5)&(hecto['feh_mean_error']<0.5))[0]
    m2fs_keep=np.where((m2fs['good_obs']==1)&(m2fs['logg_mean_error']<0.5)&(m2fs['mgfe_mean_error']<0.5)&(m2fs['feh_mean_error']<0.5))[0]
    keep=np.where((goodobs==1)&(carbon_flag==0)&(chi2_flag==0)&(agn_flag==0))[0]
    keep2=np.where((goodobs==1)&(siglogg_mean<0.5)&(sigalpha_mean<0.5)&(sigz_mean<0.5)&(carbon_flag==0)&(chi2_flag==0)&(agn_flag==0))[0]
    keep3=np.where((goodobs==1)&(siglogg_mean<0.25)&(sigalpha_mean<0.25)&(sigz_mean<0.25))[0]
    order=np.argsort(logg_mean[keep])[::-1]
    order2=np.argsort(logg_mean[keep2])[::-1]
    order3=np.argsort(logg_mean[keep3])[::-1]

    nmem_table=open('m2fs_hecto_nmem_table.tex','w')        
    logg_member=3.#maximum logg to associate with members (as crude proxy for membership)
    ngiant_total=0
    nmem_total=0

    def getmem(this,formal_name):
        this0=np.where(dsph[1].data['name']==field_object_andrew[this[0]])[0]
        if len(this0)>0:
            vlos_system=dsph[1].data['vlos'][dsph[1].data['name']==field_object_andrew[this[0]]][0]
            sigvlos_system=dsph[1].data['sigvlos'][dsph[1].data['name']==field_object_andrew[this[0]]][0]
            vdisp_system=dsph[1].data['vdisp'][dsph[1].data['name']==field_object_andrew[this[0]]][0]
            pmra_system=dsph[1].data['pmra_dr3'][dsph[1].data['name']==field_object_andrew[this[0]]][0]
            pmdec_system=dsph[1].data['pmdec_dr3'][dsph[1].data['name']==field_object_andrew[this[0]]][0]
            sigpmra_system=dsph[1].data['sigpmra_dr3'][dsph[1].data['name']==field_object_andrew[this[0]]][0]
            sigpmdec_system=dsph[1].data['sigpmdec_dr3'][dsph[1].data['name']==field_object_andrew[this[0]]][0]
        else:
            vlos_system=np.nan
            sigvlos_system=np.nan
            vdisp_system=np.nan
            pmra_system=np.nan
            pmdec_system=np.nan
            sigpmra_system=np.nan
            sigpmdec_system=np.nan
        if vdisp_system!=vdisp_system:
            vdisp_system=0.

        dvlos=np.abs(vlos_mean-vlos_system)
        sigdvlos=np.sqrt(sigvlos_mean**2+vdisp_system**2+sigvlos_system**2)
        dpm=np.sqrt((pmra-pmra_system)**2+(pmdec-pmdec_system)**2)
        sigdpm=np.sqrt((pmra-pmra_system)**2/dpm*pmra_error**2+(pmdec-pmdec_system)**2/dpm*pmdec_error**2+(pmra-pmra_system)**2/dpm*sigpmra_system**2+(pmra-pmra_system)**2/dpm*sigpmdec_system**2)
        ngal=len(np.where((goodobs[this]==1))[0])
        ngiant=len(np.where((goodobs[this]==1)&(logg_mean[this]<logg_member))[0])
        nmem=len(np.where((goodobs[this]==1)&(logg_mean[this]<logg_member*100)&(dvlos[this]<3.*sigdvlos[this])&(dpm[this]<3.*sigdpm[this]))[0])
        nobsmem=len(np.where((goodobs[this]>0)&(logg_mean[this]<logg_member)&(dvlos[this]<3.*sigdvlos[this])&(dpm[this]<3.*sigdpm[this]))[0])
        nrepeatmem=len(np.where((goodobs[this]==1)&(goodnobs[this]>1)&(logg_mean[this]<logg_member)&(dvlos[this]<3.*sigdvlos[this])&(dpm[this]<3.*sigdpm[this]))[0])
        string=formal_name+' & '+str(ngal)+' & '+str(ngiant)+' & '+str(nmem)+' & '+'\\\ \n'
        nmem_table.write(string)
        print(string)
        return ngiant,nmem
        
    this=np.where(obj=='ant2')[0]
    formal_name='Antlia 2'
    getmem(this,formal_name)
    
    this=np.where(obj=='boo1')[0]
    formal_name='Bootes I'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='boo2')[0]
    formal_name='Bootes II'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='boo3')[0]
    formal_name='Bootes III'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='car')[0]
    formal_name='Carina'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='cra2')[0]
    formal_name='Crater II'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='cvn1')[0]
    formal_name='CVnI'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='dra')[0]
    formal_name='Draco'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='for')[0]
    formal_name='Fornax'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='gru1')[0]
    formal_name='Grus 1'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='gru2')[0]
    formal_name='Grus 2'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='hor2')[0]
    formal_name='Horologium 2'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='hyd1')[0]
    formal_name='Hydrus 1'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='leo1')[0]
    formal_name='Leo I'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='leo2')[0]
    formal_name='Leo II'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='leo4')[0]
    formal_name='Leo IV'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='leo5')[0]
    formal_name='Leo V'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='pho2')[0]
    formal_name='Phoenix 2'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='psc2')[0]
    formal_name='Pisces II'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='ret2')[0]
    formal_name='Reticulum 2'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='sgr2')[0]
    formal_name='Sagittarius 2'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='scl')[0]
    formal_name='Sculptor'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='seg1')[0]
    formal_name='Segue I'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='seg2')[0]
    formal_name='Segue II'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='sex')[0]
    formal_name='Sextans'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='tri2')[0]
    formal_name='Triangulum II'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='tuc2')[0]
    formal_name='Tucana 2'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='tuc3')[0]
    formal_name='Tucana 3'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='tuc4')[0]
    formal_name='Tucana 4'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='tuc5')[0]
    formal_name='Tucana 5'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='uma1')[0]
    formal_name='UMa I'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='uma2')[0]
    formal_name='UMa II'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='umi')[0]
    formal_name='Ursa Minor'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='cra')[0]
    formal_name='Crater'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='ind1')[0]
    formal_name='Indus 1'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='ind2')[0]
    formal_name='Indus 2'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='kgo2')[0]
    formal_name='KGO 2'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='kgo4')[0]
    formal_name='KGO 4'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='kgo7')[0]
    formal_name='KGO 7'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='kgo8')[0]
    formal_name='KGO 8'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='kgo10')[0]
    formal_name='KGO 10'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='kgo13')[0]
    formal_name='KGO 13'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='kgo22')[0]
    formal_name='KGO 22'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='kop2')[0]
    formal_name='Koposov 2'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='pal1')[0]
    formal_name='Palomar 1'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='psc')[0]
    formal_name='Pisces I'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember
    this=np.where(obj=='seg3')[0]
    formal_name='Segue 3'
    ngiant,nmember=getmem(this,formal_name)
    ngiant_total+=ngiant
    nmem_total+=nmember

    nmem_table.close()
    
    gs=plt.GridSpec(9,9)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:9,0:9])

    c1=ax1.scatter(vlos_mean[keep][order],z_mean[keep][order],c=logg_mean[keep][order],s=0.1,cmap='jet',rasterized=True)
#    ax1.scatter(hecto_vlos_mean[hecto_keep],hecto_z_mean[hecto_keep],c=hecto_logg_mean[hecto_keep],s=0.25,alpha=0.3,cmap='jet',rasterized=True)
#    ax1.scatter(m2fs_vlos_mean[m2fs_keep],m2fs_z_mean[m2fs_keep],c=m2fs_logg_mean[m2fs_keep],s=0.25,alpha=0.3,cmap='jet',rasterized=True)
    clb=plt.colorbar(c1,location='right',ax=ax1)#,label=r'$\log_{10}[g/(\mathrm{cm}\,\mathrm{s}^{-2})]$',ax=ax1)
    clb.ax.tick_params(labelsize=7)
    clb.ax.set_title(label=r'$\log g$',fontsize=7)

    ax1.set_xlabel(r'$V_{\rm LOS}$ [km/s]',fontsize=7)
    ax1.set_ylabel(r'[Fe/H]',fontsize=7)
    ax1.set_xlim([-450,450])
    ax1.set_ylim([-4,0.5])
    ax1.set_xticks([-400,-200,0,200,400])
    ax1.set_xticklabels(['-400','-200','0','200','400'],fontsize=7)
    ax1.set_yticks([-4,-3,-2,-1,0],fontsize=6)
    ax1.set_yticklabels(['-4','-3','-2','-1','0'],fontsize=7)

    plt.savefig('field_of_halos_nsf.pdf',dpi=300)
    plt.show()
    plt.close()
    
    gs=plt.GridSpec(9,9)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:3,0:5])
    ax2=fig.add_subplot(gs[6:9,0:5])
#    ax3=fig.add_subplot(gs[6:9,5:9])

    c1=ax1.scatter(vlos_mean[keep][order],z_mean[keep][order],c=logg_mean[keep][order],s=0.75,lw=0,cmap='jet',rasterized=True)
#    ax1.scatter(hecto_vlos_mean[hecto_keep],hecto_z_mean[hecto_keep],c=hecto_logg_mean[hecto_keep],s=0.25,alpha=0.3,cmap='jet',rasterized=True)
#    ax1.scatter(m2fs_vlos_mean[m2fs_keep],m2fs_z_mean[m2fs_keep],c=m2fs_logg_mean[m2fs_keep],s=0.25,alpha=0.3,cmap='jet',rasterized=True)
    clb=plt.colorbar(c1,location='right',ax=ax1)#,label=r'$\log_{10}[g/(\mathrm{cm}\,\mathrm{s}^{-2})]$',ax=ax1)
    clb.ax.tick_params(labelsize=7)
    clb.ax.set_title(label=r'$\log g$',fontsize=7)

    ax1.set_xlabel(r'$V_{\rm LOS}$ [km/s]',fontsize=7)
    ax1.set_ylabel(r'[Fe/H]',fontsize=7)
    ax1.set_xlim([-450,450])
    ax1.set_ylim([-4,0.5])
    ax1.set_xticks([-400,-200,0,200,400])
    ax1.set_xticklabels(['-400','-200','0','200','400'],fontsize=7)
    ax1.set_yticks([-4,-3,-2,-1,0],fontsize=6)
    ax1.set_yticklabels(['-4','-3','-2','-1','0'],fontsize=7)

    c2=ax2.scatter(z_mean[keep2][order2],alpha_mean[keep2][order2],c=logg_mean[keep2][order2],s=0.75,lw=0,cmap='jet',rasterized=True)
#    ax2.scatter(hecto_z_mean[hecto_keep],hecto_alpha_mean[hecto_keep],c=hecto_logg_mean[hecto_keep],s=1,alpha=0.3,cmap='jet',rasterized=True)
#    ax2.scatter(m2fs_z_mean[m2fs_keep],m2fs_alpha_mean[m2fs_keep],c=m2fs_logg_mean[m2fs_keep],s=1,alpha=0.3,cmap='jet',rasterized=True)

    clb=plt.colorbar(c2,location='right',ax=ax2)#label=r'$\log_{10}[g/(\mathrm{cm}\,\mathrm{s}^{-2})]$',ax=ax2)
    clb.ax.tick_params(labelsize=10)
    clb.ax.set_title(label=r'$\log g$',fontsize=10)
    ax2.set_xlabel(r'[Fe/H]',fontsize=10)
    ax2.set_ylabel(r'[Mg/Fe]',labelpad=-5,fontsize=10)
    ax2.set_xticks([-4,-3,-2,-1,0])
    ax2.set_xticklabels(['-4','-3','-2','-1','0'])
    ax2.set_xlim([-4,0.5])
    ax2.set_ylim([-0.8,1])
    
    plt.savefig('field_of_halos.pdf',dpi=500)
    plt.show()
    plt.close()

    gs=plt.GridSpec(9,9)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax3=fig.add_subplot(gs[0:4,0:4])
    
    c3=ax3.scatter(teff_mean[keep2][order2],logg_mean[keep2][order2],c=z_mean[keep2][order2],s=0.75,lw=0,cmap='jet',rasterized=True)
#    ax3.scatter(hecto_teff_mean[hecto_keep],hecto_logg_mean[hecto_keep],c=hecto_z_mean[hecto_keep],s=1,alpha=0.3,cmap='jet',rasterized=True)
#    ax3.scatter(m2fs_teff_mean[m2fs_keep],m2fs_logg_mean[m2fs_keep],c=m2fs_z_mean[m2fs_keep],s=1,alpha=0.3,cmap='jet',rasterized=True)
    clb=plt.colorbar(c3,location='right',ax=ax3)
    clb.ax.tick_params(labelsize=10)
    clb.ax.set_title(label='[Fe/H]',fontsize=10)
    ax3.set_xlabel(r'$T_{\rm eff}$ [K]',fontsize=10)
    ax3.set_ylabel(r'$\log_{10}[$g/(cm s$^{-1})$]',fontsize=10)
    ax3.set_xticks([7000,6000,5000,4000])
    ax3.set_xticklabels(['7000','6000','5000','4000'])
    ax3.set_xlim([7500,3900])
    ax3.set_ylim([5,0])

    mist=get_ichrone('mist')
    eep0=np.linspace(202,605,400)
    iso_rich=[]
    iso_poor=[]
    iso_med=[]
    iso_solar=[]
    feh_rich=-1.
    feh_poor=-3.
    feh_med=-2.
    feh_solar=0.
    for j in range(0,len(eep0)):
        iso_rich.append(mist.interp_value([eep0[j],np.log10(10.e+9),feh_rich],['Teff','logg','radius','density']))
        iso_poor.append(mist.interp_value([eep0[j],np.log10(10.e+9),feh_poor],['Teff','logg','radius','density']))
        iso_med.append(mist.interp_value([eep0[j],np.log10(10.e+9),feh_med],['Teff','logg','radius','density']))
        iso_solar.append(mist.interp_value([eep0[j],np.log10(10.e+9),feh_solar],['Teff','logg','radius','density']))
    iso_rich=np.array(iso_rich)
    iso_poor=np.array(iso_poor)
    iso_med=np.array(iso_med)
    iso_solar=np.array(iso_solar)
    c=matplotlib.cm.get_cmap('jet')
    c_rich=c(0.6)
    c_poor=c(0.4)
#    iso_young=np.array(iso_young)
    plt.plot(iso_solar.T[0],iso_solar.T[1],'0.5',linestyle='-',lw=1,label='[Fe/H]=0')
    plt.plot(iso_rich.T[0],iso_rich.T[1],'0.5',linestyle='--',lw=1,label='[Fe/H]=-1')
    plt.plot(iso_med.T[0],iso_med.T[1],color='0.5',linestyle='-.',lw=1,label='[Fe/H]=-2')
    plt.plot(iso_poor.T[0],iso_poor.T[1],color='0.5',linestyle=':',lw=1,label='[Fe/H]=-3')
    #    plt.plot(iso_young.T[0],iso_young.T[1],color='c',linestyle='-',lw=0.5,label='[Fe/H]=0')
    plt.legend(loc=2,fontsize=6,handlelength=4)
    plt.savefig('hr_diagram.pdf',dpi=300)
    plt.show()
    plt.close()

    #c4=ax4.scatter(hecto_ra[((hecto_sigv<5.)&(hecto_sigz<0.51)&(hecto_sigalpha<1.4))]*np.pi/180.,hecto_dec[((hecto_sigv<5.)&(hecto_sigz<0.51)&(hecto_sigalpha<1.4))]*np.pi/180.,c=hecto_z[((hecto_sigv<5.)&(hecto_sigz<0.51)&(hecto_sigalpha<1.4))],s=1,cmap='jet')
    #ax4.scatter(hecto_ra[((hecto_sigv<5.)&(hecto_sigz<0.51)&(hecto_sigalpha<1.4))]*np.pi/180.,hecto_dec[((hecto_sigv<5.)&(hecto_sigz<0.51)&(hecto_sigalpha<1.4))]*np.pi/180.,c=hecto_z[((hecto_sigv<5.)&(hecto_sigz<0.51)&(hecto_sigalpha<1.4))],s=1,alpeha=0.3,cmap='jet')
    #ax4.scatter(m2fs_ra[((m2fs_sigv<5.)&(m2fs_sigz<0.57)&(m2fs_sigalpha<0.38))]*np.pi/180.,m2fs_dec[((m2fs_sigv<5.)&(m2fs_sigz<0.57)&(m2fs_sigalpha<0.38))]*np.pi/180.,c=m2fs_z[((m2fs_sigv<5.)&(m2fs_sigz<0.57)&(m2fs_sigalpha<0.38))],s=1,alpha=0.3,cmap='jet')
    #plt.colorbar(c3,location='right',label=r'[Fe/H]',ax=ax4)
    #ax4.set_xlabel(r'$T_{\rm eff}$ [K]')
    #ax4.set_ylabel(r'$\log_{10}$g/(cm s$^{-1}$)')
    #ax4.set_xlim([7500,3900])
    #ax4.set_ylim([5,0])

    m2fs_systems=[]
    both_systems=[]
    for i in range(0,len(m2fs_catalog)):
        if not (m2fs_catalog['target_system'][i] in m2fs_systems):
            m2fs_systems.append(m2fs_catalog['target_system'][i])
        if not (m2fs_catalog['target_system'][i] in both_systems):
            both_systems.append(m2fs_catalog['target_system'][i])
    m2fs_obs=np.where(m2fs_catalog['obs']>0)[0]
    m2fs_goodobs=np.where(m2fs_catalog['good_obs']>0)[0]
    m2fs_goodstar=np.where(m2fs_catalog['good_obs']==1)[0]
    m2fs_goodzstar=np.where((m2fs_catalog['good_obs']==1)&(m2fs_catalog['feh_mean_error']<0.5)&(m2fs_catalog['logg_mean_error']<0.5)&(m2fs_catalog['mgfe_mean_error']<0.5))[0]
    m2fs_nobs=m2fs_catalog['good_n_obs'][m2fs_goodstar]
    m2fs_repeated=np.where((m2fs_catalog['good_n_obs']>1)&(m2fs_catalog['good_obs']==1))[0]
    m2fs_goodrrl=np.where((m2fs_catalog['good_obs']==1)&(m2fs_catalog['gaia_rrl']==1))[0]
    m2fs_maxgoodobs=np.max(m2fs_catalog['good_obs'])
    m2fs_wavflag=np.where(m2fs_catalog['wav_cal_flag']==True)[0]
    m2fs_wavflaggood=np.where((m2fs_catalog['wav_cal_flag']==True)&(m2fs_catalog['good_obs']>0))[0]

    hecto_systems=[]
    for i in range(0,len(hecto_catalog)):
        if not (hecto_catalog['target_system'][i] in hecto_systems):
            hecto_systems.append(hecto_catalog['target_system'][i])
        if not (hecto_catalog['target_system'][i] in both_systems):
            both_systems.append(hecto_catalog['target_system'][i])
    hecto_obs=np.where(hecto_catalog['obs']>0)[0]
    hecto_goodobs=np.where(hecto_catalog['good_obs']>0)[0]
    hecto_goodstar=np.where(hecto_catalog['good_obs']==1)[0]
    hecto_goodzstar=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['feh_mean_error']<0.5)&(hecto_catalog['logg_mean_error']<0.5)&(hecto_catalog['mgfe_mean_error']<0.5))[0]
    hecto_nobs=hecto_catalog['good_n_obs'][hecto_goodstar]
    hecto_repeated=np.where((hecto_catalog['good_n_obs']>1)&(hecto_catalog['good_obs']==1))[0]
    hecto_goodrrl=np.where((hecto_catalog['good_obs']==1)&(hecto_catalog['gaia_rrl']==1))[0]
    hecto_maxgoodobs=np.max(hecto_catalog['good_obs'])

    both_vlos_error=np.concatenate([m2fs_catalog['vlos_error'][m2fs_goodobs],hecto_catalog['vlos_error'][hecto_goodobs]])
    both_teff_error=np.concatenate([m2fs_catalog['teff_error'][m2fs_goodobs],hecto_catalog['teff_error'][hecto_goodobs]])
    both_logg_error=np.concatenate([m2fs_catalog['logg_error'][m2fs_goodobs],hecto_catalog['logg_error'][hecto_goodobs]])
    both_feh_error=np.concatenate([m2fs_catalog['feh_error'][m2fs_goodobs],hecto_catalog['feh_error'][hecto_goodobs]])
    both_mgfe_error=np.concatenate([m2fs_catalog['mgfe_error'][m2fs_goodobs],hecto_catalog['mgfe_error'][hecto_goodobs]])

    g1=open('summary_stats.tex','w')
    string='\\newcommand{\mtwofsmedressystems}{$'+str(len(m2fs_systems))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresobs}{$'+str(len(m2fs_obs))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresobsk}{$'+str(int(len(m2fs_obs)/1000.))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresgoodobs}{$'+str(len(m2fs_goodobs))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresgoodobsk}{$'+str(round(len(m2fs_goodobs)/1000.,1))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresgoodstar}{$'+str(len(m2fs_goodstar))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresgoodstark}{$'+str(round(len(m2fs_goodstar)/1000.,1))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresgoodzstar}{$'+str(len(m2fs_goodzstar))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresrepeated}{$'+str(len(m2fs_repeated))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresrepeatedk}{$'+str(round(len(m2fs_repeated)/1000.,1))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\bothrepeated}{$'+str(len(m2fs_repeated)+len(hecto_repeated))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\bothrepeatedk}{$'+str(round(len((m2fs_repeated)+len(hecto_repeated))/1000.,1))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresgoodrrl}{$'+str(len(m2fs_goodrrl))+'$} \n'
    g1.write(string)
    string='\\newcommand{\mtwofsmedresmaxgoodobs}{$'+str(m2fs_maxgoodobs)+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectosystems}{$'+str(len(hecto_systems))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectoobs}{$'+str(len(hecto_obs))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectoobsk}{$'+str(round(len(hecto_obs)/1000.,1))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectogoodobs}{$'+str(len(hecto_goodobs))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectogoodobsk}{$'+str(round(len(hecto_goodobs)/1000.,1))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectogoodstar}{$'+str(len(hecto_goodstar))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectogoodstark}{$'+str(round(len(hecto_goodstar)/1000.,1))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectogoodzstar}{$'+str(len(hecto_goodzstar))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectorepeated}{$'+str(len(hecto_repeated))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectorepeatedk}{$'+str(round(len(hecto_repeated)/1000.,1))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectogoodrrl}{$'+str(len(hecto_goodrrl))+'$} \n'
    g1.write(string)
    string='\\newcommand{\hectomaxgoodobs}{$'+str(hecto_maxgoodobs)+'$} \n'
    g1.write(string)
    string='\\newcommand{\\bothmaxgoodobs}{$'+str(np.max([m2fs_maxgoodobs,hecto_maxgoodobs]))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\bothsystems}{$'+str(len(both_systems))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\bothgoodstar}{$'+str(len(m2fs_goodstar)+len(hecto_goodstar))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\bothgoodstark}{$'+str(round((len(m2fs_goodstar)+len(hecto_goodstar))/1000.,1))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\bothgoodzstar}{$'+str(len(m2fs_goodzstar)+len(hecto_goodzstar))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\mtwofsmedreswavflagn}{$'+str(len(m2fs_wavflag))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\mtwofsmedreswavflaggoodn}{$'+str(len(m2fs_wavflaggood))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\vlosmedianerror}{$'+str(round(np.median(both_vlos_error),1))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\teffmedianerror}{$'+str(int(np.median(both_teff_error)))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\loggmedianerror}{$'+str(round(np.median(both_logg_error),2))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\fehmedianerror}{$'+str(round(np.median(both_feh_error),2))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\mgfemedianerror}{$'+str(round(np.median(both_mgfe_error),2))+'$} \n'
    g1.write(string)
    string='\\newcommand{\\ngianttotal}{$'+str(ngiant_total)+'$} \n'
    g1.write(string)
    string='\\newcommand{\\nmembertotal}{$'+str(nmem_total)+'$} \n'
    g1.write(string)
    g1.close()
    
if plot_spectra:

    m2fs=fits.open(m2fs_fits_calibrated_filename)[1].data
    hecto=fits.open(hecto_fits_calibrated_filename)[1].data

    gs=plt.GridSpec(15,15)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:3,0:7])
    ax2=fig.add_subplot(gs[3:6,0:7])
    ax3=fig.add_subplot(gs[6:9,0:7])
    ax4=fig.add_subplot(gs[9:12,0:7])
    ax5=fig.add_subplot(gs[12:15,0:7])
    ax11=fig.add_subplot(gs[0:3,8:15])
    ax12=fig.add_subplot(gs[3:6,8:15])
    ax13=fig.add_subplot(gs[6:9,8:15])
    ax14=fig.add_subplot(gs[9:12,8:15])
    ax15=fig.add_subplot(gs[12:15,8:15])

    mags=np.array([17.,18.,19.,20.,21.])
    for i in range(0,len(mags)):
        giant=np.where((np.abs(m2fs['gaia_gmag']-mags[i])<0.2)&(m2fs['vlos_error']<5.)&(m2fs['logg_error']<0.6)&(m2fs['feh_error']<0.6)&(m2fs['mgfe_error']<0.6)&(m2fs['logg']<2.)&(np.abs(m2fs['vlos'])<500.)&(np.abs(m2fs['vlos'])>30.))[0]
        dwarf=np.where((np.abs(m2fs['gaia_gmag']-mags[i])<0.2)&(m2fs['vlos_error']<5.)&(m2fs['logg_error']<0.6)&(m2fs['feh_error']<0.6)&(m2fs['mgfe_error']<0.6)&(m2fs['logg']>3.)&(np.abs(m2fs['vlos'])<500.)&(np.abs(m2fs['vlos'])>30.))[0]
#        this2=np.where(np.abs(hecto['gaia_gmag']-mags[i])<0.1)[0]
        giant_spec=fits.open(m2fs['fits_filename'][giant[0]])
        giant_wav=giant_spec[1].data[m2fs['fits_index'][giant[0]]]
        giant_skysub=giant_spec[2].data[m2fs['fits_index'][giant[0]]]
        giant_mask=giant_spec[4].data[m2fs['fits_index'][giant[0]]]
        giant_bestfit=giant_spec[5].data[m2fs['fits_index'][giant[0]]]

        dwarf_spec=fits.open(m2fs['fits_filename'][dwarf[0]])
        dwarf_wav=dwarf_spec[1].data[m2fs['fits_index'][dwarf[0]]]
        dwarf_skysub=dwarf_spec[2].data[m2fs['fits_index'][dwarf[0]]]
        dwarf_mask=dwarf_spec[4].data[m2fs['fits_index'][dwarf[0]]]
        dwarf_bestfit=dwarf_spec[5].data[m2fs['fits_index'][dwarf[0]]]

        if i==0:
            rastring,decstring=mycode.coordstring(m2fs['ra_deg'][giant[0]],m2fs['dec_deg'][giant[0]])
            coordstring=rastring+decstring
            ymin=np.min(giant_skysub[giant_mask==0])-0.3*(np.max(giant_skysub[giant_mask==0])-np.min(giant_skysub[giant_mask==0]))
            ymax=1.25*np.max(giant_skysub[giant_mask==0])
            magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][giant[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][giant[0]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][giant[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][giant[0]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][giant[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][giant[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][giant[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][giant[0]],2))+'$'
            ax1.plot(giant_wav[giant_mask==0],giant_skysub[giant_mask==0],color='k',lw=0.5)
            ax1.plot(giant_wav[giant_mask==0],giant_bestfit[giant_mask==0],color='r',lw=0.5)
            ax1.set_xlim([5130,5190])
            ax1.set_ylim([ymin,ymax])
            ax1.set_yticks([])
            ax1.set_yticklabels([])
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            ax1.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
            ax1.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
            ax1.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)

            ax11.plot(dwarf_wav[dwarf_mask==0],dwarf_skysub[dwarf_mask==0],color='k',lw=0.5)
            ax11.plot(dwarf_wav[dwarf_mask==0],dwarf_bestfit[dwarf_mask==0],color='r',lw=0.5)
            ax11.set_xlim([5130,5190])
            ymin=np.min(dwarf_skysub[dwarf_mask==0])-0.3*(np.max(dwarf_skysub[dwarf_mask==0])-np.min(dwarf_skysub[dwarf_mask==0]))
            ymax=1.25*np.max(dwarf_skysub[dwarf_mask==0])
            ax11.set_ylim([ymin,ymax])
            ax11.set_yticks([])
            ax11.set_yticklabels([])
            ax11.set_xticks([])
            ax11.set_xticklabels([])
            rastring,decstring=mycode.coordstring(m2fs['ra_deg'][dwarf[0]],m2fs['dec_deg'][dwarf[0]])
            coordstring=rastring+decstring
            magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][dwarf[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][dwarf[0]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][dwarf[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][dwarf[0]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][dwarf[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][dwarf[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][dwarf[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][dwarf[0]],2))+'$'
            ax11.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax11.transAxes,fontsize=6)
            ax11.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax11.transAxes,fontsize=6)
            ax11.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)            
        
        if i==1:
            rastring,decstring=mycode.coordstring(m2fs['ra_deg'][giant[0]],m2fs['dec_deg'][giant[0]])
            coordstring=rastring+decstring
            ymin=np.min(giant_skysub[giant_mask==0])-0.3*(np.max(giant_skysub[giant_mask==0])-np.min(giant_skysub[giant_mask==0]))
            ymax=1.25*np.max(giant_skysub[giant_mask==0])
            magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][giant[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][giant[0]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][giant[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][giant[0]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][giant[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][giant[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][giant[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][giant[0]],2))+'$'
            ax2.plot(giant_wav[giant_mask==0],giant_skysub[giant_mask==0],color='k',lw=0.5)
            ax2.plot(giant_wav[giant_mask==0],giant_bestfit[giant_mask==0],color='r',lw=0.5)
            ax2.set_xlim([5130,5190])
            ax2.set_ylim([ymin,ymax])
            ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)

            ax12.plot(dwarf_wav[dwarf_mask==0],dwarf_skysub[dwarf_mask==0],color='k',lw=0.5)
            ax12.plot(dwarf_wav[dwarf_mask==0],dwarf_bestfit[dwarf_mask==0],color='r',lw=0.5)
            ax12.set_xlim([5130,5190])
            ymin=np.min(dwarf_skysub[dwarf_mask==0])-0.3*(np.max(dwarf_skysub[dwarf_mask==0])-np.min(dwarf_skysub[dwarf_mask==0]))
            ymax=1.25*np.max(dwarf_skysub[dwarf_mask==0])
            ax12.set_ylim([ymin,ymax])
            ax12.set_yticks([])
            ax12.set_yticklabels([])
            ax12.set_xticks([])
            ax12.set_xticklabels([])
            rastring,decstring=mycode.coordstring(m2fs['ra_deg'][dwarf[0]],m2fs['dec_deg'][dwarf[0]])
            coordstring=rastring+decstring
            magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][dwarf[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][dwarf[0]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][dwarf[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][dwarf[0]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][dwarf[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][dwarf[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][dwarf[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][dwarf[0]],2))+'$'
            ax12.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax12.transAxes,fontsize=6)
            ax12.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax12.transAxes,fontsize=5)
            ax12.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax12.transAxes,fontsize=6)
            ax12.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax12.transAxes,fontsize=5)
            ax12.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax12.transAxes,fontsize=5)
            ax12.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax12.transAxes,fontsize=5)
            ax12.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax12.transAxes,fontsize=5)            
        
        if i==2:
            rastring,decstring=mycode.coordstring(m2fs['ra_deg'][giant[0]],m2fs['dec_deg'][giant[0]])
            coordstring=rastring+decstring
            ymin=np.min(giant_skysub[giant_mask==0])-0.3*(np.max(giant_skysub[giant_mask==0])-np.min(giant_skysub[giant_mask==0]))
            ymax=1.25*np.max(giant_skysub[giant_mask==0])
            magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][giant[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][giant[0]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][giant[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][giant[0]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][giant[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][giant[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][giant[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][giant[0]],2))+'$'
            ax3.plot(giant_wav[giant_mask==0],giant_skysub[giant_mask==0],color='k',lw=0.5)
            ax3.plot(giant_wav[giant_mask==0],giant_bestfit[giant_mask==0],color='r',lw=0.5)
            ax3.set_ylabel('counts')
            ax3.set_xlim([5130,5190])
            ax3.set_ylim([ymin,ymax])
            ax3.set_yticks([])
            ax3.set_yticklabels([])
            ax3.set_xticks([])
            ax3.set_xticklabels([])
            ax3.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=6)
            ax3.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax3.transAxes,fontsize=6)
            ax3.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)

            ax13.plot(dwarf_wav[dwarf_mask==0],dwarf_skysub[dwarf_mask==0],color='k',lw=0.5)
            ax13.plot(dwarf_wav[dwarf_mask==0],dwarf_bestfit[dwarf_mask==0],color='r',lw=0.5)
            ax13.set_xlim([5130,5190])
            ymin=np.min(dwarf_skysub[dwarf_mask==0])-0.3*(np.max(dwarf_skysub[dwarf_mask==0])-np.min(dwarf_skysub[dwarf_mask==0]))
            ymax=1.25*np.max(dwarf_skysub[dwarf_mask==0])
            ax13.set_ylim([ymin,ymax])
            ax13.set_yticks([])
            ax13.set_yticklabels([])
            ax13.set_xticks([])
            ax13.set_xticklabels([])
            rastring,decstring=mycode.coordstring(m2fs['ra_deg'][dwarf[0]],m2fs['dec_deg'][dwarf[0]])
            coordstring=rastring+decstring
            magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][dwarf[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][dwarf[0]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][dwarf[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][dwarf[0]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][dwarf[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][dwarf[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][dwarf[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][dwarf[0]],2))+'$'
            ax13.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax13.transAxes,fontsize=6)
            ax13.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax13.transAxes,fontsize=5)
            ax13.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax13.transAxes,fontsize=6)
            ax13.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax13.transAxes,fontsize=5)
            ax13.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax13.transAxes,fontsize=5)
            ax13.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax13.transAxes,fontsize=5)
            ax13.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax13.transAxes,fontsize=5)            
        
        if i==3:
            rastring,decstring=mycode.coordstring(m2fs['ra_deg'][giant[0]],m2fs['dec_deg'][giant[0]])
            coordstring=rastring+decstring
            ymin=np.min(giant_skysub[giant_mask==0])-0.3*(np.max(giant_skysub[giant_mask==0])-np.min(giant_skysub[giant_mask==0]))
            ymax=1.25*np.max(giant_skysub[giant_mask==0])
            magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][giant[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][giant[0]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][giant[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][giant[0]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][giant[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][giant[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][giant[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][giant[0]],2))+'$'
            ax4.plot(giant_wav[giant_mask==0],giant_skysub[giant_mask==0],color='k',lw=0.5)
            ax4.plot(giant_wav[giant_mask==0],giant_bestfit[giant_mask==0],color='r',lw=0.5)
            ax4.set_xlim([5130,5190])
            ax4.set_ylim([ymin,ymax])
            ax4.set_yticks([])
            ax4.set_yticklabels([])
            ax4.set_xticks([])
            ax4.set_xticklabels([])
            ax4.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes,fontsize=6)
            ax4.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax4.transAxes,fontsize=6)
            ax4.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)

            ax14.plot(dwarf_wav[dwarf_mask==0],dwarf_skysub[dwarf_mask==0],color='k',lw=0.5)
            ax14.plot(dwarf_wav[dwarf_mask==0],dwarf_bestfit[dwarf_mask==0],color='r',lw=0.5)
            ax14.set_xlim([5130,5190])
            ymin=np.min(dwarf_skysub[dwarf_mask==0])-0.3*(np.max(dwarf_skysub[dwarf_mask==0])-np.min(dwarf_skysub[dwarf_mask==0]))
            ymax=1.25*np.max(dwarf_skysub[dwarf_mask==0])
            ax14.set_ylim([ymin,ymax])
            ax14.set_yticks([])
            ax14.set_yticklabels([])
            ax14.set_xticks([])
            ax14.set_xticklabels([])
            rastring,decstring=mycode.coordstring(m2fs['ra_deg'][dwarf[0]],m2fs['dec_deg'][dwarf[0]])
            coordstring=rastring+decstring
            magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][dwarf[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][dwarf[0]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][dwarf[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][dwarf[0]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][dwarf[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][dwarf[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][dwarf[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][dwarf[0]],2))+'$'
            ax14.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax14.transAxes,fontsize=6)
            ax14.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax14.transAxes,fontsize=5)
            ax14.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax14.transAxes,fontsize=6)
            ax14.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax14.transAxes,fontsize=5)
            ax14.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax14.transAxes,fontsize=5)
            ax14.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax14.transAxes,fontsize=5)
            ax14.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax14.transAxes,fontsize=5)            
        
        if i==4:
            rastring,decstring=mycode.coordstring(m2fs['ra_deg'][giant[0]],m2fs['dec_deg'][giant[0]])
            coordstring=rastring+decstring
            ymin=np.min(giant_skysub[giant_mask==0])-0.3*(np.max(giant_skysub[giant_mask==0])-np.min(giant_skysub[giant_mask==0]))
            ymax=1.25*np.max(giant_skysub[giant_mask==0])
            magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][giant[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][giant[0]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][giant[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][giant[0]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][giant[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][giant[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][giant[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][giant[0]],2))+'$'
            ax5.plot(giant_wav[giant_mask==0],giant_skysub[giant_mask==0],color='k',lw=0.5)
            ax5.plot(giant_wav[giant_mask==0],giant_bestfit[giant_mask==0],color='r',lw=0.5)
            ax5.set_xlim([5130,5190])
            ax5.set_ylim([ymin,ymax])
            ax5.set_yticks([])
            ax5.set_yticklabels([])
            ax5.set_xlabel(r'$\lambda$ [Angstroms]')
#            ax5.set_xticks([])
#            ax5.set_xticklabels([])
            ax5.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax5.transAxes,fontsize=6)
            ax5.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax5.transAxes,fontsize=6)
            ax5.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)

            ax15.plot(dwarf_wav[dwarf_mask==0],dwarf_skysub[dwarf_mask==0],color='k',lw=0.5)
            ax15.plot(dwarf_wav[dwarf_mask==0],dwarf_bestfit[dwarf_mask==0],color='r',lw=0.5)
            ax15.set_xlim([5130,5190])
            ymin=np.min(dwarf_skysub[dwarf_mask==0])-0.3*(np.max(dwarf_skysub[dwarf_mask==0])-np.min(dwarf_skysub[dwarf_mask==0]))
            ymax=1.25*np.max(dwarf_skysub[dwarf_mask==0])
            ax15.set_ylim([ymin,ymax])
            ax15.set_yticks([])
            ax15.set_yticklabels([])
            ax15.set_xlabel(r'$\lambda$ [Angstroms]')
#            ax15.set_xticks([])
#            ax15.set_xticklabels([])
            rastring,decstring=mycode.coordstring(m2fs['ra_deg'][dwarf[0]],m2fs['dec_deg'][dwarf[0]])
            coordstring=rastring+decstring
            magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][dwarf[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][dwarf[0]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][dwarf[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][dwarf[0]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][dwarf[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][dwarf[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][dwarf[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][dwarf[0]],2))+'$'
            ax15.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax15.transAxes,fontsize=6)
            ax15.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax15.transAxes,fontsize=5)
            ax15.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax15.transAxes,fontsize=6)
            ax15.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax15.transAxes,fontsize=5)
            ax15.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax15.transAxes,fontsize=5)
            ax15.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax15.transAxes,fontsize=5)
            ax15.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax15.transAxes,fontsize=5)            

    plt.savefig('m2fs_spectra.pdf',dpi=200)
    plt.show()
    plt.close()


    gs=plt.GridSpec(15,15)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:3,0:7])
    ax2=fig.add_subplot(gs[3:6,0:7])
    ax3=fig.add_subplot(gs[6:9,0:7])
    ax4=fig.add_subplot(gs[9:12,0:7])
    ax5=fig.add_subplot(gs[12:15,0:7])
    ax11=fig.add_subplot(gs[0:3,8:15])
    ax12=fig.add_subplot(gs[3:6,8:15])
    ax13=fig.add_subplot(gs[6:9,8:15])
    ax14=fig.add_subplot(gs[9:12,8:15])
    ax15=fig.add_subplot(gs[12:15,8:15])

    mags=np.array([17.,18.,19.,20.,21.])
    for i in range(0,len(mags)):
#        giant=np.where((np.abs(hecto['gaia_gmag']-mags[i])<0.5)&(hecto['vlos_error']<5.)&(hecto['logg_error']<0.5)&(hecto['feh_error']<0.5)&(hecto['mgfe_error']<0.5)&(hecto['logg']<1.)&(np.abs(hecto['vlos'])<500.)&(np.abs(hecto['vlos'])>30.))[0]
#        dwarf=np.where((np.abs(hecto['gaia_gmag']-mags[i])<0.5)&(hecto['vlos_error']<5.)&(hecto['logg_error']<0.5)&(hecto['feh_error']<0.5)&(hecto['mgfe_error']<0.5)&(hecto['logg']>4.)&(np.abs(hecto['vlos'])<500.)&(np.abs(hecto['vlos'])>30.))[0]
        giant=np.where((np.abs(hecto['gaia_gmag']-mags[i])<0.5)&(hecto['vlos_error']<5.)&(hecto['logg_error']<0.5)&(hecto['feh_error']<0.5)&(hecto['mgfe_error']<0.5)&(hecto['logg']<1.)&(np.abs(hecto['vlos'])<500.)&(np.abs(hecto['vlos'])>30.))[0]
        dwarf=np.where((np.abs(hecto['gaia_gmag']-mags[i])<0.5)&(hecto['vlos_error']<5.)&(hecto['logg_error']<0.5)&(hecto['feh_error']<0.5)&(hecto['mgfe_error']<0.5)&(hecto['logg']>4.)&(np.abs(hecto['vlos'])<500.)&(np.abs(hecto['vlos'])>30.))[0]
#        this2=np.where(np.abs(hecto['gaia_gmag']-mags[i])<0.1)[0]
        giant_spec=fits.open(hecto['fits_filename'][giant[0]])
        giant_wav=giant_spec[0].data[hecto['fits_index'][giant[0]]]
        giant_skysub=giant_spec[1].data[hecto['fits_index'][giant[0]]]
        giant_mask=giant_spec[3].data[hecto['fits_index'][giant[0]]]
        giant_bestfit=giant_spec[4].data[hecto['fits_index'][giant[0]]]

        dwarf_spec=fits.open(hecto['fits_filename'][dwarf[0]])
        dwarf_wav=dwarf_spec[0].data[hecto['fits_index'][dwarf[0]]]
        dwarf_skysub=dwarf_spec[1].data[hecto['fits_index'][dwarf[0]]]
        dwarf_mask=dwarf_spec[3].data[hecto['fits_index'][dwarf[0]]]
        dwarf_bestfit=dwarf_spec[4].data[hecto['fits_index'][dwarf[0]]]

        if i==0:
            rastring,decstring=mycode.coordstring(hecto['ra_deg'][giant[0]],hecto['dec_deg'][giant[0]])
            coordstring=rastring+decstring
            ymin=np.min(giant_skysub[giant_mask==0])-0.3*(np.max(giant_skysub[giant_mask==0])-np.min(giant_skysub[giant_mask==0]))
            ymax=1.25*np.max(giant_skysub[giant_mask==0])
            magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][giant[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][giant[0]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][giant[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][giant[0]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][giant[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][giant[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][giant[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][giant[0]],2))+'$'
            ax1.plot(giant_wav[giant_mask==0],giant_skysub[giant_mask==0],color='k',lw=0.5)
            ax1.plot(giant_wav[giant_mask==0],giant_bestfit[giant_mask==0],color='r',lw=0.5)
            ax1.set_xlim([5155,5295])
            ax1.set_ylim([ymin,ymax])
            ax1.set_yticks([])
            ax1.set_yticklabels([])
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            ax1.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
            ax1.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
            ax1.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)

            ax11.plot(dwarf_wav[dwarf_mask==0],dwarf_skysub[dwarf_mask==0],color='k',lw=0.5)
            ax11.plot(dwarf_wav[dwarf_mask==0],dwarf_bestfit[dwarf_mask==0],color='r',lw=0.5)
            ax11.set_xlim([5155,5295])
            ymin=np.min(dwarf_skysub[dwarf_mask==0])-0.3*(np.max(dwarf_skysub[dwarf_mask==0])-np.min(dwarf_skysub[dwarf_mask==0]))
            ymax=1.25*np.max(dwarf_skysub[dwarf_mask==0])
            ax11.set_ylim([ymin,ymax])
            ax11.set_yticks([])
            ax11.set_yticklabels([])
            ax11.set_xticks([])
            ax11.set_xticklabels([])
            rastring,decstring=mycode.coordstring(hecto['ra_deg'][dwarf[0]],hecto['dec_deg'][dwarf[0]])
            coordstring=rastring+decstring
            magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][dwarf[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][dwarf[0]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][dwarf[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][dwarf[0]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][dwarf[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][dwarf[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][dwarf[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][dwarf[0]],2))+'$'
            ax11.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax11.transAxes,fontsize=6)
            ax11.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax11.transAxes,fontsize=6)
            ax11.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)            
        
        if i==1:
            rastring,decstring=mycode.coordstring(hecto['ra_deg'][giant[0]],hecto['dec_deg'][giant[0]])
            coordstring=rastring+decstring
            ymin=np.min(giant_skysub[giant_mask==0])-0.3*(np.max(giant_skysub[giant_mask==0])-np.min(giant_skysub[giant_mask==0]))
            ymax=1.25*np.max(giant_skysub[giant_mask==0])
            magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][giant[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][giant[0]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][giant[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][giant[0]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][giant[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][giant[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][giant[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][giant[0]],2))+'$'
            ax2.plot(giant_wav[giant_mask==0],giant_skysub[giant_mask==0],color='k',lw=0.5)
            ax2.plot(giant_wav[giant_mask==0],giant_bestfit[giant_mask==0],color='r',lw=0.5)
            ax2.set_xlim([5155,5295])
            ax2.set_ylim([ymin,ymax])
            ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)

            ax12.plot(dwarf_wav[dwarf_mask==0],dwarf_skysub[dwarf_mask==0],color='k',lw=0.5)
            ax12.plot(dwarf_wav[dwarf_mask==0],dwarf_bestfit[dwarf_mask==0],color='r',lw=0.5)
            ax12.set_xlim([5155,5295])
            ymin=np.min(dwarf_skysub[dwarf_mask==0])-0.3*(np.max(dwarf_skysub[dwarf_mask==0])-np.min(dwarf_skysub[dwarf_mask==0]))
            ymax=1.25*np.max(dwarf_skysub[dwarf_mask==0])
            ax12.set_ylim([ymin,ymax])
            ax12.set_yticks([])
            ax12.set_yticklabels([])
            ax12.set_xticks([])
            ax12.set_xticklabels([])
            rastring,decstring=mycode.coordstring(hecto['ra_deg'][dwarf[0]],hecto['dec_deg'][dwarf[0]])
            coordstring=rastring+decstring
            magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][dwarf[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][dwarf[0]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][dwarf[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][dwarf[0]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][dwarf[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][dwarf[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][dwarf[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][dwarf[0]],2))+'$'
            ax12.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax12.transAxes,fontsize=6)
            ax12.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax12.transAxes,fontsize=5)
            ax12.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax12.transAxes,fontsize=6)
            ax12.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax12.transAxes,fontsize=5)
            ax12.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax12.transAxes,fontsize=5)
            ax12.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax12.transAxes,fontsize=5)
            ax12.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax12.transAxes,fontsize=5)            
        
        if i==2:
            rastring,decstring=mycode.coordstring(hecto['ra_deg'][giant[0]],hecto['dec_deg'][giant[0]])
            coordstring=rastring+decstring
            ymin=np.min(giant_skysub[giant_mask==0])-0.3*(np.max(giant_skysub[giant_mask==0])-np.min(giant_skysub[giant_mask==0]))
            ymax=1.25*np.max(giant_skysub[giant_mask==0])
            magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][giant[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][giant[0]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][giant[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][giant[0]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][giant[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][giant[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][giant[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][giant[0]],2))+'$'
            ax3.plot(giant_wav[giant_mask==0],giant_skysub[giant_mask==0],color='k',lw=0.5)
            ax3.plot(giant_wav[giant_mask==0],giant_bestfit[giant_mask==0],color='r',lw=0.5)
            ax3.set_xlim([5155,5295])
            ax3.set_ylim([ymin,ymax])
            ax3.set_yticks([])
            ax3.set_yticklabels([])
            ax3.set_xticks([])
            ax3.set_xticklabels([])
            ax3.set_ylabel('counts')
            ax3.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=6)
            ax3.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax3.transAxes,fontsize=6)
            ax3.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)

            ax13.plot(dwarf_wav[dwarf_mask==0],dwarf_skysub[dwarf_mask==0],color='k',lw=0.5)
            ax13.plot(dwarf_wav[dwarf_mask==0],dwarf_bestfit[dwarf_mask==0],color='r',lw=0.5)
            ax13.set_xlim([5155,5295])
            ymin=np.min(dwarf_skysub[dwarf_mask==0])-0.3*(np.max(dwarf_skysub[dwarf_mask==0])-np.min(dwarf_skysub[dwarf_mask==0]))
            ymax=1.25*np.max(dwarf_skysub[dwarf_mask==0])
            ax13.set_ylim([ymin,ymax])
            ax13.set_yticks([])
            ax13.set_yticklabels([])
            ax13.set_xticks([])
            ax13.set_xticklabels([])
            rastring,decstring=mycode.coordstring(hecto['ra_deg'][dwarf[0]],hecto['dec_deg'][dwarf[0]])
            coordstring=rastring+decstring
            magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][dwarf[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][dwarf[0]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][dwarf[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][dwarf[0]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][dwarf[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][dwarf[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][dwarf[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][dwarf[0]],2))+'$'
            ax13.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax13.transAxes,fontsize=6)
            ax13.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax13.transAxes,fontsize=5)
            ax13.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax13.transAxes,fontsize=6)
            ax13.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax13.transAxes,fontsize=5)
            ax13.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax13.transAxes,fontsize=5)
            ax13.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax13.transAxes,fontsize=5)
            ax13.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax13.transAxes,fontsize=5)            
        
        if i==3:
            rastring,decstring=mycode.coordstring(hecto['ra_deg'][giant[0]],hecto['dec_deg'][giant[0]])
            coordstring=rastring+decstring
            ymin=np.min(giant_skysub[giant_mask==0])-0.3*(np.max(giant_skysub[giant_mask==0])-np.min(giant_skysub[giant_mask==0]))
            ymax=1.25*np.max(giant_skysub[giant_mask==0])
            magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][giant[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][giant[0]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][giant[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][giant[0]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][giant[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][giant[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][giant[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][giant[0]],2))+'$'
            ax4.plot(giant_wav[giant_mask==0],giant_skysub[giant_mask==0],color='k',lw=0.5)
            ax4.plot(giant_wav[giant_mask==0],giant_bestfit[giant_mask==0],color='r',lw=0.5)
            ax4.set_xlim([5155,5295])
            ax4.set_ylim([ymin,ymax])
            ax4.set_yticks([])
            ax4.set_yticklabels([])
            ax4.set_xticks([])
            ax4.set_xticklabels([])
            ax4.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes,fontsize=6)
            ax4.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax4.transAxes,fontsize=6)
            ax4.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)

            ax14.plot(dwarf_wav[dwarf_mask==0],dwarf_skysub[dwarf_mask==0],color='k',lw=0.5)
            ax14.plot(dwarf_wav[dwarf_mask==0],dwarf_bestfit[dwarf_mask==0],color='r',lw=0.5)
            ax14.set_xlim([5155,5295])
            ymin=np.min(dwarf_skysub[dwarf_mask==0])-0.3*(np.max(dwarf_skysub[dwarf_mask==0])-np.min(dwarf_skysub[dwarf_mask==0]))
            ymax=1.25*np.max(dwarf_skysub[dwarf_mask==0])
            ax14.set_ylim([ymin,ymax])
            ax14.set_yticks([])
            ax14.set_yticklabels([])
            ax14.set_xticks([])
            ax14.set_xticklabels([])
            rastring,decstring=mycode.coordstring(hecto['ra_deg'][dwarf[0]],hecto['dec_deg'][dwarf[0]])
            coordstring=rastring+decstring
            magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][dwarf[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][dwarf[0]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][dwarf[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][dwarf[0]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][dwarf[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][dwarf[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][dwarf[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][dwarf[0]],2))+'$'
            ax14.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax14.transAxes,fontsize=6)
            ax14.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax14.transAxes,fontsize=5)
            ax14.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax14.transAxes,fontsize=6)
            ax14.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax14.transAxes,fontsize=5)
            ax14.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax14.transAxes,fontsize=5)
            ax14.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax14.transAxes,fontsize=5)
            ax14.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax14.transAxes,fontsize=5)            
        
        if i==4:
            rastring,decstring=mycode.coordstring(hecto['ra_deg'][giant[0]],hecto['dec_deg'][giant[0]])
            coordstring=rastring+decstring
            ymin=np.min(giant_skysub[giant_mask==0])-0.3*(np.max(giant_skysub[giant_mask==0])-np.min(giant_skysub[giant_mask==0]))
            ymax=1.25*np.max(giant_skysub[giant_mask==0])
            magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][giant[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][giant[0]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][giant[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][giant[0]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][giant[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][giant[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][giant[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][giant[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][giant[0]],2))+'$'
            ax5.plot(giant_wav[giant_mask==0],giant_skysub[giant_mask==0],color='k',lw=0.5)
            ax5.plot(giant_wav[giant_mask==0],giant_bestfit[giant_mask==0],color='r',lw=0.5)
            ax5.set_xlim([5150,5300])
            ax5.set_ylim([ymin,ymax])
            ax5.set_yticks([])
            ax5.set_yticklabels([])
            ax5.set_xlabel(r'$\lambda$ [Angstroms]')
            ax5.set_xticks([5150,5200,5250])
            ax5.set_xticklabels(['5150','5200','5250'])
            ax5.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax5.transAxes,fontsize=6)
            ax5.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax5.transAxes,fontsize=6)
            ax5.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)

            ax15.plot(dwarf_wav[dwarf_mask==0],dwarf_skysub[dwarf_mask==0],color='k',lw=0.5)
            ax15.plot(dwarf_wav[dwarf_mask==0],dwarf_bestfit[dwarf_mask==0],color='r',lw=0.5)
            ax15.set_xlim([5150,5300])
            ymin=np.min(dwarf_skysub[dwarf_mask==0])-0.3*(np.max(dwarf_skysub[dwarf_mask==0])-np.min(dwarf_skysub[dwarf_mask==0]))
            ymax=1.25*np.max(dwarf_skysub[dwarf_mask==0])
            ax15.set_ylim([ymin,ymax])
            ax15.set_yticks([])
            ax15.set_yticklabels([])
            ax15.set_xlabel(r'$\lambda$ [Angstroms]')
            ax15.set_xticks([5150,5200,5250,5300])
            ax15.set_xticklabels(['5150','5200','5250','5300'])
            rastring,decstring=mycode.coordstring(hecto['ra_deg'][dwarf[0]],hecto['dec_deg'][dwarf[0]])
            coordstring=rastring+decstring
            magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][dwarf[0]],2))
            vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][dwarf[0]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][dwarf[0]],2))+'$ km/s'
            teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][dwarf[0]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][dwarf[0]],2))+'$ K'
            loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][dwarf[0]],2))+'$'
            zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][dwarf[0]],2))+'$'
            alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][dwarf[0]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][dwarf[0]],2))+'$'
            ax15.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax15.transAxes,fontsize=6)
            ax15.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax15.transAxes,fontsize=5)
            ax15.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax15.transAxes,fontsize=6)
            ax15.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax15.transAxes,fontsize=5)
            ax15.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax15.transAxes,fontsize=5)
            ax15.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax15.transAxes,fontsize=5)
            ax15.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax15.transAxes,fontsize=5)            

    plt.savefig('hecto_spectra.pdf',dpi=200)
    plt.show()
    plt.close()

if plot_weird_spectra:

    m2fs=fits.open(m2fs_fits_calibrated_filename)[1].data
    hecto=fits.open(hecto_fits_calibrated_filename)[1].data

    gs=plt.GridSpec(9,10)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:3,0:2])
    ax2=fig.add_subplot(gs[0:3,2:4])
    ax3=fig.add_subplot(gs[0:3,4:6])
    ax4=fig.add_subplot(gs[0:3,6:8])
    ax5=fig.add_subplot(gs[0:3,8:10])

    good_obs=np.concatenate([m2fs['good_obs'],hecto['good_obs']])
    gaia_rrl=np.concatenate([m2fs['gaia_rrl'],hecto['gaia_rrl']])
    good_n_obs=np.concatenate([m2fs['good_n_obs'],hecto['good_n_obs']])
    keep_not_rrl=np.where((gaia_rrl==0)&(good_obs==1)&(good_n_obs>1))[0]
    keep_rrl=np.where((gaia_rrl==1)&(good_obs==1)&(good_n_obs>1))[0]
    vlos_mean_error=np.concatenate([m2fs['vlos_mean_error'],hecto['vlos_mean_error']])
    teff_mean_error=np.concatenate([m2fs['teff_mean_error'],hecto['teff_mean_error']])
    logg_mean_error=np.concatenate([m2fs['logg_mean_error'],hecto['logg_mean_error']])
    feh_mean_error=np.concatenate([m2fs['feh_mean_error'],hecto['feh_mean_error']])
    mgfe_mean_error=np.concatenate([m2fs['mgfe_mean_error'],hecto['mgfe_mean_error']])
    vlos_mean_scatter=np.concatenate([m2fs['vlos_mean_scatter'],hecto['vlos_mean_scatter']])
    teff_mean_scatter=np.concatenate([m2fs['teff_mean_scatter'],hecto['teff_mean_scatter']])
    logg_mean_scatter=np.concatenate([m2fs['logg_mean_scatter'],hecto['logg_mean_scatter']])
    feh_mean_scatter=np.concatenate([m2fs['feh_mean_scatter'],hecto['feh_mean_scatter']])
    mgfe_mean_scatter=np.concatenate([m2fs['mgfe_mean_scatter'],hecto['mgfe_mean_scatter']])
    
#    m2fs_keep_not_rrl=np.where((m2fs['gaia_rrl']==0)&(m2fs['good_obs']==1)&(m2fs['good_n_obs']>1))[0]
#    m2fs_keep_rrl=np.where((m2fs['gaia_rrl']==1)&(m2fs['good_obs']==1)&(m2fs['good_n_obs']>1))[0]
#    hecto_keep_not_rrl=np.where((hecto['gaia_rrl']==0)&(hecto['good_obs']==1)&(hecto['good_n_obs']>1))[0]
#    hecto_keep_rrl=np.where((hecto['gaia_rrl']==1)&(hecto['good_obs']==1)&(hecto['good_n_obs']>1))[0]

    ax1.hist(vlos_mean_scatter[keep_not_rrl]/vlos_mean_error[keep_not_rrl],color='navy',align='mid',density=True,histtype='stepfilled',alpha=0.5,range=[0,100],bins=100,label='Not RRL')
    ax1.hist(vlos_mean_scatter[keep_rrl]/vlos_mean_error[keep_rrl],color='r',align='mid',density=True,histtype='stepfilled',alpha=0.5,range=[0,100],bins=100,label='RRL')
    ax1.set_xlim([0,14.9])
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set_xticks([0,5,10])
    ax1.set_xticklabels([0,5,10],fontsize=9)
    ax1.set_ylabel('N (normalized)',fontsize=9)
#    ax1.set_xlabel(r'$\frac{\mathrm{scatter}_{V_{\rm LOS}}}{\sigma_{V_{\rm LOS}}}$',fontsize=9)
    ax1.legend(loc=5,fontsize=6)
    ax1.text(0.9,0.95,r'$V_{\rm LOS}$',horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=9)
    
    ax2.hist(teff_mean_scatter[keep_not_rrl]/teff_mean_error[keep_not_rrl],color='navy',align='mid',density=True,histtype='stepfilled',alpha=0.5,range=[0,30],bins=100)
    ax2.hist(teff_mean_scatter[keep_rrl]/teff_mean_error[keep_rrl],color='r',align='mid',density=True,histtype='stepfilled',alpha=0.5,range=[0,30],bins=100)
    ax2.set_xlim([0,5])
    ax2.set_xticks([1,2,3,4])
    ax2.set_xticklabels([1,2,3,4],fontsize=9)
    ax2.set_yticklabels([])
    ax2.text(0.9,0.95,r'$T_{\rm eff}$',horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=9)
#    ax2.set_xlabel(r'$\sigma_{T_{\rm eff}}$ [100 K]',fontsize=9)

    ax3.hist(logg_mean_scatter[keep_not_rrl]/logg_mean_error[keep_not_rrl],color='navy',align='mid',density=True,histtype='stepfilled',alpha=0.5,range=[0,30],bins=100)
    ax3.hist(logg_mean_scatter[keep_rrl]/logg_mean_error[keep_rrl],color='r',align='mid',density=True,histtype='stepfilled',alpha=0.5,range=[0,30],bins=100)
    ax3.set_xlim([0,5])
    ax3.set_xticks([1,2,3,4])
    ax3.set_xticklabels([1,2,3,4],fontsize=9)
    ax3.set_yticklabels([])
    ax3.set_xlabel('weighted standard deviation / weighted error',fontsize=9)
    ax3.text(0.9,0.95,r'$\log g$',horizontalalignment='right',verticalalignment='top',transform=ax3.transAxes,fontsize=9)
#    ax3.set_xlabel(r'$\sigma_{\log g}$',fontsize=9)

    ax4.hist(feh_mean_scatter[keep_not_rrl]/feh_mean_error[keep_not_rrl],color='navy',align='mid',density=True,histtype='stepfilled',alpha=0.5,range=[0,30],bins=100)
    ax4.hist(feh_mean_scatter[keep_rrl]/feh_mean_error[keep_rrl],color='r',align='mid',density=True,histtype='stepfilled',alpha=0.5,range=[0,30],bins=100)
    ax4.set_xlim([0,5])
    ax4.set_xticks([1,2,3,4])
    ax4.set_xticklabels([1,2,3,4],fontsize=9)
    ax4.set_yticklabels([])
    ax4.text(0.9,0.95,'[Fe/H]',horizontalalignment='right',verticalalignment='top',transform=ax4.transAxes,fontsize=9)
#    ax4.set_xlabel(r'$\sigma_{\mathrm{[Fe/H]}}$',fontsize=9)

    ax5.hist(mgfe_mean_scatter[keep_not_rrl]/mgfe_mean_error[keep_not_rrl],color='navy',align='mid',density=True,histtype='stepfilled',alpha=0.5,range=[0,30],bins=100)
    ax5.hist(mgfe_mean_scatter[keep_rrl]/mgfe_mean_error[keep_rrl],color='r',align='mid',density=True,histtype='stepfilled',alpha=0.5,range=[0,30],bins=100)
    ax5.set_xlim([0,5])
    ax5.set_xticks([1,2,3,4,5])
    ax5.set_xticklabels([1,2,3,4,5],fontsize=9)
    ax5.set_yticklabels([])
    ax5.text(0.9,0.95,'[Mg/Fe]',horizontalalignment='right',verticalalignment='top',transform=ax5.transAxes,fontsize=9)
#    ax5.set_xlabel(r'$\sigma_{\mathrm{[Mg/Fe]}}$',fontsize=9)

    plt.savefig('rrl_scatter.pdf',dpi=300)
    plt.show()
    plt.close()
        
    
    m2fs_good=np.where(m2fs['good_obs']>0)[0]
    hecto_good=np.where(hecto['good_obs']>0)[0]
 
    m2fs_rho0=0.9
    m2fs_gamma=0.
    m2fs_beta=3.
    m2fs_rs=25.
    m2fs_alpha=1.

    hecto_rho0=4.
    hecto_gamma=0.
    hecto_beta=3.
    hecto_rs=75.
    hecto_alpha=1.

    m2fs_phot_variable=np.where((m2fs['gaia_phot_variable_flag']=='VARIABLE'))[0]
    m2fs_phot_variable_good=np.where((m2fs['gaia_phot_variable_flag']=='VARIABLE')&(m2fs['sn_ratio']>=1.))[0]
    m2fs_rrl=np.where((m2fs['gaia_rrl']==1)&(m2fs['obs']==1))[0]
    m2fs_rrl_good=np.where((m2fs['gaia_rrl']==1)&(m2fs['good_obs']==1))[0]
    m2fs_cepheid=np.where((m2fs['gaia_cepheid']==1)&(m2fs['obs']==1))[0]
    m2fs_cepheid_good=np.where((m2fs['gaia_cepheid']==1)&(m2fs['good_obs']==1))[0]
    m2fs_rv_variable=np.where((m2fs['gaia_rv_variable']==1)&(m2fs['obs']==1))[0]
    m2fs_rv_variable_good=np.where((m2fs['gaia_rv_variable']==1)&(m2fs['good_obs']==1))[0]
    m2fs_compact_companion=np.where((m2fs['gaia_compact_companion']==1)&(m2fs['obs']==1))[0]
    m2fs_compact_companion_good=np.where((m2fs['gaia_compact_companion']==1)&(m2fs['good_obs']==1))[0]
    m2fs_agn_wrong=np.where((m2fs['gaia_agn']==1)&(m2fs['good_obs']==1))[0]

    m2fs_chi2_outlier=np.where(m2fs['chi2_flag'])[0]
    m2fs_chi2_outlier_good=np.where((m2fs['chi2_flag'])&(m2fs['sn_ratio']>=1.))[0]
    m2fs_carbon=np.where((m2fs['carbon_flag']))[0]
    m2fs_carbon_good=np.where((m2fs['carbon_flag'])&(m2fs['sn_ratio']>=1.))[0]
    m2fs_agn=np.where((m2fs['gaia_agn']==1))[0]                      
    m2fs_agn_good=np.where((m2fs['gaia_agn']==1)&(m2fs['sn_ratio']>=1.))[0]
    m2fs_phot_variable_n=0
    m2fs_phot_variable_good_n=0
    m2fs_chi2_outlier_n=0
    m2fs_chi2_outlier_good_n=0
    m2fs_carbon_n=0
    m2fs_carbon_good_n=0
    m2fs_agn_n=0
    m2fs_agn_good_n=0
    
    m2fs_anychi2=np.where((m2fs['any_chi2_flag'])&(m2fs['good_obs']==1))[0]
    m2fs_anycarbon=np.where((m2fs['any_carbon_flag'])&(m2fs['good_obs']==1))[0]
                      
    coords=[]
    for i in range(0,len(m2fs)):
        split=m2fs['obj_id'][i].split('_')
        coords.append(split[0]+split[1])
    coords=np.array(coords)
    
    used=np.zeros(len(m2fs))
    for i in range(0,len(m2fs_phot_variable)):
        if used[m2fs_phot_variable[i]]==0:
            this=np.where(coords==coords[m2fs_phot_variable[i]])[0]
            used[this]=1
            m2fs_phot_variable_n+=1

    used=np.zeros(len(m2fs))
    for i in range(0,len(m2fs_phot_variable_good)):
        if used[m2fs_phot_variable_good[i]]==0:
            this=np.where(coords==coords[m2fs_phot_variable_good[i]])[0]
            used[this]=1
            m2fs_phot_variable_good_n+=1            
            
    used=np.zeros(len(m2fs))
    for i in range(0,len(m2fs_carbon)):
        if used[m2fs_carbon[i]]==0:
            this=np.where(coords==coords[m2fs_carbon[i]])[0]
            used[this]=1
            m2fs_carbon_n+=1

    used=np.zeros(len(m2fs))
    for i in range(0,len(m2fs_carbon_good)):
        if used[m2fs_carbon_good[i]]==0:
            this=np.where(coords==coords[m2fs_carbon_good[i]])[0]
            used[this]=1
            m2fs_carbon_good_n+=1            

    used=np.zeros(len(m2fs))
    for i in range(0,len(m2fs_agn)):
        if used[m2fs_agn[i]]==0:
            this=np.where(coords==coords[m2fs_agn[i]])[0]
            used[this]=1
            m2fs_agn_n+=1

    used=np.zeros(len(m2fs))
    for i in range(0,len(m2fs_agn_good)):
        if used[m2fs_agn_good[i]]==0:
            this=np.where(coords==coords[m2fs_agn_good[i]])[0]
            used[this]=1
            m2fs_agn_good_n+=1            
            
    hecto_phot_variable=np.where((hecto['gaia_phot_variable_flag']=='VARIABLE'))[0]
    hecto_phot_variable_good=np.where((hecto['gaia_phot_variable_flag']=='VARIABLE')&(hecto['sn_ratio']>=1.))[0]
    hecto_rrl=np.where((hecto['gaia_rrl']==1)&(hecto['obs']==1))[0]
    hecto_rrl_good=np.where((hecto['gaia_rrl']==1)&(hecto['good_obs']==1))[0]
    hecto_cepheid=np.where((hecto['gaia_cepheid']==1)&(hecto['obs']==1))[0]
    hecto_cepheid_good=np.where((hecto['gaia_cepheid']==1)&(hecto['good_obs']==1))[0]
    hecto_rv_variable=np.where((hecto['gaia_rv_variable']==1)&(hecto['obs']==1))[0]
    hecto_rv_variable_good=np.where((hecto['gaia_rv_variable']==1)&(hecto['good_obs']==1))[0]
    hecto_compact_companion=np.where((hecto['gaia_compact_companion']==1)&(hecto['obs']==1))[0]
    hecto_compact_companion_good=np.where((hecto['gaia_compact_companion']==1)&(hecto['good_obs']==1))[0]
    hecto_agn_wrong=np.where((hecto['gaia_agn']==1)&(hecto['good_obs']==1))[0]

    hecto_chi2_outlier=np.where((hecto['chi2_flag']))[0]
    hecto_chi2_outlier_good=np.where((hecto['chi2_flag'])&(hecto['sn_ratio']>1.))[0]
    hecto_carbon=np.where((hecto['carbon_flag']))[0]
    hecto_carbon_good=np.where((hecto['carbon_flag'])&(hecto['sn_ratio']>=1.))[0]
    hecto_agn=np.where((hecto['gaia_agn']==1))[0]
    hecto_agn_good=np.where((hecto['gaia_agn']==1)&(hecto['sn_ratio']>=1.))[0]
    hecto_chi2_outlier_n=0
    hecto_chi2_outlier_good_n=0
    hecto_carbon_n=0
    hecto_carbon_good_n=0
    hecto_agn_n=0
    hecto_agn_good_n=0
    hecto_phot_variable_n=0
    hecto_phot_variable_good_n=0

    hecto_anychi2=np.where((hecto['any_chi2_flag'])&(hecto['good_obs']==1))[0]
    hecto_anycarbon=np.where((hecto['any_carbon_flag'])&(hecto['good_obs']==1))[0]
    
    coords=[]
    for i in range(0,len(hecto)):
        split=hecto['obj_id'][i].split('_')
        coords.append(split[0]+split[1])
    coords=np.array(coords)
    
    used=np.zeros(len(hecto))
    for i in range(0,len(hecto_phot_variable)):
        if used[hecto_phot_variable[i]]==0:
            this=np.where(coords==coords[hecto_phot_variable[i]])[0]
            used[this]=1
            hecto_phot_variable_n+=1

    used=np.zeros(len(hecto))
    for i in range(0,len(hecto_phot_variable_good)):
        if used[hecto_phot_variable_good[i]]==0:
            this=np.where(coords==coords[hecto_phot_variable_good[i]])[0]
            used[this]=1
            hecto_phot_variable_good_n+=1            
            
    used=np.zeros(len(hecto))
    for i in range(0,len(hecto_carbon)):
        if used[hecto_carbon[i]]==0:
            this=np.where(coords==coords[hecto_carbon[i]])[0]
            used[this]=1
            hecto_carbon_n+=1

    used=np.zeros(len(hecto))
    for i in range(0,len(hecto_carbon_good)):
        if used[hecto_carbon_good[i]]==0:
            this=np.where(coords==coords[hecto_carbon_good[i]])[0]
            used[this]=1
            hecto_carbon_good_n+=1            

    used=np.zeros(len(hecto))
    for i in range(0,len(hecto_agn)):
        if used[hecto_agn[i]]==0:
            this=np.where(coords==coords[hecto_agn[i]])[0]
            used[this]=1
            hecto_agn_n+=1

    used=np.zeros(len(hecto))
    for i in range(0,len(hecto_agn_good)):
        if used[hecto_agn_good[i]]==0:
            this=np.where(coords==coords[hecto_agn_good[i]])[0]
            used[this]=1
            hecto_agn_good_n+=1            

    g0=open('flags_newcommands.tex','w')
    
    string='\\newcommand{\mtwofsmedresagnwrong}{$'+str(len(m2fs_agn_wrong))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedresphotvariablen}{$'+str(m2fs_phot_variable_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedresphotvariablegoodn}{$'+str(m2fs_phot_variable_good_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedresrrln}{$'+str(len(m2fs_rrl))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedresrrlgoodn}{$'+str(len(m2fs_rrl_good))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedrescepheidn}{$'+str(len(m2fs_cepheid))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedrescepheidgoodn}{$'+str(len(m2fs_cepheid_good))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedresagnn}{$'+str(m2fs_agn_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedresagngoodn}{$'+str(m2fs_agn_good_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedresrvvariablen}{$'+str(len(m2fs_rv_variable))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedresrvvariablegoodn}{$'+str(len(m2fs_rv_variable_good))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedrescompactcompanionn}{$'+str(len(m2fs_compact_companion))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedrescompactcompaniongoodn}{$'+str(len(m2fs_compact_companion))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedreschitwooutliern}{$'+str(len(m2fs_chi2_outlier))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedreschitwooutliergoodn}{$'+str(len(m2fs_chi2_outlier_good))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedrescarbonn}{$'+str(m2fs_carbon_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedrescarbongoodn}{$'+str(m2fs_carbon_good_n)+'$} \n'
    g0.write(string)

    string='\\newcommand{\mtwofsmedresanychitwon}{$'+str(len(m2fs_anychi2))+'$} \n'
    g0.write(string)
    string='\\newcommand{\mtwofsmedresanycarbonn}{$'+str(len(m2fs_anycarbon))+'$} \n'
    g0.write(string)

    string='\\newcommand{\hectoagnwrong}{$'+str(len(hecto_agn_wrong))+'$} \n'
    g0.write(string)    
    string='\\newcommand{\hectophotvariablen}{$'+str(hecto_phot_variable_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectophotvariablegoodn}{$'+str(hecto_phot_variable_good_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectorrln}{$'+str(len(hecto_rrl))+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectorrlgoodn}{$'+str(len(hecto_rrl_good))+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectocepheidn}{$'+str(len(hecto_cepheid))+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectocepheidgoodn}{$'+str(len(hecto_cepheid_good))+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectoagnn}{$'+str(hecto_agn_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectoagngoodn}{$'+str(hecto_agn_good_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectorvvariablen}{$'+str(len(hecto_rv_variable))+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectorvvariablegoodn}{$'+str(len(hecto_rv_variable_good))+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectocompactcompanionn}{$'+str(len(hecto_compact_companion))+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectocompactcompaniongoodn}{$'+str(len(hecto_compact_companion))+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectochitwooutliern}{$'+str(hecto_chi2_outlier_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectochitwooutliergoodn}{$'+str(hecto_chi2_outlier_good_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectocarbonn}{$'+str(hecto_carbon_n)+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectocarbongoodn}{$'+str(hecto_carbon_good_n)+'$} \n'
    g0.write(string)

    string='\\newcommand{\hectoanychitwon}{$'+str(len(hecto_anychi2))+'$} \n'
    g0.write(string)
    string='\\newcommand{\hectoanycarbonn}{$'+str(len(hecto_anycarbon))+'$} \n'
    g0.write(string)
             
    g0.close()

    gs=plt.GridSpec(13,13)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:3,0:5])
    ax2=fig.add_subplot(gs[3:6,0:5])
    ax3=fig.add_subplot(gs[7:10,0:5])
    ax4=fig.add_subplot(gs[10:13,0:5])
    
    ax1.scatter(m2fs['sn_ratio'],m2fs['chi2'],s=1,color='k',alpha=0.3,lw=0,rasterized=True)
    ax1.scatter(m2fs['sn_ratio'][m2fs_good],m2fs['chi2'][m2fs_good],s=1,color='r',alpha=0.5,lw=0,rasterized=True)
    logx=np.linspace(-3,3,100)
    x=10.**logx
    y=m2fs_rho0/(x/m2fs_rs)**m2fs_gamma/(1.+(x/m2fs_rs)**m2fs_alpha)**((m2fs_gamma-m2fs_beta)/m2fs_alpha)
    ax1.plot(x,y,color='k',linestyle='--',lw=0.5)
    ax1.set_xlim([0.1,1000])
    ax1.set_ylim([0.3,100])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([])
    ax1.set_yticks([1,10,100])
    ax1.set_yticklabels([1,10,100],fontsize=10)
    ax1.text(0.05,0.95,'M2FS',horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=10)
    ax1.set_ylabel(r'$\chi^2/$pix')

    ax2.scatter(hecto['sn_ratio'],hecto['chi2'],s=1,color='k',alpha=0.3,lw=0,rasterized=True)
    ax2.scatter(hecto['sn_ratio'][hecto_good],hecto['chi2'][hecto_good],s=1,color='r',alpha=0.5,lw=0,rasterized=True)
    logx=np.linspace(-3,3,100)
    x=10.**logx
    y=hecto_rho0/(x/hecto_rs)**hecto_gamma/(1.+(x/hecto_rs)**hecto_alpha)**((hecto_gamma-hecto_beta)/hecto_alpha)
    ax2.plot(x,y,color='k',linestyle='--',lw=0.5)
    ax2.set_xlim([0.1,1000])
    ax2.set_ylim([0.3,900])
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks([0.1,1,10,100,1000])
    ax2.set_xticklabels([0.1,1,10,100,1000],fontsize=10)
    ax2.set_yticks([1,10,100])
    ax2.set_yticklabels([1,10,100],fontsize=10)
    ax2.text(0.05,0.95,'Hectochelle',horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=10)
#    ax2.set_xlabel('median S/N/pix')
    ax2.set_ylabel(r'$\chi^2/$pix')
    
    ax3.scatter(m2fs['sn_ratio'],m2fs['w5163']/m2fs['w5180'],s=1,color='k',alpha=0.3,lw=0,rasterized=True)
    ax3.scatter(m2fs['sn_ratio'][m2fs_good],m2fs['w5163'][m2fs_good]/m2fs['w5180'][m2fs_good],s=1,color='r',alpha=0.5,lw=0,rasterized=True)
    ax3.set_xlim([0.1,1000])
    ax3.set_ylim([0.05,100])
    ax3.set_yticks([0.1,1,10,100])
    ax3.set_yticklabels([0.1,1,10,100])
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xticks([])
    ax3.text(0.95,0.15,'M2FS',horizontalalignment='right',verticalalignment='top',transform=ax3.transAxes,fontsize=10)
    ax3.set_ylabel(r'$W_{5163}/W_{5180}$')
    logx=np.linspace(-1,3,100)
    x=10.**logx
    logy=-0.1*np.exp(-1.*(logx-2.5))
    y=1.8*10.**(-0.7*(1./x)**0.434)
    y[y>0.8]=0.8
    y2=0.3*10.**(1.8*(1./x)**0.434)+0.8
    ax3.plot(x,y,linestyle='--',color='k',lw=0.5)
    ax3.scatter([1.e-12],[1.e-12],s=10,color='k',lw=0,label=r'$\sigma_{V_{\rm LOS}}>5$ km/s',rasterized=True)
    ax3.scatter([1.e-12],[1.e-12],s=10,color='r',lw=0,label=r'$\sigma_{V_{\rm LOS}}>5$ km/s',rasterized=True)
    ax3.legend(loc=1,fontsize=5.5,borderaxespad=0)

    ax4.scatter(hecto['sn_ratio'],hecto['w5163']/hecto['w5180'],s=1,color='k',alpha=0.3,lw=0,rasterized=True)
    ax4.scatter(hecto['sn_ratio'][hecto_good],hecto['w5163'][hecto_good]/hecto['w5180'][hecto_good],s=1,color='r',alpha=0.5,lw=0,rasterized=True)
    ax4.set_xlim([0.1,1000])
    ax4.set_ylim([0.05,90])
    ax3.set_yticks([0.1,1,10])
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('median S/N')
    ax4.text(0.95,0.15,'Hectochelle',horizontalalignment='right',verticalalignment='top',transform=ax4.transAxes,fontsize=10)
    ax4.set_ylabel(r'$W_{5163}/W_{5180}$')
    logx=np.linspace(-1,3,100)
    x=10.**logx
    logy=-0.1*np.exp(-1.*(logx-2.5))
    y=1.8*10.**(-1.0*(1./x)**0.434)
    y[y>0.6]=0.6
    y2=0.3*10.**(1.8*(1./x)**0.434)+0.6
    ax4.plot(x,y,linestyle='--',color='k',lw=0.5)
    
    plt.savefig('m2fs_hecto_weird.pdf',dpi=300)
    plt.show()
    plt.close()
    #    for i in range(0,len(hecto_weird)):

    gs=plt.GridSpec(12,20)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:3,11:20])
    ax2=fig.add_subplot(gs[3:6,11:20])
    ax3=fig.add_subplot(gs[6:9,11:20])
    ax4=fig.add_subplot(gs[9:12,11:20])    

    ax5=fig.add_subplot(gs[0:3,0:9])
    ax6=fig.add_subplot(gs[3:6,0:9])
    ax7=fig.add_subplot(gs[6:9,0:9])
    ax8=fig.add_subplot(gs[9:12,0:9])    
    
    hecto_plot=[1,6,27,22]
    for i in range(0,len(hecto_plot)):
        spec=fits.open(hecto['fits_filename'][hecto_chi2_outlier_good[hecto_plot[i]]])
        wav=spec[0].data[hecto['fits_index'][hecto_chi2_outlier_good[hecto_plot[i]]]]
        skysub=spec[1].data[hecto['fits_index'][hecto_chi2_outlier_good[hecto_plot[i]]]]
        mask=spec[3].data[hecto['fits_index'][hecto_chi2_outlier_good[hecto_plot[i]]]]
        mask[skysub>5.*np.median(skysub[mask==0])]=1
        mask[skysub<-100.]=1
        bestfit=spec[4].data[hecto['fits_index'][hecto_chi2_outlier_good[hecto_plot[i]]]]
        print ((np.max(wav[mask==0]-np.min(wav[mask==0])))/(np.max(np.arange(len(mask))[mask==0])-np.min(np.arange(len(mask))[mask==0])))
        rastring,decstring=mycode.coordstring(hecto['ra_deg'][hecto_chi2_outlier_good[hecto_plot[i]]],hecto['dec_deg'][hecto_chi2_outlier_good[hecto_plot[i]]])
        coordstring=rastring+decstring
        ymin=np.min(skysub[mask==0])-0.3*(np.max(skysub[mask==0])-np.min(skysub[mask==0]))
        ymax=1.25*np.max(skysub[mask==0])
        magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][hecto_chi2_outlier_good[hecto_plot[i]]],2))
        hjdstring=str.format('{0:.2f}',round(hecto['hjd'][hecto_chi2_outlier_good[hecto_plot[i]]],3))
        vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][hecto_chi2_outlier_good[hecto_plot[i]]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][hecto_chi2_outlier_good[hecto_plot[i]]],2))+'$ km/s'
        teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][hecto_chi2_outlier_good[hecto_plot[i]]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][hecto_chi2_outlier_good[hecto_plot[i]]],2))+'$ K'
        loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][hecto_chi2_outlier_good[hecto_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][hecto_chi2_outlier_good[hecto_plot[i]]],2))+'$'
        zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][hecto_chi2_outlier_good[hecto_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][hecto_chi2_outlier_good[hecto_plot[i]]],2))+'$'
        alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][hecto_chi2_outlier_good[hecto_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][hecto_chi2_outlier_good[hecto_plot[i]]],2))+'$'
        print(i)
        if i==0:
#
            ax1.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
#            ax1.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax1.set_xlim([5150,5300])
            ax1.set_ylim([ymin,ymax])
#            ax1.set_xlabel(r'$\lambda$ [Angs.]')
            ax1.set_xticks([])
            ax1.set_ylabel('counts')
            ax1.text(0.01,0.98,coordstring+'_'+hjdstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
#            ax1.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
            ax1.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=6)
#            ax1.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=6)
#            ax1.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=6)
#            ax1.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=6)
#            ax1.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=6)

        if i==1:
            ymax=1.3*np.max(skysub)
            ax2.plot(wav,skysub,color='k',lw=0.5)
#            ax2.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
#            ax2.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax2.set_xlim([5150,5300])
            ax2.set_ylim([ymin,ymax])
#            ax2.set_xlabel(r'$\lambda$ [Angs.]')
            ax2.set_xticks([])
            ax2.set_ylabel('counts')
            ax2.text(0.01,0.98,coordstring+'_'+hjdstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
#            ax2.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
#            ax2.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
#            ax2.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=6)
#            ax2.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=6)
#            ax2.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=6)
#            ax2.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=6)

        if i==2:
            ax3.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
#            ax3.plot(wav,skysub,color='k',lw=0.5)
#            ax3.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax3.set_xlim([5150,5300])
            ax3.set_ylim([ymin,ymax])
            ax3.set_xticks([])
#            ax3.set_xlabel(r'$\lambda$ [Angs.]')
            ax3.set_ylabel('counts')            
            ax3.text(0.01,0.98,coordstring+'_'+hjdstring,horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=6)
#            ax3.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=6)
            ax3.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax3.transAxes,fontsize=6)
#            ax3.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax3.transAxes,fontsize=6)
#            ax3.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax3.transAxes,fontsize=6)
#            ax3.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax3.transAxes,fontsize=6)
#            ax3.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax3.transAxes,fontsize=6)

        if i==3:
            ax4.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
#            ax4.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax4.set_xlim([5150,5300])
            ax4.set_ylim([ymin,ymax])
            ax4.set_xlabel(r'$\lambda$ [Angs.]')
            ax4.set_ylabel('counts')
#            ax4.set_xticks([])
            ax4.text(0.01,0.98,coordstring+'_'+hjdstring,horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes,fontsize=6)
#            ax4.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes,fontsize=6)
            ax4.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax4.transAxes,fontsize=6)
#            ax4.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax4.transAxes,fontsize=6)
#            ax4.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax4.transAxes,fontsize=6)
#            ax4.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax4.transAxes,fontsize=6)
#            ax4.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax4.transAxes,fontsize=6)


    m2fs_plot=[29,9,35,69]
    for i in range(0,len(m2fs_plot)):
        spec=fits.open(m2fs['fits_filename'][m2fs_chi2_outlier[m2fs_plot[i]]])
        wav=spec[1].data[m2fs['fits_index'][m2fs_chi2_outlier[m2fs_plot[i]]]]
        skysub=spec[2].data[m2fs['fits_index'][m2fs_chi2_outlier[m2fs_plot[i]]]]
        mask=spec[4].data[m2fs['fits_index'][m2fs_chi2_outlier[m2fs_plot[i]]]]
        mask[skysub>5.*np.median(skysub[mask==0])]=1
        mask[skysub<-100.]=1
        bestfit=spec[5].data[m2fs['fits_index'][m2fs_chi2_outlier[m2fs_plot[i]]]]
        print ((np.max(wav[mask==0]-np.min(wav[mask==0])))/(np.max(np.arange(len(mask))[mask==0])-np.min(np.arange(len(mask))[mask==0])))
        rastring,decstring=mycode.coordstring(m2fs['ra_deg'][m2fs_chi2_outlier[m2fs_plot[i]]],m2fs['dec_deg'][m2fs_chi2_outlier[m2fs_plot[i]]])
        coordstring=rastring+decstring
        ymin=np.min(skysub[mask==0])-0.3*(np.max(skysub[mask==0])-np.min(skysub[mask==0]))
        ymax=1.3*np.max(skysub[mask==0])
        magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][m2fs_chi2_outlier[m2fs_plot[i]]],2))
        hjdstring=str.format('{0:.2f}',round(m2fs['hjd'][m2fs_chi2_outlier[m2fs_plot[i]]],3))
        vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][m2fs_chi2_outlier[m2fs_plot[i]]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][m2fs_chi2_outlier[m2fs_plot[i]]],2))+'$ km/s'
        teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][m2fs_chi2_outlier[m2fs_plot[i]]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][m2fs_chi2_outlier[m2fs_plot[i]]],2))+'$ K'
        loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][m2fs_chi2_outlier[m2fs_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][m2fs_chi2_outlier[m2fs_plot[i]]],2))+'$'
        zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][m2fs_chi2_outlier[m2fs_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][m2fs_chi2_outlier[m2fs_plot[i]]],2))+'$'
        alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][m2fs_chi2_outlier[m2fs_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][m2fs_chi2_outlier[m2fs_plot[i]]],2))+'$'

        if i==0:
            ymax=1.3*np.max(skysub)
            ax5.plot(wav,skysub,color='k',lw=0.5)
#            ax5.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
#            ax5.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax5.set_xlim([5125,5190])
            ax5.set_ylim([ymin,ymax])
#            ax5.set_xlabel(r'$\lambda$ [Angs.]')
            ax5.set_xticks([])
            ax5.set_ylabel('counts')
            ax5.text(0.01,0.98,coordstring+'_'+hjdstring,horizontalalignment='left',verticalalignment='top',transform=ax5.transAxes,fontsize=6)
#            ax5.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax5.transAxes,fontsize=6)
            ax5.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax5.transAxes,fontsize=6)
#            ax5.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax5.transAxes,fontsize=6)
#            ax5.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax5.transAxes,fontsize=6)
#            ax5.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax5.transAxes,fontsize=6)
#            ax5.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax5.transAxes,fontsize=6)

        if i==1:
            ax6.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
#            ax6.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax6.set_xlim([5125,5190])
            ax6.set_ylim([ymin,ymax])
#            ax6.set_xlabel(r'$\lambda$ [Angs.]')
            ax6.set_xticks([])
            ax6.set_ylabel('counts')
            ax6.text(0.01,0.98,coordstring+'_'+hjdstring,horizontalalignment='left',verticalalignment='top',transform=ax6.transAxes,fontsize=6)
#            ax6.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax6.transAxes,fontsize=6)
            ax6.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax6.transAxes,fontsize=6)
#            ax6.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax6.transAxes,fontsize=6)
#            ax6.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax6.transAxes,fontsize=6)
#            ax6.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax6.transAxes,fontsize=6)
#            ax6.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax6.transAxes,fontsize=6)

        if i==2:
            ax7.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
#            ax7.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax7.set_xlim([5125,5190])
            ax7.set_ylim([ymin,ymax])
            ax7.set_xticks([])
#            ax7.set_xlabel(r'$\lambda$ [Angs.]')
            ax7.set_ylabel('counts')            
            ax7.text(0.01,0.98,coordstring+'_'+hjdstring,horizontalalignment='left',verticalalignment='top',transform=ax7.transAxes,fontsize=6)
#            ax7.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax7.transAxes,fontsize=6)
            ax7.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax7.transAxes,fontsize=6)
#            ax7.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax7.transAxes,fontsize=6)
#            ax7.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax7.transAxes,fontsize=6)
#            ax7.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax7.transAxes,fontsize=6)
#            ax7.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax7.transAxes,fontsize=6)

        if i==3:
            ax8.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
#            ax8.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax8.set_xlim([5125,5190])
            ax8.set_ylim([ymin,ymax])
            ax8.set_xlabel(r'$\lambda$ [Angs.]')
            ax8.set_ylabel('counts')
#            ax8.set_xticks([])
            ax8.text(0.01,0.98,coordstring+'_'+hjdstring,horizontalalignment='left',verticalalignment='top',transform=ax8.transAxes,fontsize=6)
#            ax8.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax8.transAxes,fontsize=6)
            ax8.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax8.transAxes,fontsize=6)
#            ax8.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax8.transAxes,fontsize=6)
#            ax8.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax8.transAxes,fontsize=6)
#            ax8.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax8.transAxes,fontsize=6)
#            ax8.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax8.transAxes,fontsize=6)

    plt.savefig('m2fs_hecto_chi2_outlier_spectra.pdf',dpi=300)
    plt.show()
    plt.close()

    hecto_weird=hecto_agn_wrong
    for i in range(0,len(hecto_weird)):
        print(i)
        gs=plt.GridSpec(15,15)
        gs.update(wspace=0,hspace=0)
        fig=plt.figure(figsize=(6,6))
        ax1=fig.add_subplot(gs[0:15,0:15])

        spec=fits.open(hecto['fits_filename'][hecto_weird[i]])
        wav=spec[0].data[hecto['fits_index'][hecto_weird[i]]]
        skysub=spec[1].data[hecto['fits_index'][hecto_weird[i]]]
        mask=spec[3].data[hecto['fits_index'][hecto_weird[i]]]
        bestfit=spec[4].data[hecto['fits_index'][hecto_weird[i]]]
        
        rastring,decstring=mycode.coordstring(hecto['ra_deg'][hecto_weird[i]],hecto['dec_deg'][hecto_weird[i]])
        coordstring=rastring+decstring
        ymin=np.min(skysub[mask==0])-0.3*(np.max(skysub[mask==0])-np.min(skysub[mask==0]))
        ymax=1.25*np.max(skysub[mask==0])
        magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][hecto_weird[i]],2))
        targetstring='target='+hecto['target_system'][hecto_weird[i]]
        vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][hecto_weird[i]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][hecto_weird[i]],2))+'$ km/s'
        teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][hecto_weird[i]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][hecto_weird[i]],2))+'$ K'
        loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][hecto_weird[i]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][hecto_weird[i]],2))+'$'
        zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][hecto_weird[i]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][hecto_weird[i]],2))+'$'
        alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][hecto_weird[i]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][hecto_weird[i]],2))+'$'
        ax1.plot(wav,skysub,color='b',lw=0.5)
        ax1.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
        ax1.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
        ax1.set_xlim([5150,5300])
        ax1.set_ylim([ymin,ymax])
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax1.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=12)
        ax1.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=12)
        ax1.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=12)
        ax1.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=12)
        ax1.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=12)
        ax1.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=12)
        ax1.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=12)
        ax1.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=12)

        plt.show()
        plt.close()    

    gs=plt.GridSpec(9,10)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:3,0:10])
    ax2=fig.add_subplot(gs[3:6,0:10])
    ax3=fig.add_subplot(gs[6:9,0:10])

    m2fs_weird=np.where((m2fs['gaia_rrl']==1)&(m2fs['good_n_obs']>2)&(m2fs['sn_ratio']>5.))[0]
    for i in range(0,len(m2fs_weird)):
        print(i)

        spec=fits.open(m2fs['fits_filename'][m2fs_weird[i]])
        wav=spec[1].data[m2fs['fits_index'][m2fs_weird[i]]]
        skysub=spec[2].data[m2fs['fits_index'][m2fs_weird[i]]]
        mask=spec[4].data[m2fs['fits_index'][m2fs_weird[i]]]
        bestfit=spec[5].data[m2fs['fits_index'][m2fs_weird[i]]]
        
        rastring,decstring=mycode.coordstring(m2fs['ra_deg'][m2fs_weird[i]],m2fs['dec_deg'][m2fs_weird[i]])
        coordstring=rastring+decstring
        ymin=np.min(skysub[mask==0])-0.3*(np.max(skysub[mask==0])-np.min(skysub[mask==0]))
        ymax=1.25*np.max(skysub[mask==0])
        magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][m2fs_weird[i]],2))
        targetstring='target='+m2fs['target_system'][m2fs_weird[i]]
        vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][m2fs_weird[i]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][m2fs_weird[i]],2))+'$ km/s'
        teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][m2fs_weird[i]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][m2fs_weird[i]],2))+'$ K'
        loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][m2fs_weird[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][m2fs_weird[i]],2))+'$'
        zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][m2fs_weird[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][m2fs_weird[i]],2))+'$'
        alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][m2fs_weird[i]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][m2fs_weird[i]],2))+'$'

        if i==0:
            ax1.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax1.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax1.set_xlim([5127,5190])
            ax1.set_ylim([ymin,ymax])
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            ax1.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=9)
            ax1.set_ylabel('counts')
        if i==1:
            ax2.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax2.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax2.set_xlim([5127,5190])
            ax2.set_ylim([ymin,ymax])
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=9)
            ax2.set_ylabel('counts')
        if i==2:
            ax3.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax3.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax3.set_xlim([5127,5190])
            ax3.set_ylim([ymin,ymax])
            ax3.set_xlabel(r'$\lambda$ [Angstroms]')
            ax3.set_ylabel('counts')
            ax3.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=9)
            ax3.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=9)
            ax3.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax3.transAxes,fontsize=9)
            ax3.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax3.transAxes,fontsize=9)
            ax3.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax3.transAxes,fontsize=9)
            ax3.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax3.transAxes,fontsize=9)
            ax3.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax3.transAxes,fontsize=9)
            ax3.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax3.transAxes,fontsize=9)
    plt.savefig('m2fs_rrl.pdf',dpi=300)
    plt.show()
    plt.close()    

    gs=plt.GridSpec(9,10)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:3,0:10])
    ax2=fig.add_subplot(gs[3:6,0:10])

    hecto_weird=np.where((hecto['gaia_rrl']==1)&(hecto['good_n_obs']>1)&(hecto['sn_ratio']>10.))[0]
    for i in range(0,len(hecto_weird)):
        print(i)

        spec=fits.open(hecto['fits_filename'][hecto_weird[i]])
        wav=spec[0].data[hecto['fits_index'][hecto_weird[i]]]
        skysub=spec[1].data[hecto['fits_index'][hecto_weird[i]]]
        mask=spec[3].data[hecto['fits_index'][hecto_weird[i]]]
        bestfit=spec[4].data[hecto['fits_index'][hecto_weird[i]]]
        
        rastring,decstring=mycode.coordstring(hecto['ra_deg'][hecto_weird[i]],hecto['dec_deg'][hecto_weird[i]])
        coordstring=rastring+decstring
        ymin=np.min(skysub[mask==0])-0.3*(np.max(skysub[mask==0])-np.min(skysub[mask==0]))
        ymax=1.25*np.max(skysub[mask==0])
        magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][hecto_weird[i]],2))
        targetstring='target='+hecto['target_system'][hecto_weird[i]]
        vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][hecto_weird[i]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][hecto_weird[i]],2))+'$ km/s'
        teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][hecto_weird[i]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][hecto_weird[i]],2))+'$ K'
        loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][hecto_weird[i]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][hecto_weird[i]],2))+'$'
        zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][hecto_weird[i]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][hecto_weird[i]],2))+'$'
        alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][hecto_weird[i]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][hecto_weird[i]],2))+'$'

        if i==0:
            ax1.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax1.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax1.set_xlim([5150,5300])
            ax1.set_ylim([ymin,ymax])
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            ax1.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=9)
            ax1.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=9)
            ax1.set_ylabel('counts')
        if i==1:
            ax2.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax2.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax2.set_xlim([5150,5300])
            ax2.set_xlabel(r'$\lambda$ [Angstroms]')
            ax2.set_ylim([ymin,ymax])
            ax2.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=9)
            ax2.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=9)
            ax2.set_ylabel('counts')
    plt.savefig('hecto_rrl.pdf',dpi=300)
    plt.show()
    plt.close()    

    
    gs=plt.GridSpec(12,12)
    gs.update(wspace=0,hspace=0)
    fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(gs[0:2,0:4])
    ax2=fig.add_subplot(gs[3:5,0:4])
    ax3=fig.add_subplot(gs[5:7,0:4])
    ax4=fig.add_subplot(gs[7:9,0:4])
    ax5=fig.add_subplot(gs[9:11,0:4])
    ax6=fig.add_subplot(gs[0:2,6:10])
    ax7=fig.add_subplot(gs[2:4,6:10])
    ax8=fig.add_subplot(gs[4:6,6:10])
    ax9=fig.add_subplot(gs[6:8,6:10])
    ax10=fig.add_subplot(gs[8:10,6:10])
    ax11=fig.add_subplot(gs[10:12,6:10])

    hecto_weird=np.where((hecto['good_obs']==1)&(hecto['feh_mean']<-3.6)&(hecto['logg_mean']>4.5))[0]#3,4,5,7,10,11
    m2fs_weird=np.where((m2fs['good_obs']==1)&(m2fs['feh_mean']<-3.6)&(m2fs['logg_mean']>4.5))[0]#3,4,5,7,10,11
    
    m2fs_plot=[0]
    hecto_plot=[0,1,2,3,4,5,6,7,8,9]
    for i in range(0,len(hecto_plot)):
#        plt.plot(wav[mask==0],skysub[mask==0],color='k')
#        plt.plot(wav[mask==0],bestfit[mask==0],color='r')
#        plt.show()
#        plt.close()

        print(i)

        spec=fits.open(hecto['fits_filename'][hecto_weird[hecto_plot[i]]])
        wav=spec[0].data[hecto['fits_index'][hecto_weird[hecto_plot[i]]]]
        skysub=spec[1].data[hecto['fits_index'][hecto_weird[hecto_plot[i]]]]
        mask=spec[3].data[hecto['fits_index'][hecto_weird[hecto_plot[i]]]]
        bestfit=spec[4].data[hecto['fits_index'][hecto_weird[hecto_plot[i]]]]
        
        rastring,decstring=mycode.coordstring(hecto['ra_deg'][hecto_weird[hecto_plot[i]]],hecto['dec_deg'][hecto_weird[hecto_plot[i]]])
        coordstring=rastring+decstring
        ymin=np.min(skysub[mask==0])-0.3*(np.max(skysub[mask==0])-np.min(skysub[mask==0]))
        ymax=1.25*np.max(skysub[mask==0])
        magstring='G='+str.format('{0:.2f}',round(hecto['gaia_gmag'][hecto_weird[hecto_plot[i]]],2))
        targetstring='target='+hecto['target_system'][hecto_weird[hecto_plot[i]]]
        vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(hecto['vlos'][hecto_weird[hecto_plot[i]]],2))+'\pm'+str.format('{0:.1f}',round(hecto['vlos_error'][hecto_weird[hecto_plot[i]]],2))+'$ km/s'
        teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(hecto['teff'][hecto_weird[hecto_plot[i]]],0))+'\pm'+str.format('{0:.0f}',round(hecto['teff_error'][hecto_weird[hecto_plot[i]]],2))+'$ K'
        loggstring=r'$\log$g$='+str.format('{0:.2f}',round(hecto['logg'][hecto_weird[hecto_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(hecto['logg_error'][hecto_weird[hecto_plot[i]]],2))+'$'
        zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(hecto['feh'][hecto_weird[hecto_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(hecto['feh_error'][hecto_weird[hecto_plot[i]]],2))+'$'
        alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(hecto['mgfe'][hecto_weird[hecto_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(hecto['mgfe_error'][hecto_weird[hecto_plot[i]]],2))+'$'

        if i==0:
            ax2.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax2.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax2.set_xlim([5150,5300])
            ax2.set_ylim([ymin,ymax])
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=5)
#            ax2.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)
            ax2.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=5)
            ax2.set_ylabel('counts')
        if i==1:
            ax3.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax3.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax3.set_xlim([5150,5300])
            ax3.set_ylim([ymin,ymax])
            ax3.set_xticks([])
            ax3.set_xticklabels([])
            ax3.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax3.transAxes,fontsize=5)
#            ax3.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)
            ax3.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax3.transAxes,fontsize=5)
            ax3.set_ylabel('counts')
        if i==2:
            ax4.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax4.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax4.set_xlim([5150,5300])
#            ax4.set_xlabel(r'$\lambda$ [Angstroms]')
            ax4.set_xticks([])
            ax4.set_xticklabels([])
            ax4.set_ylim([ymin,ymax])
            ax4.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax4.transAxes,fontsize=5)
#            ax4.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)
            ax4.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax4.transAxes,fontsize=5)
            ax4.set_ylabel('counts')

        if i==3:
            ax5.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax5.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax5.set_xlim([5150,5300])
            ax5.set_xlabel(r'$\lambda$ [Angstroms]')
            ax5.set_ylim([ymin,ymax])
            ax5.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax5.transAxes,fontsize=5)
#            ax5.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)
            ax5.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax5.transAxes,fontsize=5)
            ax5.set_ylabel('counts')

        if i==4:
            ax6.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax6.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax6.set_xlim([5150,5300])
#            ax6.set_xlabel(r'$\lambda$ [Angstroms]')
            ax6.set_xticks([])
            ax6.set_xticklabels([])
            ax6.set_ylim([ymin,ymax])
            ax6.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax6.transAxes,fontsize=5)
            ax6.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax6.transAxes,fontsize=5)
            ax6.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax6.transAxes,fontsize=5)
#            ax6.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax6.transAxes,fontsize=5)
            ax6.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax6.transAxes,fontsize=5)
            ax6.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax6.transAxes,fontsize=5)
            ax6.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax6.transAxes,fontsize=5)
            ax6.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax6.transAxes,fontsize=5)
            ax6.set_ylabel('counts')

        if i==5:
            ax7.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax7.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax7.set_xlim([5150,5300])
#            ax7.set_xlabel(r'$\lambda$ [Angstroms]')
            ax7.set_xticks([])
            ax7.set_xticklabels([])
            ax7.set_ylim([ymin,ymax])
            ax7.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax7.transAxes,fontsize=5)
            ax7.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax7.transAxes,fontsize=5)
            ax7.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax7.transAxes,fontsize=5)
#            ax7.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax7.transAxes,fontsize=5)
            ax7.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax7.transAxes,fontsize=5)
            ax7.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax7.transAxes,fontsize=5)
            ax7.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax7.transAxes,fontsize=5)
            ax7.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax7.transAxes,fontsize=5)
            ax7.set_ylabel('counts')

        if i==6:
            ax8.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax8.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax8.set_xlim([5150,5300])
#            ax8.set_xlabel(r'$\lambda$ [Angstroms]')
            ax8.set_xticks([])
            ax8.set_xticklabels([])
            ax8.set_ylim([ymin,ymax])
            ax8.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax8.transAxes,fontsize=5)
            ax8.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax8.transAxes,fontsize=5)
            ax8.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax8.transAxes,fontsize=5)
#            ax8.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax8.transAxes,fontsize=5)
            ax8.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax8.transAxes,fontsize=5)
            ax8.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax8.transAxes,fontsize=5)
            ax8.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax8.transAxes,fontsize=5)
            ax8.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax8.transAxes,fontsize=5)
            ax8.set_ylabel('counts')

        if i==7:
            ax9.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax9.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax9.set_xlim([5150,5300])
#            ax9.set_xlabel(r'$\lambda$ [Angstroms]')
            ax9.set_xticks([])
            ax9.set_xticklabels([])
            ax9.set_ylim([ymin,ymax])
            ax9.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax9.transAxes,fontsize=5)
            ax9.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax9.transAxes,fontsize=5)
            ax9.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax9.transAxes,fontsize=5)
#            ax9.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax9.transAxes,fontsize=5)
            ax9.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax9.transAxes,fontsize=5)
            ax9.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax9.transAxes,fontsize=5)
            ax9.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax9.transAxes,fontsize=5)
            ax9.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax9.transAxes,fontsize=5)
            ax9.set_ylabel('counts')

        if i==8:
            ax10.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax10.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax10.set_xlim([5150,5300])
#            ax10.set_xlabel(r'$\lambda$ [Angstroms]')
            ax10.set_xticks([])
            ax10.set_xticklabels([])
            ax10.set_ylim([ymin,ymax])
            ax10.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax10.transAxes,fontsize=5)
            ax10.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax10.transAxes,fontsize=5)
            ax10.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax10.transAxes,fontsize=5)
#            ax10.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax10.transAxes,fontsize=5)
            ax10.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax10.transAxes,fontsize=5)
            ax10.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax10.transAxes,fontsize=5)
            ax10.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax10.transAxes,fontsize=5)
            ax10.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax10.transAxes,fontsize=5)
            ax10.set_ylabel('counts')

        if i==9:
            ax11.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax11.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax11.set_xlim([5150,5300])
            ax11.set_xlabel(r'$\lambda$ [Angstroms]')
#            ax11.set_xticks([])
#            ax11.set_xticklabels([])
            ax11.set_ylim([ymin,ymax])
            ax11.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax11.transAxes,fontsize=5)
#            ax11.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)
            ax11.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax11.transAxes,fontsize=5)
            ax11.set_ylabel('counts')
            
    for i in range(0,len(m2fs_plot)):
        print(i)

        spec=fits.open(m2fs['fits_filename'][m2fs_weird[m2fs_plot[i]]])
        wav=spec[1].data[m2fs['fits_index'][m2fs_weird[m2fs_plot[i]]]]
        skysub=spec[2].data[m2fs['fits_index'][m2fs_weird[m2fs_plot[i]]]]
        mask=spec[4].data[m2fs['fits_index'][m2fs_weird[m2fs_plot[i]]]]
        bestfit=spec[5].data[m2fs['fits_index'][m2fs_weird[m2fs_plot[i]]]]
        
        rastring,decstring=mycode.coordstring(m2fs['ra_deg'][m2fs_weird[m2fs_plot[i]]],m2fs['dec_deg'][m2fs_weird[m2fs_plot[i]]])
        coordstring=rastring+decstring
        ymin=np.min(skysub[mask==0])-0.3*(np.max(skysub[mask==0])-np.min(skysub[mask==0]))
        ymax=1.25*np.max(skysub[mask==0])
        magstring='G='+str.format('{0:.2f}',round(m2fs['gaia_gmag'][m2fs_weird[m2fs_plot[i]]],2))
        targetstring='target='+m2fs['target_system'][m2fs_weird[m2fs_plot[i]]]
        vstring=r'$V_{\rm LOS}='+str.format('{0:.1f}',round(m2fs['vlos'][m2fs_weird[m2fs_plot[i]]],2))+'\pm'+str.format('{0:.1f}',round(m2fs['vlos_error'][m2fs_weird[m2fs_plot[i]]],2))+'$ km/s'
        teffstring=r'$T_{\rm eff}='+str.format('{0:.0f}',round(m2fs['teff'][m2fs_weird[m2fs_plot[i]]],0))+'\pm'+str.format('{0:.0f}',round(m2fs['teff_error'][m2fs_weird[m2fs_plot[i]]],2))+'$ K'
        loggstring=r'$\log$g$='+str.format('{0:.2f}',round(m2fs['logg'][m2fs_weird[m2fs_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['logg_error'][m2fs_weird[m2fs_plot[i]]],2))+'$'
        zstring=r'[Fe/H]$='+str.format('{0:.2f}',round(m2fs['feh'][m2fs_weird[m2fs_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['feh_error'][m2fs_weird[m2fs_plot[i]]],2))+'$'
        alphastring=r'[Mg/Fe]$='+str.format('{0:.2f}',round(m2fs['mgfe'][m2fs_weird[m2fs_plot[i]]],2))+'\pm'+str.format('{0:.2f}',round(m2fs['mgfe_error'][m2fs_weird[m2fs_plot[i]]],2))+'$'

        if i==0:
            ax1.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax1.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax1.set_xlim([5125,5190])
            ax1.set_ylim([ymin,ymax])
#            ax1.set_xlabel(r'$\lambda$ [Angstroms]')
#            ax1.set_xticks([])
#            ax1.set_xticklabels([])
            ax1.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=5)
#            ax1.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)
            ax1.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes,fontsize=5)
            ax1.set_ylabel('counts')
        if i==1:
            ax2.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax2.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax2.set_xlim([5125,5190])
#            ax2.set_xlabel(r'$\lambda$ [Angstroms]')
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            ax2.set_ylim([ymin,ymax])
            ax2.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=6)
            ax2.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax2.transAxes,fontsize=6)
            ax2.set_ylabel('counts')
        if i==2:
            ax7.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax7.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax7.set_xlim([5125,5190])
#            ax7.set_xlabel(r'$\lambda$ [Angstroms]')
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            ax7.set_ylim([ymin,ymax])
            ax7.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax7.transAxes,fontsize=6)
            ax7.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax7.transAxes,fontsize=6)
            ax7.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax7.transAxes,fontsize=6)
            ax7.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax7.transAxes,fontsize=6)
            ax7.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax7.transAxes,fontsize=6)
            ax7.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax7.transAxes,fontsize=6)
            ax7.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax7.transAxes,fontsize=6)
            ax7.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax7.transAxes,fontsize=6)
            ax7.set_ylabel('counts')
        if i==3:
            ax8.plot(wav[mask==0],skysub[mask==0],color='k',lw=0.5)
            ax8.plot(wav[mask==0],bestfit[mask==0],color='r',lw=0.5)
            ax8.set_xlim([5125,5190])
            ax8.set_xlabel(r'$\lambda$ [Angstroms]')
            ax8.set_ylim([ymin,ymax])
            ax8.text(0.01,0.98,coordstring,horizontalalignment='left',verticalalignment='top',transform=ax8.transAxes,fontsize=6)
            ax8.text(0.01,0.9,vstring,horizontalalignment='left',verticalalignment='top',transform=ax8.transAxes,fontsize=6)
            ax8.text(0.99,0.98,magstring,horizontalalignment='right',verticalalignment='top',transform=ax8.transAxes,fontsize=6)
            ax8.text(0.99,0.9,targetstring,horizontalalignment='right',verticalalignment='top',transform=ax8.transAxes,fontsize=6)
            ax8.text(0.01,0.09,teffstring,horizontalalignment='left',verticalalignment='bottom',transform=ax8.transAxes,fontsize=6)
            ax8.text(0.01,0.00,loggstring,horizontalalignment='left',verticalalignment='bottom',transform=ax8.transAxes,fontsize=6)
            ax8.text(0.99,0.09,zstring,horizontalalignment='right',verticalalignment='bottom',transform=ax8.transAxes,fontsize=6)
            ax8.text(0.99,0.00,alphastring,horizontalalignment='right',verticalalignment='bottom',transform=ax8.transAxes,fontsize=6)
            ax2.set_ylabel('counts')

    plt.savefig('m2fs_hecto_weird_lowz.pdf',dpi=300)
    plt.show()
    plt.close()    
    
if plot_cmdmap:

#    m2fs_data_table=open('m2fs_data_table.tex','w')
#    m2fs_data_table_short=open('m2fs_data_table_short.tex','w')
    m2fs_field_center=pickle.load(open('m2fs_field_center.pkl','rb'))
    hecto_field_center=pickle.load(open('../nelson/hecto_field_center.pkl','rb'))
    field_center=np.concatenate([m2fs_field_center,hecto_field_center])
    labels=np.array(['name','host','ra','dec','rhalf','sigrhalf','pa','sigpa','ellipticity','sigellipticity','ref_structure','dmodulus','sigdmodulus','distance','sigdistance','ref_distance','rhalfpc','rhalfpcsphere','absvmag','sigabsvmag','ref_abvsmag','vlos','sigvlos','vdisp','sigvdisp','vdisplimit','ref_vlos','pmra_dr3','sigpmra_dr3','pmdec_dr3','sigpmdec_dr3','ref_pm_dr3','pmra_dr2','sigpmra_dr2','pmdec_dr2','sigpmdec_dr2','ref_pm_dr2','feh','sigfeh','fehdisp','sigfehdisp','fehdisplimit','ref_z'],dtype='str')
    cols=[]
    for i in range(0,len(labels)):
        in_csvfile=open('general_dsph_info.csv','r',newline='')
        in_obj=csv.reader(in_csvfile)
        xxx=[]
        for row in in_obj:
            if in_obj.line_num>1:
                xxx.append(row[i])
        xxx=np.array(xxx)
        if ((not 'ref' in labels[i])&(not 'name' in labels[i])&(not 'host' in labels[i])):
            bad=np.where(xxx=='')[0]
            xxx[bad]=np.nan
            cols.append(fits.Column(name=labels[i],format='D',array=xxx))
        else:
            bad=np.where(xxx=='')[0]
            xxx[bad]='none'
            cols.append(fits.Column(name=labels[i],format='A100',array=xxx))
    coldefs=fits.ColDefs([cols[q] for q in range(0,len(cols))])
    table_hdu=fits.BinTableHDU.from_columns([cols[q] for q in range(0,len(cols))])
    table_hdu.writeto('dsph_parameters.fits',overwrite=True)
    m2fs=fits.open(m2fs_fits_calibrated_filename)[1].data
    hecto=fits.open(hecto_fits_calibrated_filename)[1].data
    dsph=fits.open('dsph_parameters.fits')

    m2fs_coords=SkyCoord(m2fs['ra_deg'],m2fs['dec_deg'],unit=(u.deg,u.deg))
    hecto_coords=SkyCoord(hecto['ra_deg'],hecto['dec_deg'],unit=(u.deg,u.deg))

    age=np.zeros(len(dsph[1].data['name']))+1.e+10

    m2fs_field_object_andrew=np.empty(len(m2fs['target_system']),dtype='U100')
    m2fs_field_object_formal=np.empty(len(m2fs['target_system']),dtype='U100')
    hecto_field_object_andrew=np.empty(len(hecto['target_system']),dtype='U100')
    hecto_field_object_formal=np.empty(len(hecto['target_system']),dtype='U100')

    m2fs_names=np.array([['ant2','antlia_2','Antlia 2'],['car','carina_1','Carina'],['cra','crater_1','Crater'],['for','fornax_1','Fornax'],['gru1','grus_1','Grus 1'],\
                   ['gru2','grus_2','Grus 2'],['hor2','horologium_2','Horologium 2'],['hyd1','hydrus_1','Hydrus 1'],['ind1','indus_1','Indus 1'],['ind2','indus_2','Indus 2'],\
                   ['kgo2','kgo_2','KGO 2'],['kgo4','kgo_4','KGO 4'],['kgo7','kgo_7','KGO 7'],['kgo8','kgo_8','KGO 8'],['kgo10','kgo_10','KGO 10'],['kgo13','kgo_13','KGO 13'],\
                   ['kgo22','kgo_22','KGO 22'],['kop2','koposov_2','Koposov 2'],['pal5','palomoar_5','Palomar 5'],['pho2','phoenix_2','Phoenix 2'],\
                   ['ret2','reticulum_2','Reticulum 2'],['scl','sculptor_1','Sculptor'],['sex','sextans_1','Sextans'],\
                         ['tuc2','tucana_2','Tucana 2'],['tuc3','tucana_3','Tucana 3'],['tuc4','tucana_4','Tucana 4'],['tuc5','tucana_5','Tucana 5']],dtype='U100')

    hecto_names=np.array([['seg3','segue_3','Segue 3'],['umi','ursa_minor_1','Ursa Minor'],['dra','draco_1','Draco'],['leo4','leo_4','Leo IV'],['kgo13','kgo_13','KGO 13'],['cvn1','canes_venatici_1','Canes Venatici I'],['psc2','pisces_2','Pisces 2'],['boo1','bootes_1','Bootes 1'],['tri2','triangulum_2','Triangulum 2'],['sex','sextans_1','Sextans'],['sgr2','sagittarius_2','Sagittarius 2'],\
                          ['cra2','crater_2','Crater 2'],['leo1','leo_1','Leo I'],['leo2','leo_2','Leo II'],['boo2','bootes_2','Bootes 2'],['boo3','bootes_3','Bootes 3'],\
                          ['uma2','ursa_major_2','Ursa Major II'],['uma1','ursa_major_1','Ursa Major I'],['seg1','segue_1','Segue 1'],['seg2','segue_2','Segue 2'],['psc','pisces_1','Pisces 1'],\
                          ['kop2','koposov_2','Koposov 2'],['leo5','leo_5','Leo V']],dtype='U100')
    
    for i in range(0,len(m2fs_names)):
        this=np.where(m2fs['target_system']==m2fs_names[i][0])[0]
        m2fs_field_object_andrew[this]=m2fs_names[i][1]
        m2fs_field_object_formal[this]=m2fs_names[i][2]
    for i in range(0,len(hecto_names)):
        this=np.where(hecto['target_system']==hecto_names[i][0])[0]
        hecto_field_object_andrew[this]=hecto_names[i][1]
        hecto_field_object_formal[this]=hecto_names[i][2]

    def get_cmdmap(catalog,coords,field_object_andrew,field_object_formal,dsph,i,instrument):

        age=1.e+10
        maglim=[21,15]
        print(i,dsph[1].data['name'][i],dsph[1].data['rhalf'][i],dsph[1].data['feh'][i],dsph[1].data['ellipticity'][i],dsph[1].data['pa'][i],dsph[1].data['pmra_dr3'][i],dsph[1].data['pmdec_dr3'][i],)
        if dsph[1].data['feh'][i]!=dsph[1].data['feh'][i]:
            dsph[1].data['feh'][i]=-2.
            dsph[1].data['feh'][i]=999.
        if dsph[1].data['ellipticity'][i]>2.:
            dsph[1].data['ellipticity'][i]=0.
        if dsph[1].data['feh'][i]>2.:
            dsph[1].data['feh'][i]=-2.
        if dsph[1].data['vdisp'][i]>900.:
            dsph[1].data['vdisp'][i]=5.
        center=SkyCoord(dsph[1].data['ra'][i],dsph[1].data['dec'][i],unit=(u.deg,u.deg))
        xi,eta=mycode.etaxiarr(coords.ra.rad,coords.dec.rad,center.ra.rad,center.dec.rad)
        r=np.sqrt(xi**2+eta**2)
        if instrument=='m2fs':
            this=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['filter_name']=='HiRes'))[0]
            this1=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['good_obs']==1)&(catalog['filter_name']=='HiRes'))[0]
            thismem=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['good_obs']==1)&(catalog['logg_mean']<=2.)&(catalog['filter_name']=='HiRes'))[0]
            thismem3=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['good_obs']==1)&(catalog['logg_mean']<=2.)&(r>dsph[1].data['rhalf'][i]*3)&(catalog['filter_name']=='HiRes'))[0]
            thismem4=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['good_obs']==1)&(catalog['logg_mean']<=2.)&(r>dsph[1].data['rhalf'][i]*4)&(catalog['filter_name']=='HiRes'))[0]
            thismem5=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['good_obs']==1)&(catalog['logg_mean']<=2.)&(r>dsph[1].data['rhalf'][i]*5)&(catalog['filter_name']=='HiRes'))[0]
        else:
            this=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.))[0]
            this1=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['good_obs']==1))[0]
            thismem=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['good_obs']==1)&(catalog['logg_mean']<=2.))[0]
            thismem3=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['good_obs']==1)&(catalog['logg_mean']<=2.)&(r>dsph[1].data['rhalf'][i]*3))[0]
            thismem4=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['good_obs']==1)&(catalog['logg_mean']<=2.)&(r>dsph[1].data['rhalf'][i]*4))[0]
            thismem5=np.where((field_object_andrew==dsph[1].data['name'][i])&(catalog['vlos_error']<=5.)&(catalog['good_obs']==1)&(catalog['logg_mean']<=2.)&(r>dsph[1].data['rhalf'][i]*5))[0]
    
        rmax=np.max(r[this])/60.
        rsearch=np.min(np.array([6.,10.*rmax]))
        if instrument=='m2fs':
            rplot=np.max(np.array([3.*dsph[1].data['rhalf'][i]/60.,1.1*np.max(r[this])/60.,1.1*15./60.]))
        if instrument=='hecto':
            rplot=np.max(np.array([3.*dsph[1].data['rhalf'][i]/60.,1.1*np.max(r[this])/60.,1.1*30./60.]))
        if dsph[1].data['name'][i]=='grus_1':
            age=1.3e10
            maglim=[21,16]
        if dsph[1].data['name'][i]=='grus_2':
            maglim=[21,16]
        if dsph[1].data['name'][i]=='horologium_2':
            maglim=[21,16]
        if dsph[1].data['name'][i]=='phoenix_2':
            maglim=[21,16]
        if dsph[1].data['name'][i]=='carina_1':
            maglim=[21,15]
        if dsph[1].data['name'][i]=='fornax_1':
            age=5.e9
            maglim=[21,16]
        if dsph[1].data['name'][i]=='hydrus_1':
            maglim=[21,14]
        if dsph[1].data['name'][i]=='reticulum_2':
            maglim=[21,14]
            age=1.3e10
        if dsph[1].data['name'][i]=='tucana_2':
            maglim=[21,15]
        if dsph[1].data['name'][i]=='tucana_3':
            maglim=[21,14]
            age=1.3e10
        if dsph[1].data['name'][i]=='tucana_4':
            maglim=[21,14]
        if dsph[1].data['name'][i]=='antlia_2':
            rplot=1.7
            maglim=[21,16]
        if dsph[1].data['name'][i]=='draco_1':
            maglim=[21,14]
            age=1.3e10
        if dsph[1].data['name'][i]=='triangulum_2':
            maglim=[21,14]
        if dsph[1].data['name'][i]=='pisces_2':
            maglim=[21,17]
        if dsph[1].data['name'][i]=='crater_2':
            maglim=[21,16]
        if dsph[1].data['name'][i]=='leo_1':
            maglim=[21,18]
            age=5.e9
        if dsph[1].data['name'][i]=='leo_4':
            maglim=[21,17]
        if dsph[1].data['name'][i]=='leo_5':
            maglim=[21,17]
        if dsph[1].data['name'][i]=='leo_2':
            maglim=[21,18]
            age=5.e+9
        if dsph[1].data['name'][i]=='ursa_major_2':
            maglim=[21,14]
        if dsph[1].data['name'][i]=='ursa_major_1':
            maglim=[21,16]
        if dsph[1].data['name'][i]=='segue_1':
            maglim=[21,14]
            age=1.3e10
        if dsph[1].data['name'][i]=='canes_venatici_1':
            maglim=[21,18]
        if dsph[1].data['name'][i]=='bootes_2':
            maglim=[21,14]

        maglimticks=(np.arange(maglim[0]-maglim[1]+1)+maglim[1])[::-1]
        Q=str.format('''select source_id,ra,dec,pmra,pmdec,pmra_error,pmdec_error,pmra_pmdec_corr,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,phot_g_mean_flux,phot_g_mean_flux_error,phot_bp_mean_flux,phot_bp_mean_flux_error,phot_rp_mean_flux,phot_rp_mean_flux_error,parallax,parallax_error,astrometric_excess_noise,duplicated_source from gaia_edr3.gaia_source where phot_g_mean_mag=phot_g_mean_mag and q3c_radial_query(ra,dec,{center.ra.deg},{center.dec.deg},{rsearch}) and astrometric_excess_noise<exp(greatest(1.5,1.5+0.3*(phot_g_mean_mag-18))) and parallax<3*parallax_error limit 10000000''', **locals())
        gaia_id,gaia_radeg,gaia_decdeg,gaia_pmra,gaia_pmdec,gaia_sigpmra,gaia_sigpmdec,gaia_pmra_pmdec_corr,gaia_gmag,gaia_bpmag,gaia_rpmag,gaia_gflux,gaia_siggflux,gaia_bpflux,gaia_sigbpflux,gaia_rpflux,gaia_sigrpflux,gaia_parallax,gaia_sigparallax,gaia_astrometric_excess_noise,gaia_duplicated_source0=sqlutil.get(Q,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')
        gaia_coords=SkyCoord(gaia_radeg,gaia_decdeg,unit=(u.deg,u.deg))
        gaia_xi,gaia_eta=mycode.etaxiarr(gaia_coords.ra.rad,gaia_coords.dec.rad,center.ra.rad,center.dec.rad)
        gaia_r=np.sqrt(gaia_xi**2+gaia_eta**2)
        gaia_ebv=dustmaps.sfd.SFDQuery().query_equ(gaia_radeg,gaia_decdeg)
        gaia_siggmag=(2.5/gaia_gflux/2.30258)**2*gaia_siggflux**2
        gaia_sigbpmag=(2.5/gaia_bpflux/2.30258)**2*gaia_sigbpflux**2
        gaia_sigrpmag=(2.5/gaia_rpflux/2.30258)**2*gaia_sigrpflux**2
        a0=3.1*gaia_ebv*0.86#use Eq. 1 from Babusiaux et al 2018 (1804.09378) to get extinction coefficients, apply extinction to update BP-RP, then interate again to get final estimate of extinction coefficient.  The factor of 0.86 comes from Sergey, as explained to me in email from Andrew on July 11, 2022: "I've recently realized that Babusieux's formula was written to be applied to "True" EBV. But the SFD EBV is overestimated according to Schlafly11. So if you use SFD EBV's for the getDust, then the ebv needs to be multiplied by 0.86. (when you work with other surveys it's usually not a problem, because the coefficients are already given wrt SFD, but that I learned doesn't apply for gaia's code."
        kg_c=np.array([0.9761,-0.1704,0.0086,0.0011,-0.0438,0.0013,0.0099])
        kbp_c=np.array([1.1517,-0.0871,-0.0333,0.0173,-0.0230,0.0006,0.0043])
        krp_c=np.array([0.6104,-0.0170,-0.0026,-0.0017,-0.0078,0.00005,0.0006])
        col0=gaia_bpmag-gaia_rpmag
        kg=kg_c[0]+kg_c[1]*col0+kg_c[2]*col0**2+kg_c[3]*col0**3+kg_c[4]*a0+kg_c[5]*a0**2+kg_c[6]*col0*a0
        kbp=kbp_c[0]+kbp_c[1]*col0+kbp_c[2]*col0**2+kbp_c[3]*col0**3+kbp_c[4]*a0+kbp_c[5]*a0**2+kbp_c[6]*col0*a0
        krp=krp_c[0]+krp_c[1]*col0+krp_c[2]*col0**2+krp_c[3]*col0**3+krp_c[4]*a0+krp_c[5]*a0**2+krp_c[6]*col0*a0
        ag=kg*a0
        abp=kbp*a0
        arp=krp*a0
        col1=gaia_bpmag-abp-(gaia_rpmag-arp)
        kg=kg_c[0]+kg_c[1]*col1+kg_c[2]*col1**2+kg_c[3]*col1**3+kg_c[4]*a0+kg_c[5]*a0**2+kg_c[6]*col1*a0
        kbp=kbp_c[0]+kbp_c[1]*col1+kbp_c[2]*col1**2+kbp_c[3]*col1**3+kbp_c[4]*a0+kbp_c[5]*a0**2+kbp_c[6]*col1*a0
        krp=krp_c[0]+krp_c[1]*col1+krp_c[2]*col1**2+krp_c[3]*col1**3+krp_c[4]*a0+krp_c[5]*a0**2+krp_c[6]*col1*a0
        gaia_ag=kg*a0
        gaia_abp=kbp*a0
        gaia_arp=krp*a0
        gaia_gmag_dered=gaia_gmag-gaia_ag
        gaia_bpmag_dered=gaia_bpmag-gaia_abp
        gaia_rpmag_dered=gaia_rpmag-gaia_arp

        g0=[]
        bp0=[]
        rp0=[]
        eep0=np.linspace(202,707,400)
        for j in range(0,len(eep0)):
            pars=[eep0[j],np.log10(age),dsph[1].data['feh'][i]]
            shite=mist.interp_mag(pars+[10.**((dsph[1].data['dmodulus'][i]+5.)/5.),0.0],['G','BP','RP'])
            if shite[3][0]==shite[3][0]:
                g0.append(shite[3][0])
                bp0.append(shite[3][1])
                rp0.append(shite[3][2])
        g0=np.array(g0)
        bp0=np.array(bp0)
        rp0=np.array(rp0)

        iso_color=bp0-rp0
        iso_mag=g0
        gaia_mag=gaia_gmag_dered
        gaia_color=gaia_bpmag_dered-gaia_rpmag_dered
        gaia_err=np.sqrt(gaia_siggmag**2+gaia_sigbpmag**2+gaia_sigrpmag**2)

        phot_target=np.zeros(len(gaia_id),dtype='int')
        for j in range(0,len(gaia_id)):
            if gaia_color[j]==gaia_color[j]:
                mindist=np.min(np.sqrt((gaia_color[j]-iso_color)**2+(gaia_mag[j]-iso_mag)**2))
                if mindist<=np.max(np.array([0.15,gaia_err[j]])):
                    phot_target[j]=1

        plt.rc('xtick',labelsize='6')
        plt.rc('ytick',labelsize='6')
        gs=plt.GridSpec(22,22)
        gs.update(wspace=0,hspace=0)
        fig=plt.figure(figsize=(6,6))
        ax1=fig.add_subplot(gs[0:4,0:4])
        ax2=fig.add_subplot(gs[0:4,6:10])
        ax3=fig.add_subplot(gs[0:4,12:16])
        ax4=fig.add_subplot(gs[0:4,18:22])
            
        fake_x=np.array([-999,-999])
        fake_y=np.array([-999,-999])
        fake_logg=np.array([0,5])

        print(len(this),len(this1),len(thismem),len(thismem3),len(thismem4),len(thismem5),' bbbbbbbbbbbbbbb')
        
        order=np.argsort(catalog['logg_mean'][this1])[::-1]
        ax2.scatter(gaia_bpmag_dered[gaia_r<60.]-gaia_rpmag_dered[gaia_r<60.],gaia_gmag_dered[gaia_r<60.],s=0.25,alpha=0.3,color='0.7',label=r'$R<1^{\circ}$',rasterized=True,lw=0)
        c2=ax2.scatter(np.concatenate([catalog['gaia_bpmag_dered'][this1][order]-catalog['gaia_rpmag_dered'][this1][order],fake_x]),np.concatenate([catalog['gaia_gmag_dered'][this1][order],fake_y]),s=1,c=np.concatenate([catalog['logg_mean'][this1][order],fake_logg]),alpha=0.5,cmap='jet',rasterized=True,lw=0)
        ax2.plot(iso_color,iso_mag,color='k',lw=0.5,linestyle='--')
        ax2.set_xlim([-0.5,2])
        ax2.set_ylim(maglim)
        ax2.set_yticks(maglimticks)
        ax2.set_yticklabels(maglimticks,fontsize=6)
        ax2.set_xticks([0,1,2])
        ax2.set_xticklabels([0,1,2],fontsize=6)
        ax2.set_xlabel('BP-RP',fontsize=6)
        ax2.set_ylabel('G',fontsize=6,labelpad=-0.5)

        for j in range(0,len(field_center)):
            field_center_xi,field_center_eta=mycode.etaxiarr(field_center[j].ra.rad,field_center[j].dec.rad,center.ra.rad,center.dec.rad)
            if instrument=='m2fs':
                axis_major=0.5
                axis_minor=0.5
            if instrument=='hecto':
                axis_major=1.
                axis_minor=1.
                
            ell=Ellipse(xy=[field_center_xi/60.,field_center_eta/60],height=axis_major,width=axis_minor,angle=0,fc='0.7',ec=None,fill=True,alpha=1)
            if ((instrument=='m2fs')&(j<len(m2fs_field_center))):
                ax1.add_artist(ell)
            if ((instrument=='hecto')&(j>=len(m2fs_field_center))):
                ax1.add_artist(ell)                
            axis_major=2.*2.*dsph[1].data['rhalf'][i]/np.sqrt(1.-dsph[1].data['ellipticity'][i])/60.#twice Rhalf
            axis_minor=axis_major*(1.-dsph[1].data['ellipticity'][i])
            ell=Ellipse(xy=[0,0],height=axis_major,width=axis_minor,angle=-dsph[1].data['PA'][i],fc=None,ec='k',fill=False,linestyle='--',lw=0.5)

        ax1.scatter(gaia_xi[phot_target==1]/60.,gaia_eta[phot_target==1]/60.,s=0.25,alpha=0.3,color='0.5',label='isochrone',rasterized=True,lw=0)
        c1=ax1.scatter(np.concatenate([xi[this1][order]/60.,fake_x]),np.concatenate([eta[this1][order]/60.,fake_y]),c=np.concatenate([catalog['logg_mean'][this1][order],fake_logg]),s=1,alpha=0.5,cmap='jet',rasterized=True,lw=0)
        ax1.add_artist(ell)
        ax1.set_xlim([rplot,-rplot])
        ax1.set_ylim([-rplot,rplot])
        ax1.set_xlabel(r'$\Delta$R.A. [deg]',fontsize=6)
        ax1.set_ylabel(r'$\Delta$Dec. [deg]',labelpad=-2,fontsize=6)
        ax1.text(0.05,1.15,field_object_formal[this1[0]],horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=6)


        xxx=np.concatenate([catalog['gaia_pmra'][this1][order],fake_x])
        yyy=np.concatenate([catalog['gaia_pmdec'][this1][order],fake_y])
        ggg=np.concatenate([catalog['logg_mean'][this1][order],fake_logg])
        c3=ax3.scatter(xxx,yyy,s=1,c=ggg,alpha=0.5,cmap='jet',rasterized=True,vmin=0,vmax=5,lw=0)
        ax3.set_xlim([-5,5])
        ax3.set_ylim([-5,5])
        axis_major=2.*dsph[1].data['sigpmra_dr3'][i]
        axis_minor=2.*dsph[1].data['sigpmdec_dr3'][i]
        if dsph[1].data['pmra_dr3'][i]==dsph[1].data['pmra_dr3'][i]:
            ax3.plot([dsph[1].data['pmra_dr3'][i],dsph[1].data['pmra_dr3'][i]],[-999,999],color='k',linestyle='--',lw=0.5)
            ax3.plot([-999,999],[dsph[1].data['pmdec_dr3'][i],dsph[1].data['pmdec_dr3'][i]],color='k',linestyle='--',lw=0.5)

        ax3.set_xlabel(r'$\mu_{\alpha *}$ [mas/year]',fontsize=6)
        ax3.set_ylabel(r'$\mu_{\delta}$ [mas/year]',fontsize=6,labelpad=-2)
        
        xxx=np.concatenate([catalog['vlos_mean'][this1][order],fake_x])
        yyy=np.concatenate([catalog['feh_mean'][this1][order],fake_y])
        ggg=np.concatenate([catalog['logg_mean'][this1][order],fake_logg])
        c4=ax4.scatter(xxx,yyy,s=1,c=ggg,alpha=0.5,cmap='jet',rasterized=True,vmin=0,vmax=5,lw=0)
        clb=plt.colorbar(c4,location='right',ax=ax4,ticks=[0,1,2,3,4,5])
        clb.ax.set_title(label=r'$\log g$',fontsize=6)
        xmin=dsph[1].data['vlos'][i]-100.
        xmax=dsph[1].data['vlos'][i]+100.
        if(dsph[1].data['vlos'][i]==dsph[1].data['vlos'][i]):
            plt.plot([dsph[1].data['vlos'][i],dsph[1].data['vlos'][i]],[-999,999],color='k',linestyle='--',lw=0.5)
            ax4.set_xlim([xmin,xmax])
        else:
            ax4.set_xlim([-300,300])
        ax4.set_ylim([-4,1])
        ax4.set_yticks([-4,-3,-2,-1,0,1])
        ax4.set_yticklabels([-4,-3,-2,-1,0,1],fontsize=6)
        ax4.set_xlabel(r'$V_{\rm LOS}$ [km/s]',fontsize=6)
        ax4.set_ylabel('[Fe/H]',fontsize=6)

        plt.savefig(dsph[1].data['name'][i]+'_'+instrument+'_cmdmap.pdf',dpi=500)
#        plt.show()
        plt.close()
        return 0

    for j in range(0,len(hecto_names)):
        this=np.where(dsph[1].data['name']==hecto_names[j][1])[0]
        if len(this)>0:
            get_cmdmap(hecto,hecto_coords,hecto_field_object_andrew,hecto_field_object_formal,dsph,this[0],'hecto')

    for j in range(0,len(m2fs_names)):
        this=np.where(dsph[1].data['name']==m2fs_names[j][1])[0]
        if len(this)>0:
            get_cmdmap(m2fs,m2fs_coords,m2fs_field_object_andrew,m2fs_field_object_formal,dsph,this[0],'m2fs')
    

            
    
