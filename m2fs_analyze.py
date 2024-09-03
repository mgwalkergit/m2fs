import numpy as np
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
import mycode
matplotlib.use('TkAgg')
#matplotlib.use('pdf')

samestar=1.#arcsec

m2fs=fits.open('all_m2fs.fits')
m2fs_ra_deg=m2fs[1].data['ra_deg']
m2fs_dec_deg=m2fs[1].data['dec_deg']
m2fs_v=m2fs[1].data['vlos']
m2fs_sig_v=m2fs[1].data['vlos_error']
m2fs_skew_v=m2fs[1].data['vlos_skew']
m2fs_kurt_v=m2fs[1].data['vlos_kurtosis']
m2fs_teff=m2fs[1].data['teff']
m2fs_sig_teff=m2fs[1].data['teff_error']
m2fs_skew_teff=m2fs[1].data['teff_skew']
m2fs_kurt_teff=m2fs[1].data['teff_kurtosis']
m2fs_logg=m2fs[1].data['logg']
m2fs_sig_logg=m2fs[1].data['logg_error']
m2fs_skew_logg=m2fs[1].data['logg_skew']
m2fs_kurt_logg=m2fs[1].data['logg_kurtosis']
m2fs_z=m2fs[1].data['z']
m2fs_sig_z=m2fs[1].data['z_error']
m2fs_skew_z=m2fs[1].data['z_skew']
m2fs_kurt_z=m2fs[1].data['z_kurtosis']

m2fs_coords=SkyCoord(m2fs_ra_deg*u.deg,m2fs_dec_deg*u.deg)
m2fs_good=np.where((m2fs_sig_v<5.)&(m2fs_sig_v>0.)&(np.abs(m2fs_skew_v)<5.)&(np.abs(m2fs_kurt_v)<50.))[0]
m2fs_obs=np.zeros(len(m2fs_ra_deg),dtype='int')
m2fs_goodobs=np.zeros(len(m2fs_ra_deg),dtype='int')
m2fs_nobs=np.zeros(len(m2fs_ra_deg),dtype='int')
m2fs_ngoodobs=np.zeros(len(m2fs_ra_deg),dtype='int')
m2fs_mean_v=np.zeros(len(m2fs_ra_deg),dtype='float')
m2fs_sig_mean_v=np.zeros(len(m2fs_ra_deg),dtype='float')
m2fs_mean_teff=np.zeros(len(m2fs_ra_deg),dtype='float')
m2fs_sig_mean_teff=np.zeros(len(m2fs_ra_deg),dtype='float')
m2fs_mean_logg=np.zeros(len(m2fs_ra_deg),dtype='float')
m2fs_sig_mean_logg=np.zeros(len(m2fs_ra_deg),dtype='float')
m2fs_mean_z=np.zeros(len(m2fs_ra_deg),dtype='float')
m2fs_sig_mean_z=np.zeros(len(m2fs_ra_deg),dtype='float')

for i in range(0,len(m2fs_ra_deg)):
    if m2fs_obs[i]==0:
        sep=m2fs_coords[i].separation(m2fs_coords)
        same=np.where(sep.value*3600.<samestar)[0]
        m2fs_nobs[same]=len(same)
        ggg=0
        use=np.zeros(len(same),dtype='int')
        for j in range(0,len(same)):
            m2fs_obs[same[j]]=j+1
            if same[j] in m2fs_good:
                use[j]=1
                ggg=ggg+1
                m2fs_goodobs[same[j]]=ggg
        m2fs_ngoodobs[same]=ggg
        keep=np.where(use==1)[0]
        if len(keep)>0:
            m2fs_mean_v0,m2fs_sig_mean_v0=mycode.weightedmean(m2fs_v[same[keep]],m2fs_sig_v[same[keep]])
            m2fs_mean_teff0,m2fs_sig_mean_teff0=mycode.weightedmean(m2fs_teff[same[keep]],m2fs_sig_teff[same[keep]])
            m2fs_mean_logg0,m2fs_sig_mean_logg0=mycode.weightedmean(m2fs_logg[same[keep]],m2fs_sig_logg[same[keep]])
            m2fs_mean_z0,m2fs_sig_mean_z0=mycode.weightedmean(m2fs_z[same[keep]],m2fs_sig_z[same[keep]])
            m2fs_mean_v[same]=m2fs_mean_v0
            m2fs_sig_mean_v[same]=m2fs_sig_mean_v0
            m2fs_mean_teff[same]=m2fs_mean_teff0
            m2fs_sig_mean_teff[same]=m2fs_sig_mean_teff0
            m2fs_mean_logg[same]=m2fs_mean_logg0
            m2fs_sig_mean_logg[same]=m2fs_sig_mean_logg0
            m2fs_mean_z[same]=m2fs_mean_z0
            m2fs_sig_mean_z[same]=m2fs_sig_mean_z0



hecto=fits.open('/nfs/nas-0-9/mgwalker.proj/nelson/nelson_all.fits')
hecto_ra_deg=hecto[1].data['ra_deg']
hecto_dec_deg=hecto[1].data['dec_deg']
hecto_v=hecto[1].data['vlos']
hecto_sig_v=hecto[1].data['vlos_error']
hecto_skew_v=hecto[1].data['vlos_skew']
hecto_kurt_v=hecto[1].data['vlos_kurtosis']
hecto_teff=hecto[1].data['teff']
hecto_sig_teff=hecto[1].data['teff_error']
hecto_skew_teff=hecto[1].data['teff_skew']
hecto_kurt_teff=hecto[1].data['teff_kurtosis']
hecto_logg=hecto[1].data['logg']
hecto_sig_logg=hecto[1].data['logg_error']
hecto_skew_logg=hecto[1].data['logg_skew']
hecto_kurt_logg=hecto[1].data['logg_kurtosis']
hecto_z=hecto[1].data['z']
hecto_sig_z=hecto[1].data['z_error']
hecto_skew_z=hecto[1].data['z_skew']
hecto_kurt_z=hecto[1].data['z_kurtosis']

hecto_coords=SkyCoord(hecto_ra_deg*u.deg,hecto_dec_deg*u.deg)
hecto_good=np.where((hecto_sig_v<5.)&(hecto_sig_v>0.)&(np.abs(hecto_skew_v)<5.)&(np.abs(hecto_kurt_v)<50.))[0]
hecto_obs=np.zeros(len(hecto_ra_deg),dtype='int')
hecto_goodobs=np.zeros(len(hecto_ra_deg),dtype='int')
hecto_nobs=np.zeros(len(hecto_ra_deg),dtype='int')
hecto_ngoodobs=np.zeros(len(hecto_ra_deg),dtype='int')
hecto_mean_v=np.zeros(len(hecto_ra_deg),dtype='float')
hecto_sig_mean_v=np.zeros(len(hecto_ra_deg),dtype='float')
hecto_mean_teff=np.zeros(len(hecto_ra_deg),dtype='float')
hecto_sig_mean_teff=np.zeros(len(hecto_ra_deg),dtype='float')
hecto_mean_logg=np.zeros(len(hecto_ra_deg),dtype='float')
hecto_sig_mean_logg=np.zeros(len(hecto_ra_deg),dtype='float')
hecto_mean_z=np.zeros(len(hecto_ra_deg),dtype='float')
hecto_sig_mean_z=np.zeros(len(hecto_ra_deg),dtype='float')

for i in range(0,len(hecto_ra_deg)):
    if hecto_obs[i]==0:
        sep=hecto_coords[i].separation(hecto_coords)
        same=np.where(sep.value*3600.<samestar)[0]
        hecto_nobs[same]=len(same)
        ggg=0
        use=np.zeros(len(same),dtype='int')
        for j in range(0,len(same)):
            hecto_obs[same[j]]=j+1
            if same[j] in hecto_good:
                use[j]=1
                ggg=ggg+1
                hecto_goodobs[same[j]]=ggg
        hecto_ngoodobs[same]=ggg
        keep=np.where(use==1)[0]
        if len(keep)>0:
            hecto_mean_v0,hecto_sig_mean_v0=mycode.weightedmean(hecto_v[same[keep]],hecto_sig_v[same[keep]])
            hecto_mean_teff0,hecto_sig_mean_teff0=mycode.weightedmean(hecto_teff[same[keep]],hecto_sig_teff[same[keep]])
            hecto_mean_logg0,hecto_sig_mean_logg0=mycode.weightedmean(hecto_logg[same[keep]],hecto_sig_logg[same[keep]])
            hecto_mean_z0,hecto_sig_mean_z0=mycode.weightedmean(hecto_z[same[keep]],hecto_sig_z[same[keep]])
            hecto_mean_v[same]=hecto_mean_v0
            hecto_sig_mean_v[same]=hecto_sig_mean_v0
            hecto_mean_teff[same]=hecto_mean_teff0
            hecto_sig_mean_teff[same]=hecto_sig_mean_teff0
            hecto_mean_logg[same]=hecto_mean_logg0
            hecto_sig_mean_logg[same]=hecto_sig_mean_logg0
            hecto_mean_z[same]=hecto_mean_z0
            hecto_sig_mean_z[same]=hecto_sig_mean_z0
        

m2fs_keep=np.where(m2fs_goodobs==1)[0]
hecto_keep=np.where(hecto_goodobs==1)[0]

plt.scatter(m2fs_mean_v[m2fs_keep],m2fs_mean_z[m2fs_keep],c=m2fs_mean_logg[m2fs_keep],s=1,alpha=0.2,cmap='inferno',vmin=0,vmax=5)
plt.scatter(hecto_mean_v[hecto_keep],hecto_mean_z[hecto_keep],c=hecto_mean_logg[hecto_keep],s=1,alpha=0.2,cmap='inferno',vmin=0,vmax=5)
plt.xlim([-3000,3000])
plt.colorbar(cmap='inferno')
plt.xlabel(r'$v_{\rm helio}$ [km/s]')
plt.ylabel(r'[Fe/H]')
plt.savefig('field_of_halos.pdf',dpi=200)
plt.show()
plt.close()
