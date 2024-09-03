import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
#matplotlib.use('pdf')

scl=fits.open('all_m2fshires.fits')
old=fits.open('scl_m2fs.fits')
sex=fits.open('../nelson/nelson_sex.fits')
mike=fits.open('../mike/mike_data.fits')

scl_v=scl[1].data['vlos']
scl_sigv=scl[1].data['vlos_error']
scl_skewv=scl[1].data['vlos_skew']
scl_kurtv=scl[1].data['vlos_kurtosis']
scl_teff=scl[1].data['teff']
scl_sigteff=scl[1].data['teff_error']
scl_logg=scl[1].data['logg']
scl_siglogg=scl[1].data['logg_error']
scl_z=scl[1].data['z']
scl_sigz=scl[1].data['z_error']
scl_alpha=scl[1].data['alpha']
scl_sigalpha=scl[1].data['alpha_error']
scl_chi2=scl[1].data['chi2']
scl_sigma=scl[1].data['resolution_sigma']

sex_v=sex[1].data['vlos']
sex_sigv=sex[1].data['vlos_error']
sex_skewv=sex[1].data['vlos_skew']
sex_kurtv=sex[1].data['vlos_kurtosis']
sex_teff=sex[1].data['teff']
sex_sigteff=sex[1].data['teff_error']
sex_logg=sex[1].data['logg']
sex_siglogg=sex[1].data['logg_error']
sex_z=sex[1].data['z']
sex_sigz=sex[1].data['z_error']
sex_alpha=sex[1].data['alpha']
sex_sigalpha=sex[1].data['alpha_error']
sex_chi2=sex[1].data['chi2']
sex_sigma=sex[1].data['resolution_sigma']

old_v=old[1].data['vlos']
old_sigv=old[1].data['vlos_error']
old_skewv=old[1].data['vlos_skew']
old_kurtv=old[1].data['vlos_kurtosis']
old_teff=old[1].data['teff']
old_sigteff=old[1].data['teff_error']
old_logg=old[1].data['logg']
old_siglogg=old[1].data['logg_error']
old_z=old[1].data['z']
old_sigz=old[1].data['z_error']
old_chi2=old[1].data['chi2']
old_sigma=old[1].data['resolution_sigma']

mike_v=mike[1].data['vlos']
mike_sigv=mike[1].data['vlos_error']
mike_skewv=mike[1].data['vlos_skew']
mike_kurtv=mike[1].data['vlos_kurtosis']
mike_teff=mike[1].data['teff']
mike_sigteff=mike[1].data['teff_error']
mike_logg=mike[1].data['logg']
mike_siglogg=mike[1].data['logg_error']
mike_z=mike[1].data['z']
mike_sigz=mike[1].data['z_error']
mike_alpha=mike[1].data['alpha']
mike_sigalpha=mike[1].data['alpha_error']

#keep=np.where((scl_sigv<5.)&(np.abs(scl_skewv)<1.)&(np.abs(scl_kurtv)<1.)*(old_sigv<5.)&(np.abs(old_skewv)<1.)&(np.abs(old_kurtv)<1.))[0]
scl_keep=np.where((scl_sigv<5.)&(np.abs(scl_skewv)<1.)&(np.abs(scl_kurtv)<1.))[0]
scl_keep2=np.where((scl_sigv<1.)&(np.abs(scl_skewv)<1.)&(np.abs(scl_kurtv)<1.))[0]
old_keep=np.where((old_sigv<5.)&(np.abs(old_skewv)<1.)&(np.abs(old_kurtv)<1.))[0]
#keep1=np.where((scl_sigv<1.)&(np.abs(scl_skewv)<1.)&(np.abs(scl_kurtv)<1.)&(old_sigv<1.)&(np.abs(old_skewv)<1.)&(np.abs(old_kurtv)<1.))[0]
#keep2=np.where((scl_sigv<5.)&(np.abs(scl_skewv)<1.)&(np.abs(scl_kurtv)<1.)&(old_sigv<5.)&(np.abs(old_skewv)<1.)&(np.abs(old_kurtv)<1.)&(old_teff>4800)&(old_teff<5000))[0]

scl_mem=np.where((scl_sigv<5.)&(np.abs(scl_skewv)<1.)&(np.abs(scl_kurtv)<1.)&(np.abs(scl_v-110.)<30.))[0]
scl_non=np.where((scl_sigv<5.)&(np.abs(scl_skewv)<1.)&(np.abs(scl_kurtv)<1.)&(np.abs(scl_v-110.)>30.))[0]
sex_mem=np.where((sex_sigv<5.)&(np.abs(sex_skewv)<1.)&(np.abs(sex_kurtv)<1.)&(np.abs(sex_v-224.)<30.))[0]
sex_non=np.where((sex_sigv<5.)&(np.abs(sex_skewv)<1.)&(np.abs(sex_kurtv)<1.)&(np.abs(sex_v-224.)>30.))[0]

scl_mem2=np.where((scl_sigv<1.)&(np.abs(scl_skewv)<1.)&(np.abs(scl_kurtv)<1.)&(np.abs(scl_v-110.)<30.))[0]
scl_non2=np.where((scl_sigv<1.)&(np.abs(scl_skewv)<1.)&(np.abs(scl_kurtv)<1.)&(np.abs(scl_v-110.)>30.))[0]
sex_mem2=np.where((sex_sigv<1.)&(np.abs(sex_skewv)<1.)&(np.abs(sex_kurtv)<1.)&(np.abs(sex_v-224.)<30.))[0]
sex_non2=np.where((sex_sigv<1.)&(np.abs(sex_skewv)<1.)&(np.abs(sex_kurtv)<1.)&(np.abs(sex_v-224.)>30.))[0]

mike_keep=np.where((mike_sigv<5.)&(np.abs(mike_skewv)<1.)&(np.abs(mike_kurtv)<1.)&(np.abs(mike_v-110.)<30000.))[0]

#plt.scatter(scl_z[keep],old_z[keep],s=3)
#plt.xlabel('Ian fe/h [K]')
#plt.ylabel(r'old fe/h [K]$')
#plt.xlim([-4,1])
#plt.ylim([-4,1])
#plt.show()
#plt.close()

#plt.plot([-5,],[-5,5],color='k',linestyle='--')
#plt.scatter(scl_alpha[keep],scl_z[keep]-old_z[keep],s=3,color='0.7')
#plt.scatter(scl_alpha[keep2],scl_z[keep2]-old_z[keep2],s=3,color='r')
#plt.xlabel('Ian [alpha/Fe] ')
#plt.ylabel(r'Ian Fe/H - SSPP [Fe/H]')
#plt.xlim([-1,1])
#plt.ylim([-1,1])
#plt.savefig('scl_compare.pdf',dpi=200)
#plt.show()
#plt.close()

plt.plot([-10,10],[0,0],linestyle='--',color='k')
plt.scatter(scl[1].data['alpha'][keep],scl[1].data['chi2'][keep]/scl[1].data['n'][keep]-old[1].data['chi2'][keep]/old[1].data['n'][keep],s=1,alpha=0.3,color='k')
plt.ylim([-0.03,0.03])
plt.xlim([-1,1])
plt.xlabel('Ian [Mg/Fe]')
plt.ylabel(r'$(\chi^2_{\rm Ian}-\chi^2_{\rm SSPP})/N$')
plt.savefig('ian_compare2.pdf',dpi=200)
plt.show()
plt.close()




fig,axs=plt.subplots(3,1,sharex=True,sharey=True)

axs[0].scatter(scl_z[scl_non],scl_alpha[scl_non],s=3,color='k',label='MW',rasterized=True)
axs[0].scatter(sex_z[sex_non],sex_alpha[sex_non],s=3,color='k',rasterized=True)
axs[0].scatter(scl_z[scl_mem],scl_alpha[scl_mem],s=3,color='b',label='Scl',rasterized=True)
axs[0].scatter(sex_z[sex_mem],sex_alpha[sex_mem],s=3,color='r',label='Sex',rasterized=True)

axs[1].scatter(scl_z[scl_non2],scl_alpha[scl_non2],s=3,color='k',label='MW',rasterized=True)
axs[1].scatter(sex_z[sex_non2],sex_alpha[sex_non2],s=3,color='k',rasterized=True)
axs[1].scatter(scl_z[scl_mem2],scl_alpha[scl_mem2],s=3,color='b',label='Scl',rasterized=True)
axs[1].scatter(sex_z[sex_mem2],sex_alpha[sex_mem2],s=3,color='r',label='Sex',rasterized=True)
axs[1].legend()

axs[2].scatter(mike_z[mike_keep],mike_alpha[mike_keep],s=3,color='k',label='MIKE',rasterized=True)
axs[2].legend()

axs[0].set_xlim([-4,1])
axs[0].set_ylim([-1,1])
axs[0].set_xlabel('[Fe/H]')
axs[0].set_ylabel(r'[alpha/Fe]')

axs[2].set_xlim([-4,1])
axs[2].set_ylim([-1,1])
axs[2].set_xlabel('[Fe/H]')
axs[2].set_ylabel(r'[alpha/Fe]')

axs[1].set_xlim([-4,1])
axs[1].set_ylim([-1,1])
#axs[1].set_xlabel('[Fe/H]')
axs[1].set_ylabel(r'[alpha/Fe]')
#plt.xlim([-4,1])
#plt.ylim([-1,1])
plt.savefig('alpha_vs_fe.pdf',dpi=200)
plt.show()
plt.close()



plt.show()
plt.close()
