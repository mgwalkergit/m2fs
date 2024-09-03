import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

dsph=fits.open('dsph_parameters.fits')
this_dsph=np.where(dsph[1].data['name']=='fornax_1')[0]
                   
gs=plt.GridSpec(22,22)
gs.update(wspace=0,hspace=0)
fig=plt.figure(figsize=(6,6))
ax1=fig.add_subplot(gs[0:11,0:11])

shite=fits.open('m2fs_HiRes_catalog.fits')[1].data
fehcut=-2.5
mp=np.where((shite.target_system=='for')&(shite.good_obs==1)&(shite.feh_mean<fehcut))[0]
mr=np.where((shite.target_system=='for')&(shite.good_obs==1)&(shite.feh_mean>=fehcut))[0]

axis_major=2.*2.*dsph[1].data['rhalf'][this_dsph]/np.sqrt(1.-dsph[1].data['ellipticity'][this_dsph])/60.#twice Rhalf
axis_minor=axis_major*(1.-dsph[1].data['ellipticity'][this_dsph])
ell=Ellipse(xy=[dsph[1].data.ra[this_dsph],dsph[1].data.dec[this_dsph]],height=axis_major,width=axis_minor,angle=-dsph[1].data['PA'][this_dsph],fc=None,ec='k',fill=False,linestyle='--',lw=0.5)
ax1.add_artist(ell)
ax1.scatter(shite.ra[mr],shite.dec[mr],s=1,color='r',alpha=0.3,label=r'[Fe/H]$\geq -2.5$',rasterized=True)
ax1.scatter(shite.ra[mp],shite.dec[mp],s=5,color='b',alpha=0.3,label=r'[Fe/H]$< -2.5$',rasterized=True)
ax1.legend(loc=3,fontsize=7)
ax1.set_xlabel(r'R.A. [deg.]')
ax1.set_ylabel(r'Dec. [deg.]')
ax1.set_title('Fornax (M2FS)')
ax1.set_xlim([41.5,38.5])
ax1.set_ylim([-35.8,-33.2])
plt.savefig('fornax_scatter.pdf',dpi=300)
plt.show()
plt.close()
