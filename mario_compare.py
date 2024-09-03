import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle
matplotlib.use('TkAgg')

mario_spec1=fits.open('/nfs/nas-0-9/mgwalker.proj/m2fs/tuc2/Tuc2_HiRes/MayJun2017/Tuc20_1/r0093.norm.ss.fits')
mario_varspec1=fits.open('/nfs/nas-0-9/mgwalker.proj/m2fs/tuc2/Tuc2_HiRes/MayJun2017/Tuc20_1/r0093_var.norm.ss.fits')
mario_spec2=fits.open('/nfs/nas-0-9/mgwalker.proj/m2fs/tuc2/Tuc2_HiRes/MayJun2017/Tuc20_1/r0094.norm.ss.fits')
mario_varspec2=fits.open('/nfs/nas-0-9/mgwalker.proj/m2fs/tuc2/Tuc2_HiRes/MayJun2017/Tuc20_1/r0094_var.norm.ss.fits')
mario_spec3=fits.open('/nfs/nas-0-9/mgwalker.proj/m2fs/tuc2/Tuc2_HiRes/MayJun2017/Tuc20_1/r0097.norm.ss.fits')
mario_varspec3=fits.open('/nfs/nas-0-9/mgwalker.proj/m2fs/tuc2/Tuc2_HiRes/MayJun2017/Tuc20_1/r0097_var.norm.ss.fits')
mario_spec0=fits.open('/nfs/nas-0-9/mgwalker.proj/m2fs/tuc2/Tuc2_HiRes/MayJun2017/Tuc20_1/rTuc20_1.norm.ss.fits')
mario_varspec0=fits.open('/nfs/nas-0-9/mgwalker.proj/m2fs/tuc2/Tuc2_HiRes/MayJun2017/Tuc20_1/rTuc20_1_var.norm.ss.fits')

mario_wav=np.linspace(5130.,5188.811,925)

matt_spec1=pickle.load(open('/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2017/ut20170521/r0093_extract1d_array_noflat.pickle','rb'))
matt_spec2=pickle.load(open('/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2017/ut20170521/r0094_extract1d_array_noflat.pickle','rb'))
matt_spec3=pickle.load(open('/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2017/ut20170521/r0097_extract1d_array_noflat.pickle','rb'))
matt_wav1=pickle.load(open('/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2017/ut20170521/r0093_wavcal_array_noflat.pickle','rb'))
matt_wav2=pickle.load(open('/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2017/ut20170521/r0094_wavcal_array_noflat.pickle','rb'))
matt_wav3=pickle.load(open('/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2017/ut20170521/r0097_wavcal_array_noflat.pickle','rb'))
matt_spec0=pickle.load(open('/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2017/ut20170521/r_tuc2_medres0_may17_stack_array_noflat.pickle','rb'))
matt_wav0=pickle.load(open('/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2017/ut20170521/r_tuc2_medres0_may17_stack_wavcal_array_noflat.pickle','rb'))

gs=plt.GridSpec(12,10) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
ax1matt=fig.add_subplot(gs[0:3,0:5])
ax1mario=fig.add_subplot(gs[0:3,5:10])
ax2matt=fig.add_subplot(gs[3:6,0:5])
ax2mario=fig.add_subplot(gs[3:6,5:10])
ax3matt=fig.add_subplot(gs[6:9,0:5])
ax3mario=fig.add_subplot(gs[6:9,5:10])
ax0matt=fig.add_subplot(gs[9:12,0:5])
ax0mario=fig.add_subplot(gs[9:12,5:10])

ax0matt.plot(matt_wav0[2].wav[matt_spec0[2].spec1d_mask==False],matt_spec0[2].spec1d_flux.value[matt_spec0[2].spec1d_mask==False]*1.0+0.,lw=0.1,color='k')
ax0matt.fill_between(matt_wav0[2].wav[matt_spec0[2].spec1d_mask==False],matt_spec0[2].spec1d_flux.value[matt_spec0[2].spec1d_mask==False]*1.0+0.-matt_spec0[2].spec1d_uncertainty.quantity.value[matt_spec0[2].spec1d_mask==False]*1.0,matt_spec0[2].spec1d_flux.value[matt_spec0[2].spec1d_mask==False]*1.0+0.+matt_spec0[2].spec1d_uncertainty.quantity.value[matt_spec0[2].spec1d_mask==False]*1.0,color='r',alpha=0.1)
ax0mario.plot(mario_wav,mario_spec0[0].data[125],lw=0.1,color='k')
ax0matt.set_ylim([-50,150])
ax0mario.set_ylim([-50,150])
ax0mario.yaxis.set_major_formatter(plt.NullFormatter())
ax0mario.fill_between(mario_wav,mario_spec0[0].data[125]-np.sqrt(mario_varspec1[0].data[125]),mario_spec0[0].data[125]+np.sqrt(mario_varspec1[0].data[125]),color='r',alpha=0.1)

ax1matt.plot(matt_wav0[2].wav[matt_spec1[2].spec1d_mask==False],matt_spec1[2].spec1d_flux.value[matt_spec1[2].spec1d_mask==False]*1.0+10.,lw=0.1,color='k')
ax1matt.fill_between(matt_wav1[2].wav[matt_spec1[2].spec1d_mask==False],matt_spec1[2].spec1d_flux.value[matt_spec1[2].spec1d_mask==False]*1.0+10.-matt_spec1[2].spec1d_uncertainty.quantity.value[matt_spec1[2].spec1d_mask==False]*1.0,matt_spec1[2].spec1d_flux.value[matt_spec1[2].spec1d_mask==False]*1.0+10.+matt_spec1[2].spec1d_uncertainty.quantity.value[matt_spec1[2].spec1d_mask==False]*1.0,color='r',alpha=0.1)
ax1mario.plot(mario_wav,mario_spec1[0].data[125],lw=0.1,color='k')
ax1matt.set_ylim([-50,100])
ax1mario.set_ylim([-50,100])
ax1mario.yaxis.set_major_formatter(plt.NullFormatter())
ax1mario.fill_between(mario_wav,mario_spec1[0].data[125]-np.sqrt(mario_varspec1[0].data[125]),mario_spec1[0].data[125]+np.sqrt(mario_varspec1[0].data[125]),color='r',alpha=0.1)

ax2matt.plot(matt_wav2[2].wav[matt_spec2[2].spec1d_mask==False],matt_spec2[2].spec1d_flux.value[matt_spec2[2].spec1d_mask==False]*1.0+10.,lw=0.1,color='k')
ax2matt.fill_between(matt_wav2[2].wav[matt_spec2[2].spec1d_mask==False],matt_spec2[2].spec1d_flux.value[matt_spec2[2].spec1d_mask==False]*1.0+10.-matt_spec2[2].spec1d_uncertainty.quantity.value[matt_spec2[2].spec1d_mask==False]*1.0,matt_spec2[2].spec1d_flux.value[matt_spec2[2].spec1d_mask==False]*1.0+10.+matt_spec2[2].spec1d_uncertainty.quantity.value[matt_spec2[2].spec1d_mask==False]*1.0,color='r',alpha=0.1)
ax2mario.plot(mario_wav,mario_spec2[0].data[125],lw=0.1,color='k')
ax2matt.set_ylim([-50,100])
ax2mario.set_ylim([-50,100])
ax2mario.yaxis.set_major_formatter(plt.NullFormatter())
ax2mario.fill_between(mario_wav,mario_spec2[0].data[125]-np.sqrt(mario_varspec2[0].data[125]),mario_spec2[0].data[125]+np.sqrt(mario_varspec2[0].data[125]),color='r',alpha=0.1)

ax3matt.plot(matt_wav3[2].wav[matt_spec3[2].spec1d_mask==False],matt_spec3[2].spec1d_flux.value[matt_spec3[2].spec1d_mask==False]*1.0+10.,lw=0.1,color='k')
ax3matt.fill_between(matt_wav3[2].wav[matt_spec3[2].spec1d_mask==False],matt_spec3[2].spec1d_flux.value[matt_spec3[2].spec1d_mask==False]*1.0+10.-matt_spec3[2].spec1d_uncertainty.quantity.value[matt_spec3[2].spec1d_mask==False]*1.0,matt_spec3[2].spec1d_flux.value[matt_spec3[2].spec1d_mask==False]*1.0+10.+matt_spec3[2].spec1d_uncertainty.quantity.value[matt_spec3[2].spec1d_mask==False]*1.0,color='r',alpha=0.1)
ax3mario.plot(mario_wav,mario_spec3[0].data[125],lw=0.1,color='k')
ax3matt.set_ylim([-50,100])
ax3mario.set_ylim([-50,100])
ax3mario.yaxis.set_major_formatter(plt.NullFormatter())
ax3mario.fill_between(mario_wav,mario_spec3[0].data[125]-np.sqrt(mario_varspec3[0].data[125]),mario_spec3[0].data[125]+np.sqrt(mario_varspec3[0].data[125]),color='r',alpha=0.1)

plt.savefig('mario_compare.pdf',dpi=200)
plt.show()
plt.close()

