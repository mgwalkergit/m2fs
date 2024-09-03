import astropy
from astropy import units
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import CCDData
import astropy.units as u
from astropy.modeling import models
import os
import mycode
import specutils
from specutils import Spectrum1D
import scipy
from astropy.nddata import StdDevUncertainty
from astropy.visualization import quantity_support
import numpy as np
from specutils.fitting import find_lines_threshold
from specutils.fitting import find_lines_derivative
from specutils.fitting import fit_lines
from specutils.analysis import centroid
from specutils.analysis import fwhm
from specutils.analysis import line_flux
from specutils.analysis import equivalent_width
from specutils import SpectralRegion
from specutils.manipulation import extract_region
from specutils.fitting import fit_generic_continuum
from astropy.modeling import models,fitting
from ccdproc import Combiner
matplotlib.use('pdf')
matplotlib.use('TkAgg')

n_apertures=128
n_lines=10#lines to combine
min_separation=5.#pixels, minimum separation between aperture centers
window=5#pixels, width of aperture window for fitting (gaussian) aperture profiles (perpendicular to spectral axis)
stddev=2.#pixels, initialization of standard deviation parameter for estimating stddev of aperture profile (perpendicular to spectral axis)
amplitude=1000.#electrons, initialization of amplitude for estimating gaussian fit to aperture profile (perpendicular to spectral axis)

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'
m2fsrun='nov18'
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/NovDec2018/'

utdate=[]
file1=[]
file2=[]
flat=[]
thar=[]
field=[]
frames=[]
with open(directory+m2fsrun+'_science_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(int(p[1]))
    file2.append(int(p[2]))
    flat.append(int(p[3]))
    thar.append(int(p[4]))
    field.append(str(p[5]))
    frames.append(str(p[6:]))
with open(directory+m2fsrun+'_fibermap_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(int(p[1]))
    file2.append(int(p[2]))
    flat.append(int(p[3]))
    thar.append(int(p[4]))
    field.append(str(p[5]))
    frames.append(str(p[6:]))
with open(directory+m2fsrun+'_twilight_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(int(p[1]))
    file2.append(int(p[2]))
    flat.append(int(p[3]))
    thar.append(int(p[4]))
    field.append(str(p[5]))
    frames.append(p[6:])
utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flat=np.array(flat)
thar=np.array(thar)
field=np.array(field)

for i in range(0,len(utdate)):
    data=astropy.nddata.CCDData.read(datadir+utdate[i]+'/b'+str(flat[i]).zfill(4)+'_stitched.fits')

    n_cols=np.shape(data)[1]
    col=np.arange(n_lines,dtype='int')+np.long(n_cols/2-n_lines/2)+1#identify the central n_lines columns to sum
    column=data.data[:,col]
    column_uncertainty=data.uncertainty[:,col]
    column_mask=data.mask[:,col]

    shite=[]
    sig_shite=[]
    for j in range(0,len(column[0])):
        shite.append(data[:,col[j]])
        sig_shite.append(data.uncertainty._array[:,col[j]])
    sig_shite=np.array(sig_shite)

    c=Combiner(shite)
    c.weights=1./sig_shite**2
    comb=c.average_combine(uncertainty_func=mycode.stdmean)

    lamb=(np.arange(len(comb.data),dtype='float'))*u.AA#unit is pixels, but specutils apparently can't handle that, so we lie and say Angs.
    spec1d=0.
    aperture_centers=0.
    apertures=0.
#spec1d=Spectrum1D(spectral_axis=lamb,flux=column*u.electron,uncertainty=column_uncertainty,mask=column_mask)
    spec1d=Spectrum1D(spectral_axis=lamb,flux=comb.data*u.electron,uncertainty=comb.uncertainty,mask=comb.mask)

    p_init=models.Polynomial1D(degree=10)
    fitter=fitting.LinearLSQFitter()
    x=lamb.value
    for q in range(0,10):
        y=np.ma.masked_array(spec1d.data,mask=spec1d.mask)
        p=fitter(p_init,x,y)
        outlier=(np.where(spec1d.data-p(x)>1.*spec1d.uncertainty.array))[0]
        print('rejected '+str(len(outlier))+' from continuum fit in iteration '+str(q+1))
        y.mask[outlier]=True

    apertures=find_lines_threshold(spec1d-p(x),noise_factor=5)

###identify false-positive apertures as those that violate minimum aperture separation
    aperture_centers=apertures['line_center'].value
    dcenters=[]
    for k in range(0,len(aperture_centers)-1):
        dcenters.append(aperture_centers[k+1]-aperture_centers[k])
    dcenters=np.array(dcenters)
    keep_apertures=np.concatenate(((dcenters>min_separation),np.array([True])))
    apertures=apertures[keep_apertures]
    keep_apertures=np.array(keep_apertures)

    print('found '+str(len(apertures))+' apertures in column '+str(col))

    gs=plt.GridSpec(10,10) # define multi-panel plot
    gs.update(wspace=0,hspace=0) # specify inter-panel spacing
    fig=plt.figure(figsize=(6,6)) # define plot size
    ax1=fig.add_subplot(gs[0:4,0:10])
    ax2=fig.add_subplot(gs[6:10,0:4])
    ax3=fig.add_subplot(gs[6:10,6:10])

    ax1.plot(spec1d.spectral_axis,spec1d.flux,lw=0.1,color='k')
    ax1.plot(spec1d.spectral_axis.value,p(x),color='r')
    ax1.set_xlim([0,np.max(lamb.value)])
    eqw=[]
    std=[]#
    mean=[]
    for j in range(0,len(apertures)):
        g_init=models.Gaussian1D(amplitude=amplitude*u.electron,mean=apertures[j][0].value*u.AA,stddev=stddev*u.AA)
        g_fit=fit_lines(spec1d,g_init,window=window*u.AA)
        val1=apertures[j][0].value-window/2.
        val2=apertures[j][0].value+window/2.
        sub_region=SpectralRegion(val1*u.AA,val2*u.AA)
        sub_spectrum=extract_region(spec1d,sub_region)
        y_fit=g_fit(sub_spectrum.spectral_axis)
        ax1.axvline(x=g_fit.mean.value,color='b',lw=0.1,alpha=0.7)
        ax1.fill_betweenx([0.,np.max(spec1d.data)],g_fit.mean.value-2.5*g_fit.stddev.value,g_fit.mean.value+2.5*g_fit.stddev.value,color='r',alpha=0.3)
        std.append(g_fit.stddev.value)
        mean.append(g_fit.mean.value)
        eqw.append(equivalent_width(spec1d,regions=sub_region).value)
        ax1.plot(sub_spectrum.spectral_axis,y_fit,color='r',lw=0.05,alpha=0.7)
        diff=g_fit.mean.value-apertures[j][0].value
#    mean=np.array(mean)
#    std=np.array(std)
#    eqw=np.array(eqw)
#
    ax2.imshow(data,vmin=0,vmax=1000,cmap='gray')
    ax2.axvline(x=col[0],color='r',lw=1,alpha=0.5)
#    ax3.scatter(np.arange(len(apertures)),std,s=5,color='k')
    plt.savefig('test'+str(col)+'.pdf',dpi=200)
    plt.show()
    plt.close()
