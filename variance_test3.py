import astropy
import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
from astropy.nddata import NDData
from astropy.nddata import CCDData
import astropy.units as u
from astropy.modeling import models
import m2fs_process as m2fs
import specutils
from specutils.spectra import Spectrum1D
from astropy.nddata import StdDevUncertainty
matplotlib.use('TkAgg')

scatteredlightcorr_order=1
scatteredlightcorr_rejection_iterations=10
scatteredlightcorr_rejection_sigma=3.

def get_adu(source,bias,dark,rdnoise,n,gain):
    n_e=np.random.poisson(source+dark,size=n)
    n_adu0=n_e/gain
    n_adu=np.random.normal(loc=bias+n_adu0,scale=rdnoise,size=n)
    n_adu_int=np.round(n_adu).astype(int)
#    n_adu_int=np.trunc(n_adu)
    return n_adu_int

def get_image_adu(source,scattered,bias,dark,rdnoise,gain):
    n_e=np.random.poisson(source+scattered+dark)
    n_adu0=n_e/gain
    n_adu=np.random.normal(loc=bias+n_adu0,scale=rdnoise)
    n_adu_int=np.round(n_adu).astype(int)
#    n_adu_int=np.trunc(n_adu)
    return n_adu_int

nx=200
ny=200
nsample=nx*ny
ncalibrate_dark=250
ncalibrate_bias=250

gain=0.68#electrons/adu
rdnoise_adu=3.82#adu
bias_level=0.#adu
signal_dark=np.array([10.,0.,100.])#electrons
logsignal_source=np.linspace(-1,3,100)
signal_source=10.**logsignal_source#electrons

scattered0=1000.
scattered_x0=0.
scattered_y0=0.
scattered_gx=nx/100.
scattered_gy=ny/500.

source_signal=np.zeros((nx,ny),dtype='float')
scattered_signal=np.zeros((nx,ny),dtype='float')
mask=np.zeros((nx,ny),dtype='bool')#mask==False in between-aperture regions with only scattered light

for i in range(0,nx):
    odd=False
    if i & 1:
        odd=True
    for j in range(0,ny):
        scattered_signal[i,j]=scattered0+(i-scattered_x0)*scattered_gx+(j-scattered_y0)*scattered_gy
        if odd:
            source_signal[i,j]=100.
            mask[i,j]=True

readnoise_estimate=rdnoise_adu
dark_estimate=signal_dark[0]
bias_estimate=bias_level
            
image2d=get_image_adu(source=source_signal,scattered=scattered_signal,bias=bias_level,dark=signal_dark[0],rdnoise=rdnoise_adu,gain=gain)
count_estimate=image2d-bias_estimate-dark_estimate
count_estimate_e=count_estimate*gain

err_bias_estimate=0.
err_dark_estimate=0.

var_read_estimate=(readnoise_estimate*gain)**2
var_bias_estimate=(err_bias_estimate*gain)**2
var_dark_estimate=(err_dark_estimate*gain)**2

var_count_estimate_e=np.zeros((nx,ny),dtype='float')
for i in range(0,nx):
    for j in range(0,ny):
        var_count_estimate_e[i,j]=np.max([count_estimate_e[i,j]+dark_estimate*gain+2.+var_read_estimate+var_bias_estimate+var_dark_estimate,0.6*(var_read_estimate+var_bias_estimate+var_dark_estimate)])

count_estimate_e_ccd=astropy.nddata.CCDData(count_estimate_e*u.electron,mask=mask,uncertainty=StdDevUncertainty(np.sqrt(var_count_estimate_e)))
                      
scatteredlightfunc=m2fs.get_scatteredlightfunc(count_estimate_e_ccd,mask,scatteredlightcorr_order,scatteredlightcorr_rejection_iterations,scatteredlightcorr_rejection_sigma)
y,x=np.mgrid[:len(count_estimate_e_ccd.data),:len(count_estimate_e_ccd.data[0])]
scattered_model=CCDData(scatteredlightfunc.func(x,y)*u.electron,mask=np.full((len(count_estimate_e_ccd.data),len(count_estimate_e_ccd.data[0])),False,dtype=bool),uncertainty=StdDevUncertainty(np.full((len(count_estimate_e_ccd.data),len(count_estimate_e_ccd.data[0])),scatteredlightfunc.rms,dtype='float')))

piss=scattered_signal-scattered_model.data

#np.pause()

scattercorr=count_estimate_e_ccd.subtract(scattered_model)
shite=(scattercorr.data-source_signal)/scattercorr.uncertainty.quantity.value
#plt.imshow(shite)
plt.hist(shite[mask==True].flatten(),bins=30,density=True)
x=np.linspace(-4,4,100)
y=1./np.sqrt(2.*np.pi)*np.exp(-x**2)
plt.plot(x,y)
plt.show()
plt.close()
np.pause()
#bias_calibrate=get_adu(source=0.,bias=bias_level,dark=0.,rdnoise=rdnoise_adu,n=ncalibrate_bias,gain=gain)

#dark_calibrate=[]
#for j in range(0,len(signal_dark)):
#    dark_calibrate.append(get_adu(source=0.,bias=bias_level,dark=signal_dark[j],rdnoise=rdnoise_adu,n=ncalibrate_dark,gain=gain))
#dark_calibrate=np.array(dark_calibrate)

#bias_estimate=np.mean(bias_calibrate)
#err_bias_estimate=np.std(bias_calibrate)/np.sqrt(ncalibrate_bias)
#readnoise_estimate=np.std(bias_calibrate)

#dark_estimate=[]
#err_dark_estimate=[]
#for j in range(0,len(signal_dark)):
#    dark_estimate.append(np.mean(dark_calibrate[j]-bias_estimate))
#    err_dark_estimate.append(np.std(dark_calibrate[j]-bias_estimate)/np.sqrt(ncalibrate_dark))
#dark_estimate=np.array(dark_estimate)
#err_dark_estimate=np.array(err_dark_estimate)

chi2_obs=[]
chi2_obs_fudge=[]
chi2_true=[]

for i in range(0,len(signal_source)):

    bias_calibrate=get_adu(source=0.,bias=bias_level,dark=0.,rdnoise=rdnoise_adu,n=ncalibrate_bias,gain=gain)
    dark_calibrate=[]
    for j in range(0,len(signal_dark)):
        dark_calibrate.append(get_adu(source=0.,bias=bias_level,dark=signal_dark[j],rdnoise=rdnoise_adu,n=ncalibrate_dark,gain=gain))
    dark_calibrate=np.array(dark_calibrate)

    bias_estimate=np.mean(bias_calibrate)
    err_bias_estimate=np.std(bias_calibrate)/np.sqrt(ncalibrate_bias)
    readnoise_estimate=np.std(bias_calibrate)

    dark_estimate=[]
    err_dark_estimate=[]
    for j in range(0,len(signal_dark)):
        dark_estimate.append(np.mean(dark_calibrate[j]-bias_estimate))
        err_dark_estimate.append(np.std(dark_calibrate[j]-bias_estimate)/np.sqrt(ncalibrate_dark))
    dark_estimate=np.array(dark_estimate)
    err_dark_estimate=np.array(err_dark_estimate)

    print(i)
    count_estimate=[]
    for j in range(0,len(signal_dark)):
        count=get_adu(source=signal_source[i],bias=bias_level,dark=signal_dark[j],rdnoise=rdnoise_adu,n=nsample,gain=gain)
        count_estimate.append(count-bias_estimate-dark_estimate[j])
    count_estimate=np.array(count_estimate)

    count_estimate_e=count_estimate*gain
    
    var_dark=signal_dark
    var_source=signal_source[i]
    var_read=(rdnoise_adu*gain)**2
    var_read_estimate=(readnoise_estimate*gain)**2
    var_bias_estimate=(err_bias_estimate*gain)**2
    var_dark_estimate=(err_dark_estimate*gain)**2

    var_obs=[]
    var_obs_fudge=[]
    var_true=[]
    for j in range(0,len(signal_dark)):
        var_true.append(var_source+var_dark[j]+var_read+var_bias_estimate+var_dark_estimate[j])
        var_obs.append([np.max([count_estimate_e[j][q]+dark_estimate[j]*gain+var_read_estimate+var_bias_estimate+var_dark_estimate[j],var_read_estimate+var_bias_estimate+var_dark_estimate[j]]) for q in range(0,nsample)])
        var_obs_fudge.append([np.max([count_estimate_e[j][q]+dark_estimate[j]*gain+2.+var_read_estimate+var_bias_estimate+var_dark_estimate[j],0.6*(var_read_estimate+var_bias_estimate+var_dark_estimate[j])]) for q in range(0,nsample)])
    var_obs=np.array(var_obs)
    var_obs_fudge=np.array(var_obs_fudge)
    var_true=np.array(var_true)

    chi2_obs0=[]
    chi2_obs_fudge0=[]
    chi2_true0=[]
    for j in range(0,len(signal_dark)):
        chi2_true0.append(np.sum((count_estimate_e[j]-signal_source[i])**2/var_true[j])/nsample)
        chi2_obs0.append(np.sum((count_estimate_e[j]-signal_source[i])**2/var_obs[j])/nsample)
        chi2_obs_fudge0.append(np.sum((count_estimate_e[j]-signal_source[i])**2/var_obs_fudge[j])/nsample)
    chi2_obs0=np.array(chi2_obs0)
    chi2_obs_fudge0=np.array(chi2_obs_fudge0)
    chi2_true0=np.array(chi2_true0)

    chi2_obs.append(chi2_obs0)
    chi2_obs_fudge.append(chi2_obs_fudge0)
    chi2_true.append(chi2_true0)

chi2_obs=np.transpose(np.array(chi2_obs))
chi2_obs_fudge=np.transpose(np.array(chi2_obs_fudge))
chi2_true=np.transpose(np.array(chi2_true))

gs=plt.GridSpec(6,6) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
ax1=fig.add_subplot(gs[0:2,0:3])
ax2=fig.add_subplot(gs[2:4,0:3])
ax3=fig.add_subplot(gs[4:6,0:3])

for j in range(0,len(signal_dark)):
#for j in range(2,3):
    if j==0:
        ax=ax1
    if j==1:
        ax=ax2
    if j==2:
        ax=ax3
    ax.plot(signal_source,chi2_true[j],color='k')#,label=r'$\mathrm{Var}=\overline{S}+\overline{D}+\sigma^2$')
    ax.plot(signal_source,chi2_obs[j],color='b')#,label=r'$\mathrm{Var}=\mathrm{max}[C-B+\sigma^2,\sigma^2]$')
    ax.plot(signal_source,chi2_obs_fudge[j],color='r')#,label=r'$\mathrm{Var}=\mathrm{max}[C-B+2\sigma^2,0.6\sigma^2]$')
    ax.set_xlim([0.1,1000])
    ax.set_xscale('log')
ax3.plot([0,100],[0,100],color='k',label=r'$\hat{\sigma}^2_N=N_{\rm in}+D_{\rm in}+\sigma^2$')
ax3.plot([0,100],[0,100],color='b',label=r'$\hat{\sigma}^2_N=\mathrm{max}[N+D+\sigma^2,\sigma^2]$')
ax3.plot([0,100],[0,100],color='r',label=r'$\hat{\sigma}^2_N=\mathrm{max}[N+D+2\sigma^2,0.6\sigma^2]$')
ax3.legend(loc=1,fontsize=6,borderaxespad=0)
ax1.text(80,0.85,r'$\sigma_R=2.6$ e$^-$',fontsize=8)
ax2.text(80,0.85,r'$\sigma_R=2.6$ e$^-$',fontsize=8)
ax3.text(80,0.85,r'$\sigma_R=2.6$ e$^-$',fontsize=8)
ax1.text(30,0.77,r'$D_{\rm in}=1$ e$^-$',fontsize=8)
ax2.text(30,0.77,r'$D_{\rm in}=10$ e$^-$',fontsize=8)
ax3.text(30,0.77,r'$D_{\rm in}=100$ e$^-$',fontsize=8)
ax1.tick_params(right=False,top=False,left=True,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=8)
ax1.tick_params(right=False,top=False,left=True,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',which='minor',length=4)
ax2.tick_params(right=False,top=True,left=True,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=8)
ax2.tick_params(right=False,top=True,left=True,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=4,which='minor')
ax3.tick_params(left=True,bottom=True,right=False,top=True,labelbottom=True,labelleft=True,reset=True,direction='inout',length=8)
ax3.tick_params(left=True,bottom=True,right=False,top=True,labelbottom=True,labelleft=True,reset=True,direction='inout',length=4,which='minor')
ax1.set_ylim([0.71,1.3])
ax2.set_ylim([0.71,1.29])
ax3.set_ylim([0.7,1.29])
ax3.set_xlabel(r'$N_{\rm in}$ [e$^-$]')
ax2.set_ylabel(r'$\overline{\chi_1^2}\equiv (N-N_{\rm in})^2/\hat{\sigma}^2_N$')
plt.savefig('variance_test.pdf',dpi=200)
plt.show()
plt.close()
