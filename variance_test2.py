import astropy
import matplotlib
import matplotlib.pyplot as plt
import mycode
import scipy
import numpy as np
matplotlib.use('TkAgg')

def get_adu(source,bias,dark,rdnoise,n,gain):
    n_e=np.random.poisson(source+dark,size=n)
    n_adu0=n_e/gain
    n_adu=np.random.normal(loc=bias+n_adu0,scale=rdnoise,size=n)
    n_adu_int=np.round(n_adu).astype(int)
    return n_adu_int
    
nsample=10000
ncalibrate_dark=250
ncalibrate_bias=250

gain=0.68#electrons/adu
rdnoise_adu=3.82#adu
bias_level=500.#adu
signal_dark=0.#electrons
logsignal_source=np.linspace(-1,3,100)
signal_source=10.**logsignal_source#electrons

chi2_obs=[]
chi2_obs_fudge=[]
chi2_true=[]

for i in range(0,len(signal_source)):

    bias_calibrate=get_adu(source=0.,bias=bias_level,dark=0.,rdnoise=rdnoise_adu,n=ncalibrate_bias,gain=gain)
    dark_calibrate=get_adu(source=0.,bias=bias_level,dark=signal_dark,rdnoise=rdnoise_adu,n=ncalibrate_dark,gain=gain)

    bias_estimate=np.mean(bias_calibrate)
    err_bias_estimate=np.std(bias_calibrate)/np.sqrt(ncalibrate_bias)
    readnoise_estimate=np.std(bias_calibrate)

    dark_estimate=np.mean(dark_calibrate-bias_estimate)
    err_dark_estimate=np.std(dark_calibrate-bias_estimate)/np.sqrt(ncalibrate_dark)

    dark_estimate=signal_dark/gain
    err_dark_estimate=0.
    

    print(i)
    count=get_adu(source=signal_source[i],bias=bias_level,dark=signal_dark,rdnoise=rdnoise_adu,n=nsample,gain=gain)
    count_estimate=count-bias_estimate-dark_estimate
    count_estimate_e=count_estimate*gain
    
    var_dark=signal_dark
    var_source=signal_source[i]
    var_read=(rdnoise_adu*gain)**2
    var_read_estimate=(readnoise_estimate*gain)**2
    var_bias_estimate=(err_bias_estimate*gain)**2
    var_dark_estimate=(err_dark_estimate*gain)**2

    var_true=var_source+var_dark+var_read+var_bias_estimate+var_dark_estimate
    var_obs=[np.max([count_estimate_e[q]+dark_estimate*gain+var_read_estimate+var_bias_estimate+var_dark_estimate,var_read_estimate+var_bias_estimate+var_dark_estimate]) for q in range(0,nsample)]
    var_obs_fudge=[np.max([count_estimate_e[q]+dark_estimate*gain+2.+var_read_estimate+var_bias_estimate+var_dark_estimate,0.6*(var_read_estimate+var_bias_estimate+var_dark_estimate)]) for q in range(0,nsample)]

    chi2_true0=np.sum((count_estimate_e-signal_source[i])**2/var_true)/nsample
    chi2_obs0=np.sum((count_estimate_e-signal_source[i])**2/var_obs)/nsample
    chi2_obs_fudge0=np.sum((count_estimate_e-signal_source[i])**2/var_obs_fudge)/nsample

    chi2_obs.append(chi2_obs0)
    chi2_obs_fudge.append(chi2_obs_fudge0)
    chi2_true.append(chi2_true0)

chi2_obs=np.transpose(np.array(chi2_obs))
chi2_obs_fudge=np.transpose(np.array(chi2_obs_fudge))
chi2_true=np.transpose(np.array(chi2_true))

gs=plt.GridSpec(6,6) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
ax3=fig.add_subplot(gs[0:6,0:6])

ax=ax3
ax.plot(signal_source,chi2_true,color='k')#,label=r'$\mathrm{Var}=\overline{S}+\overline{D}+\sigma^2$')
ax.plot(signal_source,chi2_obs,color='b')#,label=r'$\mathrm{Var}=\mathrm{max}[C-B+\sigma^2,\sigma^2]$')
ax.plot(signal_source,chi2_obs_fudge,color='r')#,label=r'$\mathrm{Var}=\mathrm{max}[C-B+2\sigma^2,0.6\sigma^2]$')
ax.set_xlim([0.1,1000])
ax.set_xscale('log')
ax3.plot([0,100],[0,100],color='k',label=r'$\hat{\sigma}^2_N=N_{\rm in}+D_{\rm in}+\sigma^2$')
ax3.plot([0,100],[0,100],color='b',label=r'$\hat{\sigma}^2_N=\mathrm{max}[N+D+\sigma^2,\sigma^2]$')
ax3.plot([0,100],[0,100],color='r',label=r'$\hat{\sigma}^2_N=\mathrm{max}[N+D+2\sigma^2,0.6\sigma^2]$')
ax3.legend(loc=1,fontsize=6,borderaxespad=0)
ax3.text(80,0.85,r'$\sigma_R=2.6$ e$^-$',fontsize=8)
ax3.text(30,0.77,r'$D_{\rm in}=100$ e$^-$',fontsize=8)
ax3.tick_params(left=True,bottom=True,right=False,top=True,labelbottom=True,labelleft=True,reset=True,direction='inout',length=8)
ax3.tick_params(left=True,bottom=True,right=False,top=True,labelbottom=True,labelleft=True,reset=True,direction='inout',length=4,which='minor')
ax3.set_ylim([0.7,1.29])
ax3.set_xlabel(r'$N_{\rm in}$ [e$^-$]')
plt.savefig('variance_test2.pdf',dpi=200)
plt.show()
plt.close()
