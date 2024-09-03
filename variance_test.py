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
#    n_adu_int=np.trunc(n_adu)
    return n_adu_int
    
nsample=1000000
ncalibrate_dark=250
ncalibrate_bias=250

gain=0.68#electrons/adu
rdnoise_adu=3.82#adu
bias_level=0.#adu
signal_dark=np.array([0.])#electrons
logsignal_source=np.linspace(-1,3,100)
signal_source=10.**logsignal_source#electrons

chi2_obs=[]
chi2_obs_fudge=[]
chi2_true=[]

for i in range(0,len(signal_source)):

#    bias_calibrate=get_adu(source=0.,bias=bias_level,dark=0.,rdnoise=rdnoise_adu,n=ncalibrate_bias,gain=gain)
#    dark_calibrate=[]
#    for j in range(0,len(signal_dark)):
#        dark_calibrate.append(get_adu(source=0.,bias=bias_level,dark=signal_dark[j],rdnoise=rdnoise_adu,n=ncalibrate_dark,gain=gain))
#    dark_calibrate=np.array(dark_calibrate)

#    bias_estimate=np.mean(bias_calibrate)
#    err_bias_estimate=np.std(bias_calibrate)/np.sqrt(ncalibrate_bias)
#    readnoise_estimate=np.std(bias_calibrate)
    
#    dark_estimate=[]
#    err_dark_estimate=[]
#    for j in range(0,len(signal_dark)):
#        dark_estimate.append(np.mean(dark_calibrate[j]-bias_estimate))
#        err_dark_estimate.append(np.std(dark_calibrate[j]-bias_estimate)/np.sqrt(ncalibrate_dark))
#    dark_estimate=np.array(dark_estimate)
#    err_dark_estimate=np.array(err_dark_estimate)

    bias_estimate=bias_level
    err_bias_estimate=0.
    readnoise_estimate=rdnoise_adu
    dark_estimate=signal_dark
    err_dark_estimate=signal_dark-signal_dark
    
    print(i)
    count_estimate=[]
    for j in range(0,len(signal_dark)):
        count=get_adu(source=signal_source[i],bias=bias_level,dark=signal_dark[j],rdnoise=rdnoise_adu,n=nsample,gain=gain)
        count_estimate.append(count-bias_estimate-dark_estimate[j])
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
ax1=fig.add_subplot(gs[0:3,0:4])
#ax2=fig.add_subplot(gs[2:4,0:3])
#ax3=fig.add_subplot(gs[4:6,0:3])

for j in range(0,len(signal_dark)):
#for j in range(2,3):
    if j==0:
        ax=ax1
    if j==1:
        ax=ax2
    if j==2:
        ax=ax3
    ax.plot(signal_source,chi2_true[j],color='k')#,label=r'$\mathrm{Var}=\overline{S}+\overline{D}+\delta^2$')
    ax.plot(signal_source,chi2_obs[j],color='b')#,label=r'$\mathrm{Var}=\mathrm{max}[C-B+\delta^2,\delta^2]$')
    ax.plot(signal_source,chi2_obs_fudge[j],color='r')#,label=r'$\mathrm{Var}=\mathrm{max}[C-B+2\delta^2,0.6\delta^2]$')
    ax.set_xlim([0.1,1000])
    ax.set_xscale('log')
ax1.plot([0,100],[0,100],color='k',label=r'$\hat{\sigma}^2_S=S_{\rm in}+\delta^2$')
ax1.plot([0,100],[0,100],color='b',label=r'$\hat{\sigma}^2_S=\mathrm{max}[S+\delta^2,\delta^2]$')
ax1.plot([0,100],[0,100],color='r',label=r'$\hat{\sigma}^2_S=\mathrm{max}[S+2+\delta^2,0.6\delta^2]$')
ax1.legend(loc=4,fontsize=8)
#ax1.tick_params(right=False,top=False,left=True,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',length=8)
#ax1.tick_params(right=False,top=False,left=True,bottom=True,labelbottom=False,labelleft=True,labeltop=False,labelright=False,direction='inout',which='minor',length=4)
#ax3.tick_params(left=True,bottom=True,right=False,top=True,labelbottom=True,labelleft=True,reset=True,direction='inout',length=8)
#ax3.tick_params(left=True,bottom=True,right=False,top=True,labelbottom=True,labelleft=True,reset=True,direction='inout',length=4,which='minor')
ax1.set_ylim([0.7,1.3])
#ax2.set_ylim([0.71,1.29])
ax1.set_xlabel(r'$S_{\rm in}$ [e$^-$]')
ax1.set_ylabel(r'$\overline{\chi_1^2}\equiv (S-S_{\rm in})^2/\hat{\sigma}^2_S$')
plt.savefig('variance_test.pdf',dpi=200)
plt.show()
plt.close()
