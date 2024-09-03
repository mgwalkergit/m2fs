import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits 
from scipy import stats
import random
import mycode2
np.set_printoptions(threshold='nan')

#plt.rc('grid',alpha=0.7) # modify rcparams
#plt.rc('grid',color='white')
#plt.rc('grid',linestyle='-')
plt.rc('legend',frameon='True')
plt.rc('xtick.major',size=2)
plt.rc('ytick.major',size=2)
plt.rc('axes',axisbelow='True')
plt.rc('axes',grid='False')
plt.rc('axes',facecolor='white')
plt.rc('axes',linewidth=0.5)
plt.rc('xtick.major',width=0.5)
plt.rc('ytick.major',width=0.5)
plt.rc('xtick',direction='in')
plt.rc('ytick',direction='in')
plt.rc('xtick',labelsize='6')
plt.rc('ytick',labelsize='6')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='eps')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)
            
data1='cra_compare.dat'

with open(data1) as f: # read data file
    data=f.readlines()

v1=[]
sigv1=[]
v2=[]
sigv2=[]
feh1=[]
sigfeh1=[]
feh2=[]
sigfeh2=[]
gmag1=[]
rmag1=[]
imag1=[]

for line in data: # fill arrays
    p=line.split()

    v1.append(float(p[0]))
    sigv1.append(float(p[1]))
    v2.append(float(p[2]))
    sigv2.append(float(p[3]))
    feh1.append(float(p[4]))
    sigfeh1.append(float(p[5]))
    feh2.append(float(p[6]))
    sigfeh2.append(float(p[7]))
    gmag1.append(float(p[8]))
    rmag1.append(float(p[9]))
    imag1.append(float(p[10]))

v1=np.array(v1)
sigv1=np.array(sigv1)
v2=np.array(v2)
sigv2=np.array(sigv2)
feh1=np.array(feh1)
sigfeh1=np.array(sigfeh1)
feh2=np.array(feh2)
sigfeh2=np.array(sigfeh2)
gmag1=np.array(gmag1)
rmag1=np.array(rmag1)
imag1=np.array(imag1)



data1='cra_compare2.dat'

with open(data1) as f: # read data file
    data=f.readlines()

v1b=[]
sigv1b=[]
v2b=[]
sigv2b=[]
teff1b=[]
sigteff1b=[]
teff2b=[]
sigteff2b=[]
logg1b=[]
siglogg1b=[]
logg2b=[]
siglogg2b=[]
feh1b=[]
sigfeh1b=[]
feh2b=[]
sigfeh2b=[]
gmag1b=[]
rmag1b=[]
imag1b=[]

for line in data: # fill arrays
    p=line.split()

    v1b.append(float(p[0]))
    sigv1b.append(float(p[1]))
    v2b.append(float(p[2]))
    sigv2b.append(float(p[3]))
    teff1b.append(float(p[4]))
    sigteff1b.append(float(p[5]))
    teff2b.append(float(p[6]))
    sigteff2b.append(float(p[7]))
    logg1b.append(float(p[8]))
    siglogg1b.append(float(p[9]))
    logg2b.append(float(p[10]))
    siglogg2b.append(float(p[11]))
    feh1b.append(float(p[12]))
    sigfeh1b.append(float(p[13]))
    feh2b.append(float(p[14]))
    sigfeh2b.append(float(p[15]))
    gmag1b.append(float(p[16]))
    rmag1b.append(float(p[17]))
    imag1b.append(float(p[18]))

v1b=np.array(v1b)
sigv1b=np.array(sigv1b)
v2b=np.array(v2b)
sigv2b=np.array(sigv2b)
teff1b=np.array(teff1b)
sigteff1b=np.array(sigteff1b)
teff2b=np.array(teff2b)
sigteff2b=np.array(sigteff2b)
logg1b=np.array(logg1b)
siglogg1b=np.array(siglogg1b)
logg2b=np.array(logg2b)
siglogg2b=np.array(siglogg2b)
feh1b=np.array(feh1b)
sigfeh1b=np.array(sigfeh1b)
feh2b=np.array(feh2b)
sigfeh2b=np.array(sigfeh2b)
gmag1b=np.array(gmag1b)
rmag1b=np.array(rmag1b)
imag1b=np.array(imag1b)


gs=plt.GridSpec(18,18) # define multi-panel plot
gs2=plt.GridSpec(20,20) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_
ax0_0=fig.add_subplot(gs[0:3,0:3])
ax0_1=fig.add_subplot(gs[0:3,5:8])
ax0_2=fig.add_subplot(gs[0:3,10:13])
ax0_3=fig.add_subplot(gs[0:3,15:18])
ax1_0=fig.add_subplot(gs[10:13,0:3])
ax1_1=fig.add_subplot(gs[10:13,5:8])
ax1_2=fig.add_subplot(gs[10:13,10:13])
ax1_3=fig.add_subplot(gs[10:13,15:18])

ax0_0.set_xlabel(r'$v_{\rm los,M2FS}$ [km/s]',fontsize=10,rotation=0,labelpad=5)
ax0_0.set_ylabel(r'$v_{\rm los,other}$ [km/s]',fontsize=10,rotation=90,labelpad=5)
ax0_0.set_xlim([50,275])
ax0_0.set_ylim([50,275])
ax0_0.set_xscale(u'linear')
ax0_0.set_yscale(u'linear')
ax0_0.set_yticks([50,100,150,200,250])
#ax0_0.set_xticklabels(vticks,rotation=0)
ax0_0.set_xticks([50,100,150,200,250])
#ax0_0.set_yticklabels([4000]+teffticks,rotation=0)
ax0_0.plot([-1000,1000],[-1000,1000],linewidth=0.5,linestyle=':',color='k')
ax0_0.errorbar(v1,v2,xerr=sigv1,yerr=sigv2,elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='k',rasterized=True,label='Deimos')
ax0_0.errorbar(v1b,v2b,xerr=sigv1b,yerr=sigv2b,elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='X-shooter')
ax0_0.legend(loc=2,fontsize=4,handlelength=0,numpoints=1,scatterpoints=1,shadow=False,borderpad=0.75)


ax1_0.set_xlabel(r'$v_{\rm los,M2FS}$ [km/s]',fontsize=10,rotation=0,labelpad=5)
ax1_0.set_ylabel(r'$v_{\rm los,other}-v_{\rm los,M2FS}$ [km/s]',fontsize=10,rotation=90,labelpad=5)
ax1_0.set_xlim([50,275])
ax1_0.set_ylim([-25,60])
ax1_0.set_xscale(u'linear')
ax1_0.set_yscale(u'linear')
ax1_0.set_yticks([-25,0,25,50])
#ax1_0.set_xticklabels(vticks,rotation=0)
ax1_0.set_xticks([50,100,150,200,250])
#ax1_0.set_yticklabels([4000]+teffticks,rotation=0)
ax1_0.plot([-1000,1000],[0,0],linewidth=0.5,linestyle=':',color='k')
ax1_0.errorbar(v1,v2-v1,xerr=sigv2,yerr=np.sqrt(sigv1**2+sigv2**2),elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='k',rasterized=True,label='Deimos')
ax1_0.errorbar(v1b,v2b-v1b,xerr=sigv2b,yerr=np.sqrt(sigv1b**2+sigv2b**2),elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='X-shooter')
ax1_0.legend(loc=2,fontsize=4,handlelength=0,numpoints=1,scatterpoints=1,shadow=False,borderpad=0.75)

ax0_3.set_xlabel(r'[Fe/H]$_{\rm M2FS}$',fontsize=10,rotation=0,labelpad=5)
ax0_3.set_ylabel(r'[Fe/H]$_{\rm other}$',fontsize=10,rotation=90,labelpad=5)
ax0_3.set_xlim([-3,-0.5])
ax0_3.set_ylim([-3,-0.5])
ax0_3.set_xscale(u'linear')
ax0_3.set_yscale(u'linear')
ax0_3.set_yticks([-3,-2,-1])
#ax0_3.set_xticklabels(vticks,rotation=0)
ax0_3.set_xticks([-3,-2,-1])
#ax0_3.set_yticklabels([4000]+teffticks,rotation=0)
ax0_3.plot([-1000,1000],[-1000,1000],linewidth=0.5,linestyle=':',color='k')
ax0_3.errorbar(feh1[sigfeh2 < 10.],feh2[sigfeh2 < 10.],xerr=sigfeh1[sigfeh2 < 10.],yerr=sigfeh2[sigfeh2 < 10.],elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='k',rasterized=True,label='Deimos')
ax0_3.errorbar(feh1b[sigfeh2b < 10.],feh2b[sigfeh2b < 10.],xerr=sigfeh1b[sigfeh2b < 10.],yerr=sigfeh2b[sigfeh2b < 10.],elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='X-shooter')

ax1_3.set_xlabel(r'[Fe/H]$_{\rm M2FS}$',fontsize=8,rotation=0,labelpad=5)
ax1_3.set_ylabel(r'[Fe/H]$_{\rm other}-$[Fe/H]$_{\rm M2FS}$',fontsize=8,rotation=90,labelpad=2)
ax1_3.set_xlim([-3,-0.5])
ax1_3.set_ylim([-1,1])
ax1_3.set_xscale(u'linear')
ax1_3.set_yscale(u'linear')
ax1_3.set_yticks([-1,-0.5,0,0.5,1])
#ax1_3.set_xticklabels(vticks,rotation=0)
ax1_3.set_xticks([-3,-2,-1])
#ax1_3.set_yticklabels([4000]+teffticks,rotation=0)
ax1_3.plot([-1000,1000],[0,0],linewidth=0.5,linestyle=':',color='k')
ax1_3.errorbar(feh1[sigfeh2 < 10.],feh2[sigfeh2 < 10.]-feh1[sigfeh2 < 10.],xerr=sigfeh1[sigfeh2 < 10.],yerr=np.sqrt(sigfeh1[sigfeh2 < 10.]**2+sigfeh2[sigfeh2 < 10.]**2),elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='k',rasterized=True,label='Deimos')
ax1_3.errorbar(feh1b[sigfeh2b < 10.],feh2b[sigfeh2b < 10.]-feh1b[sigfeh2b < 10.],xerr=sigfeh1b[sigfeh2b < 10.],yerr=np.sqrt(sigfeh1b[sigfeh2b < 10.]**2+sigfeh2b[sigfeh2b < 10.]**2),elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='X-shooter')


ax0_1.set_xlabel(r'$T_{\rm eff,M2FS}$ [K]',fontsize=10,rotation=0,labelpad=2)
ax0_1.set_ylabel(r'$T_{\rm eff,other}$ [K]',fontsize=10,rotation=90,labelpad=2)
ax0_1.set_xlim([4000,5000])
ax0_1.set_ylim([4000,5000])
ax0_1.set_xscale(u'linear')
ax0_1.set_yscale(u'linear')
ax0_1.set_yticks([4000,4500,5000])
#ax0_1.set_xticklabels(vticks,rotation=0)
ax0_1.set_xticks([4000,4500,5000])
#ax0_1.set_yticklabels([4000]+teffticks,rotation=0)
ax0_1.plot([-10000,10000],[-10000,10000],linewidth=0.5,linestyle=':',color='k')
ax0_1.errorbar(teff1b,teff2b,xerr=sigteff1b,yerr=sigteff2b,elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='X-shooter')

ax1_1.set_xlabel(r'$T_{\rm eff,M2FS}$ [K]',fontsize=8,rotation=0,labelpad=2)
ax1_1.set_ylabel(r'$T_{\rm eff,other}-T_{\rm eff,M2FS}$ [K]',fontsize=8,rotation=90,labelpad=1)
ax1_1.set_xlim([4000,5000])
ax1_1.set_ylim([-500,500])
ax1_1.set_xscale(u'linear')
ax1_1.set_yscale(u'linear')
ax1_1.set_yticks([-500,0,500])
#ax1_1.set_xticklabels(vticks,rotation=0)
ax1_1.set_xticks([4000,4500,5000])
#ax1_1.set_yticklabels([4000]+teffticks,rotation=0)
ax1_1.plot([-10000,10000],[0,0],linewidth=0.5,linestyle=':',color='k')
ax1_1.errorbar(teff1b,teff2b-teff1b,xerr=sigteff1b,yerr=np.sqrt(sigteff1b**2+sigteff2b**2),elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='X-shooter')


ax0_2.set_xlabel(r'$\log g_{\rm M2FS}$',fontsize=10,rotation=0,labelpad=5)
ax0_2.set_ylabel(r'$\log g_{\rm other}$',fontsize=10,rotation=90,labelpad=5)
ax0_2.set_xlim([0,2])
ax0_2.set_ylim([0,2])
ax0_2.set_xscale(u'linear')
ax0_2.set_yscale(u'linear')
ax0_2.set_yticks([0,1,2,3])
#ax0_2.set_xticklabels(vticks,rotation=0)
ax0_2.set_xticks([0,1,2])
#ax0_2.set_yticklabels([4000]+loggticks,rotation=0)
ax0_2.plot([-10000,10000],[-10000,10000],linewidth=0.5,linestyle=':',color='k')
ax0_2.errorbar(logg1b,logg2b,xerr=siglogg1b,yerr=siglogg2b,elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='X-shooter')

ax1_2.set_xlabel(r'$\log g_{\rm M2FS}$',fontsize=8,rotation=0,labelpad=5)
ax1_2.set_ylabel(r'$\log g_{\rm other}-\log g_{\rm M2FS}$',fontsize=8,rotation=90,labelpad=0.5)
ax1_2.set_xlim([0,2])
ax1_2.set_ylim([-1.5,1.5])
ax1_2.set_xscale(u'linear')
ax1_2.set_yscale(u'linear')
ax1_2.set_yticks([-1.5,0,1.5])
#ax1_3.set_xticklabels(vticks,rotation=0)
ax1_2.set_xticks([-1,0,1])
#ax123.set_yticklabels([4000]+loggticks,rotation=0)
ax1_2.plot([-10000,10000],[0,0],linewidth=0.5,linestyle=':',color='k')
ax1_2.errorbar(logg1b,logg2b-logg1b,xerr=siglogg1b,yerr=np.sqrt(siglogg1b**2+siglogg2b**2),elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='X-shooter')



plotfilename='cra_compare.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
