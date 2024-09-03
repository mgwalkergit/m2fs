import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits 
from scipy import stats
import random
import matplotlib.mlab as mlab
#import mycode2
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
plt.rc('xtick',labelsize='8')
plt.rc('ytick',labelsize='8')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='eps')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)
            
data1='ret2_subexposures2.dat'
with open(data1) as f: # read data file
    data=f.readlines()

v1=[]
sigv1=[]
v2=[]
sigv2=[]
teff1=[]
sigteff1=[]
teff2=[]
sigteff2=[]
logg1=[]
siglogg1=[]
logg2=[]
siglogg2=[]
feh1=[]
sigfeh1=[]
feh2=[]
sigfeh2=[]
pmem=[]
id1=[]
obs2=[]

for line in data: # fill arrays
    p=line.split()
    v1.append(float(p[0]))
    sigv1.append(float(p[1]))
    v2.append(float(p[2]))
    sigv2.append(float(p[3]))
    teff1.append(float(p[4]))
    sigteff1.append(float(p[5]))
    teff2.append(float(p[6]))
    sigteff2.append(float(p[7]))
    logg1.append(float(p[8]))
    siglogg1.append(float(p[9]))
    logg2.append(float(p[10]))
    siglogg2.append(float(p[11]))
    feh1.append(float(p[12]))
    sigfeh1.append(float(p[13]))
    feh2.append(float(p[14]))
    sigfeh2.append(float(p[15]))
    pmem.append(float(p[16]))
    id1.append(float(p[17]))
    obs2.append(float(p[18]))

v1=np.array(v1)
sigv1=np.array(sigv1)
v2=np.array(v2)
sigv2=np.array(sigv2)
teff1=np.array(teff1)
sigteff1=np.array(sigteff1)
teff2=np.array(teff2)
sigteff2=np.array(sigteff2)
logg1=np.array(logg1)
siglogg1=np.array(siglogg1)
logg2=np.array(logg2)
siglogg2=np.array(siglogg2)
feh1=np.array(feh1)
sigfeh1=np.array(sigfeh1)
feh2=np.array(feh2)
sigfeh2=np.array(sigfeh2)
pmem=np.array(pmem)
id1=np.array(id1)
obs2=np.array(obs2)


gs=plt.GridSpec(10,10) # define multi-panel plot
gs.update(wspace=0.25,hspace=0.) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_

ax0_0=fig.add_subplot(gs[0:4,0:3])
ax0_1=fig.add_subplot(gs[0:4,3:6])
   
vlim=[0,300]
dvlim=[-11,11]

vlabel=[r'$v_{\rm los,stack}$ [km/s]']
dvlabel=[r'$v_{\rm los,stack}-v_{\rm los,exp}$ [km/s]']

ax0_0.set_xlabel(vlabel[0],fontsize=10,rotation=0)
ax0_0.set_ylabel(dvlabel[0],fontsize=10,rotation=90,labelpad=5)
ax0_0.set_xlim(vlim)
ax0_0.set_ylim(dvlim)
ax0_0.set_xscale(u'linear')
ax0_0.set_yscale(u'linear')
#ax0_0.set_xticks([0,100,200,300])
#ax0_0.set_yticks([0,100,200,300])
ax0_0.set_xticks([0,50,100,150,200,250])
ax0_0.set_yticks([-10,-5,0,5,10])
ax0_0.plot([-1000,1000],[0,0],linestyle=':',color='k',linewidth=0.25)
ax0_0.errorbar(v1[(id1 <= 91) & (obs2 == 1)],v1[(id1 <= 91) & (obs2 == 1)]-v2[(id1 <= 91) & (obs2 == 1)],xerr=sigv1[(id1 <= 91) & (obs2 == 1)],yerr=np.sqrt(sigv1[(id1 <= 91) & (obs2 == 1)]**2+sigv2[(id1 <= 91) & (obs2 == 1)]**2),elinewidth=0.5,fmt='.',capsize=0,alpha=1,color='b',rasterized=True,label='exposure 1')
ax0_0.errorbar(v1[(id1 <= 91) & (obs2 == 2)],v1[(id1 <= 91) & (obs2 == 2)]-v2[(id1 <= 91) & (obs2 == 2)],xerr=sigv1[(id1 <= 91) & (obs2 == 2)],yerr=np.sqrt(sigv1[(id1 <= 91) & (obs2 == 2)]**2+sigv2[(id1 <= 91) & (obs2 == 2)]**2),elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='g',rasterized=True,label='exposure 2')
ax0_0.errorbar(v1[(id1 <= 91) & (obs2 == 3)],v1[(id1 <= 91) & (obs2 == 3)]-v2[(id1 <= 91) & (obs2 == 3)],xerr=sigv1[(id1 <= 91) & (obs2 == 3)],yerr=np.sqrt(sigv1[(id1 <= 91) & (obs2 == 3)]**2+sigv2[(id1 <= 91) & (obs2 == 3)]**2),elinewidth=0.2,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='exposure 3')
ax0_0.legend(loc=4,fontsize=8,handlelength=0,numpoints=1,scatterpoints=1,shadow=False)
ax0_0.text(135,8,'blue channel',fontsize=9,color='k')

ax0_1.set_xlabel(vlabel[0],fontsize=10,rotation=0)
#ax0_1.set_ylabel(dvlabel[0],fontsize=10,rotation=90,labelpad=5)
ax0_1.set_xlim(vlim)
ax0_1.set_ylim(dvlim)
ax0_1.set_xscale(u'linear')
ax0_1.set_yscale(u'linear')
#ax0_1.set_xticks([0,100,200,300])
#ax0_1.set_yticks([0,100,200,300])
ax0_1.set_xticks([50,100,150,200,250,300])
#ax0_1.set_yticks([-10,-5,0,5,10])
ax0_1.yaxis.set_major_formatter(plt.NullFormatter())
ax0_1.plot([-1000,1000],[0,0],linestyle=':',color='k',linewidth=0.25)
ax0_1.errorbar(v1[(id1 > 91) & (obs2 == 1)],v1[(id1 > 91) & (obs2 == 1)]-v2[(id1 > 91) & (obs2 == 1)],xerr=sigv1[(id1 > 91) & (obs2 == 1)],yerr=np.sqrt(sigv1[(id1 > 91) & (obs2 == 1)]**2+sigv2[(id1 > 91) & (obs2 == 1)]**2),elinewidth=0.5,fmt='.',capsize=0,alpha=1,color='b',rasterized=True,label='exposure 1')
ax0_1.errorbar(v1[(id1 > 91) & (obs2 == 2)],v1[(id1 > 91) & (obs2 == 2)]-v2[(id1 > 91) & (obs2 == 2)],xerr=sigv1[(id1 > 91) & (obs2 == 2)],yerr=np.sqrt(sigv1[(id1 > 91) & (obs2 == 2)]**2+sigv2[(id1 > 91) & (obs2 == 2)]**2),elinewidth=0.35,fmt='.',capsize=0,alpha=1,color='g',rasterized=True,label='exposure 2')
ax0_1.errorbar(v1[(id1 > 91) & (obs2 == 3)],v1[(id1 > 91) & (obs2 == 3)]-v2[(id1 > 91) & (obs2 == 3)],xerr=sigv1[(id1 > 91) & (obs2 == 3)],yerr=np.sqrt(sigv1[(id1 > 91) & (obs2 == 3)]**2+sigv2[(id1 > 91) & (obs2 == 3)]**2),elinewidth=0.2,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='exposure 3')
ax0_1.legend(loc=3,fontsize=8,handlelength=0,numpoints=1,scatterpoints=1,shadow=False)
ax0_1.text(140,8,'red channel',fontsize=9,color='k')


plotfilename='ret2_subexposures.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
