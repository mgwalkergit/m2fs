import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import StdDevUncertainty
from astropy.nddata import CCDData
from astropy.coordinates import SkyCoord, EarthLocation
import ccdproc
import astropy.units as u
from astropy.modeling import models
from ccdproc import Combiner
import os
from os import path
import scipy
import time
import m2fs_process as m2fs
import mycode
matplotlib.use('TkAgg')

with open('tuc2_m2fshires_noflat.dat') as f:
    data=f.readlines()[1:]
radeg1=[]
decdeg1=[]
hjd1=[]
snratio1=[]
v1=[]
sigv1=[]
skewv1=[]
kurtv1=[]
teff1=[]
sigteff1=[]
skewteff1=[]
kurtteff1=[]
logg1=[]
siglogg1=[]
skewlogg1=[]
kurtlogg1=[]
z1=[]
sigz1=[]
skewz1=[]
kurtz1=[]
for line in data:
    p=line.split()
    radeg1.append(float(p[1]))
    decdeg1.append(float(p[2]))
    hjd1.append(float(p[3]))
    snratio1.append(float(p[4]))
    v1.append(float(p[5]))
    sigv1.append(float(p[6]))
    skewv1.append(float(p[7]))
    kurtv1.append(float(p[8]))
    teff1.append(float(p[9]))
    sigteff1.append(float(p[10]))
    skewteff1.append(float(p[11]))
    kurtteff1.append(float(p[12]))
    logg1.append(float(p[13]))
    siglogg1.append(float(p[14]))
    skewlogg1.append(float(p[15]))
    kurtlogg1.append(float(p[16]))
    z1.append(float(p[17]))
    sigz1.append(float(p[18]))
    skewz1.append(float(p[19]))
    kurtz1.append(float(p[20]))
radeg1=np.array(radeg1)
decdeg1=np.array(decdeg1)
hjd1=np.array(hjd1)
snratio1=np.array(snratio1)
v1=np.array(v1)
sigv1=np.array(sigv1)
skewv1=np.array(skewv1)
kurtv1=np.array(kurtv1)
teff1=np.array(teff1)
sigteff1=np.array(sigteff1)
skewteff1=np.array(skewteff1)
kurtteff1=np.array(kurtteff1)
logg1=np.array(logg1)
siglogg1=np.array(siglogg1)
skewlogg1=np.array(skewlogg1)
kurtlogg1=np.array(kurtlogg1)
z1=np.array(z1)
sigz1=np.array(sigz1)
skewz1=np.array(skewz1)
kurtz1=np.array(kurtz1)



with open('tuc2/tuc2.dat') as f:
    data=f.readlines()[1:]
radeg2=[]
decdeg2=[]
hjd2=[]
snratio2=[]
v2=[]
sigv2=[]
skewv2=[]
kurtv2=[]
teff2=[]
sigteff2=[]
skewteff2=[]
kurtteff2=[]
logg2=[]
siglogg2=[]
skewlogg2=[]
kurtlogg2=[]
z2=[]
sigz2=[]
skewz2=[]
kurtz2=[]
for line in data:
    p=line.split()
    radeg2.append(float(p[0]))
    decdeg2.append(float(p[1]))
    hjd2.append(float(p[2]))
    snratio2.append(float(p[3]))
    v2.append(float(p[4]))
    sigv2.append(float(p[5]))
    skewv2.append(float(p[6]))
    kurtv2.append(float(p[7]))
    teff2.append(float(p[8]))
    sigteff2.append(float(p[9]))
    skewteff2.append(float(p[10]))
    kurtteff2.append(float(p[11]))
    logg2.append(float(p[12]))
    siglogg2.append(float(p[13]))
    skewlogg2.append(float(p[14]))
    kurtlogg2.append(float(p[15]))
    z2.append(float(p[16]))
    sigz2.append(float(p[17]))
    skewz2.append(float(p[18]))
    kurtz2.append(float(p[19]))
radeg2=np.array(radeg2)
decdeg2=np.array(decdeg2)
hjd2=np.array(hjd2)
snratio2=np.array(snratio2)
v2=np.array(v2)
sigv2=np.array(sigv2)
skewv2=np.array(skewv2)
kurtv2=np.array(kurtv2)
teff2=np.array(teff2)
sigteff2=np.array(sigteff2)
skewteff2=np.array(skewteff2)
kurtteff2=np.array(kurtteff2)
logg2=np.array(logg2)
siglogg2=np.array(siglogg2)
skewlogg2=np.array(skewlogg2)
kurtlogg2=np.array(kurtlogg2)
z2=np.array(z2)
sigz2=np.array(sigz2)
skewz2=np.array(skewz2)
kurtz2=np.array(kurtz2)

this=[]
for i in range(0,len(radeg1)):
    dist=np.sqrt((radeg1[i]-radeg2)**2+(decdeg1[i]-decdeg2)**2)*3600.
    this0=np.where((dist<0.5)&(np.abs(hjd1[i]-hjd2)<0.5))[0]
    if len(this0)>0:
        this.append(np.where((dist<0.5)&(np.abs(hjd1[i]-hjd2)<0.5))[0][0])
    else:
        this.append(0)
gs=plt.GridSpec(10,10) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
ax1=fig.add_subplot(gs[0:4,0:4])
ax2=fig.add_subplot(gs[0:4,6:10])
#ax3=fig.add_subplot(gs[6:10,0:4])
#ax4=fig.add_subplot(gs[6:10,6:10])

ax1.scatter(logg1,logg2[this],color='k',s=1)
ax1.errorbar(logg1,logg2[this],xerr=siglogg1,yerr=siglogg2[this],fmt='.',capsize=0,mew=0.1,color='k',lw=0.3)
ax1.plot([-1000,1000],[-1000,1000],lw=1,linestyle=':',color='k')
ax1.set_xlim([0,5])
ax1.set_ylim([0,5])
ax1.set_xlabel('logg from mattraf [dex]',fontsize=8)
ax1.set_ylabel('logg from marioraf [dex]',fontsize=8)

ax2.scatter(siglogg1,siglogg2[this],color='k',s=1)
ax2.plot([-1000,1000],[-1000,1000],lw=1,linestyle=':',color='k')
ax2.set_xlim([0,2])
ax2.set_ylim([0,2])
ax2.set_xlabel('logg error from mattraf [dex]',fontsize=8)
ax2.set_ylabel('logg error from marioraf [dex]',fontsize=8)

plt.savefig('mario_compare4.pdf',dpi=200)
plt.show()
plt.close()
