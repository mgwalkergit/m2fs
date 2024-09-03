import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import StdDevUncertainty
from astropy.nddata import CCDData
import ccdproc
import astropy.units as u
from astropy.modeling import models
from ccdproc import Combiner
import os
from os import path
import scipy
import m2fs_process as m2fs
import mycode
#matplotlib.use('TkAgg')
matplotlib.use('pdf')

with open('all_m2fs_files') as f:
    data=f.readlines()
fitsfile=[]
obj=[]
for line in data:
    p=line.split()
    fitsfile.append(p[0])
    obj.append(p[1])
fitsfile=np.array(fitsfile)
obj=np.array(obj)

dev=[]
for i in range(0,len(fitsfile)):
    print(i,len(fitsfile))
    cut=fitsfile[i].split('_20')
    fitsfile2=os.popen('ls '+cut[0]+'*stackskysub_noflat.fits').read().split('\n')[0]

    hdul=fits.open(fitsfile[i])
    hdul2=fits.open(fitsfile2)
    fitsobject=m2fs.m2fs_getfromfits(hdul)
    fitsobject2=m2fs.m2fs_getfromfits(hdul2)
    filtername=hdul[0].header['filtername']
    filtername2=hdul2[0].header['filtername']
    hdul.close()
    hdul2.close()

#    root=[]
#    for j in range(0,len(fitsobject.obj)):
#        root.append('m2fs_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3))
#    root=np.array(root)

    skies=np.where(fitsobject.obj=='SKY')[0]
    targets=np.where(fitsobject.icode>0)[0]
    skies2=np.where(fitsobject2.obj=='SKY')[0]
    targets2=np.where(fitsobject2.icode>0)[0]

    for j in range(0,len(targets)):
        this=np.where(fitsobject2.aperture==fitsobject.aperture[targets[j]])[0]
        if len(this)>0:
            print(fitsobject.snratio[targets[j]],fitsobject2.snratio[this][0])
            plt.scatter(np.log10(fitsobject.snratio[targets[j]]),np.log10(fitsobject2.snratio[this]),s=1,alpha=0.3,color='k',rasterized=True)

plt.plot([-10,10],[-10,10],linestyle=':',color='r')
plt.xlabel(r'$\log_{10}$[S/N/pixel], flattened')
plt.ylabel(r'$\log_{10}$[S/N/pixel], not flattened')
plt.xlim([-1,2])
plt.ylim([-1,2])
plt.savefig('yesflatnoflat.pdf',dpi=200)
#plt.show()
plt.close()
