import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
#from astropy.nddata import NDData
from astropy.nddata import CCDData
import ccdproc
import astropy.units as u
from astropy.modeling import models
from ccdproc import Combiner
import os
import mycode
matplotlib.use('TkAgg')

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'
m2fsrun='nov18'
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/nov18/m2fs.astro.lsa.umich.edu/data/NovDec2018/'

utdate=[]
file1=[]
file2=[]
flatfile=[]
tharfile=[]
#with open(directory+m2fsrun+'_fibermap_raw') as f:
#    data=f.readlines()[0:]
#for line in data:
#    p=line.split()
#    utdate.append(str(p[0]))
#    file1.append(int(p[1]))
#    file2.append(int(p[2]))
#    flatfile.append(int(p[3]))
#    tharfile.append(int(p[4]))
#with open(directory+m2fsrun+'_twilight_raw') as f:
#    data=f.readlines()[0:]
#for line in data:
#    p=line.split()
#    utdate.append(str(p[0]))
#    file1.append(int(p[1]))
#    file2.append(int(p[2]))
#    flatfile.append(int(p[3]))
#    tharfile.append(int(p[4]))
with open(directory+m2fsrun+'_science_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(int(p[1]))
    file2.append(int(p[2]))
    flatfile.append(int(p[3]))
    tharfile.append(int(p[4]))
utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flatfile=np.array(flatfile)
tharfile=np.array(tharfile)
for i in range(0,len(utdate)):
    for j in range(file1[i],file2[i]+1):
        for ccd in ('b','r'):
            filename1=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+'_stitched.fits'
            filename2=datadir+utdate[i]+'/'+ccd+str(flatfile[i]).zfill(4)+'_apflatten.fits'
            data=astropy.nddata.CCDData.read(filename1,unit=u.adu)#header is in data.meta
            flat=astropy.nddata.CCDData.read(filename2,unit=u.adu)#header is in flat.meta
#            print(filename1,data.header['object'],data.header['binning'])
#            hdul=fits.PrimaryHDU(data.data,data.header)
#            hdul.writeto(out2,overwrite=True)
#            hdul=fits.PrimaryHDU(data.uncertainty.array**2,data.header)
#            hdul.writeto(out3,overwrite=True)
            plt.imshow(flat,vmin=0,vmax=2)
            plt.show()
            plt.close()

