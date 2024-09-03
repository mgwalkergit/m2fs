import numpy as np
import astropy
from astropy import units
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
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
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/NovDec2018/'

with open(directory+m2fsrun+'_fibermap_raw') as f:
    data=f.readlines()[0:]
utdate=[]
file1=[]
file2=[]
flat=[]
thar=[]
field=[]
frames=[]
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
frames=np.array(frames)

for ccd in ('b','r'):
    master_fibermap=[]
    sig_master_fibermap=[]
    master_exptime=[]
    for i in range(0,len(utdate)):
        fibmap=[]
        sig_fibmap=[]
        exptime=[]
        for j in frames[i]:
            filename=datadir+utdate[i]+'/'+ccd+j.zfill(4)+'_stitched.fits'
            
            data=astropy.nddata.CCDData.read(filename,unit=u.electron)#header is in data.meta
            exptime.append(data.header['exptime'])
            print(filename,data.header['object'],data.header['binning'])
            fibmap.append(data)
            sig_fibmap.append(data.uncertainty._array)
        sig_fibmap=np.array(sig_fibmap)
        exptime=np.array(exptime)

        c=Combiner(fibmap)
        c.weights=1./sig_fibmap**2
#        ccdall=c.average_combine(uncertainty_func=mycode.stdmean)
        ccdall=c.median_combine()
        ccdall.header['exptime']=np.mean(exptime)
        master_exptime.append(np.mean(exptime))
        master_fibermap.append(ccdall)
            
        sig_master_fibermap.append(ccdall.uncertainty._array)
        ccdall.write(directory+m2fsrun+'_'+ccd+'_master_fibermap'+str(i+1)+'.fits',overwrite=True)
        iraf=astropy.nddata.CCDData(ccdall.data,unit=u.electron)
        iraf.header=ccdall.header
        hdu=fits.PrimaryHDU(ccdall.data)
        hdul=fits.HDUList([hdu])
        hdul.writeto(directory+m2fsrun+'_'+ccd+'_master_fibermap'+str(i+1)+'_iraf.fits',overwrite=True)
#        hdul=ccdall.to_hdu()
#        hdul.writeto(directory+m2fsrun+'_'+ccd+'_master_fibermap'+str(i+1)+'_iraf.fits',overwrite=True)

    sig_master_fibermap=np.array(sig_master_fibermap)
    master_exptime=np.array(master_exptime)
    c=Combiner(master_fibermap)
    c.weights=1./sig_master_fibermap**2
    ccdall=c.median_combine()
    ccdall.header['exptime']=np.mean(master_exptime)
    ccdall.write(directory+m2fsrun+'_'+ccd+'_master_fibermap.fits',overwrite=True)
#    hdul=ccdall.to_hdu()
#    hdul.writeto(directory+m2fsrun+'_'+ccd+'_master_fibermap'+'_iraf.fits',overwrite=True)    
    hdu=fits.PrimaryHDU(ccdall.data)
    hdul=fits.HDUList([hdu])
    hdul.writeto(directory+m2fsrun+'_'+ccd+'_master_fibermap'+'_iraf.fits',overwrite=True)
