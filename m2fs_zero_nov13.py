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
#m2fsrun='feb15'
#m2fsrun='nov18'
#m2fsrun='jul15'
#m2fsrun='may19'
#m2fsrun='may17'
#m2fsrun='nov16'
#m2fsrun='aug18'
#m2fsrun='aug19'
#m2fsrun='may18'
m2fsrun='nov13'
#m2fsrun='nov19'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Feb2015/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/NovDec2018/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Jul2015/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2019/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2017/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/NovDec2016/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Aug2018/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/AugSep2019/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/May2018/'
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Nov2013/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Nov2019/'

with open(directory+m2fsrun+'_bias_raw') as f:
    data=f.readlines()[0:]
utdate=[]
file1=[]
file2=[]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(int(p[1]))
    file2.append(int(p[2]))
utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)

for ccd in (['b','r']):
    for chip in (['c1','c2','c3','c4']):
        master_processed=[]
        sig_master_processed=[]
        for i in range(0,len(utdate)):
            processed=[]
            sig_processed=[]
            for j in range(file1[i],file2[i]+1):
                filename=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+chip+'.fits'
                
                data=astropy.nddata.CCDData.read(filename,unit=u.adu)#header is in data.meta
                print(filename,data.header['object'],data.header['binning'])

                oscan_subtracted=ccdproc.subtract_overscan(data,overscan=data[:,1024:],overscan_axis=1,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})
                trimmed1=ccdproc.trim_image(oscan_subtracted[:,:1024],add_keyword={'trim1':'Done'})
                trimmed2=ccdproc.trim_image(trimmed1[:1028,:1024],add_keyword={'trim2':'Done'})
                data_with_deviation=ccdproc.create_deviation(trimmed2,gain=data.meta['egain']*u.electron/u.adu,readnoise=data.meta['enoise']*u.electron)
                gain_corrected=ccdproc.gain_correct(data_with_deviation,data_with_deviation.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
                cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=5)
                sig_cr_cleaned=cr_cleaned.uncertainty._array

#                data_with_deviation=ccdproc.create_deviation(data,gain=data.meta['egain']*u.electron/u.adu,readnoise=data.meta['enoise']*u.electron)
#                gain_corrected=ccdproc.gain_correct(data_with_deviation,data_with_deviation.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
#                cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=5)
#                oscan_subtracted=ccdproc.subtract_overscan(cr_cleaned,overscan=cr_cleaned[:,1024:],overscan_axis=1,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})
#                trimmed1=ccdproc.trim_image(oscan_subtracted[:,:1024],add_keyword={'trim1':'Done'})
#                trimmed2=ccdproc.trim_image(trimmed1[:1028,:1024],add_keyword={'trim2':'Done'})
#                trimmed.append(trimmed2)
#                sig_trimmed.append(trimmed2.uncertainty._array)
                processed.append(cr_cleaned)
                sig_processed.append(sig_cr_cleaned)
                master_processed.append(cr_cleaned)
                sig_master_processed.append(sig_cr_cleaned)
#            processed=np.array(processed)
            sig_processed=np.array(sig_processed)

#            c=Combiner(trimmed)
            c=Combiner(processed)
#            c.weights=1./sig_trimmed**2
#            ccdall=c.average_combine(uncertainty_func=mycode.stdmean)
            ccdall=c.average_combine()
#            master_processed.append(ccdall)
#            sig_master_processed.append(ccdall.uncertainty._array)
            ccdall.write(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_bias'+str(i+1)+'.fits',overwrite=True)
        sig_master_processed=np.array(sig_master_processed)
        c=Combiner(master_processed)
#        c.weights=1./sig_master_bias**2
#        ccdall=c.average_combine(uncertainty_func=mycode.stdmean)
        ccdall=c.average_combine()
        ccdall.write(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_bias.fits',overwrite=True)
