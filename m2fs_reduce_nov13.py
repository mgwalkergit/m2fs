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
from astropy.nddata import StdDevUncertainty

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

utdate=[]
file1=[]
file2=[]
flatfile=[]
tharfile=[]
field_name=[]
scifile=[]

#with open(directory+m2fsrun+'_fibermap_raw') as f:
#    data=f.readlines()[0:]
#for line in data:
#    p=line.split()
#    if p[0]!='none':
#        utdate.append(str(p[0]))
#        file1.append(int(p[1]))
#        file2.append(int(p[2]))
#with open(directory+m2fsrun+'_twilight_raw') as f:
#    data=f.readlines()[0:]
#for line in data:
#    p=line.split()
#    if p[0]!='none':
#        utdate.append(str(p[0]))
#        file1.append(int(p[1]))
#        file2.append(int(p[2]))
with open(directory+m2fsrun+'_science_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        utdate.append(str(p[0]))
        file1.append(int(p[1]))
        file2.append(int(p[2]))
        flatfile.append(p[3])
        tharfile.append(p[4])
        field_name.append(p[5])
        scifile.append(p[6])

utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flatfile=np.array(flatfile)
tharfile=np.array(tharfile)
field_name=np.array(field_name)
scifile=np.array(scifile)

flatfile0=[]
tharfile0=[]
scifile0=[]
allfile0=[]
for i in range(0,len(tharfile)):
    flatfile0.append(flatfile[i].split('-'))
    tharfile0.append(tharfile[i].split('-'))
    scifile0.append(scifile[i].split('-'))
    allfile0.append(flatfile[i].split('-')+tharfile[i].split('-')+scifile[i].split('-'))
flatfile0=np.array(flatfile0,dtype='object')
tharfile0=np.array(tharfile0,dtype='object')
scifile0=np.array(scifile0,dtype='object')
allfile0=np.array(allfile0,dtype='object')

for i in range(0,len(utdate)):
#    for j in range(file1[i],file2[i]+1):
    for j in allfile0[i]:
        for ccd in (['b','r']):
            out=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+'_stitched.fits'
            out2=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+'_stitched_iraf.fits'
            out3=datadir+utdate[i]+'/'+ccd+str(j).zfill(4)+'_stitched_iraf_var.fits'
            for chip in (['c1','c2','c3','c4']):
#                print(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_bias.fits')
                master_bias=astropy.nddata.CCDData.read(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_bias.fits')
                master_dark=astropy.nddata.CCDData.read(directory+m2fsrun+'_'+ccd+'_'+chip+'_master_dark.fits')
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

                debiased0=ccdproc.subtract_bias(cr_cleaned,master_bias)
                dedark0=ccdproc.subtract_dark(debiased0,master_dark,exposure_time='exptime',exposure_unit=u.second,scale=True,add_keyword={'dark_corr':'Done'})
                dedark0.write(out,overwrite=True)

                if chip=='c1':
                    c1_reduce=dedark0
                if chip=='c2':
                    c2_reduce=dedark0
                if chip=='c3':
                    c3_reduce=dedark0
                if chip=='c4':
                    c4_reduce=dedark0
            left_data=np.concatenate((c1_reduce,np.flipud(c4_reduce)),axis=0)#left half of stitched image
            left_uncertainty=np.concatenate((c1_reduce.uncertainty._array,np.flipud(c4_reduce.uncertainty._array)),axis=0)
            left_mask=np.concatenate((c1_reduce.mask,np.flipud(c4_reduce.mask)),axis=0)
            right_data=np.concatenate((np.fliplr(c2_reduce),np.fliplr(np.flipud(c3_reduce))),axis=0)#right half of stitched image
            right_uncertainty=np.concatenate((np.fliplr(c2_reduce.uncertainty._array),np.fliplr(np.flipud(c3_reduce.uncertainty._array))),axis=0)
            right_mask=np.concatenate((np.fliplr(c2_reduce.mask),np.fliplr(np.flipud(c3_reduce.mask))),axis=0)

            stitched_data=np.concatenate((left_data,right_data),axis=1)
            stitched_uncertainty=np.concatenate((left_uncertainty,right_uncertainty),axis=1)
            stitched_mask=np.concatenate((left_mask,right_mask),axis=1)

            stitched=astropy.nddata.CCDData(stitched_data,unit=u.electron)

            bad=np.where(stitched_uncertainty!=stitched_uncertainty)#bad variances due to negative counts after overscan/bias/dark correction
            stitched_mask[bad]=True
            stitched_uncertainty[bad]=1.e+10
            stitched.uncertainty=stitched_uncertainty
            stitched.mask=stitched_mask
            stitched.mask[bad]=True
            stitched.header=c1_reduce.header
            stitched.write(out,overwrite=True)
            hdu=fits.PrimaryHDU(stitched.data,stitched.header)
            hdul=fits.HDUList([hdu])
            hdul.writeto(out2,overwrite=True)
#            hdul=fits.PrimaryHDU(stitched.data,stitched.header)
            hdul.writeto(out2,overwrite=True)
#            hdul=fits.PrimaryHDU(stitched.uncertainty.array**2,stitched.header)
            hdu=fits.PrimaryHDU(stitched.uncertainty.array**2,stitched.header)
            hdul=fits.HDUList([hdu])
            hdul.writeto(out3,overwrite=True)
            
