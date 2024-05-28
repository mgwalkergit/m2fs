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

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'
m2fsrun='feb19'
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/feb19/m2fs.astro.lsa.umich.edu/data/NovDec2018/'

utdate=[]
file1=[]
file2=[]
with open(directory+m2fsrun+'_science_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(long(p[1]))
    file2.append(long(p[2]))
with open(directory+m2fsrun+'_twilight_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(long(p[1]))
    file2.append(long(p[2]))
with open(directory+m2fsrun+'_fibermap_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(long(p[1]))
    file2.append(long(p[2]))
with open(directory+m2fsrun+'_dark_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(long(p[1]))
    file2.append(long(p[2]))

utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)

for i in range(0,len(utdate)):
    filename_in=[]
    filename_out=[]
    for j in range(file1[i],file2[i]+1):
        filename_in.append(datadir+utdate[i]+'/b'+str(j).zfill(4)+'c1.fits')
        filename_in.append(datadir+utdate[i]+'/b'+str(j).zfill(4)+'c2.fits')
        filename_in.append(datadir+utdate[i]+'/b'+str(j).zfill(4)+'c3.fits')
        filename_in.append(datadir+utdate[i]+'/b'+str(j).zfill(4)+'c4.fits')
        filename_in.append(datadir+utdate[i]+'/r'+str(j).zfill(4)+'c1.fits')
        filename_in.append(datadir+utdate[i]+'/r'+str(j).zfill(4)+'c2.fits')
        filename_in.append(datadir+utdate[i]+'/r'+str(j).zfill(4)+'c3.fits')
        filename_in.append(datadir+utdate[i]+'/r'+str(j).zfill(4)+'c4.fits')
        filename_out.append(datadir+utdate[i]+'/b'+str(j).zfill(4)+'c1_biascorr.fits')
        filename_out.append(datadir+utdate[i]+'/b'+str(j).zfill(4)+'c2_biascorr.fits')
        filename_out.append(datadir+utdate[i]+'/b'+str(j).zfill(4)+'c3_biascorr.fits')
        filename_out.append(datadir+utdate[i]+'/b'+str(j).zfill(4)+'c4_biascorr.fits')
        filename_out.append(datadir+utdate[i]+'/r'+str(j).zfill(4)+'c1_biascorr.fits')
        filename_out.append(datadir+utdate[i]+'/r'+str(j).zfill(4)+'c2_biascorr.fits')
        filename_out.append(datadir+utdate[i]+'/r'+str(j).zfill(4)+'c3_biascorr.fits')
        filename_out.append(datadir+utdate[i]+'/r'+str(j).zfill(4)+'c4_biascorr.fits')
    filename_in=np.array(filename_in)
    filename_out=np.array(filename_out)

    trimmed=[]
    for j in range(0,len(filename_in)):
        data=astropy.nddata.CCDData.read(filename_in[j],unit=u.adu)#header is in data.meta
        print(str(j)+' of '+str(len(filename_in)),filename_in[j],data.header['object'],data.header['binning'])
        data_with_deviation=ccdproc.create_deviation(data,gain=data.meta['egain']*u.electron/u.adu,readnoise=data.meta['enoise']*u.electron)
        gain_corrected=ccdproc.gain_correct(data_with_deviation,data_with_deviation.meta['egain']*u.electron/u.adu,add_keyword={'gain_corr':'Done'})
        cr_cleaned=ccdproc.cosmicray_lacosmic(gain_corrected,sigclip=5)
        oscan_subtracted=ccdproc.subtract_overscan(cr_cleaned,overscan=cr_cleaned[:,1024:],overscan_axis=1,model=models.Polynomial1D(3),add_keyword={'oscan_corr':'Done'})
        trimmed1=ccdproc.trim_image(oscan_subtracted[:,:1024],add_keyword={'trim1':'Done'})
        trimmed2=ccdproc.trim_image(trimmed1[:1028,:1024],add_keyword={'trim2':'Done'})
        bias_subtracted=ccdproc.subtract_bias(trimmed2,master_bias)
        os.popen('rm '+filename_out[j])
#        trimmed2.write(filename_out[j])
#        trimmed.append(trimmed2)
        bias_subtracted.write(filename_out[j]))
