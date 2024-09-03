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

directory='/hildafs/projects/phy200028p/mgwalker/m2fs/'
m2fsrun='nov18'
datadir='/hildafs/projects/phy200028p/mgwalker/m2fs/nov18/m2fs.astro.lsa.umich.edu/data/NovDec2018/'

filename_out=directory+m2fsrun+'_master_bias.fits'
os.popen('rm '+filename_out)

with open(directory+m2fsrun+'_master_bias_files') as f:
    data=f.readlines()[0:]
filename_in=[]
for line in data:
    p=line.split()
    filename_in.append(p[0])
filename_in=np.array(filename_in)

data=[]
unc=[]
for i in range(0,len(filename_in)):
    new=astropy.nddata.CCDData.read(filename_in[i])
    print(i,np.median(new.uncertainty._array))
    data.append(new)
    unc.append(new.uncertainty._array)
unc=np.array(unc)

c=Combiner(data)
c.weights=1./unc**2

#def stdmean(a,axis=None,dtype=None,out=None,ddof=0,keepdims=np._NoValue):
#    std=np.ma.std(a)
#    print(len(a))
#    return std/len(a)

ccdall=c.average_combine(uncertainty_func=mycode.stdmean)
crap=ccdproc.combine(data,scaling=0,weights=1./unc**2,combine_uncertainty_function=mycode.stdmean)
print(np.median(ccdall.uncertainty._array))
print(np.median(crap.uncertainty._array))
ccdall.write(filename_out)
