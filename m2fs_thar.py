import astropy
import dill as pickle
from copy import deepcopy
from astropy import units
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from astropy.nddata import NDDataRef
import astropy.units as u
from astropy.modeling import models
import m2fs_process as m2fs
import os
from os import path
import mycode
import specutils
from specutils.spectra import Spectrum1D
import scipy
from astropy.nddata import StdDevUncertainty
from astropy.visualization import quantity_support
import numpy as np
from specutils.fitting import fit_lines
from specutils.analysis import centroid
from specutils.analysis import fwhm
from specutils.analysis import line_flux
from specutils.analysis import equivalent_width
from specutils import SpectralRegion
from specutils.manipulation import extract_region
from specutils.fitting import fit_generic_continuum
from astropy.modeling import models,fitting
from specutils.fitting import estimate_line_parameters
from ccdproc import Combiner
from scipy import interpolate
from astropy import time, coordinates as coord, units as u
from astropy.coordinates import SkyCoord, EarthLocation
from mpl_toolkits.axes_grid1 import make_axes_locatable
#matplotlib.use('pdf')
#matplotlib.use('TkAgg')

directory='/hildafs/projects/phy200028p/mgwalker/m2fs/'
m2fsrun='jan20' 
datadir=m2fs.get_datadir(m2fsrun)

utdate=[]
file1=[]
file2=[]
flatfile=[]
tharfile=[]
field_name=[]
scifile=[]
fibermap_file=[]
fiber_changes=[]
obj=[]

map_utdate=[]
map_file=[]
do_map=True

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
        fibermap_file.append(p[7])
        fiber_changes.append(p[8])
        obj.append(p[9])
with open(directory+m2fsrun+'_fibermap_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        map_utdate.append(str(p[0]))
        map_file.append(str(p[1]))
utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flatfile=np.array(flatfile)
tharfile=np.array(tharfile)
field_name=np.array(field_name)
scifile=np.array(scifile)
fibermap_file=np.array(fibermap_file)
fiber_changes=np.array(fiber_changes)
obj=np.array(obj)

ccd='b'
i=0
root=datadir+utdate[i]+'/'+ccd+str(tharfile[i]).zfill(4)
root2=datadir+utdate[i]+'/'+ccd+'_'+field_name[i]+'_'+m2fsrun
data_file=root+'_stitched.fits'
data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
plt.imshow(data.data,vmin=0,vmax=500,cmap='gray')
#plt.imshow(data.data,vmin=100,vmax=1000,cmap='gray')
plt.xlabel('column')
plt.ylabel('row')
plt.xlim([0,2048])
plt.ylim([0,2056])
plt.savefig('m2fs_thar.pdf',dpi=200)
plt.show()
plt.close()
