import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import CCDData
import ccdproc
import astropy.units as u
from astropy.modeling import models
from ccdproc import Combiner
import m2fs_process as m2fs
import dill as pickle

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'
m2fsrun='nov18'
#m2fsrun='jul15'
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/NovDec2018/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Jul2015/'

threshold_factor=25.#multiple of continuum residual rms threshold to impose for aperture detection in find_lines_derivative
n_lines=20#columns to combine when scanning across rows to identify apertures (as 'emission lines')
continuum_rejection_iterations=10#number of iterations of outlier rejection for fitting "continuum"
window=10#pixels, width of aperture window for fitting (gaussian) aperture profiles (perpendicular to spectral axis)
trace_step=n_lines#tracing step

utdate=[]
file1=[]
file2=[]
flatfile=[]
with open(directory+m2fsrun+'_science_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        utdate.append(str(p[0]))
        file1.append(int(p[1]))
        file2.append(int(p[2]))
        flatfile.append(int(p[3]))
utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flatfile=np.array(flatfile)

for i in range(0,len(utdate)):
    for ccd in ('b','r'):
        root=datadir+utdate[i]+'/'+ccd+str(flatfile[i]).zfill(4)
        data=astropy.nddata.CCDData.read(root+'_stitched.fits')

        columnspec_array=pickle.load(open(root+'_columnspec_array.pickle','rb'))

        '''
        Make initial array of more precise aperture location fits in middle column stack
        '''
        middle_column=np.long(len(columnspec_array)/2)
        apertures_profile_middle=columnspec_array[middle_column].apertures_profile

        '''
        apertures_profile_middle object contains: 
        fit: parameters of fit to cross-aperture profile ('line' profile) *for middle column*
        subregion: pixel range of cross-aperture 'spectrum' used for fit *for middle column*
        initial: value of True means aperture was found in automatic procedure, value of False means it was inserted by hand
        realvirtual: value of True means aperture corresponds to real spectrum, value of False means virtual aperture used as place-holder (due to bad or unplugged fiber)
        '''

        '''
        display apertures initially identified in central column stack and allow user to add/delete apertures with screen input
        '''
        apertures_profile_middle=m2fs.fiddle_apertures(columnspec_array,middle_column,window,apertures_profile_middle)
        pickle.dump([apertures_profile_middle,middle_column],open(root+'_apertures_profile_middle.pickle','wb'))#save pickle to file
