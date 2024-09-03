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
        out=root+'_columnspec_array.pickle'
        columnspec_array=m2fs.get_columnspec(data,trace_step,n_lines,continuum_rejection_iterations,threshold_factor,window)
        pickle.dump(columnspec_array,open(out,'wb'))
        print('writing '+out)
        '''
        columnspec_array is array of columnspec objects.
        columnspec object contains: 
        columns: numbers of columns in original data frame that are stacked
        spec1d: 'spectrum' across stacked column, where each aperture appears as an 'emission line'
        pixel: value of pixel across stacked column 'spectrum', artificially given units of AA in order to comply with specutils requirements for fitting spectra
        continuum: parameters of 1dGaussian continuum fit to stacked column 'spectrum'
        rms: rms residuals around continuum fit (using only regions between fiber bundles)
        apertures_initial: initial aperture centers returned by specutils find_lines_derivative
        apertures_profile: contains parameters of fits to apertures detected in apertures_initial
        '''
