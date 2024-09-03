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
import os
import os.path
from os import path
matplotlib.use('TkAgg')

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'
m2fsrun='jul15'
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Jul2015/'

threshold_factor=25.#multiple of continuum residual rms threshold to impose for aperture detection in find_lines_derivative
n_lines=20#columns to combine when scanning across rows to identify apertures (as 'emission lines')
continuum_rejection_iterations=10#number of iterations of outlier rejection for fitting "continuum"
window=10#pixels, width of aperture window for fitting (gaussian) aperture profiles (perpendicular to spectral axis)
trace_step=n_lines#tracing step
trace_nlost_max=2
trace_shift_max=1.5
trace_order=4
trace_rejection_iterations=10
trace_rejection_sigma=3.#largest rms deviation to accept in fit to aperture trace
trace_rejection_iterations=10
trace_pix_min=200.#conservative limits for trace fit
trace_pix_max=1550.#conservative limits for trace fit
lsf_rejection_iterations=10#number of iterations of outlier rejection for fitting lsf sigma
lsf_nsample=50#number of points along spectral- (x-) axis at which to measure LSF sigma before performing fit of sigma(x)
lsf_order=4#order of polynomial used to fit lsf sigma as function of pixel along dispersion direction

utdate=[]
file1=[]
file2=[]
flatfile=[]
bmasterflatfile=[]
rmasterflatfile=[]
with open(directory+m2fsrun+'_science_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        utdate.append(str(p[0]))
        file1.append(int(p[1]))
        file2.append(int(p[2]))
        flatfile.append(int(p[3]))
        bmasterflatfile.append(str(p[7]))
        rmasterflatfile.append(str(p[8]))
utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flatfile=np.array(flatfile)
bmasterflatfile=np.array(bmasterflatfile)
rmasterflatfile=np.array(rmasterflatfile)

for i in range(0,len(utdate)):
    for ccd in ('b','r'):

        if ccd=='b':
            master_root=datadir+str(bmasterflatfile[i]).zfill(4)
        if ccd=='r':
            master_root=datadir+str(rmasterflatfile[i]).zfill(4)
        master_columnspec_array_file=master_root+'_columnspec_array.pickle'
        master_apertures_profile_middle_file=master_root+'_apertures_profile_middle.pickle'
        master_aperture_array_file=master_root+'_aperture_array.pickle'

        root=datadir+utdate[i]+'/'+ccd+str(flatfile[i]).zfill(4)
        columnspec_array_file=root+'_columnspec_array.pickle'
        apertures_profile_middle_file=root+'_apertures_profile_middle.pickle'
        aperture_array_file=root+'_aperture_array.pickle'

        print()
        print('finding apertures for '+ccd+str(flatfile[i]).zfill(4))
        print('reference = '+master_columnspec_array_file)
        print()

        master_exists=path.exists(master_apertures_profile_middle_file)
        exists=path.exists(apertures_profile_middle_file)

        master_columnspec_array=pickle.load(open(master_columnspec_array_file,'rb'))
        master_apertures_profile_middle,master_middle_column=pickle.load(open(master_apertures_profile_middle_file,'rb'))#has gone through the manual 'fiddle' to make sure apertures are accurate
        master_apertures_array=pickle.load(open(master_aperture_array_file,'rb'))
        columnspec_array=pickle.load(open(columnspec_array_file,'rb'))

        middle_column=master_middle_column
        apertures_profile_middle=columnspec_array[middle_column].apertures_profile

        shite=[]
        for j in range(0,len(master_apertures_profile_middle.fit)):
            crap=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]
            dist=np.sqrt((master_apertures_profile_middle.fit[j].mean.value-21.-crap)**2)
            print(j,master_apertures_profile_middle.fit[j].mean.value,master_apertures_profile_middle.realvirtual[j],np.min(dist))
            shite.append(np.min(dist))
        shite=np.array(shite)
        np.pause()

