import astropy
import dill as pickle
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
matplotlib.use('TkAgg')

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'

with open(directory+'m2fs_zero0.py') as f:
    zero_text=f.readlines()[0:]
with open(directory+'m2fs_dark0.py') as f:
    dark_text=f.readlines()[0:]
with open(directory+'m2fs_reduce0.py') as f:
    reduce_text=f.readlines()[0:]
with open(directory+'m2fs_apertures0.py') as f:
    apertures_text=f.readlines()[0:]

m2fsrun0=['dec14','feb15','jul15','sep15','feb16','jun16','aug16','nov16','feb17','may17','sep17','nov17','feb18','may18','aug18','nov18','feb19','may19','aug19','nov19']

for i in range(0,len(m2fsrun0)):
    zero_out='m2fs_zero_'+m2fsrun0[i]+'.py'
    dark_out='m2fs_dark.py'
    reduce_out='m2fs_reduce_'+m2fsrun0[i]+'.py'
    apertures_out='m2fs_apertures_'+m2fsrun0[i]+'.py'
    zero_slurm_out='/nfs/nas-0-9/mgwalker.proj/scripts/m2fs_zero_'+m2fsrun0[i]+'.slurm'
    dark_slurm_out='/nfs/nas-0-9/mgwalker.proj/scripts/m2fs_dark.slurm'
    reduce_slurm_out='/nfs/nas-0-9/mgwalker.proj/scripts/m2fs_reduce_'+m2fsrun0[i]+'.slurm'

    g1=open(zero_out,'w')
    for line in zero_text:
        if 'enter_run_here' in line:
            g1.write('m2fsrun=\''+m2fsrun0[i]+'\' \n')
        else:
            g1.write(line)
    g1.close()

    g1=open(dark_out,'w')
    for line in dark_text:
#        if 'enter_run_here' in line:
#            g1.write('m2fsrun=\''+m2fsrun0[i]+'\' \n')
#        else:
        g1.write(line)
    g1.close()

    g1=open(reduce_out,'w')
    for line in reduce_text:
        if 'enter_run_here' in line:
            g1.write('m2fsrun=\''+m2fsrun0[i]+'\' \n')
        else:
            g1.write(line)
    g1.close()

    g1=open(apertures_out,'w')
    for line in apertures_text:
        if 'enter_run_here' in line:
            g1.write('m2fsrun=\''+m2fsrun0[i]+'\' \n')
        else:
            g1.write(line)
    g1.close()

    g1=open(zero_slurm_out,'w')
    g1.write('#!/bin/bash \n')
    g1.write('#SBATCH -N 1 \n')
    g1.write('#SBATCH -c 1 \n')
    g1.write('#SBATCH -t 0-07:59 \n')
    g1.write('#SBATCH --partition=short \n')
    g1.write('#SBATCH --mem-per-cpu=100000 \n')
    g1.write('#SBATCH -o m2fs_zero_'+m2fsrun0[i]+'.o \n')
    g1.write('#SBATCH -e m2fs_zero_'+m2fsrun0[i]+'.err \n')
    g1.write(' \n')
    g1.write('python /nfs/nas-0-9/mgwalker.proj/m2fs/'+zero_out+' \n')
    g1.close()

    g1=open(dark_slurm_out,'w')
    g1.write('#!/bin/bash \n')
    g1.write('#SBATCH -N 1 \n')
    g1.write('#SBATCH -c 1 \n')
    g1.write('#SBATCH -t 0-07:59 \n')
    g1.write('#SBATCH --partition=short \n')
    g1.write('#SBATCH --mem-per-cpu=100000 \n')
    g1.write('#SBATCH -o m2fs_dark.o \n')
    g1.write('#SBATCH -e m2fs_dark.err \n')
    g1.write(' \n')
    g1.write('python /nfs/nas-0-9/mgwalker.proj/m2fs/'+dark_out+' \n')
    g1.close()

    g1=open(reduce_slurm_out,'w')
    g1.write('#!/bin/bash \n')
    g1.write('#SBATCH -N 1 \n')
    g1.write('#SBATCH -c 1 \n')
    g1.write('#SBATCH -t 0-07:59 \n')
    g1.write('#SBATCH --partition=short \n')
    g1.write('#SBATCH --mem-per-cpu=100000 \n')
    g1.write('#SBATCH -o m2fs_reduce_'+m2fsrun0[i]+'.o \n')
    g1.write('#SBATCH -e m2fs_reduce_'+m2fsrun0[i]+'.err \n')
    g1.write(' \n')
    g1.write('python /nfs/nas-0-9/mgwalker.proj/m2fs/'+reduce_out+' \n')
    g1.close()
