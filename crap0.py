import numpy as np
import astropy
import sqlutil
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import StdDevUncertainty
from astropy.nddata import CCDData
from astropy.coordinates import SkyCoord
import dustmaps.sfd
import astropy.units as u
from astropy.modeling import models
import os
import csv
from os import path
import scipy
import m2fs_process as m2fs
import mycode
import crossmatcher
matplotlib.use('TkAgg')
from matplotlib.patches import Ellipse
import dill as pickle
from isochrones.mist import MIST_Isochrone
from isochrones.mist import MIST_EvolutionTrack
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from isochrones import get_ichrone
from pymultinest.solve import solve
from pymultinest import Analyzer

data=fits.open('all_m2fs_HiRes_raw.fits')[1].data
use=[]
keep=np.where((data.hjd>2460220.)&(data.target_system=='car'))[0]
data=data[keep]

channel_cass_fib=[]
for i in range(0,len(data)):
    aperture=data.fits_index[i]+1
    #ap=129-throughputcorr_array[i].aperture
    ap=129-aperture
    if 'b_carina' in data.fits_filename[i]:
        channel0='b'
        if ((ap>=1)&(ap<=16)):
            cass='8'
            fib=str(ap).zfill(2)
        if ((ap>=17)&(ap<=32)):
            cass='7'
            fib=str(ap-16).zfill(2)
        if ((ap>=33)&(ap<=48)):
            cass='6'
            fib=str(ap-32).zfill(2)
        if ((ap>=49)&(ap<=64)):
            cass='5'
            fib=str(ap-48).zfill(2)
        if ((ap>=65)&(ap<=80)):
            cass='4'
            fib=str(ap-64).zfill(2)
        if ((ap>=81)&(ap<=96)):
            cass='3'
            fib=str(ap-80).zfill(2)
        if ((ap>=97)&(ap<=112)):
            cass='2'
            fib=str(ap-96).zfill(2)
        if ((ap>=113)&(ap<=128)):
            cass='1'
            fib=str(ap-112).zfill(2)
            
    if 'r_carina' in data.fits_filename[i]:
        channel0='r'
        if ((ap>=1)&(ap<=16)):
            cass='1'
            fib=str(ap).zfill(2)
        if ((ap>=17)&(ap<=32)):
            cass='2'
            fib=str(ap-16).zfill(2)
        if ((ap>=33)&(ap<=48)):
            cass='3'
            fib=str(ap-32).zfill(2)
        if ((ap>=49)&(ap<=64)):
            cass='4'
            fib=str(ap-48).zfill(2)
        if ((ap>=65)&(ap<=80)):
            cass='5'
            fib=str(ap-64).zfill(2)
        if ((ap>=81)&(ap<=96)):
            cass='6'
            fib=str(ap-80).zfill(2)
        if ((ap>=97)&(ap<=112)):
            cass='7'
            fib=str(ap-96).zfill(2)
        if ((ap>=113)&(ap<=128)):
            cass='8'
            fib=str(ap-112).zfill(2)
            
    #print(i,channel0+cass+'-'+fib)
    channel_cass_fib.append(str(channel0)+'-'+str(cass)+'-'+str(fib))


g0=open('carina_A_to_mario.dat','w')
for ccd in ['b','r']:
    if ccd=='b':
        ccd0='B'
    if ccd=='r':
        ccd0='R'
    for cass in range(1,9):
        for fib in range(1,17):
            this=-1
            for j in range(0,len(data)):
                split=channel_cass_fib[j].split('-')
#                print(ccd,cass,fib,split[0],split[1],split[2])
                if ((split[0]==ccd)&(int(split[1])==cass)&(int(split[2])==fib)):
                    this=j
            if this>=0:
                string=channel_cass_fib[this]+' '+str(data.vlos_raw[this])+' '+str(data.vlos_raw_error[this])
            else:
                string=ccd+'-'+str(cass)+'-'+str(fib).zfill(2)+' -999 -999'
            print(string)
            g0.write(string+' \n')
g0.close()
