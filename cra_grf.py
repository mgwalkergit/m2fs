#!/usr/bin/python
import numpy as np
import matplotlib
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy.io import fits 
from scipy import stats
import random
import mycode2
np.set_printoptions(threshold='nan')

cra_racenter=174.066
cra_deccenter=-10.8777

cra_data1='cra_gradientpost_equal_weights.dat'
out1='cra_grfcommands.tex'
g1=open(out1,'w')

usolar0=-11.1 #               KM/S, SOLAR MOTION WRT LSR; DEHNEN & BINNEY 98
vsolar0=12.24 #               KM/S
wsolar0=7.25 #               KM/S
sigusolar0=1.
sigvsolar0=2.
sigwsolar0=0.5

vlsr=220.# LSR velocity, km/s
llsr=np.pi/2. #direction of LSR motion in Galactic coordinates, radians
blsr=0. #radians
vlsrx=220.
vlsry=0.
vlsrz=0.



with open(cra_data1) as f: # read data file
    data=f.readlines()

cra_vmean=[]

for line in data: # fill arrays
    p=line.split()
    cra_vmean.append(float(p[0]))

cra_vmean=np.array(cra_vmean)

cra_coords=SkyCoord(cra_racenter,cra_deccenter,unit='deg')
cra_lcenter=cra_coords.galactic.l.radian
cra_bcenter=cra_coords.galactic.b.radian

cra_usolar=np.random.normal(loc=usolar0,scale=sigusolar0,size=len(cra_vmean))
cra_vsolar=np.random.normal(loc=vsolar0,scale=sigvsolar0,size=len(cra_vmean))
cra_wsolar=np.random.normal(loc=wsolar0,scale=sigwsolar0,size=len(cra_vmean))

cra_vsolarx=cra_vsolar
cra_vsolary=cra_usolar
cra_vsolarz=cra_wsolar

cra_x=np.sin(cra_lcenter)*np.cos(cra_bcenter)
cra_y=-np.cos(cra_bcenter)*np.cos(cra_lcenter)
cra_z=np.sin(cra_bcenter)
cra_projlsr=vlsrx*cra_x+vlsry*cra_y+vlsrz*cra_z
cra_projsolar=cra_vsolarx*cra_x+cra_vsolary*cra_y+cra_vsolarz*cra_z
cra_proj=cra_projlsr+cra_projsolar
cra_vgrf=cra_vmean+cra_proj
cra_statsvgrf=np.percentile(cra_vgrf,[2.5,16.,50.,84.,97.5])


g1.write(r'\newcommand{\tucvgrf}{$'+str(round(cra_statsvgrf[2],1))+r'_{'+str(round(cra_statsvgrf[1]-cra_statsvgrf[2],1))+r'}'+r'^{+'+str(round(cra_statsvgrf[3]-cra_statsvgrf[2],1))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\tucvgrfexpanded}{$'+str(round(cra_statsvgrf[2],1))+r'_{'+str(round(cra_statsvgrf[1]-cra_statsvgrf[2],1))+r'('+str(round(cra_statsvgrf[0]-cra_statsvgrf[2],1))+r')}'+r'^{+'+str(round(cra_statsvgrf[3]-cra_statsvgrf[2],1))+r'(+'+str(round(cra_statsvgrf[4]-cra_statsvgrf[2],1))+r')}$'+r'}'+'\n')

g1.close()
