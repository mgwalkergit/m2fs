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

tuc2_racenter=342.9796
tuc2_deccenter=-58.5689
gru1_racenter=344.1765
gru1_deccenter=-50.1633

tuc2_data1='tuc2_gradientpost_equal_weights.dat'
gru1_data1='gru1_gradientpost_equal_weights.dat'
out1='tuc2gru1_grfcommands.tex'
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



with open(tuc2_data1) as f: # read data file
    data=f.readlines()

tuc2_vmean=[]

for line in data: # fill arrays
    p=line.split()
    tuc2_vmean.append(float(p[0]))

tuc2_vmean=np.array(tuc2_vmean)

with open(gru1_data1) as f: # read data file
    data=f.readlines()

gru1_vmean=[]

for line in data: # fill arrays
    p=line.split()
    gru1_vmean.append(float(p[0]))

gru1_vmean=np.array(gru1_vmean)


tuc2_coords=SkyCoord(tuc2_racenter,tuc2_deccenter,unit='deg')
tuc2_lcenter=tuc2_coords.galactic.l.radian
tuc2_bcenter=tuc2_coords.galactic.b.radian

tuc2_usolar=np.random.normal(loc=usolar0,scale=sigusolar0,size=len(tuc2_vmean))
tuc2_vsolar=np.random.normal(loc=vsolar0,scale=sigvsolar0,size=len(tuc2_vmean))
tuc2_wsolar=np.random.normal(loc=wsolar0,scale=sigwsolar0,size=len(tuc2_vmean))

tuc2_vsolarx=tuc2_vsolar
tuc2_vsolary=tuc2_usolar
tuc2_vsolarz=tuc2_wsolar

tuc2_x=np.sin(tuc2_lcenter)*np.cos(tuc2_bcenter)
tuc2_y=-np.cos(tuc2_bcenter)*np.cos(tuc2_lcenter)
tuc2_z=np.sin(tuc2_bcenter)
tuc2_projlsr=vlsrx*tuc2_x+vlsry*tuc2_y+vlsrz*tuc2_z
tuc2_projsolar=tuc2_vsolarx*tuc2_x+tuc2_vsolary*tuc2_y+tuc2_vsolarz*tuc2_z
tuc2_proj=tuc2_projlsr+tuc2_projsolar
tuc2_vgrf=tuc2_vmean+tuc2_proj
tuc2_statsvgrf=np.percentile(tuc2_vgrf,[2.5,16.,50.,84.,97.5])


gru1_coords=SkyCoord(gru1_racenter,gru1_deccenter,unit='deg')
gru1_lcenter=gru1_coords.galactic.l.radian
gru1_bcenter=gru1_coords.galactic.b.radian

gru1_usolar=np.random.normal(loc=usolar0,scale=sigusolar0,size=len(gru1_vmean))
gru1_vsolar=np.random.normal(loc=vsolar0,scale=sigvsolar0,size=len(gru1_vmean))
gru1_wsolar=np.random.normal(loc=wsolar0,scale=sigwsolar0,size=len(gru1_vmean))

gru1_vsolarx=gru1_vsolar
gru1_vsolary=gru1_usolar
gru1_vsolarz=gru1_wsolar

gru1_x=np.sin(gru1_lcenter)*np.cos(gru1_bcenter)
gru1_y=-np.cos(gru1_bcenter)*np.cos(gru1_lcenter)
gru1_z=np.sin(gru1_bcenter)
gru1_projlsr=vlsrx*gru1_x+vlsry*gru1_y+vlsrz*gru1_z
gru1_projsolar=gru1_vsolarx*gru1_x+gru1_vsolary*gru1_y+gru1_vsolarz*gru1_z
gru1_proj=gru1_projlsr+gru1_projsolar
gru1_vgrf=gru1_vmean+gru1_proj
gru1_statsvgrf=np.percentile(gru1_vgrf,[2.5,16.,50.,84.,97.5])

g1.write(r'\newcommand{\tucvgrf}{$'+str(round(tuc2_statsvgrf[2],1))+r'_{'+str(round(tuc2_statsvgrf[1]-tuc2_statsvgrf[2],1))+r'}'+r'^{+'+str(round(tuc2_statsvgrf[3]-tuc2_statsvgrf[2],1))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\tucvgrfexpanded}{$'+str(round(tuc2_statsvgrf[2],1))+r'_{'+str(round(tuc2_statsvgrf[1]-tuc2_statsvgrf[2],1))+r'('+str(round(tuc2_statsvgrf[0]-tuc2_statsvgrf[2],1))+r')}'+r'^{+'+str(round(tuc2_statsvgrf[3]-tuc2_statsvgrf[2],1))+r'(+'+str(round(tuc2_statsvgrf[4]-tuc2_statsvgrf[2],1))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\gruvgrf}{$'+str(round(gru1_statsvgrf[2],1))+r'_{'+str(round(gru1_statsvgrf[1]-gru1_statsvgrf[2],1))+r'}'+r'^{+'+str(round(gru1_statsvgrf[3]-gru1_statsvgrf[2],1))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\gruvgrfexpanded}{$'+str(round(gru1_statsvgrf[2],1))+r'_{'+str(round(gru1_statsvgrf[1]-gru1_statsvgrf[2],1))+r'('+str(round(gru1_statsvgrf[0]-gru1_statsvgrf[2],1))+r')}'+r'^{+'+str(round(gru1_statsvgrf[3]-gru1_statsvgrf[2],1))+r'(+'+str(round(gru1_statsvgrf[4]-gru1_statsvgrf[2],1))+r')}$'+r'}'+'\n')

g1.close()
