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
import os
import mycode

with open('fibermap_files') as f:
    data=f.readlines()[0:]
filename=[]
for line in data:
    p=line.split()
    filename.append(str(p[0]))
filename=np.array(filename)

rah=[]
ram=[]
ras=[]
chardecd=[]
decm=[]
decs=[]
for i in range(0,len(filename)):
    with open(filename[i]) as f:
        data=f.readlines()[0:]
    data2=[]
    hitguide=0
    keep=0
    for line in data:
        if ' T ' in line:
            data2.append(line)
            if hitguide==0:
                keep+=1
        if '[guides]' in line:
            hitguide=1
    fiber=[]
    idnum=[]
    ra=[]
    dec=[]
    for line in data2:
        p=line.split()
        fiber.append(str(p[0]))
        idnum.append(str(p[1]))
        ra.append(str(p[2]))
        dec.append(str(p[3]))
    fiber=np.array(fiber[0:keep])
    idnum=np.array(idnum[0:keep])
    ra=np.array(ra[0:keep])
    dec=np.array(dec[0:keep])
    for j in range(0,len(ra)):
        p=ra[j].split(':')
        rah.append(int(p[0]))
        ram.append(int(p[1]))
        ras.append(float(p[2]))
        p=dec[j].split(':')
        chardecd.append((p[0]))
        decm.append(int(p[1]))
        decs.append(float(p[2]))
rah=np.array(rah,dtype='int')
ram=np.array(ram,dtype='int')
ras=np.array(ras,dtype='float')
chardecd=np.array(chardecd,dtype='str')
decm=np.array(decm,dtype='int')
decs=np.array(decs,dtype='float')

ra,dec=mycode.radecradiansarr(rah,ram,ras,chardecd,decm,decs)

g1=open('m2fs_fornaxtargets.dat','w')
radeg=ra*180./np.pi
decdeg=dec*180./np.pi
for i in range(0,len(rah)):
    string=str(round(radeg[i],10))+' '+str(round(decdeg[i],10))+' \n'
    print(string)
    g1.write(string)
g1.close()
