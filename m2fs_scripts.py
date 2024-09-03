import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
#from astropy.nddata import NDData
from astropy.nddata import CCDData
import ccdproc
import astropy.units as u
from astropy.modeling import models
from ccdproc import Combiner
import os
import mycode
matplotlib.use('TkAgg')

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'
#m2fsrun='nov18'
#m2fsrun='jul15'
m2fsrun='may19'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/NovDec2018/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Jul2015/'
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2019/'
flataps_script=m2fsrun+'_flataps.cl'
apflatten_script=m2fsrun+'_apflatten.cl'
b_masterfibermap=m2fsrun+'_b_master_fibermap1_iraf.fits'
r_masterfibermap=m2fsrun+'_r_master_fibermap1_iraf.fits'

baperture=[]
bpixel1=[]
bpixel2=[]
raperture=[]
rpixel1=[]
rpixel2=[]

with open(directory+'database/ap'+m2fsrun+'_b_master_fibermap1_iraf') as f:
    data=f.readlines()
apchar=[]
for line in data:
    p=line.split()
    apchar.append(p)
apchar=np.array(apchar)
curve=0
for i in range(0,len(apchar)):
    if 'aperture' in apchar[i]:
        if 'aperture' in apchar[i][0]:
            ap=apchar[i][1]
            curve=0
    if 'curve' in apchar[i]:
        curve=1
        c=1
    if curve==1:
        if c==4:
            pixel1=apchar[i][0]
        if c==5:
            pixel2=apchar[i][0]
            curve=0
            baperture.append(ap)
            bpixel1.append(pixel1)
            bpixel2.append(pixel2)
            print('b ',ap,pixel1,pixel2)
        c+=1   

with open(directory+'database/ap'+m2fsrun+'_r_master_fibermap1_iraf') as f:
    data=f.readlines()
apchar=[]
for line in data:
    p=line.split()
    apchar.append(p)
apchar=np.array(apchar)
curve=0
for i in range(0,len(apchar)):
    if 'aperture' in apchar[i]:
        if 'aperture' in apchar[i][0]:
            ap=apchar[i][1]
            curve=0
    if 'curve' in apchar[i]:
        curve=1
        c=1
    if curve==1:
        if c==4:
            pixel1=apchar[i][0]
        if c==5:
            pixel2=apchar[i][0]
            curve=0
            raperture.append(ap)
            rpixel1.append(pixel1)
            rpixel2.append(pixel2)
            print('r ',ap,pixel1,pixel2)
        c+=1   

utdate=[]
file1=[]
file2=[]
flat=[]
thar=[]
field=[]
frames=[]
with open(directory+m2fsrun+'_science_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(int(p[1]))
    file2.append(int(p[2]))
    flat.append(int(p[3]))
    thar.append(int(p[4]))
    field.append(str(p[5]))
    frames.append(str(p[6:]))
with open(directory+m2fsrun+'_fibermap_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(int(p[1]))
    file2.append(int(p[2]))
    flat.append(int(p[3]))
    thar.append(int(p[4]))
    field.append(str(p[5]))
    frames.append(str(p[6:]))
with open(directory+m2fsrun+'_twilight_raw') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    utdate.append(str(p[0]))
    file1.append(int(p[1]))
    file2.append(int(p[2]))
    flat.append(int(p[3]))
    thar.append(int(p[4]))
    field.append(str(p[5]))
    frames.append(p[6:])

utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flat=np.array(flat)
thar=np.array(thar)
field=np.array(field)
#frames=np.array(frames)

g1=open(flataps_script,'w')
g2=open(apflatten_script,'w')
g1.write('twod \n')
g1.write('imred \n')
g1.write('apextract \n')
g2.write('twod \n')
g2.write('imred \n')
g2.write('apextract \n')
for i in range(0,len(utdate)):
    ref_filename=b_masterfibermap
    filename=datadir+utdate[i]+'/b'+str(flat[i]).zfill(4)
#    for j in range(0,len(baperture)):
#        string1='apall input='+filename+'_stitched_iraf.fits output="" apertures= "'+baperture[j]+'" references='+ref_filename+' format=multispec profiles="" interactive- find- recenter+ resize+ edit+ trace+ fittrace+ extract- extras- review- line=INDEF nsum=10 lower=-5 upper=5 width=3 radius=2 threshold=0 nfind=1 minsep=2 maxsep=100000 order=increasing aprecenter="" npeaks=INDEF shift- llimit=INDEF ulimit=INDEF ylevel=0.1 peak+ bkg+ r_grow=0 avglimits- t_nsum=10 t_step=10 t_nlost=3 t_function=legendre t_order=4 t_sample='+bpixel1[j]+':'+bpixel2[j]+' t_naverage=1 t_niterate=10 t_low_reject=3 t_high_reject=3 t_grow=0 background=none skybox=1 weights=none pfit=fit1d clean- saturation=INDEF readnoise=0 gain=1 lsigma=4 usigma=4 nsubaps=1 mode=al \n'
    string1='apall input='+filename+'_stitched_iraf.fits output="" apertures= "1-128" references='+ref_filename+' format=multispec profiles="" interactive- find- recenter+ resize+ edit+ trace+ fittrace+ extract- extras- review- line=INDEF nsum=10 lower=-5 upper=5 width=3 radius=2 threshold=0 nfind=1 minsep=2 maxsep=100000 order=increasing aprecenter="" npeaks=INDEF shift- llimit=INDEF ulimit=INDEF ylevel=0.1 peak+ bkg+ r_grow=0 avglimits- t_nsum=10 t_step=10 t_nlost=3 t_function=legendre t_order=4 t_sample=@baps t_naverage=1 t_niterate=10 t_low_reject=3 t_high_reject=3 t_grow=0 background=none skybox=1 weights=none pfit=fit1d clean- saturation=INDEF readnoise=0 gain=1 lsigma=4 usigma=4 nsubaps=1 mode=al \n'
    g1.write(string1)
#    string2='apflatten input='+filename+'_stitched_iraf.fits output='+filename+'_apflatten apertures= "1-128" references='+ref_filename+' interactive- find- recenter+ resize+ edit+ trace+ fittrace+ flatten+ fitspec- line=INDEF nsum=10 threshold=10.0 pfit=fit1d clean- saturation=INDEF readnoise=0 gain=1 lsigma=4.0 usigma=4.0 function=legendre order=10 sample="374:1344" naverage=1 niterate=10 low_reject=3 high_reject=3 grow=0.0 mode=al \n'
#    string2='apflatten input='+filename+'_stitched_iraf.fits output='+filename+'_apflatten apertures= "1-128" references='+filename+'_stitched_iraf interactive- find- recenter- resize- edit- trace- fittrace+ flatten+ fitspec- line=INDEF nsum=10 threshold=10.0 pfit=fit1d clean- saturation=INDEF readnoise=0 gain=1 lsigma=4.0 usigma=4.0 function=legendre order=10 sample="374:1344" naverage=1 niterate=10 low_reject=3 high_reject=3 grow=0.0 mode=al \n'
    string2='apflatten input='+filename+'_stitched_iraf.fits output='+filename+'_apflatten apertures= "1-128" references=OLD interactive- find- recenter- resize- edit- trace- fittrace- flatten+ fitspec- line=INDEF nsum=10 threshold=10.0 pfit=fit1d clean- saturation=INDEF readnoise=0 gain=1 lsigma=4.0 usigma=4.0 function=legendre order=10 sample="374:1344" naverage=1 niterate=10 low_reject=3 high_reject=3 grow=0.0 mode=al \n'
    g2.write(string2)

    ref_filename=r_masterfibermap
    filename=datadir+utdate[i]+'/r'+str(flat[i]).zfill(4)
#    for j in range(0,len(raperture)):
#        string1='apall input='+filename+'_stitched_iraf.fits output="" apertures= "'+raperture[j]+'" references='+ref_filename+' format=multispec profiles="" interactive- find- recenter+ resize+ edit+ trace+ fittrace+ extract- extras- review- line=INDEF nsum=10 lower=-5 upper=5 width=3 radius=2 threshold=0 nfind=1 minsep=2 maxsep=100000 order=increasing aprecenter="" npeaks=INDEF shift- llimit=INDEF ulimit=INDEF ylevel=0.1 peak+ bkg+ r_grow=0 avglimits- t_nsum=10 t_step=10 t_nlost=3 t_function=legendre t_order=4 t_sample='+rpixel1[j]+':'+rpixel2[j]+' t_naverage=1 t_niterate=10 t_low_reject=3 t_high_reject=3 t_grow=0 background=none skybox=1 weights=none pfit=fit1d clean- saturation=INDEF readnoise=0 gain=1 lsigma=4 usigma=4 nsubaps=1 mode=al \n'
    string1='apall input='+filename+'_stitched_iraf.fits output="" apertures="1-128" references='+ref_filename+' format=multispec profiles="" interactive- find- recenter+ resize+ edit+ trace+ fittrace+ extract- extras- review- line=INDEF nsum=10 lower=-5 upper=5 width=3 radius=2 threshold=0 nfind=1 minsep=2 maxsep=100000 order=increasing aprecenter="" npeaks=INDEF shift- llimit=INDEF ulimit=INDEF ylevel=0.1 peak+ bkg+ r_grow=0 avglimits- t_nsum=10 t_step=10 t_nlost=3 t_function=legendre t_order=4 t_sample=@baps t_naverage=1 t_niterate=10 t_low_reject=3 t_high_reject=3 t_grow=0 background=none skybox=1 weights=none pfit=fit1d clean- saturation=INDEF readnoise=0 gain=1 lsigma=4 usigma=4 nsubaps=1 mode=al \n'
    g1.write(string1)
#    string2='apflatten input='+filename+'_stitched_iraf.fits output='+filename+'_apflatten apertures= "1-128" references='+filename+'_stitched_iraf interactive- find- recenter+ resize+ edit+ trace+ fittrace+ flatten+ fitspec- line=INDEF nsum=10 threshold=10.0 pfit=fit1d clean- saturation=INDEF readnoise=0 gain=1 lsigma=4.0 usigma=4.0 function=legendre order=10 sample="374:1344" naverage=1 niterate=10 low_reject=3 high_reject=3 grow=0.0 mode=al \n'
    string2='apflatten input='+filename+'_stitched_iraf.fits output='+filename+'_apflatten apertures= "1-128" references=OLD interactive- find- recenter- resize- edit- trace- fittrace- flatten+ fitspec- line=INDEF nsum=10 threshold=10.0 pfit=fit1d clean- saturation=INDEF readnoise=0 gain=1 lsigma=4.0 usigma=4.0 function=legendre order=10 sample="374:1344" naverage=1 niterate=10 low_reject=3 high_reject=3 grow=0.0 mode=al \n'
    g2.write(string2)
g1.close()
g2.close()

