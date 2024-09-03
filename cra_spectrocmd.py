import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits 
from scipy import stats
import random
#import mycode2
np.set_printoptions(threshold='nan')

#plt.rc('grid',alpha=0.7) # modify rcparams
#plt.rc('grid',color='white')
#plt.rc('grid',linestyle='-')
plt.rc('legend',frameon='False')
plt.rc('xtick.major',size=2)
plt.rc('ytick.major',size=2)
plt.rc('axes',axisbelow='True')
plt.rc('axes',grid='False')
plt.rc('axes',facecolor='white')
plt.rc('axes',linewidth=0.5)
plt.rc('xtick.major',width=0.5)
plt.rc('ytick.major',width=0.5)
plt.rc('xtick',direction='in')
plt.rc('ytick',direction='in')
plt.rc('xtick',labelsize='10')
plt.rc('ytick',labelsize='10')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='eps')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)

dmodulus=20.9
loggoffset=0.6
            
data1='cra.dat'

with open(data1) as f: # read data file
    data=f.readlines()
    
radeg=[]
decdeg=[]
xi=[]
eta=[]
r=[]
pa=[]
v=[]
sigv=[]
skewv=[]
kurtv=[]
teff=[]
sigteff=[]
skewteff=[]
kurtteff=[]
logg=[]
siglogg=[]
skewlogg=[]
kurtlogg=[]
feh=[]
sigfeh=[]
skewfeh=[]
kurtfeh=[]
pmem=[]
gmag=[]
rmag=[]
imag=[]
siggmag=[]
sigrmag=[]
sigimag=[]
gclass=[]
rclass=[]
iclass=[]

for line in data: # fill arrays
    p=line.split()
    radeg.append(float(p[0]))
    decdeg.append(float(p[1]))
    xi.append(float(p[2]))
    eta.append(float(p[3]))
    r.append(float(p[4]))
    pa.append(float(p[5]))
    v.append(float(p[6]))
    sigv.append(float(p[7]))
#    skewv.append(float(p[8]))
#    kurtv.append(float(p[9]))
    teff.append(float(p[8]))
    sigteff.append(float(p[9]))
#    skewteff.append(float(p[12]))
#    kurtteff.append(float(p[13]))
    logg.append(float(p[10]))
    siglogg.append(float(p[11]))
#    skewlogg.append(float(p[16]))
#    kurtlogg.append(float(p[17]))
    feh.append(float(p[12]))
    sigfeh.append(float(p[13]))
#    skewfeh.append(float(p[20]))
#    kurtfeh.append(float(p[21]))
    pmem.append(float(p[14]))
    gmag.append(float(p[15]))
    siggmag.append(float(p[16]))
    rmag.append(float(p[17]))
    sigrmag.append(float(p[18]))
    imag.append(float(p[19]))
    sigimag.append(float(p[20]))

radeg=np.array(radeg)
decdeg=np.array(decdeg)
xi=np.array(xi)
eta=np.array(eta)
r=np.array(r)
pa=np.array(pa)
v=np.array(v)
sigv=np.array(sigv)
skewv=np.array(skewv)
kurtv=np.array(kurtv)
teff=np.array(teff)
sigteff=np.array(sigteff)
skewteff=np.array(skewteff)
kurtteff=np.array(kurtteff)
logg=np.array(logg)
siglogg=np.array(siglogg)
skewlogg=np.array(skewlogg)
kurtlogg=np.array(kurtlogg)
feh=np.array(feh)
sigfeh=np.array(sigfeh)
skewfeh=np.array(skewfeh)
kurtfeh=np.array(kurtfeh)
pmem=np.array(pmem)
gmag=np.array(gmag)
rmag=np.array(rmag)
imag=np.array(imag)
siggmag=np.array(siggmag)
sigrmag=np.array(sigrmag)
sigimag=np.array(sigimag)
sigr=r-r
mag=rmag
col=gmag-(rmag)
sigmag=mag-mag
sigcol=col-col

keep=np.where((sigv < 5.))# & (skewv >= -1.) & (skewv <= 1.) & (kurtv >= -1) & (kurtv <= 1) )
mem=np.where((sigv < 5.) & (pmem > 0.9))#(skewv > -1.) & (skewv < 1.) & (kurtv > -1.) & (kurtv < 1.) & (pmem > 0.9))
#mem2=np.where((sigv < 5.) & (skewv > -1.) & (skewv < 1.) & (kurtv > -1.) & (kurtv < 1.) & (pmem > 0.5) & (pmem < 0.8))
#mem3=np.where((sigv < 5.) & (skewv > -1.) & (skewv < 1.) & (kurtv > -1.) & (kurtv < 1.) & (pmem > 0.5))

data2='age12fehm250.iso'
with open(data2) as g: # read data file
    iso_data=g.readlines()[9:]

iso250_g=[]
iso250_r=[]
iso250_i=[]
iso250_logg=[]
iso250_teff=[]

for line in iso_data: # fill arrays
    p=line.split()
    iso250_teff.append(float(p[2]))
    iso250_logg.append(float(p[3]))
    iso250_g.append(float(p[6]))
    iso250_r.append(float(p[7]))
    iso250_i.append(float(p[8]))

iso250_g=np.array(iso250_g)
iso250_r=np.array(iso250_r)
iso250_i=np.array(iso250_i)
iso250_logg=np.array(iso250_logg)
iso250_teff=np.array(iso250_teff)
iso250_teff=10.**iso250_teff



data2='age12fehm200.iso'
with open(data2) as g: # read data file
    iso_data=g.readlines()[9:]

iso200_g=[]
iso200_r=[]
iso200_i=[]
iso200_logg=[]
iso200_teff=[]

for line in iso_data: # fill arrays
    p=line.split()
    iso200_teff.append(float(p[2]))
    iso200_logg.append(float(p[3]))
    iso200_g.append(float(p[6]))
    iso200_r.append(float(p[7]))
    iso200_i.append(float(p[8]))

iso200_g=np.array(iso200_g)
iso200_r=np.array(iso200_r)
iso200_i=np.array(iso200_i)
iso200_logg=np.array(iso200_logg)
iso200_teff=np.array(iso200_teff)
iso200_teff=10.**iso200_teff




data2='age12fehm150.iso'
with open(data2) as g: # read data file
    iso_data=g.readlines()[9:]

iso150_g=[]
iso150_r=[]
iso150_i=[]
iso150_logg=[]
iso150_teff=[]

for line in iso_data: # fill arrays
    p=line.split()
    iso150_teff.append(float(p[2]))
    iso150_logg.append(float(p[3]))
    iso150_g.append(float(p[6]))
    iso150_r.append(float(p[7]))
    iso150_i.append(float(p[8]))

iso150_g=np.array(iso150_g)
iso150_r=np.array(iso150_r)
iso150_i=np.array(iso150_i)
iso150_logg=np.array(iso150_logg)
iso150_teff=np.array(iso150_teff)
iso150_teff=10.**iso150_teff


data2='age10fehm160.iso'
with open(data2) as g: # read data file
    iso_data=g.readlines()[9:]

iso160_g=[]
iso160_r=[]
iso160_i=[]
iso160_logg=[]
iso160_teff=[]

for line in iso_data: # fill arrays
    p=line.split()
    iso160_teff.append(float(p[2]))
    iso160_logg.append(float(p[3]))
    iso160_g.append(float(p[6]))
    iso160_r.append(float(p[7]))
    iso160_i.append(float(p[8]))

iso160_g=np.array(iso160_g)
iso160_r=np.array(iso160_r)
iso160_i=np.array(iso160_i)
iso160_logg=np.array(iso160_logg)
iso160_teff=np.array(iso160_teff)
iso160_teff=10.**iso160_teff


data2='age10fehm150.iso'
with open(data2) as g: # read data file
    iso_data=g.readlines()[9:]

iso150_g=[]
iso150_r=[]
iso150_i=[]
iso150_logg=[]
iso150_teff=[]

for line in iso_data: # fill arrays
    p=line.split()
    iso150_teff.append(float(p[2]))
    iso150_logg.append(float(p[3]))
    iso150_g.append(float(p[6]))
    iso150_r.append(float(p[7]))
    iso150_i.append(float(p[8]))

iso150_g=np.array(iso150_g)
iso150_r=np.array(iso150_r)
iso150_i=np.array(iso150_i)
iso150_logg=np.array(iso150_logg)
iso150_teff=np.array(iso150_teff)
iso150_teff=10.**iso150_teff



data2='age12fehm100.iso'
with open(data2) as g: # read data file
    iso_data=g.readlines()[9:]

iso100_g=[]
iso100_r=[]
iso100_i=[]
iso100_logg=[]
iso100_teff=[]

for line in iso_data: # fill arrays
    p=line.split()
    iso100_teff.append(float(p[2]))
    iso100_logg.append(float(p[3]))
    iso100_g.append(float(p[6]))
    iso100_r.append(float(p[7]))
    iso100_i.append(float(p[8]))

iso100_g=np.array(iso100_g)
iso100_r=np.array(iso100_r)
iso100_i=np.array(iso100_i)
iso100_logg=np.array(iso100_logg)
iso100_teff=np.array(iso100_teff)
iso100_teff=10.**iso100_teff



data2='age12fehm050.iso'
with open(data2) as g: # read data file
    iso_data=g.readlines()[9:]

iso050_g=[]
iso050_r=[]
iso050_i=[]
iso050_logg=[]
iso050_teff=[]

for line in iso_data: # fill arrays
    p=line.split()
    iso050_teff.append(float(p[2]))
    iso050_logg.append(float(p[3]))
    iso050_g.append(float(p[6]))
    iso050_r.append(float(p[7]))
    iso050_i.append(float(p[8]))

iso050_g=np.array(iso050_g)
iso050_r=np.array(iso050_r)
iso050_i=np.array(iso050_i)
iso050_logg=np.array(iso050_logg)
iso050_teff=np.array(iso050_teff)
iso050_teff=10.**iso050_teff



data2='age12fehm000.iso'
with open(data2) as g: # read data file
    iso_data=g.readlines()[9:]

iso000_g=[]
iso000_r=[]
iso000_i=[]
iso000_logg=[]
iso000_teff=[]

for line in iso_data: # fill arrays
    p=line.split()
    iso000_teff.append(float(p[2]))
    iso000_logg.append(float(p[3]))
    iso000_g.append(float(p[6]))
    iso000_r.append(float(p[7]))
    iso000_i.append(float(p[8]))

iso000_g=np.array(iso000_g)
iso000_r=np.array(iso000_r)
iso000_i=np.array(iso000_i)
iso000_logg=np.array(iso000_logg)
iso000_teff=np.array(iso000_teff)
iso000_teff=10.**iso000_teff

vmem=[142.,158.]
loggmem=[0.1,2.5]
fehmem=[-2.5,-0.4]
teffmem=[4050,7000]
rmem=[0.,4.]
magmem=[23,16]
colmem=[-0.2,2]
magmem2=magmem
colmem2=colmem

gs=plt.GridSpec(10,10) # define multi-panel plot
gs2=plt.GridSpec(20,20) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_
ax0_0=fig.add_subplot(gs[0:5,0:5])
ax1_0=fig.add_subplot(gs[5:10,0:5])
ax1_1=fig.add_subplot(gs[5:10,5:10])

ax1_0.set_xlabel(r'$T_{\rm eff}$ [K]',fontsize=12,rotation=0,labelpad=5)
ax1_0.set_ylabel(r'$\log g$',fontsize=12,rotation=90,labelpad=5)
ax1_0.set_xlim([8000,4000])
ax1_0.set_ylim([5,0.01])
ax1_0.set_xscale(u'linear')
ax1_0.set_yscale(u'linear')
ax1_0.set_yticks([5,4,3,2,1])
#ax1_0.set_xticklabels(vticks,rotation=0)
ax1_0.set_xticks([8000,7000,6000,5000])
#ax1_0.plot(iso250_teff,iso250_logg,lw=1,color='b')
ax1_0.plot(iso150_teff,iso150_logg,lw=1,color='r')
ax1_0.plot(iso150_teff,iso150_logg-loggoffset,lw=1,color='k',linestyle='--')
#ax1_0.plot(iso000_teff,iso000_logg,lw=1,color='g')
ax1_0.errorbar(teff,logg,xerr=sigteff,yerr=siglogg,elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True,label='foreground')
ax1_0.errorbar(teff[mem],logg[mem],xerr=sigteff[mem],yerr=siglogg[mem],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='member')
#ax1_0.legend(loc=2,fontsize=8,handlelength=0,shadow=False)
#ax0_0.text(-0.4,16.5,'Crater',fontsize=10)
ax1_0.legend(loc=2,fontsize=10,handlelength=0,numpoints=1,scatterpoints=1,shadow=False)


#ax0_0.set_xlabel(r'$T_{\rm eff}$ [K]',fontsize=12,rotation=0,labelpad=5)
ax0_0.set_ylabel(r'$i$ [mag]',fontsize=12,rotation=90,labelpad=5)
ax0_0.set_xlim([8000,4000])
ax0_0.set_ylim([21,17])
ax0_0.set_xscale(u'linear')
ax0_0.set_yscale(u'linear')
ax0_0.set_yticks([21,20,19,18,17])
ax0_0.set_xticks([8000,7000,6000,5000,4000])
ax0_0.xaxis.set_major_formatter(plt.NullFormatter())
#ax0_0.plot(iso250_teff,iso250_i+dmodulus,lw=1,color='b',label=r"[Fe/H]=$-2.5$")
#ax0_0.plot(iso160_teff,iso160_i+dmodulus,lw=1,color='r',label=r"[Fe/H]=$-1.6$")
ax0_0.plot(iso150_teff,iso150_i+dmodulus,lw=1,color='r',label=r"[Fe/H]=$-1.5$")
#ax0_0.plot(iso000_teff,iso000_i+dmodulus,lw=1,color='g',label=r"[Fe/H]=$0$")
#ax0_0.errorbar(teff,imag,xerr=sigteff,yerr=siglogg-siglogg,elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax0_0.errorbar(teff[mem],imag[mem],xerr=sigteff[mem],yerr=sigimag[mem],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='r',rasterized=True)
ax0_0.legend(loc=2,fontsize=12,handlelength=0,shadow=False)
#ax0_0.text(-0.4,16.5,'Crater',fontsize=10)

ax1_1.set_xlabel(r'$i$ [mag]',fontsize=12,rotation=0,labelpad=5)
#ax1_1.set_ylabel(r'$\log g$',fontsize=12,rotation=90,labelpad=5)
ax1_1.set_xlim([21,17])
ax1_1.set_ylim([5,0.01])
ax1_1.set_xscale(u'linear')
ax1_1.set_yscale(u'linear')
ax1_1.set_yticks([5,4,3,2,1])
ax1_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax1_1.set_xticklabels(vticks,rotation=0)
ax1_1.set_xticks([21,20,19,18,17])
#ax1_1.plot(iso250_i+dmodulus,iso250_logg,lw=1,color='b',label=r"[Fe/H]=$-2.5$")
ax1_1.plot(iso150_i+dmodulus,iso150_logg,lw=1,color='r',label=r"[Fe/H]=$-1.6$")
ax1_1.plot(iso150_i+dmodulus,iso150_logg-loggoffset,lw=1,color='k',label=r"[Fe/H]=$-1.6$",linestyle='--')
#ax1_1.plot(iso000_i+dmodulus,iso000_logg,lw=1,color='g',label=r"[Fe/H]=$0$")
#ax1_1.errorbar(imag,logg,xerr=sigteff-sigteff,yerr=siglogg,elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax1_1.errorbar(imag[mem],logg[mem],xerr=sigimag[mem],yerr=siglogg[mem],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='r',rasterized=True)
#ax1_1.legend(loc=2,fontsize=12,handlelength=0,shadow=False)
#ax0_0.text(-0.4,16.5,'Crater',fontsize=10)





plotfilename='cra_spectrocmd.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
