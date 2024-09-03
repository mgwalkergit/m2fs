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
#plt.rc('legend',frameon='False')
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

tuc2_dmodulus=18.8
gru1_dmodulus=20.4            

data1='tuc2teffprior.dat'
data2='gru1teffprior.dat'

with open(data1) as f: # read data file
    data=f.readlines()
    
tuc2_radeg=[]
tuc2_decdeg=[]
tuc2_xi=[]
tuc2_eta=[]
tuc2_r=[]
tuc2_hjd=[]
tuc2_v=[]
tuc2_sigv=[]
tuc2_skewv=[]
tuc2_kurtv=[]
tuc2_teff=[]
tuc2_sigteff=[]
tuc2_skewteff=[]
tuc2_kurtteff=[]
tuc2_logg=[]
tuc2_siglogg=[]
tuc2_skewlogg=[]
tuc2_kurtlogg=[]
tuc2_feh=[]
tuc2_sigfeh=[]
tuc2_skewfeh=[]
tuc2_kurtfeh=[]
tuc2_pmem=[]
tuc2_gmag=[]
tuc2_rmag=[]
tuc2_imag=[]
tuc2_extg=[]
tuc2_extr=[]
tuc2_exti=[]

for line in data: # fill arrays
    p=line.split()
    tuc2_radeg.append(float(p[0]))
    tuc2_decdeg.append(float(p[1]))
    tuc2_xi.append(float(p[2]))
    tuc2_eta.append(float(p[3]))
    tuc2_r.append(float(p[4]))
    tuc2_v.append(float(p[5]))
    tuc2_sigv.append(float(p[6]))
    tuc2_teff.append(float(p[7]))
    tuc2_sigteff.append(float(p[8]))
    tuc2_logg.append(float(p[9]))
    tuc2_siglogg.append(float(p[10]))
    tuc2_feh.append(float(p[11]))
    tuc2_sigfeh.append(float(p[12]))
    tuc2_pmem.append(float(p[13]))
    tuc2_gmag.append(float(p[14]))
    tuc2_rmag.append(float(p[15]))
    tuc2_imag.append(float(p[16]))
    tuc2_extg.append(float(p[17]))
    tuc2_extr.append(float(p[18]))
    tuc2_exti.append(float(p[19]))

tuc2_radeg=np.array(tuc2_radeg)
tuc2_decdeg=np.array(tuc2_decdeg)
tuc2_xi=np.array(tuc2_xi)
tuc2_eta=np.array(tuc2_eta)
tuc2_r=np.array(tuc2_r)
tuc2_hjd=np.array(tuc2_hjd)
tuc2_v=np.array(tuc2_v)
tuc2_sigv=np.array(tuc2_sigv)
tuc2_skewv=np.array(tuc2_skewv)
tuc2_kurtv=np.array(tuc2_kurtv)
tuc2_teff=np.array(tuc2_teff)
tuc2_sigteff=np.array(tuc2_sigteff)
tuc2_skewteff=np.array(tuc2_skewteff)
tuc2_kurtteff=np.array(tuc2_kurtteff)
tuc2_logg=np.array(tuc2_logg)
tuc2_siglogg=np.array(tuc2_siglogg)
tuc2_skewlogg=np.array(tuc2_skewlogg)
tuc2_kurtlogg=np.array(tuc2_kurtlogg)
tuc2_feh=np.array(tuc2_feh)
tuc2_sigfeh=np.array(tuc2_sigfeh)
tuc2_skewfeh=np.array(tuc2_skewfeh)
tuc2_kurtfeh=np.array(tuc2_kurtfeh)
tuc2_pmem=np.array(tuc2_pmem)
tuc2_gmag=np.array(tuc2_gmag)
tuc2_rmag=np.array(tuc2_rmag)
tuc2_imag=np.array(tuc2_imag)
tuc2_extg=np.array(tuc2_extg)
tuc2_extr=np.array(tuc2_extr)
tuc2_exti=np.array(tuc2_exti)
tuc2_sigr=tuc2_r-tuc2_r
tuc2_mag=tuc2_rmag-tuc2_extr
tuc2_col=tuc2_gmag-tuc2_extg-(tuc2_rmag-tuc2_extr)
tuc2_sigmag=tuc2_mag-tuc2_mag
tuc2_sigcol=tuc2_col-tuc2_col

#tuc2_keep=np.where((tuc2_sigv < 15.) & (tuc2_skewv >= -1.) & (tuc2_skewv <= 1.) & (tuc2_kurtv >= -1) & (tuc2_kurtv <= 1))
tuc2_keep=np.where((tuc2_sigv < 15.))# & (tuc2_skewv >= -1.) & (tuc2_skewv <= 1.) & (tuc2_kurtv >= -1) & (tuc2_kurtv <= 1))
tuc2_mem=np.where((tuc2_sigv < 15.) & (tuc2_pmem > 0.5))
tuc2_mem2=np.where((tuc2_sigv < 15.) & (tuc2_pmem < 0.8))
tuc2_mem3=np.where((tuc2_sigv < 15.) & (tuc2_pmem > 0.5))






with open(data2) as f: # read data file
    data=f.readlines()
    
gru1_radeg=[]
gru1_decdeg=[]
gru1_xi=[]
gru1_eta=[]
gru1_r=[]
gru1_hjd=[]
gru1_v=[]
gru1_sigv=[]
gru1_skewv=[]
gru1_kurtv=[]
gru1_teff=[]
gru1_sigteff=[]
gru1_skewteff=[]
gru1_kurtteff=[]
gru1_logg=[]
gru1_siglogg=[]
gru1_skewlogg=[]
gru1_kurtlogg=[]
gru1_feh=[]
gru1_sigfeh=[]
gru1_skewfeh=[]
gru1_kurtfeh=[]
gru1_pmem=[]
gru1_gmag=[]
gru1_rmag=[]
gru1_imag=[]
gru1_extg=[]
gru1_extr=[]
gru1_exti=[]

for line in data: # fill arrays
    p=line.split()
    gru1_radeg.append(float(p[0]))
    gru1_decdeg.append(float(p[1]))
    gru1_xi.append(float(p[2]))
    gru1_eta.append(float(p[3]))
    gru1_r.append(float(p[4]))
    gru1_v.append(float(p[5]))
    gru1_sigv.append(float(p[6]))
    gru1_teff.append(float(p[7]))
    gru1_sigteff.append(float(p[8]))
    gru1_logg.append(float(p[9]))
    gru1_siglogg.append(float(p[10]))
    gru1_feh.append(float(p[11]))
    gru1_sigfeh.append(float(p[12]))
    gru1_pmem.append(float(p[13]))
    gru1_gmag.append(float(p[14]))
    gru1_rmag.append(float(p[15]))
    gru1_imag.append(float(p[16]))
    gru1_extg.append(float(p[17]))
    gru1_extr.append(float(p[18]))
    gru1_exti.append(float(p[19]))

gru1_radeg=np.array(gru1_radeg)
gru1_decdeg=np.array(gru1_decdeg)
gru1_xi=np.array(gru1_xi)
gru1_eta=np.array(gru1_eta)
gru1_r=np.array(gru1_r)
gru1_hjd=np.array(gru1_hjd)
gru1_v=np.array(gru1_v)
gru1_sigv=np.array(gru1_sigv)
gru1_skewv=np.array(gru1_skewv)
gru1_kurtv=np.array(gru1_kurtv)
gru1_teff=np.array(gru1_teff)
gru1_sigteff=np.array(gru1_sigteff)
gru1_skewteff=np.array(gru1_skewteff)
gru1_kurtteff=np.array(gru1_kurtteff)
gru1_logg=np.array(gru1_logg)
gru1_siglogg=np.array(gru1_siglogg)
gru1_skewlogg=np.array(gru1_skewlogg)
gru1_kurtlogg=np.array(gru1_kurtlogg)
gru1_feh=np.array(gru1_feh)
gru1_sigfeh=np.array(gru1_sigfeh)
gru1_skewfeh=np.array(gru1_skewfeh)
gru1_kurtfeh=np.array(gru1_kurtfeh)
gru1_pmem=np.array(gru1_pmem)
gru1_gmag=np.array(gru1_gmag)
gru1_rmag=np.array(gru1_rmag)
gru1_imag=np.array(gru1_imag)
gru1_extg=np.array(gru1_extg)
gru1_extr=np.array(gru1_extr)
gru1_exti=np.array(gru1_exti)
gru1_sigr=gru1_r-gru1_r
gru1_mag=gru1_rmag-gru1_extr
gru1_col=gru1_gmag-gru1_extg-(gru1_rmag-gru1_extr)
gru1_sigmag=gru1_mag-gru1_mag
gru1_sigcol=gru1_col-gru1_col

#gru1_keep=np.where((gru1_sigv < 15.) & (gru1_skewv >= -1.) & (gru1_skewv <= 1.) & (gru1_kurtv >= -1) & (gru1_kurtv <= 1))
gru1_keep=np.where((gru1_sigv < 15.))# & (gru1_skewv >= -1.) & (gru1_skewv <= 1.) & (gru1_kurtv >= -1) & (gru1_kurtv <= 1))
gru1_mem=np.where((gru1_sigv < 15.) & (gru1_pmem > 0.5))
gru1_mem2=np.where((gru1_sigv < 15.) & (gru1_pmem < 0.8))
gru1_mem3=np.where((gru1_sigv < 15.) & (gru1_pmem > 0.5))


data3='age12fehm250.iso'
with open(data3) as g: # read data file
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



data3='age12fehm200.iso'
with open(data3) as g: # read data file
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



data3='age12fehm100.iso'
with open(data3) as g: # read data file
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


data3='age12fehm150.iso'
with open(data3) as g: # read data file
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


data3='age12fehm150.iso'
with open(data3) as g: # read data file
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



data3='age12fehm100.iso'
with open(data3) as g: # read data file
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



data3='age12fehm050.iso'
with open(data3) as g: # read data file
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



data3='age12fehm000.iso'
with open(data3) as g: # read data file
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
ax1_0.plot(iso250_teff,iso250_logg,lw=1,color='k',ls='-')
ax1_0.plot(iso100_teff,iso100_logg,lw=1,color='k',ls='--')
#ax1_0.plot(iso000_teff,iso000_logg,lw=1,color='g')
ax1_0.errorbar(tuc2_teff[tuc2_keep],tuc2_logg[tuc2_keep],xerr=tuc2_sigteff[tuc2_keep],yerr=tuc2_siglogg[tuc2_keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True,label='foreground')
ax1_0.errorbar(tuc2_teff[tuc2_mem],tuc2_logg[tuc2_mem],xerr=tuc2_sigteff[tuc2_mem],yerr=tuc2_siglogg[tuc2_mem],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='r',rasterized=True,label='Tuc II member')
ax1_0.errorbar(gru1_teff[gru1_keep],gru1_logg[gru1_keep],xerr=gru1_sigteff[gru1_keep],yerr=gru1_siglogg[gru1_keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax1_0.errorbar(gru1_teff[gru1_mem],gru1_logg[gru1_mem],xerr=gru1_sigteff[gru1_mem],yerr=gru1_siglogg[gru1_mem],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='b',rasterized=True,label='Gru I member')
#ax1_0.legend(loc=2,fontsize=8,handlelength=0,shadow=False)
#ax0_0.text(-0.4,16.5,'Crater',fontsize=10)
ax1_0.legend(loc=2,fontsize=10,handlelength=0,numpoints=1,scatterpoints=1,shadow=False)


#ax0_0.set_xlabel(r'$T_{\rm eff}$ [K]',fontsize=12,rotation=0,labelpad=5)
ax0_0.set_ylabel(r'$r-(m-M)$ [mag]',fontsize=12,rotation=90,labelpad=5)
ax0_0.set_xlim([8000,4000])
ax0_0.set_ylim([2.49,-4])
ax0_0.set_xscale(u'linear')
ax0_0.set_yscale(u'linear')
ax0_0.set_yticks([2,1,0,-1,-2,-3,-4])
ax0_0.set_xticks([8000,7000,6000,5000,4000])
ax0_0.xaxis.set_major_formatter(plt.NullFormatter())
ax0_0.plot(iso250_teff,iso250_r,lw=1,color='k',label=r"[Fe/H]=$-2.5$",ls='-')
ax0_0.plot(iso100_teff,iso100_r,lw=1,color='k',label=r"[Fe/H]=$-1.0$",ls='--')
#ax0_0.plot(iso000_teff,iso000_r+dmodulus,lw=1,color='g',label=r"[Fe/H]=$0$")
#ax0_0.errorbar(teff,rmag,xerr=sigteff,yerr=siglogg-siglogg,elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax0_0.errorbar(tuc2_teff[tuc2_mem],tuc2_mag[tuc2_mem]-tuc2_dmodulus,xerr=tuc2_sigteff[tuc2_mem],yerr=tuc2_siglogg[tuc2_mem]-tuc2_siglogg[tuc2_mem],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='r',rasterized=True)
ax0_0.errorbar(gru1_teff[gru1_mem],gru1_mag[gru1_mem]-gru1_dmodulus,xerr=gru1_sigteff[gru1_mem],yerr=gru1_siglogg[gru1_mem]-gru1_siglogg[gru1_mem],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='b',rasterized=True)

ax0_0.legend(loc=2,fontsize=12,handlelength=2,shadow=False)
#ax0_0.text(-0.4,16.5,'Crater',fontsize=10)
#ax0_0.text(7500,16.5,'Tuc 2',fontsize=12)

ax1_1.set_xlabel(r'$r-(m-M)$ [mag]',fontsize=12,rotation=0,labelpad=5)
#ax1_1.set_ylabel(r'$\log g$',fontsize=12,rotation=90,labelpad=5)
ax1_1.set_xlim([2.49,-4])
ax1_1.set_ylim([5,0.01])
ax1_1.set_xscale(u'linear')
ax1_1.set_yscale(u'linear')
ax1_1.set_yticks([5,4,3,2,1])
ax1_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax1_1.set_xticklabels(vticks,rotation=0)
ax1_1.set_xticks([2,1,0,-1,-2,-3,-4])
ax1_1.plot(iso250_r,iso250_logg,lw=1,color='k',label=r"[Fe/H]=$-2.5$",ls='-')
ax1_1.plot(iso100_r,iso100_logg,lw=1,color='k',label=r"[Fe/H]=$-1.0$",ls='--')
#ax1_1.plot(iso000_r+dmodulus,iso000_logg,lw=1,color='g',label=r"[Fe/H]=$0$")
#ax1_1.errorbar(rmag,logg,xerr=sigteff-sigteff,yerr=siglogg,elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax1_1.errorbar(tuc2_mag[tuc2_mem]-tuc2_dmodulus,tuc2_logg[tuc2_mem],xerr=tuc2_sigteff[tuc2_mem]-tuc2_sigteff[tuc2_mem],yerr=tuc2_siglogg[tuc2_mem],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='r',rasterized=True)
ax1_1.errorbar(gru1_mag[gru1_mem]-gru1_dmodulus,gru1_logg[gru1_mem],xerr=gru1_sigteff[gru1_mem]-gru1_sigteff[gru1_mem],yerr=gru1_siglogg[gru1_mem],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='b',rasterized=True)
#ax1_1.legend(loc=2,fontsize=12,handlelength=0,shadow=False)
#ax0_0.text(-0.4,16.5,'Crater',fontsize=10)





plotfilename='tuc2gru1teffprior_spectrocmd.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
