import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits 
from scipy import stats
import random
import mycode2
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
plt.rc('xtick',labelsize='6')
plt.rc('ytick',labelsize='6')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='eps')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)
            
dmodulus=18.8
data1='tuc2_vasilyphot.dat'
data2='age12fehm250.iso'
#data2='parsec.dat'
data3='tuc2.dat'
data4='tuc2_ellipsehalf.res'

with open(data2) as g: # read data file
    data=g.readlines()

iso_g=[]
iso_r=[]
iso_i=[]

for line in data: # fill arrays
    p=line.split()
    iso_g.append(float(p[6]))
    iso_r.append(float(p[7]))
    iso_i.append(float(p[8]))

iso_g=np.array(iso_g)
iso_r=np.array(iso_r)
iso_i=np.array(iso_i)

with open(data4) as g: # read data file
    data=g.readlines()

ellipse_x=[]
ellipse_y=[]

for line in data: # fill arrays
    p=line.split()
    ellipse_x.append(float(p[0]))
    ellipse_y.append(float(p[1]))

ellipse_x=np.array(ellipse_x)
ellipse_y=np.array(ellipse_y)

with open(data1) as f: # read data file
    data=f.readlines()

v_radeg=[]
v_decdeg=[]
v_gmag=[]
v_rmag=[]
v_imag=[]
v_extg=[]
v_extr=[]
v_exti=[]
v_spreadg=[]
v_spreadr=[]
v_spreadi=[]

for line in data: # fill arrays
    p=line.split()
    v_radeg.append(float(p[0]))
    v_decdeg.append(float(p[1]))
    v_rmag.append(float(p[2]))
    v_gmag.append(float(p[3]))
    v_imag.append(float(p[4]))
    v_extg.append(float(p[5]))
    v_extr.append(float(p[6]))
    v_exti.append(float(p[7]))
    v_spreadg.append(float(p[8]))
    v_spreadr.append(float(p[9]))
    v_spreadi.append(float(p[10]))

v_radeg=np.array(v_radeg)
v_decdeg=np.array(v_decdeg)
v_gmag=np.array(v_gmag)
v_rmag=np.array(v_rmag)
v_imag=np.array(v_imag)
v_extg=np.array(v_extg)
v_extr=np.array(v_extr)
v_exti=np.array(v_exti)
v_spreadg=np.array(v_spreadg)
v_spreadr=np.array(v_spreadr)
v_spreadi=np.array(v_spreadi)
v_gmag=v_gmag-v_extg
v_rmag=v_rmag-v_extr
v_imag=v_imag-v_exti

v_ra=v_radeg*np.pi/180.
v_dec=v_decdeg*np.pi/180.
v_radegcenter=342.9796
v_decdegcenter=-58.5689
v_racenter=np.zeros(len(v_ra))+v_radegcenter*np.pi/180.
v_deccenter=np.zeros(len(v_dec))+v_decdegcenter*np.pi/180.
v_xi,v_eta=mycode2.etaxiarr(v_ra,v_dec,v_racenter,v_deccenter)
v_r=np.sqrt(v_xi**2+v_eta**2)

v_cand=np.zeros(len(v_ra),dtype='int')
for i in range(0,len(v_cand)):
    dist=np.sqrt((v_rmag[i]-(iso_r+dmodulus))**2+((v_gmag[i]-v_rmag[i])-(iso_g-iso_r))**2)
    if (min(dist) < 0.2) & (v_rmag[i] < 22.):
        v_cand[i]=1

v_cands=np.where(v_cand == 1)

data1='tuc2.dat'

with open(data3) as f: # read data file
    data=f.readlines()
    
radeg=[]
decdeg=[]
xi=[]
eta=[]
r=[]
hjd=[]
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
extg=[]
extr=[]
exti=[]
snratio=[]
aperture=[]
identifier=[]
sciname=[]
resolution=[]

for line in data: # fill arrays
    p=line.split()
    radeg.append(float(p[0]))
    decdeg.append(float(p[1]))
    xi.append(float(p[2]))
    eta.append(float(p[3]))
    r.append(float(p[4]))
    hjd.append(float(p[5]))
    v.append(float(p[6]))
    sigv.append(float(p[7]))
    skewv.append(float(p[8]))
    kurtv.append(float(p[9]))
    teff.append(float(p[10]))
    sigteff.append(float(p[11]))
    skewteff.append(float(p[12]))
    kurtteff.append(float(p[13]))
    logg.append(float(p[14]))
    siglogg.append(float(p[15]))
    skewlogg.append(float(p[16]))
    kurtlogg.append(float(p[17]))
    feh.append(float(p[18]))
    sigfeh.append(float(p[19]))
    skewfeh.append(float(p[20]))
    kurtfeh.append(float(p[21]))
    pmem.append(float(p[22]))
    gmag.append(float(p[23]))
    rmag.append(float(p[24]))
    imag.append(float(p[25]))
    extg.append(float(p[26]))
    extr.append(float(p[27]))
    exti.append(float(p[28]))
    snratio.append(float(p[29]))
    aperture.append(float(p[30]))
    identifier.append(float(p[31]))
    sciname.append(p[32])
    resolution.append(p[33])

radeg=np.array(radeg)
decdeg=np.array(decdeg)
xi=np.array(xi)
eta=np.array(eta)
r=np.array(r)
hjd=np.array(hjd)
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
extg=np.array(extg)
extr=np.array(extr)
exti=np.array(exti)
gmag=gmag-extg
rmag=rmag-extr
imag=imag-exti
sigr=r-r
snratio=np.array(snratio)
aperture=np.array(aperture)
identifier=np.array(identifier)
sciname=np.array(sciname)
resolution=np.array(resolution)

keep=np.where((sigv < 10.)) #& (skewv > -1.) & (skewv < 1.) & (kurtv > -1.1) & (kurtv < 1.1))
mem1=np.where((sigv < 10.) & (pmem > 0.9))
mem2=np.where((sigv < 10.) & (pmem < 0.8))
mem3=np.where((sigv < 10.) & (pmem > 0.5))

vmem=[55.,80.]
vmem2=[50.,85.]
loggmem=[0.25,4.5]
loggmem2=[0.05,4.7]
fehmem=[-4,-1.1]
fehmem2=[-4.2,-1.]
teffmem=[4250,7000]
teffmem2=[4200,7050]
rmem=[0.,12.]
rmem2=[0.,15]

center=np.where((v_r < 15.))

gs=plt.GridSpec(10,10) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_
ax0_0=fig.add_subplot(gs[0:5,0:4])
ax0_1=fig.add_subplot(gs[0:5,5:10])

ax0_0.set_xlabel(r'$g-r$ [mag]',fontsize=8,rotation=0,labelpad=5)
ax0_0.set_ylabel(r'$r$ [mag]',fontsize=8,rotation=90,labelpad=5)
ax0_0.set_xlim([-0.5,1.5])
ax0_0.set_ylim([22.5,15.5])
ax0_0.set_xscale(u'linear')
ax0_0.set_yscale(u'linear')
ax0_0.set_yticks([22,21,20,19,18,17,16])
#ax0_0.set_xticklabels(vticks,rotation=0)
ax0_0.set_xticks([-0.5,0,0.5,1,1.5])
#ax0_0.set_yticklabels([4000]+teffticks,rotation=0)
#ax0_0.errorbar(v[keep],teff[keep],xerr=sigv[keep],yerr=sigteff[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax0_0.scatter(v_gmag[center]-v_rmag[center],v_rmag[center],s=2,lw=0,edgecolor='none',alpha=0.99,marker='o',color='0.5',rasterized=True)
#ax0_0.scatter(v_gmag[v_cands]-v_rmag[v_cands],v_rmag[v_cands],s=2,lw=0,edgecolor='none',alpha=0.99,marker='o',color='r',rasterized=True)
ax0_0.scatter(gmag-rmag,rmag,s=8,lw=0.75,facecolor='none',edgecolor='k',alpha=0.65,marker='o',color='k',rasterized=True)
ax0_0.scatter(gmag[keep]-rmag[keep],rmag[keep],s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='k',rasterized=True)
ax0_0.scatter(gmag[mem1]-rmag[mem1],rmag[mem1],s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='r',rasterized=True)
#ax0_0.scatter(gmag[mem2]-rmag[mem2],rmag[mem2],s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='b',rasterized=True)
ax0_0.text(-0.4,16.25,'Tuc 2',fontsize=12)


ax0_0.plot(iso_g-iso_r,iso_r+dmodulus,lw=1,color='r')
#ax0_0.plot(vteffx2,vteffy2,color='b',linewidth=0.25)

ax0_1.set_xlabel(r'$\xi$ [arcmin]',fontsize=8,rotation=0,labelpad=5)
ax0_1.set_ylabel(r'$\eta$ [arcmin]',fontsize=8,rotation=90,labelpad=1)
ax0_1.set_xlim([20,-20])
ax0_1.set_ylim([-20,20])
ax0_1.set_xscale(u'linear')
ax0_1.set_yscale(u'linear')
#ax0_0.set_yticks([24,23,22,21,20,19,18,17,16])
#ax0_0.set_xticklabels(vticks,rotation=0)
#ax0_0.set_xticks([0,0.5,1,1.5])
#ax0_0.set_yticklabels([4000]+teffticks,rotation=0)
#ax0_0.errorbar(v[keep],teff[keep],xerr=sigv[keep],yerr=sigteff[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
circle1=plt.Circle((1.5,2),14.75,color='0.25',alpha=0.35)
ax0_1.add_artist(circle1)
ax0_1.scatter(xi,eta,s=8,lw=0.75,facecolor='none',edgecolor='k',alpha=0.65,marker='o',color='k',rasterized=True)
ax0_1.scatter(v_xi[v_cands],v_eta[v_cands],s=2,lw=0,edgecolor='none',alpha=0.99,marker='o',color='0.5',rasterized=True)
ax0_1.scatter(xi[keep],eta[keep],s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='k',rasterized=True)
ax0_1.scatter(xi[mem1],eta[mem1],s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='r',rasterized=True)
#ax0_1.scatter(xi[mem2],eta[mem2],s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='b',rasterized=True)
#ax0_1.plot(ellipse_x,ellipse_y,color='k',linewidth=0.25)
#ax0_0.plot(vteffx2,vteffy2,color='b',linewidth=0.25)
ax0_1.arrow(-17,-17,5,0,head_width=1,head_length=1.5,fc='k',ec='k',linewidth=0.5)
ax0_1.arrow(-17,-17,0,5,head_width=1,head_length=1.5,fc='k',ec='k',linewidth=0.5)
ax0_1.text(-10,-15.25,'E',color='k')
ax0_1.text(-16,-9.5,'N',color='k')

plotfilename='tuc2_map.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
