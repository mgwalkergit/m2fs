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
            
dmodulus=21.
data1='cra_vasilyphot.dat'
data2='age12fehm180.iso'
#data2='parsec.dat'
data3='cra.dat'
data4='cra_ellipsecore.res'

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

#with open(data4) as g: # read data file
#    data=g.readlines()

#ellipse_x=[]
#ellipse_y=[]

#for line in data: # fill arrays
#    p=line.split()
#    ellipse_x.append(float(p[0]))
#    ellipse_y.append(float(p[1]))

#ellipse_x=np.array(ellipse_x)
#ellipse_y=np.array(ellipse_y)

with open(data1) as f: # read data file
    data=f.readlines()

v_radeg=[]
v_decdeg=[]
v_gmag=[]
v_rmag=[]
v_imag=[]
v_xi=[]
v_eta=[]
v_r=[]

for line in data: # fill arrays
    p=line.split()
    v_radeg.append(float(p[0]))
    v_decdeg.append(float(p[1]))
    v_gmag.append(float(p[2]))
    v_rmag.append(float(p[3]))
    v_imag.append(float(p[4]))
    v_xi.append(float(p[5]))
    v_eta.append(float(p[6]))
    v_r.append(float(p[7]))

v_radeg=np.array(v_radeg)
v_decdeg=np.array(v_decdeg)
v_gmag=np.array(v_gmag)
v_rmag=np.array(v_rmag)
v_imag=np.array(v_imag)
v_xi=np.array(v_xi)
v_eta=np.array(v_eta)
v_r=np.array(v_r)
v_ra=v_radeg*np.pi/180.
v_dec=v_decdeg*np.pi/180.

v_cand=np.zeros(len(v_ra),dtype='int')
for i in range(0,len(v_cand)):
    dist=np.sqrt((v_imag[i]-(iso_i+dmodulus))**2+((v_gmag[i]-v_imag[i])-(iso_g-iso_i))**2)
    if (min(dist) < 0.2) & (v_imag[i] < 24.):
        v_cand[i]=1

v_cands=np.where(v_cand == 1)

data1='cra.dat'

with open(data1) as f: # read data file
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
    rmag.append(float(p[17]))
    imag.append(float(p[19]))
#    extg.append(float(p[26]))
#    extr.append(float(p[27]))
#    exti.append(float(p[28]))

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
gmag=gmag#-extg
rmag=rmag#-extr
imag=imag#-exti
sigr=r-r

data1='cra_all.dat'
with open(data1) as f: # read data file
    data=f.readlines()
    
radegall=[]
decdegall=[]
xiall=[]
etaall=[]
rall=[]
hjdall=[]
vall=[]
sigvall=[]
skewvall=[]
kurtvall=[]
teffall=[]
sigteffall=[]
skewteffall=[]
kurtteffall=[]
loggall=[]
sigloggall=[]
skewloggall=[]
kurtloggall=[]
fehall=[]
sigfehall=[]
skewfehall=[]
kurtfehall=[]
pmemall=[]
gmagall=[]
rmagall=[]
imagall=[]
extgall=[]
extrall=[]
extiall=[]

for line in data: # fill arrays
    p=line.split()
    radegall.append(float(p[0]))
    decdegall.append(float(p[1]))
    xiall.append(float(p[2]))
    etaall.append(float(p[3]))
    rall.append(float(p[4]))
    hjdall.append(float(p[5]))
    vall.append(float(p[6]))
    sigvall.append(float(p[7]))
#    skewvall.append(float(p[8]))
#    kurtvall.append(float(p[9]))
    teffall.append(float(p[8]))
    sigteffall.append(float(p[9]))
#    skewteffall.append(float(p[12]))
#    kurtteffall.append(float(p[13]))
    loggall.append(float(p[10]))
    sigloggall.append(float(p[11]))
#    skewloggall.append(float(p[16]))
#    kurtloggall.append(float(p[17]))
    fehall.append(float(p[12]))
    sigfehall.append(float(p[13]))
#    skewfehall.append(float(p[20]))
#    kurtfehall.append(float(p[21]))
    pmemall.append(float(p[14]))
    gmagall.append(float(p[15]))
    rmagall.append(float(p[16]))
    imagall.append(float(p[17]))
#    extgall.append(float(p[26]))
#    extrall.append(float(p[27]))
#    extiall.append(float(p[28]))

radegall=np.array(radegall)
decdegall=np.array(decdegall)
xiall=np.array(xiall)
etaall=np.array(etaall)
rall=np.array(rall)
hjdall=np.array(hjdall)
vall=np.array(vall)
sigvall=np.array(sigvall)
skewvall=np.array(skewvall)
kurtvall=np.array(kurtvall)
teffall=np.array(teffall)
sigteffall=np.array(sigteffall)
skewteffall=np.array(skewteffall)
kurtteffall=np.array(kurtteffall)
loggall=np.array(loggall)
sigloggall=np.array(sigloggall)
skewloggall=np.array(skewloggall)
kurtloggall=np.array(kurtloggall)
fehall=np.array(fehall)
sigfehall=np.array(sigfehall)
skewfehall=np.array(skewfehall)
kurtfehall=np.array(kurtfehall)
pmemall=np.array(pmemall)
gmagall=np.array(gmagall)
rmagall=np.array(rmagall)
imagall=np.array(imagall)
extgall=np.array(extgall)
extrall=np.array(extrall)
extiall=np.array(extiall)
gmagall=gmagall#-extg
rmagall=rmagall#-extr
imagall=imagall#-exti

keep=np.where((sigv < 5.))# & (skewv > -1.) & (skewv < 1.) & (kurtv > -1.) & (kurtv < 1.))
mem1=np.where((sigv < 5.) & (pmem > 0.5))# (skewv > -1.) & (skewv < 1.) & (kurtv > -1.) & (kurtv < 1.) & (pmem > 0.5))

vmem=[142.,158.]
loggmem=[0.1,2.5]
fehmem=[-2.5,-0.4]
teffmem=[4050,7000]
rmem=[0.,4.]
magmem=[23,16]
colmem=[-0.2,2]
magmem2=magmem
colmem2=colmem

center=np.where((v_r < 15.))

gs=plt.GridSpec(10,10) # define multi-panel plot
gs2=plt.GridSpec(20,20) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_
ax0_0=fig.add_subplot(gs[0:5,0:4])
ax0_1=fig.add_subplot(gs[0:5,5:10])
ax1_1=fig.add_subplot(gs[6:10,5:9])

ax0_0.set_xlabel(r'$g-i$ [mag]',fontsize=8,rotation=0,labelpad=5)
ax0_0.set_ylabel(r'$i$ [mag]',fontsize=8,rotation=90,labelpad=5)
ax0_0.set_xlim([-0.5,2])
ax0_0.set_ylim([22,16])
ax0_0.set_xscale(u'linear')
ax0_0.set_yscale(u'linear')
ax0_0.set_yticks([22,21,20,19,18,17,16])
#ax0_0.set_xticklabels(vticks,rotation=0)
ax0_0.set_xticks([0,0.5,1,1.5,2])
#ax0_0.set_yticklabels([4000]+teffticks,rotation=0)
ax0_0.scatter(v_gmag[center]-v_imag[center],v_imag[center],s=2,lw=0,edgecolor='none',alpha=0.99,marker='o',color='0.1',rasterized=True)
#ax0_0.scatter(gmagall-imagall,imagall,s=8,lw=0.75,facecolor='none',edgecolor='k',alpha=0.65,marker='o',color='k',rasterized=True)
ax0_0.scatter(gmag-imag,imag,s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='k',rasterized=True)
ax0_0.scatter(gmag[mem1]-imag[mem1],imag[mem1],s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='r',rasterized=True)
ax0_0.text(-0.4,16.5,'Crater',fontsize=10)


ax0_0.plot(iso_g-iso_i,iso_i+dmodulus,lw=1,color='r')
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
circle1=plt.Circle((-1.3,-3),14.75,color='0.25',alpha=0.35)
ax0_1.add_artist(circle1)
ax0_1.scatter(xiall,etaall,s=8,lw=0.75,facecolor='none',edgecolor='k',alpha=0.65,marker='o',color='k',rasterized=True)
ax0_1.scatter(xi,eta,s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='k',rasterized=True)
ax0_1.scatter(v_xi[v_cands],v_eta[v_cands],s=2,lw=0,edgecolor='none',alpha=0.99,marker='o',color='0.5',rasterized=True)
#ax0_1.scatter(xi[keep],eta[keep],s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='k',rasterized=True)
ax0_1.scatter(xi[mem1],eta[mem1],s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='r',rasterized=True)
#ax0_1.scatter(xi[mem2],eta[mem2],s=15,lw=0,edgecolor='none',alpha=0.99,marker='o',color='b',rasterized=True)
#ax0_1.plot(ellipse_x,ellipse_y,color='k',linewidth=0.25)
#ax0_0.plot(vteffx2,vteffy2,color='b',linewidth=0.25)
ax0_1.arrow(10,10,5,0,head_width=1,head_length=1.5,fc='k',ec='k',linewidth=0.5)
ax0_1.arrow(10,10,0,5,head_width=1,head_length=1.5,fc='k',ec='k',linewidth=0.5)
ax0_1.text(17,11.25,'E',color='k')
ax0_1.text(9.5,15.4,'N',color='k')




ax1_1.set_xlabel(r'$\xi$ [arcmin]',fontsize=8,rotation=0,labelpad=5)
ax1_1.set_ylabel(r'$\eta$ [arcmin]',fontsize=8,rotation=90,labelpad=1)
ax1_1.set_xlim([1.5,-1.5])
ax1_1.set_ylim([-1.5,1.5])
ax1_1.set_xticks([1.5,0.5,-0.5,-1.5])
ax1_1.set_yticks([-1.5,-0.5,0.5,1.5])
ax1_1.set_xscale(u'linear')
ax1_1.set_yscale(u'linear')

ax1_1.scatter(xi[mem1],eta[mem1],s=15,c=v[mem1],lw=0,edgecolor='none',alpha=0.99,marker='o',cmap='jet',rasterized=True)
ax_cb=fig.add_subplot(gs[6:10,9:10])
xxx=ax0_0.scatter(xi[mem1],eta[mem1],s=15,c=v[mem1],lw=0,edgecolor='none',alpha=0.99,marker='o',cmap='jet',rasterized=True)
cb=plt.colorbar(xxx,ax_cb,orientation='vertical',drawedges=False)
#cb.set_ticks([140,145,150,155,160])
#barticks=[140,145,150,155,160]
#cb.set_ticklabels(barticks)
#cb.set_clim(140,160)
cb.ax.tick_params(labelsize=7,pad=3,width=1,length=1)
cb.outline.set_linewidth(0.5)
cb.solids.set_edgecolor("face")
#ax_cbtitle.axis('off')
ax1_1.text(-1.75,-1.9,r"$v_{\rm los}$ [km/s]",fontsize=10)

plotfilename='cra_map.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
