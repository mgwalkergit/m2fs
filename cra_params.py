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
plt.rc('xtick',labelsize='6')
plt.rc('ytick',labelsize='6')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='eps')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)
            
data1='cra.dat'
out1='cra_newcommands.tex'
g1=open(out1,'w')

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

vmem=[142.,158.]
loggmem=[0.1,3.5]
fehmem=[-2.5,-0.4]
teffmem=[4050,7000]
rmem=[0.,4.]
magmem=[23,16]
colmem=[-0.2,2]
magmem2=magmem
colmem2=colmem

gs=plt.GridSpec(7,5) # define multi-panel plot
gs2=plt.GridSpec(5,5) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
gs2.update(wspace=1,hspace=1) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_

#ax4_4=fig.add_subplot(gs2[3:,3:])

ax6_0=fig.add_subplot(gs[6,0])
ax5_0=fig.add_subplot(gs[5,0])
ax4_0=fig.add_subplot(gs[4,0])
ax3_0=fig.add_subplot(gs[3,0])

ax5_1=fig.add_subplot(gs[5,1])
ax4_1=fig.add_subplot(gs[4,1])
ax3_1=fig.add_subplot(gs[3,1])

ax4_2=fig.add_subplot(gs[4,2])
ax3_2=fig.add_subplot(gs[3,2])

ax3_3=fig.add_subplot(gs[3,3])

ax1_4=fig.add_subplot(gs[1,4])
ax2_4=fig.add_subplot(gs[2,4])
ax3_4=fig.add_subplot(gs[3,4])


ax0_3=fig.add_subplot(gs[0,3])
ax0_2=fig.add_subplot(gs[0,2])
ax0_1=fig.add_subplot(gs[0,1])
ax0_0=fig.add_subplot(gs[0,0])

ax1_0=fig.add_subplot(gs[1,0])
ax1_1=fig.add_subplot(gs[1,1])
ax1_2=fig.add_subplot(gs[1,2])
ax1_3=fig.add_subplot(gs[1,3])
ax2_0=fig.add_subplot(gs[2,0])
ax2_1=fig.add_subplot(gs[2,1])
ax2_2=fig.add_subplot(gs[2,2])
ax2_3=fig.add_subplot(gs[2,3])

vlim=[-25.,199.]
tefflim=[4001.,7999.] 
revtefflim=[8000.,4000.] 
logglim=[0.01,4.99]
revlogglim=[5,0]
fehlim=[-4.99,0.99]
rlim=[0.,20.]
collim=[0,2]
maglim=[16.5,23.]

vlabel=[r'$v_{\rm los}$ [km/s]']
tefflabel=[r'$T_{\rm eff}$ [K]']
logglabel=[r'$\log_{10}[g/(\mathrm{cm/s}^2)]$']
fehlabel=[r'$[\mathrm{Fe/H}]$']
rlabel=[r'$R$ [arcmin]']
collabel=[r'g-r [mag]']
maglabel=[r'r [mag]']

histticks=[3,6,9,12,15]
histlim=[0,15]
vticks=[0,50,100,150,200]
vticklabels=['0','50','100','150','200']
loggticks=[1,2,3,4]
revloggticks=[5,4,3,2,1,0]
logglabels=['1','2','3','4']
fehticks=[-4,-3,-2,-1,0]
fehlabels=['-4','-3','-2','-1','0']
teffticks=[5000,6000,7000]
revteffticks=[8000,7000,6000,5000,4000]
teffticklabels=['5000','6000','7000']
colmap=plt.get_cmap("Greys")
rticks=[5,10,15]
rticklabels=['5','10','15']
colticks=[0,0.5,1,1.5]
colticklabels=colticks
magticks=[17,18,19,20,21,22]
magticklabels=magticks

vteffx=[vmem[0],vmem[1],vmem[1],vmem[0],vmem[0]]
vteffy=[teffmem[0],teffmem[0],teffmem[1],teffmem[1],teffmem[0]]
vloggx=[vmem[0],vmem[1],vmem[1],vmem[0],vmem[0]]
vloggy=[loggmem[0],loggmem[0],loggmem[1],loggmem[1],loggmem[0]]
vfehx=[vmem[0],vmem[1],vmem[1],vmem[0],vmem[0]]
vfehy=[fehmem[0],fehmem[0],fehmem[1],fehmem[1],fehmem[0]]
vrx=[vmem[0],vmem[1],vmem[1],vmem[0],vmem[0]]
vry=[rmem[0],rmem[0],rmem[1],rmem[1],rmem[0]]
vcolx=[vmem[0],vmem[1],vmem[1],vmem[0],vmem[0]]
vcoly=[colmem[0],colmem[0],colmem[1],colmem[1],colmem[0]]
vmagx=[vmem[0],vmem[1],vmem[1],vmem[0],vmem[0]]
vmagy=[magmem[0],magmem[0],magmem[1],magmem[1],magmem[0]]

teffloggx=[teffmem[0],teffmem[1],teffmem[1],teffmem[0],teffmem[0]]
teffloggy=[loggmem[0],loggmem[0],loggmem[1],loggmem[1],loggmem[0]]
tefffehx=[teffmem[0],teffmem[1],teffmem[1],teffmem[0],teffmem[0]]
tefffehy=[fehmem[0],fehmem[0],fehmem[1],fehmem[1],fehmem[0]]
teffrx=[teffmem[0],teffmem[1],teffmem[1],teffmem[0],teffmem[0]]
teffry=[rmem[0],rmem[0],rmem[1],rmem[1],rmem[0]]
teffcolx=[teffmem[0],teffmem[1],teffmem[1],teffmem[0],teffmem[0]]
teffcoly=[colmem[0],colmem[0],colmem[1],colmem[1],colmem[0]]
teffmagx=[teffmem[0],teffmem[1],teffmem[1],teffmem[0],teffmem[0]]
teffmagy=[magmem[0],magmem[0],magmem[1],magmem[1],magmem[0]]


loggfehx=[loggmem[0],loggmem[1],loggmem[1],loggmem[0],loggmem[0]]
loggfehy=[fehmem[0],fehmem[0],fehmem[1],fehmem[1],fehmem[0]]
loggrx=[loggmem[0],loggmem[1],loggmem[1],loggmem[0],loggmem[0]]
loggry=[rmem[0],rmem[0],rmem[1],rmem[1],rmem[0]]
loggcolx=[loggmem[0],loggmem[1],loggmem[1],loggmem[0],loggmem[0]]
loggcoly=[colmem[0],colmem[0],colmem[1],colmem[1],colmem[0]]
loggmagx=[loggmem[0],loggmem[1],loggmem[1],loggmem[0],loggmem[0]]
loggmagy=[magmem[0],magmem[0],magmem[1],magmem[1],magmem[0]]


fehrx=[fehmem[0],fehmem[1],fehmem[1],fehmem[0],fehmem[0]]
fehry=[rmem[0],rmem[0],rmem[1],rmem[1],rmem[0]]
fehcolx=[fehmem[0],fehmem[1],fehmem[1],fehmem[0],fehmem[0]]
fehcoly=[colmem[0],colmem[0],colmem[1],colmem[1],colmem[0]]
fehmagx=[fehmem[0],fehmem[1],fehmem[1],fehmem[0],fehmem[0]]
fehmagy=[magmem[0],magmem[0],magmem[1],magmem[1],magmem[0]]

ax6_0.set_xlabel(vlabel[0],fontsize=8,rotation=0)
ax6_0.set_ylabel(tefflabel[0],fontsize=8,rotation=90,labelpad=5)
ax6_0.set_xlim(vlim)
ax6_0.set_ylim(tefflim)
ax6_0.set_xscale(u'linear')
ax6_0.set_yscale(u'linear')
ax6_0.set_xticks(vticks)
ax6_0.set_xticklabels(vticks,rotation=0)
ax6_0.set_yticks([4000]+teffticks)
ax6_0.set_yticklabels([4000]+teffticks,rotation=0)
ax6_0.errorbar(v[keep],teff[keep],xerr=sigv[keep],yerr=sigteff[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax6_0.plot(vteffx,vteffy,color='r',linewidth=0.25)
#ax6_0.plot(vteffx2,vteffy2,color='b',linewidth=0.25)

ax5_0.set_ylabel(logglabel[0],fontsize=8,rotation=90,labelpad=5)
ax5_0.set_xlim(vlim)
ax5_0.set_ylim(logglim)
ax5_0.set_xscale(u'linear')
ax5_0.set_yscale(u'linear')
ax5_0.set_xticks(vticks)
#ax5_0.set_xticklabels(vticks,rotation=0)
ax5_0.set_yticks(loggticks)
ax5_0.set_yticklabels(loggticks,rotation=0)
ax5_0.xaxis.set_major_formatter(plt.NullFormatter())
ax5_0.errorbar(v[keep],logg[keep],xerr=sigv[keep],yerr=siglogg[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax5_0.plot(vloggx,vloggy,color='r',linewidth=0.25)
#ax5_0.plot(vloggx2,vloggy2,color='b',linewidth=0.25)

ax4_0.set_ylabel(fehlabel[0],fontsize=8,rotation=90,labelpad=5)
ax4_0.set_xlim(vlim)
ax4_0.set_ylim(fehlim)
ax4_0.set_xscale(u'linear')
ax4_0.set_yscale(u'linear')
ax4_0.set_xticks(vticks)
#ax4_0.set_xticklabels(vticks,rotation=0)
ax4_0.set_yticks(fehticks)
ax4_0.set_yticklabels(fehticks,rotation=0)
ax4_0.xaxis.set_major_formatter(plt.NullFormatter())
ax4_0.errorbar(v[keep],feh[keep],xerr=sigv[keep],yerr=sigfeh[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax4_0.plot(vfehx,vfehy,color='r',linewidth=0.25)
#ax4_0.plot(vfehx2,vfehy2,color='b',linewidth=0.25)

ax3_0.set_ylabel(rlabel[0],fontsize=8,rotation=90,labelpad=5)
ax3_0.set_xlim(vlim)
ax3_0.set_ylim(rlim)
ax3_0.set_xscale(u'linear')
ax3_0.set_yscale(u'linear')
ax3_0.set_xticks(vticks)
#ax3_0.set_xticklabels(vticks,rotation=0)
ax3_0.set_yticks(rticks)
ax3_0.set_yticklabels(rticks,rotation=0)
ax3_0.xaxis.set_major_formatter(plt.NullFormatter())
ax3_0.errorbar(v[keep],r[keep],xerr=sigv[keep],yerr=sigr[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax3_0.plot(vrx,vry,color='r',linewidth=0.25)
#ax3_0.plot(vrx2,vry2,color='b',linewidth=0.25)

ax0_0.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax0_0.xaxis.set_major_formatter(plt.NullFormatter())
ax0_0.set_ylim(histlim)
ax0_0.set_xticks(vticks)
ax0_0.set_yticks(histticks)
#ax0_0.yaxis.set_major_formatter(plt.NullFormatter())
ax0_0.hist(v[keep],bins=30,range=vlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='k',linewidth=0.25)
#ax0_0.hist(v[mem3],bins=30,range=vlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25,alpha=0.99)
ax0_0.hist(v[mem],bins=30,range=vlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=1)


ax5_1.set_xlabel(tefflabel[0],fontsize=8,rotation=0)
#ax5_1.set_ylabel(logglabel[0],fontsize=8,rotation=90,labelpad=5)
ax5_1.set_xlim(tefflim)
ax5_1.set_ylim(logglim)
ax5_1.set_xscale(u'linear')
ax5_1.set_yscale(u'linear')
ax5_1.set_xticks(teffticks+[8000])
ax5_1.set_xticklabels(teffticks+[8000],rotation=0)
ax5_1.set_yticks(loggticks)
ax5_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax5_1.set_yticklabels(loggticks,rotation=0)
ax5_1.errorbar(teff[keep],logg[keep],xerr=sigteff[keep],yerr=siglogg[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax5_1.plot(teffloggx,teffloggy,color='r',linewidth=0.25)
#ax5_1.plot(teffloggx2,teffloggy2,color='b',linewidth=0.25)

#ax4_1.set_ylabel(fehlabel[0],fontsize=8,rotation=90,labelpad=5)
ax4_1.set_xlim(tefflim)
ax4_1.set_ylim(fehlim)
ax4_1.set_xscale(u'linear')
ax4_1.set_yscale(u'linear')
ax4_1.set_xticks(teffticks)
#ax4_1.set_xticklabels(teffticks,rotation=0)
ax4_1.set_yticks(fehticks)
#ax4_1.set_yticklabels(fehticks,rotation=0)
ax4_1.xaxis.set_major_formatter(plt.NullFormatter())
ax4_1.yaxis.set_major_formatter(plt.NullFormatter())
ax4_1.errorbar(teff[keep],feh[keep],xerr=sigteff[keep],yerr=sigfeh[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax4_1.plot(tefffehx,tefffehy,color='r',linewidth=0.25)
#ax4_1.plot(tefffehx2,tefffehy2,color='b',linewidth=0.25)

#ax3_1.set_ylabel(rlabel[0],fontsize=8,rotation=90,labelpad=5)
ax3_1.set_xlim(tefflim)
ax3_1.set_ylim(rlim)
ax3_1.set_xscale(u'linear')
ax3_1.set_yscale(u'linear')
ax3_1.set_xticks(teffticks)
#ax3_1.set_xticklabels(teffticks,rotation=0)
ax3_1.set_yticks(rticks)
#ax3_1.set_yticklabels(rticks,rotation=0)
ax3_1.xaxis.set_major_formatter(plt.NullFormatter())
ax3_1.yaxis.set_major_formatter(plt.NullFormatter())
ax3_1.errorbar(teff[keep],r[keep],xerr=sigteff[keep],yerr=sigr[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax3_1.plot(teffrx,teffry,color='r',linewidth=0.25)
#ax3_1.plot(teffrx2,teffry2,color='b',linewidth=0.25)

ax0_1.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax0_1.xaxis.set_major_formatter(plt.NullFormatter())
ax0_1.yaxis.set_major_formatter(plt.NullFormatter())
ax0_1.set_ylim(histlim)
ax0_1.set_xticks(teffticks)
ax0_1.set_yticks(histticks)
ax0_1.hist(teff[keep],bins=30,range=tefflim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='k',linewidth=0.25)
#ax0_1.hist(teff[mem3],bins=30,range=tefflim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25,alpha=0.99)
ax0_1.hist(teff[mem],bins=30,range=tefflim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=1)



ax4_2.set_xlabel(logglabel[0],fontsize=8,rotation=0)
#ax4_2.set_ylabel(fehlabel[0],fontsize=8,rotation=90,labelpad=5)
ax4_2.set_xlim(logglim)
ax4_2.set_ylim(fehlim)
ax4_2.set_xscale(u'linear')
ax4_2.set_yscale(u'linear')
ax4_2.set_xticks(loggticks+[5])
ax4_2.set_xticklabels(loggticks+[5],rotation=0)
ax4_2.set_yticks(fehticks)
ax4_2.yaxis.set_major_formatter(plt.NullFormatter())
#ax4_2.set_yticklabels(fehticks,rotation=0)
ax4_2.errorbar(logg[keep],feh[keep],xerr=siglogg[keep],yerr=sigfeh[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax4_2.plot(loggfehx,loggfehy,color='r',linewidth=0.25)
#ax4_2.plot(loggfehx2,loggfehy2,color='b',linewidth=0.25)

#ax3_2.set_ylabel(rlabel[0],fontsize=8,rotation=90,labelpad=5)
ax3_2.set_xlim(logglim)
ax3_2.set_ylim(rlim)
ax3_2.set_xscale(u'linear')
ax3_2.set_yscale(u'linear')
ax3_2.set_xticks(loggticks)
#ax3_2.set_xticklabels(loggticks,rotation=0)
ax3_2.set_yticks(rticks)
#ax3_2.set_yticklabels(rticks,rotation=0)
ax3_2.xaxis.set_major_formatter(plt.NullFormatter())
ax3_2.yaxis.set_major_formatter(plt.NullFormatter())
ax3_2.errorbar(logg[keep],r[keep],xerr=siglogg[keep],yerr=sigr[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax3_2.plot(loggrx,loggry,color='r',linewidth=0.25)
#ax3_2.plot(loggrx2,loggry2,color='b',linewidth=0.25)

ax0_2.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax0_2.xaxis.set_major_formatter(plt.NullFormatter())
ax0_2.yaxis.set_major_formatter(plt.NullFormatter())
ax0_2.set_ylim(histlim)
ax0_2.set_xticks(loggticks)
ax0_2.set_yticks(histticks)
ax0_2.hist(logg[keep],bins=30,range=logglim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='k',linewidth=0.25)
#ax0_2.hist(logg[mem3],bins=30,range=logglim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25,alpha=0.99)
ax0_2.hist(logg[mem],bins=30,range=logglim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=1)




ax3_3.set_xlabel(fehlabel[0],fontsize=8,rotation=0)
#ax3_3.set_ylabel(rlabel[0],fontsize=8,rotation=90,labelpad=5)
ax3_3.set_xlim(fehlim)
ax3_3.set_ylim(rlim)
ax3_3.set_xscale(u'linear')
ax3_3.set_yscale(u'linear')
ax3_3.set_xticks(fehticks)
ax3_3.set_xticklabels(fehticks,rotation=0)
ax3_3.set_yticks(rticks)
ax3_3.yaxis.set_major_formatter(plt.NullFormatter())
#ax3_3.set_yticklabels(rticks,rotation=0)
ax3_3.errorbar(feh[keep],r[keep],xerr=sigfeh[keep],yerr=sigr[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
ax3_3.plot(fehrx,fehry,color='r',linewidth=0.25)
#ax3_3.plot(fehrx2,fehry2,color='b',linewidth=0.25)

ax0_3.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax0_3.xaxis.set_major_formatter(plt.NullFormatter())
ax0_3.yaxis.set_major_formatter(plt.NullFormatter())
ax0_3.set_ylim(histlim)
ax0_3.set_xticks(fehticks)
ax0_3.set_yticks(histticks)
ax0_3.hist(feh[keep],bins=30,range=fehlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='k',linewidth=0.25)
#ax0_3.hist(feh[mem3],bins=30,range=fehlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25,alpha=0.99)
ax0_3.hist(feh[mem],bins=30,range=fehlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=1)


#ax1_4.set_xlabel('N',fontsize=8,rotation=0,labelpad=5)
#ax1_4.xaxis.set_major_formatter(plt.NullFormatter())
ax1_4.yaxis.set_major_formatter(plt.NullFormatter())
ax1_4.xaxis.set_major_formatter(plt.NullFormatter())
ax1_4.set_xlim(histlim)
ax1_4.set_yticks(magticks)
ax1_4.set_xticks(histticks)
ax1_4.hist(mag[keep],bins=30,range=maglim,normed=False,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='k',linewidth=0.25)
#ax1_4.hist(mag[mem3],bins=30,range=maglim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.25,alpha=0.99)
ax1_4.hist(mag[mem],bins=30,range=maglim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.25,alpha=1)

#ax2_4.set_xlabel('N',fontsize=8,rotation=0,labelpad=5)
#ax2_4.xaxis.set_major_formatter(plt.NullFormatter())
ax2_4.yaxis.set_major_formatter(plt.NullFormatter())
ax2_4.xaxis.set_major_formatter(plt.NullFormatter())
ax2_4.set_xlim(histlim)
ax2_4.set_yticks(colticks)
ax2_4.set_xticks(histticks)
ax2_4.hist(col[keep],bins=30,range=collim,normed=False,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='k',linewidth=0.25)
#ax2_4.hist(col[mem3],bins=30,range=collim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.25,alpha=0.99)
ax2_4.hist(col[mem],bins=30,range=collim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.25,alpha=1)

ax3_4.set_xlabel('N',fontsize=8,rotation=0,labelpad=5)
#ax3_4.xaxis.set_major_formatter(plt.NullFormatter())
ax3_4.yaxis.set_major_formatter(plt.NullFormatter())
ax3_4.set_xlim(histlim)
ax3_4.set_yticks(rticks)
ax3_4.set_xticks(histticks)
ax3_4.hist(r[keep],bins=30,range=rlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='k',linewidth=0.25)
#ax3_4.hist(r[mem3],bins=30,range=rlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.25,alpha=0.99)
ax3_4.hist(r[mem],bins=30,range=rlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.25,alpha=1)


#ax1_0.set_xlabel(vlabel[0],fontsize=8,rotation=0)
ax2_0.set_ylabel(collabel[0],fontsize=8,rotation=90,labelpad=5)
ax2_0.set_xlim(vlim)
ax2_0.set_ylim(collim)
ax2_0.set_xscale(u'linear')
ax2_0.set_yscale(u'linear')
ax2_0.set_xticks(vticks)
ax2_0.set_xticklabels(vticks,rotation=0)
ax2_0.set_yticks(colticks)
ax2_0.xaxis.set_major_formatter(plt.NullFormatter())
#ax2_0.set_yticklabels(rticks,rotation=0)
ax2_0.errorbar(v[keep],col[keep],xerr=sigv[keep],yerr=sigcol[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax2_0.plot(vcolx,vcoly,color='r',linewidth=0.25)
#ax2_0.plot(vcolx2,vcoly2,color='b',linewidth=0.25)

ax1_0.set_ylabel(maglabel[0],fontsize=8,rotation=90,labelpad=5)
ax1_0.set_xlim(vlim)
ax1_0.set_ylim(maglim)
ax1_0.set_xscale(u'linear')
ax1_0.set_yscale(u'linear')
ax1_0.set_xticks(vticks)
ax1_0.set_xticklabels(vticks,rotation=0)
ax1_0.set_yticks(magticks)
ax1_0.xaxis.set_major_formatter(plt.NullFormatter())
#ax1_0.set_yticklabels(rticks,rotation=0)
ax1_0.errorbar(v[keep],mag[keep],xerr=sigv[keep],yerr=sigmag[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax1_0.plot(vmagx,vmagy,color='r',linewidth=0.25)
#ax1_0.plot(vmagx2,vmagy2,color='b',linewidth=0.25)

#ax2_1.set_ylabel(maglabel[0],fontsize=8,rotation=90,labelpad=5)
ax2_1.set_xlim(tefflim)
ax2_1.set_ylim(collim)
ax2_1.set_xscale(u'linear')
ax2_1.set_yscale(u'linear')
ax2_1.set_xticks(teffticks)
ax2_1.set_xticklabels(teffticks,rotation=0)
ax2_1.set_yticks(colticks)
ax2_1.xaxis.set_major_formatter(plt.NullFormatter())
ax2_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax2_1.set_yticklabels(rticks,rotation=0)
ax2_1.errorbar(teff[keep],col[keep],xerr=sigteff[keep],yerr=sigcol[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax2_1.plot(teffcolx,teffcoly,color='r',linewidth=0.25)
#ax2_1.plot(teffcolx2,teffcoly2,color='b',linewidth=0.25)

#ax1_1.set_ylabel(maglabel[0],fontsize=8,rotation=90,labelpad=5)
ax1_1.set_xlim(tefflim)
ax1_1.set_ylim(maglim)
ax1_1.set_xscale(u'linear')
ax1_1.set_yscale(u'linear')
ax1_1.set_xticks(teffticks)
ax1_1.set_xticklabels(teffticks,rotation=0)
ax1_1.set_yticks(magticks)
ax1_1.xaxis.set_major_formatter(plt.NullFormatter())
ax1_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax1_1.set_yticklabels(rticks,rotation=0)
ax1_1.errorbar(teff[keep],mag[keep],xerr=sigteff[keep],yerr=sigmag[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax1_1.plot(teffmagx,teffmagy,color='r',linewidth=0.25)
#ax1_1.plot(teffmagx2,teffmagy2,color='b',linewidth=0.25)

#ax2_2.set_ylabel(collabel[0],fontsize=8,rotation=90,labelpad=5)
ax2_2.set_xlim(logglim)
ax2_2.set_ylim(collim)
ax2_2.set_xscale(u'linear')
ax2_2.set_yscale(u'linear')
ax2_2.set_xticks(loggticks)
ax2_2.set_xticklabels(loggticks,rotation=0)
ax2_2.set_yticks(colticks)
ax2_2.xaxis.set_major_formatter(plt.NullFormatter())
ax2_2.yaxis.set_major_formatter(plt.NullFormatter())
#ax2_2.set_yticklabels(rticks,rotation=0)
ax2_2.errorbar(logg[keep],col[keep],xerr=siglogg[keep],yerr=sigcol[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax2_2.plot(loggcolx,loggcoly,color='r',linewidth=0.25)
#ax2_2.plot(loggcolx2,loggcoly2,color='b',linewidth=0.25)

#ax1_2.set_ylabel(maglabel[0],fontsize=8,rotation=90,labelpad=5)
ax1_2.set_xlim(logglim)
ax1_2.set_ylim(maglim)
ax1_2.set_xscale(u'linear')
ax1_2.set_yscale(u'linear')
ax1_2.set_xticks(loggticks)
ax1_2.set_xticklabels(loggticks,rotation=0)
ax1_2.set_yticks(magticks)
ax1_2.xaxis.set_major_formatter(plt.NullFormatter())
ax1_2.yaxis.set_major_formatter(plt.NullFormatter())
#ax1_2.set_yticklabels(rticks,rotation=0)
ax1_2.errorbar(logg[keep],mag[keep],xerr=siglogg[keep],yerr=sigmag[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax1_2.plot(loggmagx,loggmagy,color='r',linewidth=0.25)
#ax1_2.plot(loggmagx2,loggmagy2,color='b',linewidth=0.25)

#ax2_3.set_ylabel(collabel[0],fontsize=8,rotation=90,labelpad=5)
ax2_3.set_xlim(fehlim)
ax2_3.set_ylim(collim)
ax2_3.set_xscale(u'linear')
ax2_3.set_yscale(u'linear')
ax2_3.set_xticks(fehticks)
ax2_3.set_xticklabels(fehticks,rotation=0)
ax2_3.set_yticks(colticks)
ax2_3.xaxis.set_major_formatter(plt.NullFormatter())
ax2_3.yaxis.set_major_formatter(plt.NullFormatter())
#ax2_3.set_yticklabels(rticks,rotation=0)
ax2_3.errorbar(feh[keep],col[keep],xerr=sigfeh[keep],yerr=sigcol[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax2_3.plot(fehcolx,fehcoly,color='r',linewidth=0.25)
#ax2_3.plot(fehcolx2,fehcoly2,color='b',linewidth=0.25)

#ax1_3.set_ylabel(maglabel[0],fontsize=8,rotation=90,labelpad=5)
ax1_3.set_xlim(fehlim)
ax1_3.set_ylim(maglim)
ax1_3.set_xscale(u'linear')
ax1_3.set_yscale(u'linear')
ax1_3.set_xticks(fehticks)
ax1_3.set_xticklabels(fehticks,rotation=0)
ax1_3.set_yticks(magticks)
ax1_3.xaxis.set_major_formatter(plt.NullFormatter())
ax1_3.yaxis.set_major_formatter(plt.NullFormatter())
#ax1_3.set_yticklabels(rticks,rotation=0)
ax1_3.errorbar(feh[keep],mag[keep],xerr=sigfeh[keep],yerr=sigmag[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax1_3.plot(fehmagx,fehmagy,color='r',linewidth=0.25)
#ax1_3.plot(fehmagx2,fehmagy2,color='b',linewidth=0.25)

#ax4_4.set_xlabel(tefflabel[0],fontsize=8,rotation=0)
#ax4_4.set_ylabel(logglabel[0],fontsize=8,rotation=90,labelpad=5)
#ax4_4.set_xlim(revtefflim)
#ax4_4.set_ylim(revlogglim)
#ax4_4.set_xscale(u'linear')
#ax4_4.set_yscale(u'linear')
#ax4_4.set_xticks(revteffticks)
#ax4_4.set_xticklabels(revteffticks,rotation=0)
#ax4_4.set_yticks(revloggticks)
#ax4_4.yaxis.set_major_formatter(plt.NullFormatter())
#ax4_4.set_yticklabels(revloggticks,rotation=0)
#ax4_4.errorbar(teff[keep],logg[keep],xerr=sigteff[keep],yerr=siglogg[keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax4_4.errorbar(teff[mem],logg[mem],xerr=sigteff[mem],yerr=siglogg[mem],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='r',rasterized=True)
#ax4_4.errorbar(teff[mem2],logg[mem2],xerr=sigteff[mem2],yerr=siglogg[mem2],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='b',rasterized=True)

g1.write(r'\newcommand{\nobs}{$'+str(len(v))+r'$}'+'\n')
g1.write(r'\newcommand{\goodnobs}{$'+str(len(v[keep])-1)+r'$}'+'\n')
g1.write(r'\newcommand{\nred}{$'+str(len(v[mem]))+r'$}'+'\n')
#g1.write(r'\newcommand{\nblue}{$'+str(len(v[mem3]))+r'$}'+'\n')
g1.write(r'\newcommand{\medsigv}{$'+str(round(np.median(sigv[keep]),2))+r'$}'+'\n')
g1.write(r'\newcommand{\medsigteff}{$'+str(int(np.median(sigteff[keep])))+r'$}'+'\n')
g1.write(r'\newcommand{\medsiglogg}{$'+str(round(np.median(siglogg[keep]),2))+r'$}'+'\n')
g1.write(r'\newcommand{\medsigfeh}{$'+str(round(np.median(sigfeh[keep]),2))+r'$}'+'\n')
g1.write(r'\newcommand{\members}{$'+str(np.size(mem))+r'$}'+'\n')
g1.write(r'\newcommand{\minsigv}{$'+str(round(min(sigv[keep]),2))+r'$}'+'\n')
g1.write(r'\newcommand{\maxsigv}{$'+str(round(max(sigv[keep]),2))+r'$}'+'\n')
g1.write(r'\newcommand{\minsigteff}{$'+str(round(min(sigteff[keep]),2))+r'$}'+'\n')
g1.write(r'\newcommand{\maxsigteff}{$'+str(round(max(sigteff[keep]),2))+r'$}'+'\n')
g1.write(r'\newcommand{\minsiglogg}{$'+str(round(min(siglogg[keep]),2))+r'$}'+'\n')
g1.write(r'\newcommand{\maxsiglogg}{$'+str(round(max(siglogg[keep]),2))+r'$}'+'\n')
g1.write(r'\newcommand{\minsigfeh}{$'+str(round(min(sigfeh[keep]),2))+r'$}'+'\n')
g1.write(r'\newcommand{\maxsigfeh}{$'+str(round(max(sigfeh[keep]),2))+r'$}'+'\n')
g1.close()

plotfilename='cra_params.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
