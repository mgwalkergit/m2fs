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
plt.rc('xtick',labelsize='7')
plt.rc('ytick',labelsize='7')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='eps')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)
plt.rc('path',simplify='True')
            
tuc2_data1='/physics2/mgwalker/chains/tuc2teffprior_mixturepost_equal_weights.dat'

with open(tuc2_data1) as f: # read data file
    data=f.readlines()

tuc2_fmem=[]
tuc2_vmean2=[]
tuc2_vdisp2=[]
tuc2_fehmean2=[]
tuc2_fehdisp2=[]
tuc2_vmean=[]
tuc2_vdisp=[]
tuc2_vgrad=[]
tuc2_vtheta=[]
tuc2_fehmean=[]
tuc2_fehdisp=[]
tuc2_fehgrad=[]
tuc2_like=[]

for line in data: # fill arrays
    p=line.split()
    tuc2_fmem.append(float(p[0]))
    tuc2_vmean.append(float(p[1]))
    tuc2_vdisp.append(float(p[2]))
    tuc2_fehmean.append(float(p[3]))
    tuc2_fehdisp.append(float(p[4]))
    tuc2_vmean2.append(float(p[5]))
    tuc2_vdisp2.append(float(p[6]))
    tuc2_fehmean2.append(float(p[7]))
    tuc2_fehdisp2.append(float(p[8]))
    tuc2_vgrad.append(float(p[9]))
    tuc2_vtheta.append(float(p[10]))
    tuc2_fehgrad.append(float(p[11]))
    tuc2_like.append(float(p[12]))

tuc2_fmem=np.array(tuc2_fmem)
tuc2_vmean2=np.array(tuc2_vmean2)
tuc2_vdisp2=np.array(tuc2_vdisp2)
tuc2_fehmean2=np.array(tuc2_fehmean2)
tuc2_fehdisp2=np.array(tuc2_fehdisp2)
tuc2_vmean=np.array(tuc2_vmean)
tuc2_vdisp=np.array(tuc2_vdisp)
tuc2_vgrad=np.array(tuc2_vgrad)
tuc2_vtheta=np.array(tuc2_vtheta)
tuc2_fehmean=np.array(tuc2_fehmean)
tuc2_fehdisp=np.array(tuc2_fehdisp)
tuc2_fehgrad=np.array(tuc2_fehgrad)
tuc2_like=np.array(tuc2_like)

#tuc2_vdisp=10.**tuc2_vdisp
#tuc2_fehdisp=10.**tuc2_fehdisp
#tuc2_vdisp2=10.**tuc2_vdisp2
#tuc2_fehdisp2=10.**tuc2_fehdisp2
gru1_data1='gru1teffprior_mixturepost_equal_weights.dat'

with open(gru1_data1) as f: # read data file
    data=f.readlines()

gru1_fmem=[]
gru1_vmean2=[]
gru1_vdisp2=[]
gru1_fehmean2=[]
gru1_fehdisp2=[]
gru1_vmean=[]
gru1_vdisp=[]
gru1_vgrad=[]
gru1_vtheta=[]
gru1_fehmean=[]
gru1_fehdisp=[]
gru1_fehgrad=[]
gru1_like=[]

for line in data: # fill arrays
    p=line.split()
    gru1_fmem.append(float(p[0]))
    gru1_vmean.append(float(p[1]))
    gru1_vdisp.append(float(p[2]))
    gru1_fehmean.append(float(p[3]))
    gru1_fehdisp.append(float(p[4]))
    gru1_vmean2.append(float(p[5]))
    gru1_vdisp2.append(float(p[6]))
    gru1_fehmean2.append(float(p[7]))
    gru1_fehdisp2.append(float(p[8]))
    gru1_vgrad.append(float(p[9]))
    gru1_vtheta.append(float(p[10]))
    gru1_fehgrad.append(float(p[11]))
    gru1_like.append(float(p[12]))

gru1_fmem=np.array(gru1_fmem)
gru1_vmean2=np.array(gru1_vmean2)
gru1_vdisp2=np.array(gru1_vdisp2)
gru1_fehmean2=np.array(gru1_fehmean2)
gru1_fehdisp2=np.array(gru1_fehdisp2)
gru1_vmean=np.array(gru1_vmean)
gru1_vdisp=np.array(gru1_vdisp)
gru1_vgrad=np.array(gru1_vgrad)
gru1_vtheta=np.array(gru1_vtheta)
gru1_fehmean=np.array(gru1_fehmean)
gru1_fehdisp=np.array(gru1_fehdisp)
gru1_fehgrad=np.array(gru1_fehgrad)
gru1_like=np.array(gru1_like)

#gru1_vdisp=10.**gru1_vdisp
#gru1_fehdisp=10.**gru1_fehdisp
#gru1_vdisp2=10.**gru1_vdisp2
#gru1_fehdisp2=10.**gru1_fehdisp2

gs=plt.GridSpec(16,16) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_

tuc2_statsvmean=np.percentile(tuc2_vmean,[2.5,16.,50.,84.,97.5])
tuc2_statsvdisp=np.percentile(tuc2_vdisp,[2.5,16.,50.,84.,97.5])
tuc2_statsvgrad=np.percentile(tuc2_vgrad,[2.5,16.,50.,84.,97.5])
tuc2_statsvtheta=np.percentile(tuc2_vtheta,[2.5,16.,50.,84.,97.5])
tuc2_statsfehdisp=np.percentile(tuc2_fehdisp,[2.5,16.,50.,84.,97.5])
tuc2_statsfehmean=np.percentile(tuc2_fehmean,[2.5,16.,50.,84.,97.5])
tuc2_statsfehgrad=np.percentile(tuc2_fehgrad,[2.5,16.,50.,84.,97.5])

gru1_statsvmean=np.percentile(gru1_vmean,[2.5,16.,50.,84.,97.5])
gru1_statsvdisp=np.percentile(gru1_vdisp,[2.5,16.,50.,84.,97.5])
gru1_statsvgrad=np.percentile(gru1_vgrad,[2.5,16.,50.,84.,97.5])
gru1_statsvtheta=np.percentile(gru1_vtheta,[2.5,16.,50.,84.,97.5])
gru1_statsfehdisp=np.percentile(gru1_fehdisp,[2.5,16.,50.,84.,97.5])
gru1_statsfehmean=np.percentile(gru1_fehmean,[2.5,16.,50.,84.,97.5])
gru1_statsfehgrad=np.percentile(gru1_fehgrad,[2.5,16.,50.,84.,97.5])

ax11_0=fig.add_subplot(gs[2:4,0:4])
ax10_0=fig.add_subplot(gs[4:8,0:4])
ax11_1=fig.add_subplot(gs[4:8,4:6])

ax0_0=fig.add_subplot(gs[10:12,0:4])
ax1_0=fig.add_subplot(gs[12:16,0:4])
ax1_1=fig.add_subplot(gs[12:16,4:6])

#    ax0_12=fig.add_subplot(gs[0,12])
vmeanlim=[-155,-110]
vdisplim=[0,19.9]
vgradlim=[0,4.5]
vthetalim=[-179,179]
fehmeanlim=[-3,-1] 
fehdisplim=[0.01,1.]
fehgradlim=[-0.99,0.99]  

vmeanticks=[-150,-140,-130,-120,-110]
vdispticks=[0,5,10,15]
vgradticks=[1,2,3,4]
vthetaticks=[-150,0,150]
fehmeanticks=[-3,-2.5,-2,-1.5,-1]
fehdispticks=[0.,0.25,0.5,0.75]
fehgradticks=[-0.5,0,0.5]

vmeanlabel=[r'$\langle v_{\rm los}\rangle $ [km/s]']
vdisplabel=[r'$\sigma_{v_{\rm los}}$ [km/s]']
vgradlabel=[r'$k_{v_{\rm los}} $ $\bigl[ \frac{\rm km/s}{\prime}\bigr ]$']
vthetalabel=[r'$\theta_0$ $[^{\circ}]$']
fehmeanlabel=[r'$\langle [\mathrm{Fe/H}]\rangle $ [dex]']
fehdisplabel=[r'$\sigma_{[\mathrm{Fe/H}]}$ [dex]']
fehgradlabel=[r'$k_{[\mathrm{Fe/H}]} $ $\bigl [ \frac{\rm dex}{\prime}\bigr ]$']

ax1_0.set_xlabel(vmeanlabel[0],fontsize=10,rotation=0)
ax1_0.set_ylabel(vdisplabel[0],fontsize=10,rotation=90,labelpad=1)
ax1_0.set_xlim(vmeanlim)
ax1_0.set_ylim(vdisplim)
ax1_0.set_xscale(u'linear')
ax1_0.set_yscale(u'linear')
ax1_0.set_xticks(vmeanticks)
ax1_0.set_xticklabels(vmeanticks,rotation=0)
ax1_0.set_yticks(vdispticks)
ax1_0.set_yticklabels(vdispticks,rotation=0)
#ax1_0.scatter(vmean2,vdisp2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax1_0.scatter(tuc2_vmean,tuc2_vdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)
ax1_0.scatter(gru1_vmean,gru1_vdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='b',rasterized=True)

ax0_0.set_ylabel('probability',fontsize=8,rotation=90,labelpad=1)
ax0_0.xaxis.set_major_formatter(plt.NullFormatter())
ax0_0.set_xticks(vmeanticks)
ax0_0.yaxis.set_major_formatter(plt.NullFormatter())
#ax0_0.hist(vmean2,bins=50,range=vmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)
ax0_0.hist(tuc2_vmean,bins=50,range=vmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.5)
ax0_0.hist(gru1_vmean,bins=50,range=vmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)

ax1_1.set_xlabel('probability',fontsize=8,rotation=00,labelpad=10)
ax1_1.xaxis.set_major_formatter(plt.NullFormatter())
ax1_1.set_yticks(vdispticks)
ax1_1.set_ylim(vdisplim)
ax1_1.yaxis.set_major_formatter(plt.NullFormatter())
ax1_1.hist(tuc2_vdisp,bins=50,range=vdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.5)
ax1_1.hist(gru1_vdisp,bins=50,range=vdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.5)





ax10_0.set_xlabel(fehmeanlabel[0],fontsize=10,rotation=0)
ax10_0.set_ylabel(fehdisplabel[0],fontsize=10,rotation=90,labelpad=1)
ax10_0.set_xlim(fehmeanlim)
ax10_0.set_ylim(fehdisplim)
ax10_0.set_xscale(u'linear')
ax10_0.set_yscale(u'linear')
ax10_0.set_xticks(fehmeanticks)
ax10_0.set_xticklabels(fehmeanticks,rotation=0)
ax10_0.set_yticks(fehdispticks)
ax10_0.set_yticklabels(fehdispticks,rotation=0)
#ax10_0.scatter(fehmean2,fehdisp2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax10_0.scatter(gru1_fehmean,gru1_fehdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='b',rasterized=True)
ax10_0.scatter(tuc2_fehmean,tuc2_fehdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

ax11_0.set_ylabel('probability',fontsize=8,rotation=90,labelpad=1)
ax11_0.xaxis.set_major_formatter(plt.NullFormatter())
ax11_0.set_xticks(fehmeanticks)
ax11_0.yaxis.set_major_formatter(plt.NullFormatter())
#ax11_0.hist(fehmean2,bins=50,range=fehmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)
ax11_0.hist(tuc2_fehmean,bins=50,range=fehmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.5,label='Tuc 2')
ax11_0.hist(gru1_fehmean,bins=50,range=fehmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5,label='Gru 1')
ax11_0.legend(loc=1,fontsize=7,handlelength=0.5,numpoints=1,scatterpoints=1,shadow=False)

ax11_1.set_xlabel('probability',fontsize=8,rotation=00,labelpad=10)
ax11_1.xaxis.set_major_formatter(plt.NullFormatter())
ax11_1.set_yticks(fehdispticks)
ax11_1.set_ylim(fehdisplim)
ax11_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax11_1.hist(fehdisp2,bins=50,range=fehdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.5)
ax11_1.hist(tuc2_fehdisp,bins=50,range=fehdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.5,label='Tuc 2')
ax11_1.hist(gru1_fehdisp,bins=50,range=fehdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.5,label='Gru 1')


plotfilename='tuc2gru1teffprior_meansdispersions.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
