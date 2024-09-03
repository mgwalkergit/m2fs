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
plt.rc('xtick',labelsize='8')
plt.rc('ytick',labelsize='8')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='eps')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)
plt.rc('path',simplify='True')
            
data1='ret2cr_gradientpost_equal_weights.dat'
data2='ret2cr_gradient2post_equal_weights.dat'

with open(data1) as f: # read data file
    data=f.readlines()

vmean=[]
vdisp=[]
vgrad=[]
vtheta=[]
fehmean=[]
fehdisp=[]
fehgrad=[]
like=[]

for line in data: # fill arrays
    p=line.split()
    vmean.append(float(p[0]))
    vdisp.append(float(p[1]))
    vgrad.append(float(p[2]))
    vtheta.append(float(p[3]))
    fehmean.append(float(p[4]))
    fehdisp.append(float(p[5]))
    fehgrad.append(float(p[6]))
    like.append(float(p[7]))

vmean=np.array(vmean)
vdisp=np.array(vdisp)
vgrad=np.array(vgrad)
vtheta=np.array(vtheta)
fehmean=np.array(fehmean)
fehdisp=np.array(fehdisp)
fehgrad=np.array(fehgrad)
like=np.array(like)

with open(data2) as g: # read data file
    data2=g.readlines()

vmean2=[]
vdisp2=[]
vgrad2=[]
vtheta2=[]
fehmean2=[]
fehdisp2=[]
fehgrad2=[]
like2=[]

for line in data2: # fill arrays
    p=line.split()
    vmean2.append(float(p[0]))
    vdisp2.append(float(p[1]))
    vgrad2.append(float(p[2]))
    vtheta2.append(float(p[3]))
    fehmean2.append(float(p[4]))
    fehdisp2.append(float(p[5]))
    fehgrad2.append(float(p[6]))
    like2.append(float(p[7]))

vmean2=np.array(vmean2)
vdisp2=np.array(vdisp2)
vgrad2=np.array(vgrad2)
vtheta2=np.array(vtheta2)
fehmean2=np.array(fehmean2)
fehdisp2=np.array(fehdisp2)
fehgrad2=np.array(fehgrad2)
like2=np.array(like2)

gs=plt.GridSpec(16,16) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_

statsvmean=np.percentile(vmean,[2.5,16.,50.,85.,97.5])
statsvmean2=np.percentile(vmean2,[2.5,16.,50.,85.,97.5])
statsvdisp=np.percentile(vdisp,[2.5,16.,50.,85.,97.5])
statsvdisp2=np.percentile(vdisp2,[2.5,16.,50.,85.,97.5])
statsfehdisp=np.percentile(fehdisp,[2.5,16.,50.,85.,97.5])
statsfehdisp2=np.percentile(fehdisp2,[2.5,16.,50.,85.,97.5])
statsfehmean=np.percentile(fehmean,[2.5,16.,50.,85.,97.5])
statsfehmean2=np.percentile(fehmean2,[2.5,16.,50.,85.,97.5])

ax0_0=fig.add_subplot(gs[2:4,0:4])
ax1_0=fig.add_subplot(gs[4:8,0:4])
ax1_1=fig.add_subplot(gs[4:8,4:6])

ax11_0=fig.add_subplot(gs[2:4,8:12])
ax10_0=fig.add_subplot(gs[4:8,8:12])
ax11_1=fig.add_subplot(gs[4:8,12:14])

#    ax0_12=fig.add_subplot(gs[0,12])
vmeanlim=[60,75]
vdisplim=[0.01,14.9]
vgradlim=[0,4.5]
vthetalim=[-179,179]
fehmeanlim=[-4.99,0.99] 
fehdisplim=[0.01,1.5]
fehgradlim=[-0.99,0.99]  

vmeanticks=[60,65,70,75]
vdispticks=[0,5,10]
vgradticks=[1,2,3,4]
vthetaticks=[-150,0,150]
fehmeanticks=[-4,-3,-2,-1,0]
fehdispticks=[0.,0.5,1]
fehgradticks=[-0.5,0,0.5]

vmeanlabel=[r'$\overline{v_{\rm los}}$ [km/s]']
vdisplabel=[r'$\sigma_{v_{\rm los}}$ [km/s]']
vgradlabel=[r'$\frac{\Delta\overline{v}_{\rm los}}{\Delta R} $ $\bigl[ \frac{\rm km/s}{\prime}\bigr ]$']
vthetalabel=[r'$\theta_0$ $[^{\circ}]$']
fehmeanlabel=[r'$\overline{[\mathrm{Fe/H}]}$ [dex]']
fehdisplabel=[r'$\sigma_{[\mathrm{Fe/H}]}$ [dex]']
fehgradlabel=[r'$\frac{\Delta\overline{[\mathrm{Fe/H}]}}{\Delta R} $ $\bigl [ \frac{\rm dex}{\prime}\bigr ]$']

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
ax1_0.scatter(vmean2,vdisp2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax1_0.scatter(vmean,vdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)
#ax1_0.text(56,17,r'$\frac{\overline{v}_{\rm los}}{\mathrm{km/s}}=',fontsize=5,color='k')
#ax1_0.text(62,17.2,r'$'+str(round(statsvmean[2],2))+r'_{'+str(round(statsvmean[1]-statsvmean[2],2))+r'}'+r'^{+'+str(round(statsvmean[3]-statsvmean[2],2))+r'},$',fontsize=4.6,color='r')
#ax1_0.text(69,17.2,r'$'+str(round(statsvmean2[2],2))+r'_{'+str(round(statsvmean2[1]-statsvmean2[2],2))+r'}'+r'^{+'+str(round(statsvmean2[3]-statsvmean2[2],2))+r'},$',fontsize=4.6,color='b')
#ax1_0.text(56,14.,r'$\frac{\sigma_{v_{\rm los}}}{\mathrm{km/s}}=$',fontsize=5,color='k')
#ax1_0.text(62,14.5,r'$'+str(round(statsvdisp[2],2))+r'_{'+str(round(statsvdisp[1]-statsvdisp[2],2))+r'}'+r'^{+'+str(round(statsvdisp[3]-statsvdisp[2],2))+r'},$',fontsize=4.6,color='r')
#ax1_0.text(69,14.5,r'$'+str(round(statsvdisp2[2],2))+r'_{'+str(round(statsvdisp2[1]-statsvdisp2[2],2))+r'}'+r'^{+'+str(round(statsvdisp2[3]-statsvdisp2[2],2))+r'},$',fontsize=4.6,color='b')


#ax1_0.text(57,14,r'$\sigma_{v_{\rm los}}='+str(round(statsvdisp[2],2))+r'_{'+str(round(statsvdisp[1]-statsvdisp[2],2))+r'}'+r'^{+'+str(round(statsvdisp[3]-statsvdisp[2],2))+r'}$'+' km/s',fontsize=5)

ax0_0.set_ylabel('probability',fontsize=8,rotation=90,labelpad=1)
ax0_0.xaxis.set_major_formatter(plt.NullFormatter())
ax0_0.set_xticks(vmeanticks)
ax0_0.yaxis.set_major_formatter(plt.NullFormatter())
ax0_0.hist(vmean2,bins=50,range=vmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)
ax0_0.hist(vmean,bins=50,range=vmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.5)

ax1_1.set_xlabel('probability',fontsize=8,rotation=00,labelpad=10)
ax1_1.xaxis.set_major_formatter(plt.NullFormatter())
ax1_1.set_yticks(vdispticks)
ax1_1.set_ylim(vdisplim)
ax1_1.yaxis.set_major_formatter(plt.NullFormatter())
ax1_1.hist(vdisp2,bins=50,range=vdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.5)
ax1_1.hist(vdisp,bins=50,range=vdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.5)





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
ax10_0.scatter(fehmean2,fehdisp2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax10_0.scatter(fehmean,fehdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)
#ax10_0.text(-4.7,1.75,r'$\overline{\mathrm{[Fe/H]}}=$',fontsize=5,color='k')
#ax10_0.text(-2.89,1.72,r'$'+str(round(statsfehmean[2],2))+r'_{'+str(round(statsfehmean[1]-statsfehmean[2],2))+r'}'+r'^{+'+str(round(statsfehmean[3]-statsfehmean[2],2))+r'},$',fontsize=4.6,color='r')
#ax10_0.text(-0.9,1.72,r'$'+str(round(statsfehmean2[2],2))+r'_{'+str(round(statsfehmean2[1]-statsfehmean2[2],2))+r'}'+r'^{+'+str(round(statsfehmean2[3]-statsfehmean2[2],2))+r'},$',fontsize=4.6,color='b')
#ax10_0.text(-4.7,1.4,r'$\sigma_{\mathrm{[Fe/H]}}=$',fontsize=5,color='k')
#ax10_0.text(-2.49,1.4,r'$'+str(round(statsfehdisp[2],2))+r'_{'+str(round(statsfehdisp[1]-statsfehdisp[2],2))+r'}'+r'^{+'+str(round(statsfehdisp[3]-statsfehdisp[2],2))+r'},$',fontsize=4.6,color='r')
#ax10_0.text(-0.65,1.4,r'$'+str(round(statsfehdisp2[2],2))+r'_{'+str(round(statsfehdisp2[1]-statsfehdisp2[2],2))+r'}'+r'^{+'+str(round(statsfehdisp2[3]-statsfehdisp2[2],2))+r'},$',fontsize=4.6,color='b')

#ax10_0.text(-4.2,1.7,r'$\overline{\rm [Fe/H]}='+str(round(statsfehmean[2],2))+r'_{'+str(round(statsfehmean[1]-statsfehmean[2],2))+r'}'+r'^{+'+str(round(statsfehmean[3]-statsfehmean[2],2))+r'}$',fontsize=5)
#ax10_0.text(-4.2,1.4,r'$\sigma_{\rm [Fe/H]}='+str(round(statsfehdisp[2],2))+r'_{'+str(round(statsfehdisp[1]-statsfehdisp[2],2))+r'}'+r'^{+'+str(round(statsfehdisp[3]-statsfehdisp[2],2))+r'}$',fontsize=5)

ax11_0.set_ylabel('probability',fontsize=8,rotation=90,labelpad=1)
ax11_0.xaxis.set_major_formatter(plt.NullFormatter())
ax11_0.set_xticks(fehmeanticks)
ax11_0.yaxis.set_major_formatter(plt.NullFormatter())
ax11_0.hist(fehmean2,bins=50,range=fehmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)
ax11_0.hist(fehmean,bins=50,range=fehmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.5)

ax11_1.set_xlabel('probability',fontsize=8,rotation=00,labelpad=10)
ax11_1.xaxis.set_major_formatter(plt.NullFormatter())
ax11_1.set_yticks(fehdispticks)
ax11_1.set_ylim(fehdisplim)
ax11_1.yaxis.set_major_formatter(plt.NullFormatter())
ax11_1.hist(fehdisp2,bins=50,range=fehdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.5)
ax11_1.hist(fehdisp,bins=50,range=fehdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.5)


plotfilename='ret2cr_meansdispersions.pdf'
plt.savefig(plotfilename,dpi=400)
#plt.show()
plt.close()
