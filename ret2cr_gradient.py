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
plt.rc('xtick',labelsize='5')
plt.rc('ytick',labelsize='5')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='eps')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)
            
data1='ret2cr_gradientpost_equal_weights.dat'
data2='ret2cr_gradient2post_equal_weights.dat'

out1='ret2cr_gradientcommands.tex'
out2='ret2cr_gradient_table.tex'
g1=open(out1,'w')
g2=open(out2,'w')

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

gs=plt.GridSpec(7,7) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_

statsvmean=np.percentile(vmean,[2.5,16.,50.,85.,97.5])
statsvdisp=np.percentile(vdisp,[2.5,16.,50.,85.,97.5])
statsvgrad=np.percentile(vgrad,[2.5,16.,50.,85.,97.5])
statsvtheta=np.percentile(vtheta,[2.5,16.,50.,85.,97.5])
statsfehdisp=np.percentile(fehdisp,[2.5,16.,50.,85.,97.5])
statsfehmean=np.percentile(fehmean,[2.5,16.,50.,85.,97.5])
statsfehgrad=np.percentile(fehgrad,[2.5,16.,50.,85.,97.5])

statsvmean2=np.percentile(vmean2,[2.5,16.,50.,85.,97.5])
statsvdisp2=np.percentile(vdisp2,[2.5,16.,50.,85.,97.5])
statsvgrad2=np.percentile(vgrad2,[2.5,16.,50.,85.,97.5])
statsvtheta2=np.percentile(vtheta2,[2.5,16.,50.,85.,97.5])
statsfehdisp2=np.percentile(fehdisp2,[2.5,16.,50.,85.,97.5])
statsfehmean2=np.percentile(fehmean2,[2.5,16.,50.,85.,97.5])
statsfehgrad2=np.percentile(fehgrad2,[2.5,16.,50.,85.,97.5])

ax6_0=fig.add_subplot(gs[6,0])
ax5_0=fig.add_subplot(gs[5,0])
ax4_0=fig.add_subplot(gs[4,0])
ax3_0=fig.add_subplot(gs[3,0])
ax2_0=fig.add_subplot(gs[2,0])
ax1_0=fig.add_subplot(gs[1,0])

ax5_1=fig.add_subplot(gs[5,1])
ax4_1=fig.add_subplot(gs[4,1])
ax3_1=fig.add_subplot(gs[3,1])
ax2_1=fig.add_subplot(gs[2,1])
ax1_1=fig.add_subplot(gs[1,1])

ax4_2=fig.add_subplot(gs[4,2])
ax3_2=fig.add_subplot(gs[3,2])
ax2_2=fig.add_subplot(gs[2,2])
ax1_2=fig.add_subplot(gs[1,2])

ax3_3=fig.add_subplot(gs[3,3])
ax2_3=fig.add_subplot(gs[2,3])
ax1_3=fig.add_subplot(gs[1,3])

ax2_4=fig.add_subplot(gs[2,4])
ax1_4=fig.add_subplot(gs[1,4])

ax1_5=fig.add_subplot(gs[1,5])



#ax1_4=fig.add_subplot(gs[1,4])

ax1_6=fig.add_subplot(gs[1,6])
ax0_5=fig.add_subplot(gs[0,5])
ax0_4=fig.add_subplot(gs[0,4])
ax0_3=fig.add_subplot(gs[0,3])
ax0_2=fig.add_subplot(gs[0,2])
ax0_1=fig.add_subplot(gs[0,1])
ax0_0=fig.add_subplot(gs[0,0])

#    ax0_12=fig.add_subplot(gs[0,12])
vmeanlim=[55,75]
vdisplim=[0.01,19.9]
vgradlim=[0,4.5]
vthetalim=[-179,179]
fehmeanlim=[-4.99,0.99] 
fehdisplim=[0.01,2]
fehgradlim=[-0.99,0.99]  

vmeanticks=[55,60,65,70,75]
vdispticks=[5,10,15]
vgradticks=[1,2,3,4]
vthetaticks=[-150,0,150]
fehmeanticks=[-4,-3,-2,-1,0]
fehdispticks=[0.5,1,1.5]
fehgradticks=[-0.5,0,0.5]

vmeanlabel=[r'$\overline{v}_{\rm los}$ [km/s]']
vdisplabel=[r'$\sigma_v$ [km/s]']
vgradlabel=[r'$\frac{\Delta\overline{v}_{\rm los}}{\Delta R} $ $\bigl[ \frac{\rm km/s}{\prime}\bigr ]$']
vthetalabel=[r'$\theta_0$ $[^{\circ}]$']
fehmeanlabel=[r'$\overline{[\mathrm{Fe/H}]}$ [dex]']
fehdisplabel=[r'$\sigma_{[\mathrm{Fe/H}]}$ [dex]']
fehgradlabel=[r'$\frac{\Delta\overline{[\mathrm{Fe/H}]}}{\Delta R} $ $\bigl [ \frac{\rm dex}{\prime}\bigr ]$']

ax6_0.set_xlabel(vmeanlabel[0],fontsize=7,rotation=0)
ax6_0.set_ylabel(vdisplabel[0],fontsize=7,rotation=90,labelpad=1)
ax6_0.set_xlim(vmeanlim)
ax6_0.set_ylim(vdisplim)
ax6_0.set_xscale(u'linear')
ax6_0.set_yscale(u'linear')
ax6_0.set_xticks(vmeanticks)
ax6_0.set_xticklabels(vmeanticks,rotation=0)
ax6_0.set_yticks(vdispticks)
ax6_0.set_yticklabels(vdispticks,rotation=0)
ax6_0.scatter(vmean2,vdisp2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax6_0.scatter(vmean,vdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax6_0.set_xlabel(vmeanlabel[0],fontsize=7,rotation=0)
ax5_0.set_ylabel(vgradlabel[0],fontsize=7,rotation=90,labelpad=15)
ax5_0.set_xlim(vmeanlim)
ax5_0.set_ylim(vgradlim)
ax5_0.set_xscale(u'linear')
ax5_0.set_yscale(u'linear')
ax5_0.set_xticks(vmeanticks)
#ax6_0.set_xticklabels(vmeanticks,rotation=0)
ax5_0.set_yticks(vgradticks)
ax5_0.set_yticklabels(vgradticks,rotation=0)
ax5_0.xaxis.set_major_formatter(plt.NullFormatter())
ax5_0.scatter(vmean2,vgrad2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax5_0.scatter(vmean,vgrad,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax5_0.set_xlabel(vmeanlabel[0],fontsize=7,rotation=0)
ax4_0.set_ylabel(vthetalabel[0],fontsize=7,rotation=90,labelpad=1)
ax4_0.set_xlim(vmeanlim)
ax4_0.set_ylim(vthetalim)
ax4_0.set_xscale(u'linear')
ax4_0.set_yscale(u'linear')
ax5_0.set_xticks(vmeanticks)
#ax5_0.set_xticklabels(vmeanticks,rotation=0)
ax4_0.set_yticks(vthetaticks)
ax4_0.set_yticklabels(vthetaticks,rotation=0)
ax4_0.xaxis.set_major_formatter(plt.NullFormatter())
ax4_0.scatter(vmean2,vtheta2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax4_0.scatter(vmean,vtheta,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax4_0.set_xlabel(vmeanlabel[0],fontsize=7,rotation=0)
ax3_0.set_ylabel(fehmeanlabel[0],fontsize=7,rotation=90,labelpad=15)
ax3_0.set_xlim(vmeanlim)
ax3_0.set_ylim(fehmeanlim)
ax3_0.set_xscale(u'linear')
ax3_0.set_yscale(u'linear')
ax3_0.set_xticks(vmeanticks)
#ax4_0.set_xticklabels(vmeanticks,rotation=0)
ax3_0.set_yticks(fehmeanticks)
ax3_0.set_yticklabels(fehmeanticks,rotation=0)
ax3_0.xaxis.set_major_formatter(plt.NullFormatter())
ax3_0.scatter(vmean2,fehmean2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax3_0.scatter(vmean,fehmean,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax3_0.set_xlabel(vmeanlabel[0],fontsize=7,rotation=0)
ax2_0.set_ylabel(fehdisplabel[0],fontsize=7,rotation=90,labelpad=1)
ax2_0.set_xlim(vmeanlim)
ax2_0.set_ylim(fehdisplim)
ax2_0.set_xscale(u'linear')
ax2_0.set_yscale(u'linear')
ax2_0.set_xticks(vmeanticks)
#ax3_0.set_xticklabels(vmeanticks,rotation=0)
ax2_0.set_yticks(fehdispticks)
ax2_0.set_yticklabels(fehdispticks,rotation=0)
ax2_0.xaxis.set_major_formatter(plt.NullFormatter())
ax2_0.scatter(vmean2,fehdisp2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax2_0.scatter(vmean,fehdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax2_0.set_xlabel(vmeanlabel[0],fontsize=7,rotation=0)
ax1_0.set_ylabel(fehgradlabel[0],fontsize=7,rotation=90,labelpad=15)
ax1_0.set_xlim(vmeanlim)
ax1_0.set_ylim(fehgradlim)
ax1_0.set_xscale(u'linear')
ax1_0.set_yscale(u'linear')
ax1_0.set_xticks(vmeanticks)
#ax2_0.set_xticklabels(vmeanticks,rotation=0)
ax1_0.set_yticks(fehgradticks)
ax1_0.set_yticklabels(fehgradticks,rotation=0)
ax1_0.xaxis.set_major_formatter(plt.NullFormatter())
ax1_0.scatter(vmean2,fehgrad2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax1_0.scatter(vmean,fehgrad,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

ax0_0.set_ylabel('probability',fontsize=7,rotation=90,labelpad=1)
ax0_0.xaxis.set_major_formatter(plt.NullFormatter())
ax0_0.set_xticks(vmeanticks)
ax0_0.yaxis.set_major_formatter(plt.NullFormatter())
ax0_0.hist(vmean2,bins=50,range=vmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)
ax0_0.hist(vmean,bins=50,range=vmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.5)



ax5_1.set_xlabel(vdisplabel[0],fontsize=7,rotation=0)
#ax5_1.set_ylabel(vgradlabel[0],fontsize=7,rotation=90,labelpad=5)
ax5_1.set_xlim(vdisplim)
ax5_1.set_ylim(vgradlim)
ax5_1.set_xscale(u'linear')
ax5_1.set_yscale(u'linear')
ax5_1.set_xticks(vdispticks)
ax5_1.set_xticklabels(vdispticks,rotation=0)
ax5_1.set_yticks(vgradticks)
ax5_1.set_yticklabels(vgradticks,rotation=0)
ax5_1.yaxis.set_major_formatter(plt.NullFormatter())
ax5_1.scatter(vdisp2,vgrad2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax5_1.scatter(vdisp,vgrad,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax5_1.set_xlabel(vdisplabel[0],fontsize=7,rotation=0)
#ax4_1.set_ylabel(vthetalabel[0],fontsize=7,rotation=90,labelpad=5)
ax4_1.set_xlim(vdisplim)
ax4_1.set_ylim(vthetalim)
ax4_1.set_xscale(u'linear')
ax4_1.set_yscale(u'linear')
ax4_1.set_xticks(vdispticks)
#ax4_1.set_xticklabels(vdispticks,rotation=0)
ax4_1.set_yticks(vthetaticks)
#ax4_1.set_yticklabels(vthetaticks,rotation=0)
ax4_1.xaxis.set_major_formatter(plt.NullFormatter())
ax4_1.yaxis.set_major_formatter(plt.NullFormatter())
ax4_1.scatter(vdisp2,vtheta2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax4_1.scatter(vdisp,vtheta,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax3_1.set_xlabel(vdisplabel[0],fontsize=7,rotation=0)
#ax3_1.set_ylabel(fehmeanlabel[0],fontsize=7,rotation=90,labelpad=5)
ax3_1.set_xlim(vdisplim)
ax3_1.set_ylim(fehmeanlim)
ax3_1.set_xscale(u'linear')
ax3_1.set_yscale(u'linear')
ax3_1.set_xticks(vdispticks)
#ax3_1.set_xticklabels(vdispticks,rotation=0)
ax3_1.set_yticks(fehmeanticks)
#ax3_1.set_yticklabels(fehmeanticks,rotation=0)
ax3_1.yaxis.set_major_formatter(plt.NullFormatter())
ax3_1.xaxis.set_major_formatter(plt.NullFormatter())
ax3_1.scatter(vdisp2,fehmean2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax3_1.scatter(vdisp,fehmean,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax2_1.set_xlabel(vdisplabel[0],fontsize=7,rotation=0)
#ax2_1.set_ylabel(fehdisplabel[0],fontsize=7,rotation=90,labelpad=5)
ax2_1.set_xlim(vdisplim)
ax2_1.set_ylim(fehdisplim)
ax2_1.set_xscale(u'linear')
ax2_1.set_yscale(u'linear')
ax2_1.set_xticks(vdispticks)
#ax2_1.set_xticklabels(vdispticks,rotation=0)
ax2_1.set_yticks(fehdispticks)
#ax2_1.set_yticklabels(fehdispticks,rotation=0)
ax2_1.yaxis.set_major_formatter(plt.NullFormatter())
ax2_1.xaxis.set_major_formatter(plt.NullFormatter())
ax2_1.scatter(vdisp2,fehdisp2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax2_1.scatter(vdisp,fehdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax1_1.set_xlabel(vdisplabel[0],fontsize=7,rotation=0)
#ax1_1.set_ylabel(fehgradlabel[0],fontsize=7,rotation=90,labelpad=5)
ax1_1.set_xlim(vdisplim)
ax1_1.set_ylim(fehgradlim)
ax1_1.set_xscale(u'linear')
ax1_1.set_yscale(u'linear')
ax1_1.set_xticks(vdispticks)
#ax1_1.set_xticklabels(vdispticks,rotation=0)
ax1_1.set_yticks(fehgradticks)
#ax1_1.set_yticklabels(fehgradticks,rotation=0)
ax1_1.yaxis.set_major_formatter(plt.NullFormatter())
ax1_1.xaxis.set_major_formatter(plt.NullFormatter())
ax1_1.scatter(vdisp2,fehgrad2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax1_1.scatter(vdisp,fehgrad,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax0_1.set_ylabel('probability',fontsize=7,rotation=90,labelpad=5)
ax0_1.xaxis.set_major_formatter(plt.NullFormatter())
ax0_1.set_xticks(vdispticks)
ax0_1.yaxis.set_major_formatter(plt.NullFormatter())
ax0_1.hist(vdisp2,bins=50,range=vdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)
ax0_1.hist(vdisp,bins=50,range=vdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.5)



ax4_2.set_xlabel(vgradlabel[0],fontsize=7,rotation=0)
#ax4_2.set_ylabel(vthetalabel[0],fontsize=7,rotation=90,labelpad=5)
ax4_2.set_xlim(vgradlim)
ax4_2.set_ylim(vthetalim)
ax4_2.set_xscale(u'linear')
ax4_2.set_yscale(u'linear')
ax4_2.set_xticks(vgradticks)
#ax4_2.set_xticklabels(vgradticks,rotation=0)
ax4_2.set_yticks(vthetaticks)
#ax4_2.set_yticklabels(vthetaticks,rotation=0)
ax4_2.yaxis.set_major_formatter(plt.NullFormatter())
ax4_2.scatter(vgrad2,vtheta2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax4_2.scatter(vgrad,vtheta,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax3_2.set_xlabel(vgradlabel[0],fontsize=7,rotation=0)
#ax3_2.set_ylabel(fehmeanlabel[0],fontsize=7,rotation=90,labelpad=5)
ax3_2.set_xlim(vgradlim)
ax3_2.set_ylim(fehmeanlim)
ax3_2.set_xscale(u'linear')
ax3_2.set_yscale(u'linear')
ax3_2.set_xticks(vgradticks)
#ax3_2.set_xticklabels(vgradticks,rotation=0)
ax3_2.set_yticks(fehmeanticks)
#ax3_2.set_yticklabels(fehmeanticks,rotation=0)
ax3_2.yaxis.set_major_formatter(plt.NullFormatter())
ax3_2.xaxis.set_major_formatter(plt.NullFormatter())
ax3_2.scatter(vgrad2,fehmean2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax3_2.scatter(vgrad,fehmean,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax2_2.set_xlabel(vgradlabel[0],fontsize=7,rotation=0)
#ax2_2.set_ylabel(fehdisplabel[0],fontsize=7,rotation=90,labelpad=5)
ax2_2.set_xlim(vgradlim)
ax2_2.set_ylim(fehdisplim)
ax2_2.set_xscale(u'linear')
ax2_2.set_yscale(u'linear')
ax2_2.set_xticks(vgradticks)
#ax2_2.set_xticklabels(vgradticks,rotation=0)
ax2_2.set_yticks(fehdispticks)
#ax2_2.set_yticklabels(fehdispticks,rotation=0)
ax2_2.yaxis.set_major_formatter(plt.NullFormatter())
ax2_2.xaxis.set_major_formatter(plt.NullFormatter())
ax2_2.scatter(vgrad2,fehdisp2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax2_2.scatter(vgrad,fehdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax1_2.set_xlabel(vgradlabel[0],fontsize=7,rotation=0)
#ax1_2.set_ylabel(fehgradlabel[0],fontsize=7,rotation=90,labelpad=5)
ax1_2.set_xlim(vgradlim)
ax1_2.set_ylim(fehgradlim)
ax1_2.set_xscale(u'linear')
ax1_2.set_yscale(u'linear')
ax1_2.set_xticks(vgradticks)
#ax1_2.set_xticklabels(vgradticks,rotation=0)
ax1_2.set_yticks(fehgradticks)
#ax1_2.set_yticklabels(fehgradticks,rotation=0)
ax1_2.yaxis.set_major_formatter(plt.NullFormatter())
ax1_2.xaxis.set_major_formatter(plt.NullFormatter())
ax1_2.scatter(vgrad2,fehgrad2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax1_2.scatter(vgrad,fehgrad,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax0_2.set_ylabel('probability',fontsize=7,rotation=90,labelpad=5)
ax0_2.xaxis.set_major_formatter(plt.NullFormatter())
ax0_2.set_xticks(vgradticks)
ax0_2.yaxis.set_major_formatter(plt.NullFormatter())
ax0_2.hist(vgrad2,bins=50,range=vgradlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)
ax0_2.hist(vgrad,bins=50,range=vgradlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.5)


ax3_3.set_xlabel(vthetalabel[0],fontsize=7,rotation=0)
#ax3_3.set_ylabel(fehmeanlabel[0],fontsize=7,rotation=90,labelpad=5)
ax3_3.set_xlim(vthetalim)
ax3_3.set_ylim(fehmeanlim)
ax3_3.set_xscale(u'linear')
ax3_3.set_yscale(u'linear')
ax3_3.set_xticks(vthetaticks)
#ax3_3.set_xticklabels(vthetaticks,rotation=0)
ax3_3.set_yticks(fehmeanticks)
#ax3_3.set_yticklabels(fehmeanticks,rotation=0)
ax3_3.yaxis.set_major_formatter(plt.NullFormatter())
ax3_3.scatter(vtheta2,fehmean2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax3_3.scatter(vtheta,fehmean,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax2_3.set_xlabel(vthetalabel[0],fontsize=7,rotation=0)
#ax2_3.set_ylabel(fehdisplabel[0],fontsize=7,rotation=90,labelpad=5)
ax2_3.set_xlim(vthetalim)
ax2_3.set_ylim(fehdisplim)
ax2_3.set_xscale(u'linear')
ax2_3.set_yscale(u'linear')
ax2_3.set_xticks(vthetaticks)
#ax2_3.set_xticklabels(vthetaticks,rotation=0)
ax2_3.set_yticks(fehdispticks)
#ax2_3.set_yticklabels(fehdispticks,rotation=0)
ax2_3.yaxis.set_major_formatter(plt.NullFormatter())
ax2_3.xaxis.set_major_formatter(plt.NullFormatter())
ax2_3.scatter(vtheta2,fehdisp2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax2_3.scatter(vtheta,fehdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax1_3.set_xlabel(vthetalabel[0],fontsize=7,rotation=0)
#ax1_3.set_ylabel(fehgradlabel[0],fontsize=7,rotation=90,labelpad=5)
ax1_3.set_xlim(vthetalim)
ax1_3.set_ylim(fehgradlim)
ax1_3.set_xscale(u'linear')
ax1_3.set_yscale(u'linear')
ax1_3.set_xticks(vthetaticks)
#ax1_3.set_xticklabels(vthetaticks,rotation=0)
ax1_3.set_yticks(fehgradticks)
#ax1_3.set_yticklabels(fehgradticks,rotation=0)
ax1_3.yaxis.set_major_formatter(plt.NullFormatter())
ax1_3.xaxis.set_major_formatter(plt.NullFormatter())
ax1_3.scatter(vtheta2,fehgrad2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax1_3.scatter(vtheta,fehgrad,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax0_3.set_ylabel('probability',fontsize=7,rotation=90,labelpad=5)
ax0_3.xaxis.set_major_formatter(plt.NullFormatter())
ax0_3.set_xticks(vthetaticks)
ax0_3.yaxis.set_major_formatter(plt.NullFormatter())
ax0_3.hist(vtheta2,bins=50,range=vthetalim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)
ax0_3.hist(vtheta,bins=50,range=vthetalim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.5)


ax2_4.set_xlabel(fehmeanlabel[0],fontsize=7,rotation=0)
#ax2_4.set_ylabel(fehdisplabel[0],fontsize=7,rotation=90,labelpad=5)
ax2_4.set_xlim(fehmeanlim)
ax2_4.set_ylim(fehdisplim)
ax2_4.set_xscale(u'linear')
ax2_4.set_yscale(u'linear')
ax2_4.set_xticks(fehmeanticks)
#ax2_4.set_xticklabels(fehmeanticks,rotation=0)
ax2_4.set_yticks(fehdispticks)
#ax2_4.set_yticklabels(fehdispticks,rotation=0)
ax2_4.yaxis.set_major_formatter(plt.NullFormatter())
ax2_4.scatter(fehmean2,fehdisp2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax2_4.scatter(fehmean,fehdisp,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax1_4.set_xlabel(fehmeanlabel[0],fontsize=7,rotation=0)
#ax1_4.set_ylabel(fehgradlabel[0],fontsize=7,rotation=90,labelpad=5)
ax1_4.set_xlim(fehmeanlim)
ax1_4.set_ylim(fehgradlim)
ax1_4.set_xscale(u'linear')
ax1_4.set_yscale(u'linear')
ax1_4.set_xticks(fehmeanticks)
#ax1_4.set_xticklabels(fehmeanticks,rotation=0)
ax1_4.set_yticks(fehgradticks)
#ax1_4.set_yticklabels(fehgradticks,rotation=0)
ax1_4.yaxis.set_major_formatter(plt.NullFormatter())
ax1_4.xaxis.set_major_formatter(plt.NullFormatter())
ax1_4.scatter(fehmean2,fehgrad2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax1_4.scatter(fehmean,fehgrad,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax0_4.set_ylabel('probability',fontsize=7,rotation=90,labelpad=5)
ax0_4.xaxis.set_major_formatter(plt.NullFormatter())
ax0_4.set_xticks(fehmeanticks)
ax0_4.yaxis.set_major_formatter(plt.NullFormatter())
ax0_4.hist(fehmean2,bins=50,range=fehmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)
ax0_4.hist(fehmean,bins=50,range=fehmeanlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.5)


ax1_5.set_xlabel(fehdisplabel[0],fontsize=7,rotation=0)
#ax1_5.set_ylabel(fehgradlabel[0],fontsize=7,rotation=90,labelpad=5)
ax1_5.set_xlim(fehdisplim)
ax1_5.set_ylim(fehgradlim)
ax1_5.set_xscale(u'linear')
ax1_5.set_yscale(u'linear')
ax1_5.set_xticks(fehdispticks)
#ax1_5.set_xticklabels(fehdispticks,rotation=0)
ax1_5.set_yticks(fehgradticks)
#ax1_5.set_yticklabels(fehgradticks,rotation=0)
ax1_5.yaxis.set_major_formatter(plt.NullFormatter())
ax1_5.scatter(fehdisp2,fehgrad2,s=1,lw=0,edgecolor='none',alpha=0.75,marker='.',color='b',rasterized=True)
ax1_5.scatter(fehdisp,fehgrad,s=1,lw=0,edgecolor='none',alpha=0.35,marker='.',color='r',rasterized=True)

#ax0_5.set_ylabel('probability',fontsize=7,rotation=90,labelpad=5)
ax0_5.xaxis.set_major_formatter(plt.NullFormatter())
ax0_5.set_xticks(fehdispticks)
ax0_5.yaxis.set_major_formatter(plt.NullFormatter())
ax0_5.hist(fehdisp2,bins=50,range=fehdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.5)
ax0_5.hist(fehdisp,bins=50,range=fehdisplim,normed=True,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.5)

ax1_6.set_xlabel('probability',fontsize=7,rotation=0,labelpad=5)
ax1_6.yaxis.set_major_formatter(plt.NullFormatter())
ax1_6.xaxis.set_major_formatter(plt.NullFormatter())
ax1_6.set_yticks(fehgradticks)
ax1_6.hist(fehgrad2,bins=50,range=fehgradlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.5)
ax1_6.hist(fehgrad,bins=50,range=fehgradlim,normed=True,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.5)

g1.write(r'\newcommand{\vmean}{$'+str(round(statsvmean[2],2))+r'_{'+str(round(statsvmean[1]-statsvmean[2],2))+r'}'+r'^{+'+str(round(statsvmean[3]-statsvmean[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\vdisp}{$'+str(round(statsvdisp[2],2))+r'_{'+str(round(statsvdisp[1]-statsvdisp[2],2))+r'}'+r'^{+'+str(round(statsvdisp[3]-statsvdisp[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\vgrad}{$'+str(round(statsvgrad[2],2))+r'_{'+str(round(statsvgrad[1]-statsvgrad[2],2))+r'}'+r'^{+'+str(round(statsvgrad[3]-statsvgrad[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\vtheta}{$'+str(round(statsvtheta[2],2))+r'_{'+str(round(statsvtheta[1]-statsvtheta[2],2))+r'}'+r'^{+'+str(round(statsvtheta[3]-statsvtheta[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehmean}{$'+str(round(statsfehmean[2],2))+r'_{'+str(round(statsfehmean[1]-statsfehmean[2],2))+r'}'+r'^{+'+str(round(statsfehmean[3]-statsfehmean[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehdisp}{$'+str(round(statsfehdisp[2],2))+r'_{'+str(round(statsfehdisp[1]-statsfehdisp[2],2))+r'}'+r'^{+'+str(round(statsfehdisp[3]-statsfehdisp[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehgrad}{$'+str(round(statsfehgrad[2],2))+r'_{'+str(round(statsfehgrad[1]-statsfehgrad[2],2))+r'}'+r'^{+'+str(round(statsfehgrad[3]-statsfehgrad[2],2))+r'}$'+r'}'+'\n')

g1.write(r'\newcommand{\vmeanexpanded}{$'+str(round(statsvmean[2],2))+r'_{'+str(round(statsvmean[1]-statsvmean[2],2))+r'('+str(round(statsvmean[0]-statsvmean[2],2))+r')}'+r'^{+'+str(round(statsvmean[3]-statsvmean[2],2))+r'(+'+str(round(statsvmean[4]-statsvmean[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\vdispexpanded}{$'+str(round(statsvdisp[2],2))+r'_{'+str(round(statsvdisp[1]-statsvdisp[2],2))+r'('+str(round(statsvdisp[0]-statsvdisp[2],2))+r')}'+r'^{+'+str(round(statsvdisp[3]-statsvdisp[2],2))+r'(+'+str(round(statsvdisp[4]-statsvdisp[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\vgradexpanded}{$'+str(round(statsvgrad[2],2))+r'_{'+str(round(statsvgrad[1]-statsvgrad[2],2))+r'('+str(round(statsvgrad[0]-statsvgrad[2],2))+r')}'+r'^{+'+str(round(statsvgrad[3]-statsvgrad[2],2))+r'(+'+str(round(statsvgrad[4]-statsvgrad[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\vthetaexpanded}{$'+str(round(statsvtheta[2],2))+r'_{'+str(round(statsvtheta[1]-statsvtheta[2],2))+r'('+str(round(statsvtheta[0]-statsvtheta[2],2))+r')}'+r'^{+'+str(round(statsvtheta[3]-statsvtheta[2],2))+r'(+'+str(round(statsvtheta[4]-statsvtheta[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehmeanexpanded}{$'+str(round(statsfehmean[2],2))+r'_{'+str(round(statsfehmean[1]-statsfehmean[2],2))+r'('+str(round(statsfehmean[0]-statsfehmean[2],2))+r')}'+r'^{+'+str(round(statsfehmean[3]-statsfehmean[2],2))+r'(+'+str(round(statsfehmean[4]-statsfehmean[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehdispexpanded}{$'+str(round(statsfehdisp[2],2))+r'_{'+str(round(statsfehdisp[1]-statsfehdisp[2],2))+r'('+str(round(statsfehdisp[0]-statsfehdisp[2],2))+r')}'+r'^{+'+str(round(statsfehdisp[3]-statsfehdisp[2],2))+r'(+'+str(round(statsfehdisp[4]-statsfehdisp[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehgradexpanded}{$'+str(round(statsfehgrad[2],2))+r'_{'+str(round(statsfehgrad[1]-statsfehgrad[2],2))+r'('+str(round(statsfehgrad[0]-statsfehgrad[2],2))+r')}'+r'^{+'+str(round(statsfehgrad[3]-statsfehgrad[2],2))+r'(+'+str(round(statsfehgrad[4]-statsfehgrad[2],2))+r')}$'+r'}'+'\n')

g1.write(r'\newcommand{\vmeantwo}{$'+str(round(statsvmean2[2],2))+r'_{'+str(round(statsvmean2[1]-statsvmean2[2],2))+r'}'+r'^{+'+str(round(statsvmean2[3]-statsvmean2[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\vdisptwo}{$'+str(round(statsvdisp2[2],2))+r'_{'+str(round(statsvdisp2[1]-statsvdisp2[2],2))+r'}'+r'^{+'+str(round(statsvdisp2[3]-statsvdisp2[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\vgradtwo}{$'+str(round(statsvgrad2[2],2))+r'_{'+str(round(statsvgrad2[1]-statsvgrad2[2],2))+r'}'+r'^{+'+str(round(statsvgrad2[3]-statsvgrad2[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\vthetatwo}{$'+str(round(statsvtheta2[2],2))+r'_{'+str(round(statsvtheta2[1]-statsvtheta2[2],2))+r'}'+r'^{+'+str(round(statsvtheta2[3]-statsvtheta2[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehmeantwo}{$'+str(round(statsfehmean2[2],2))+r'_{'+str(round(statsfehmean2[1]-statsfehmean2[2],2))+r'}'+r'^{+'+str(round(statsfehmean2[3]-statsfehmean2[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehdisptwo}{$'+str(round(statsfehdisp2[2],2))+r'_{'+str(round(statsfehdisp2[1]-statsfehdisp2[2],2))+r'}'+r'^{+'+str(round(statsfehdisp2[3]-statsfehdisp2[2],2))+r'}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehgradtwo}{$'+str(round(statsfehgrad2[2],2))+r'_{'+str(round(statsfehgrad2[1]-statsfehgrad2[2],2))+r'}'+r'^{+'+str(round(statsfehgrad2[3]-statsfehgrad2[2],2))+r'}$'+r'}'+'\n')

g1.write(r'\newcommand{\vmeantwoexpanded}{$'+str(round(statsvmean2[2],2))+r'_{'+str(round(statsvmean2[1]-statsvmean2[2],2))+r'('+str(round(statsvmean2[0]-statsvmean2[2],2))+r')}'+r'^{+'+str(round(statsvmean2[3]-statsvmean2[2],2))+r'(+'+str(round(statsvmean2[4]-statsvmean2[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\vdisptwoexpanded}{$'+str(round(statsvdisp2[2],2))+r'_{'+str(round(statsvdisp2[1]-statsvdisp2[2],2))+r'('+str(round(statsvdisp2[0]-statsvdisp2[2],2))+r')}'+r'^{+'+str(round(statsvdisp2[3]-statsvdisp2[2],2))+r'(+'+str(round(statsvdisp2[4]-statsvdisp2[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\vgradtwoexpanded}{$'+str(round(statsvgrad2[2],2))+r'_{'+str(round(statsvgrad2[1]-statsvgrad2[2],2))+r'('+str(round(statsvgrad2[0]-statsvgrad2[2],2))+r')}'+r'^{+'+str(round(statsvgrad2[3]-statsvgrad2[2],2))+r'(+'+str(round(statsvgrad2[4]-statsvgrad2[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\vthetatwoexpanded}{$'+str(round(statsvtheta2[2],2))+r'_{'+str(round(statsvtheta2[1]-statsvtheta2[2],2))+r'('+str(round(statsvtheta2[0]-statsvtheta2[2],2))+r')}'+r'^{+'+str(round(statsvtheta2[3]-statsvtheta2[2],2))+r'(+'+str(round(statsvtheta2[4]-statsvtheta2[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehmeantwoexpanded}{$'+str(round(statsfehmean2[2],2))+r'_{'+str(round(statsfehmean2[1]-statsfehmean2[2],2))+r'('+str(round(statsfehmean2[0]-statsfehmean2[2],2))+r')}'+r'^{+'+str(round(statsfehmean2[3]-statsfehmean2[2],2))+r'(+'+str(round(statsfehmean2[4]-statsfehmean2[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehdisptwoexpanded}{$'+str(round(statsfehdisp2[2],2))+r'_{'+str(round(statsfehdisp2[1]-statsfehdisp2[2],2))+r'('+str(round(statsfehdisp2[0]-statsfehdisp2[2],2))+r')}'+r'^{+'+str(round(statsfehdisp2[3]-statsfehdisp2[2],2))+r'(+'+str(round(statsfehdisp2[4]-statsfehdisp2[2],2))+r')}$'+r'}'+'\n')
g1.write(r'\newcommand{\fehgradtwoexpanded}{$'+str(round(statsfehgrad2[2],2))+r'_{'+str(round(statsfehgrad2[1]-statsfehgrad2[2],2))+r'('+str(round(statsfehgrad2[0]-statsfehgrad2[2],2))+r')}'+r'^{+'+str(round(statsfehgrad2[3]-statsfehgrad2[2],2))+r'(+'+str(round(statsfehgrad2[4]-statsfehgrad2[2],2))+r')}$'+r'}'+'\n')

g2.write(r'\begin{table*}'+'\n')
g2.write(r'\scriptsize'+'\n')
g2.write(r'\centering'+'\n')
g2.write(r'\caption{Probability distribution functions for chemodynamical parameters}'+'\n')
g2.write(r'\begin{tabular}{@{}llllllllll@{}}'+'\n')
g2.write(r'\hline'+'\n')
g2.write(r'parameter & prior & posterior & posterior & description\\'+'\n')
g2.write(r'&  & (red sample) & (blue sample) &\\'+'\n')
g2.write(r'\hline'+'\n')
g2.write(r'\smallskip'+'\n')
g2.write(r'$\overline{v_{\rm los}}$ [km s$^{-1}$] & uniform between -500 and +500 & \vmeanexpanded & \vmeantwoexpanded '+r'& mean velocity at center' +r'\\'+'\n')
g2.write(r'\smallskip'+'\n')
g2.write(r'$\sigma_{v_{\rm los}}$ [km s$^{-1}$] & uniform between 0 and +500 & \vdispexpanded & \vdisptwoexpanded '+r'& velocity dispersion' +r'\\'+'\n')
g2.write(r'\smallskip'+'\n')
g2.write(r'$k_{v_{\rm los}}$ [km s$^{-1}$ arcmin$^{-1}$] & uniform between 0 and +10 & \vgradexpanded & \vgradtwoexpanded '+r'& magnitude of maximum velocity gradient' +r'\\'+'\n')
g2.write(r'\smallskip'+'\n')
g2.write(r'$\theta_{v_{\rm los}}$ [$\degr$] & uniform between -180 and +180 & \vthetaexpanded & \vthetatwoexpanded '+r'& direction of maximum velocity gradient' +r'\\'+'\n')
g2.write(r'\smallskip'+'\n')
g2.write(r'$\overline{\feh}$ & uniform between -5 and +1 & \fehmeanexpanded & \fehmeantwoexpanded '+r'& mean metallicity at center' +r'\\'+'\n')
g2.write(r'\smallskip'+'\n')
g2.write(r'$\sigma_{\feh}$ & uniform between 0 and +2 & \fehdispexpanded & \fehdisptwoexpanded '+r'& metallicity dispersion' +r'\\'+'\n')
g2.write(r'\smallskip'+'\n')
g2.write(r'$k_{\feh}$ [dex arcmin$^{-1}$] & uniform between -1 and +1 & \fehgradexpanded & \fehgradtwoexpanded '+r'& magnitude of metallicity gradient' +r'\\'+'\n')
#g2.write(r'$\Gamma$ & see Figure \ref{fig:dra_wp11_params} & '+string[13]+r'& $\Delta\log(R_{\rm h}\sigma^2)/\Delta\log(R_{\rm h})$, slope of dynamical mass profile' +r'\\'+'\n')


g2.write(r'\hline'+'\n')
g2.write(r'\end{tabular}'+'\n')
g2.write(r'\\'+'\n')
g2.write(r'\label{tab:gradient}'+'\n')
g2.write(r'\end{table*}'+'\n')



g1.close()
g2.close()
plotfilename='ret2cr_gradient.pdf'
plt.savefig(plotfilename,dpi=400)
#plt.show()
plt.close()
