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
plt.rc('xtick',labelsize='9')
plt.rc('ytick',labelsize='9')
plt.rc('text',usetex='True')
plt.rc('font',family='serif')
plt.rc('font',style='normal')
plt.rc('font',serif='Century')
plt.rc('savefig',format='eps')
plt.rc('lines',markeredgewidth=0)
plt.rc('lines',markersize=2)
            
tuc4_dmodulus=18.41
gru1_dmodulus=20.4

tuc4_data1='plates/tuc4/tuc4_ascii.txt'
tuc4_data2='age12fehm250.iso'
tuc4_data3='tuc4.dat'
tuc4_data4='tuc4_ellipsehalf.res'

with open(tuc4_data2) as g: # read data file
    data=g.readlines()

tuc4_iso_g=[]
tuc4_iso_r=[]
tuc4_iso_i=[]

for line in data: # fill arrays
    p=line.split()
    tuc4_iso_g.append(float(p[6]))
    tuc4_iso_r.append(float(p[7]))
    tuc4_iso_i.append(float(p[8]))

tuc4_iso_g=np.array(tuc4_iso_g)
tuc4_iso_r=np.array(tuc4_iso_r)
tuc4_iso_i=np.array(tuc4_iso_i)

with open(tuc4_data4) as g: # read data file
    data=g.readlines()

tuc4_ellipse_x=[]
tuc4_ellipse_y=[]

for line in data: # fill arrays
    p=line.split()
    tuc4_ellipse_x.append(float(p[0]))
    tuc4_ellipse_y.append(float(p[1]))

tuc4_ellipse_x=np.array(tuc4_ellipse_x)
tuc4_ellipse_y=np.array(tuc4_ellipse_y)

with open(tuc4_data1) as f: # read data file
    data=f.readlines()

tuc4_v_radeg=[]
tuc4_v_decdeg=[]
tuc4_v_gmag=[]
tuc4_v_rmag=[]
tuc4_v_extg=[]
tuc4_v_extr=[]
tuc4_v_spreadg=[]
tuc4_v_spreadr=[]
tuc4_v_spreadi=[]

for line in data: # fill arrays
    p=line.split()
    tuc4_v_radeg.append(float(p[0]))
    tuc4_v_decdeg.append(float(p[1]))
    tuc4_v_gmag.append(float(p[2]))
    tuc4_v_rmag.append(float(p[3]))
    tuc4_v_extg.append(float(p[4]))
    tuc4_v_extr.append(float(p[5]))
    tuc4_v_spreadg.append(float(p[7]))
    tuc4_v_spreadr.append(float(p[8]))

tuc4_v_radeg=np.array(tuc4_v_radeg)
tuc4_v_decdeg=np.array(tuc4_v_decdeg)
tuc4_v_gmag=np.array(tuc4_v_gmag)
tuc4_v_rmag=np.array(tuc4_v_rmag)
tuc4_v_extg=np.array(tuc4_v_extg)
tuc4_v_extr=np.array(tuc4_v_extr)
tuc4_v_spreadg=np.array(tuc4_v_spreadg)
tuc4_v_spreadr=np.array(tuc4_v_spreadr)
tuc4_v_gmag=tuc4_v_gmag-tuc4_v_extg
tuc4_v_rmag=tuc4_v_rmag-tuc4_v_extr

tuc4_v_ra=tuc4_v_radeg*np.pi/180.
tuc4_v_dec=tuc4_v_decdeg*np.pi/180.
tuc4_v_radegcenter=0.73
tuc4_v_decdegcenter=-60.85
tuc4_v_racenter=np.zeros(len(tuc4_v_ra))+tuc4_v_radegcenter*np.pi/180.
tuc4_v_deccenter=np.zeros(len(tuc4_v_dec))+tuc4_v_decdegcenter*np.pi/180.
tuc4_v_xi,tuc4_v_eta=mycode2.etaxiarr(tuc4_v_ra,tuc4_v_dec,tuc4_v_racenter,tuc4_v_deccenter)
tuc4_v_r=np.sqrt(tuc4_v_xi**2+tuc4_v_eta**2)

tuc4_v_cand=np.zeros(len(tuc4_v_ra),dtype='int')
for i in range(0,len(tuc4_v_cand)):
    tuc4_dist=np.sqrt((tuc4_v_rmag[i]-(tuc4_iso_r+tuc4_dmodulus))**2+((tuc4_v_gmag[i]-tuc4_v_rmag[i])-(tuc4_iso_g-tuc4_iso_r))**2)
    if (min(tuc4_dist) < 0.2) & (tuc4_v_rmag[i] < 22.):
        tuc4_v_cand[i]=1

tuc4_v_cands=np.where(tuc4_v_cand == 1)

tuc4_data1='tuc4.dat'

with open(tuc4_data3) as f: # read data file
    data=f.readlines()[1:]
    
tuc4_radeg=[]
tuc4_decdeg=[]
tuc4_xi=[]
tuc4_eta=[]
tuc4_r=[]
tuc4_v=[]
tuc4_sigv=[]
tuc4_teff=[]
tuc4_sigteff=[]
tuc4_logg=[]
tuc4_siglogg=[]
tuc4_feh=[]
tuc4_sigfeh=[]
tuc4_gmag=[]
tuc4_rmag=[]
for line in data: # fill arrays
    p=line.split()
    tuc4_radeg.append(float(p[0]))
    tuc4_decdeg.append(float(p[1]))
    tuc4_xi.append(float(p[2]))
    tuc4_eta.append(float(p[3]))
    tuc4_r.append(float(p[4]))
    tuc4_v.append(float(p[5]))
    tuc4_sigv.append(float(p[6]))
    tuc4_teff.append(float(p[7]))
    tuc4_sigteff.append(float(p[8]))
    tuc4_logg.append(float(p[9]))
    tuc4_siglogg.append(float(p[10]))
    tuc4_feh.append(float(p[11]))
    tuc4_sigfeh.append(float(p[12]))
    tuc4_gmag.append(float(p[13]))
    tuc4_rmag.append(float(p[14]))
tuc4_radeg=np.array(tuc4_radeg)
tuc4_decdeg=np.array(tuc4_decdeg)
tuc4_xi=np.array(tuc4_xi)
tuc4_eta=np.array(tuc4_eta)
tuc4_r=np.array(tuc4_r)
tuc4_v=np.array(tuc4_v)
tuc4_sigv=np.array(tuc4_sigv)
tuc4_teff=np.array(tuc4_teff)
tuc4_sigteff=np.array(tuc4_sigteff)
tuc4_logg=np.array(tuc4_logg)
tuc4_siglogg=np.array(tuc4_siglogg)
tuc4_feh=np.array(tuc4_feh)
tuc4_sigfeh=np.array(tuc4_sigfeh)
tuc4_sigr=tuc4_r-tuc4_r
tuc4_gmag=np.array(tuc4_gmag)
tuc4_rmag=np.array(tuc4_rmag)
tuc4_pmem=tuc4_r-tuc4_r
#tuc4_keep=np.where((tuc4_sigv < 15.) & (tuc4_skewv > -1.) & (tuc4_skewv < 1.) & (tuc4_kurtv > -1.1) & (tuc4_kurtv < 1.1))
tuc4_keep=np.where((tuc4_sigv < 15.))# & (tuc4_skewv > -1.) & (tuc4_skewv < 1.) & (tuc4_kurtv > -1.1) & (tuc4_kurtv < 1.1))
tuc4_mem=np.where((tuc4_sigv < 15.) & (tuc4_pmem > 0.5))
tuc4_goodmem=np.where((tuc4_sigv < 15.) & (tuc4_pmem > 0.9))
tuc4_nonmem=np.where((tuc4_sigv < 15.) & (tuc4_pmem < 0.5))
tuc4_goodnonmem=np.where((tuc4_sigv < 15.) & (tuc4_pmem < 0.5))
tuc4_bad=np.where(tuc4_sigv >= 15.)

tuc4_vmem=[-150.,-110.]
tuc4_loggmem=[0.,4.0]
tuc4_fehmem=[-5.,0.]
tuc4_teffmem=[4000,8000]
tuc4_rmem=[0.,12.]
tuc4_magmem=[21,16]
tuc4_colmem=[-0.2,1]

tuc4_center=np.where((tuc4_v_r < 15.))

gs=plt.GridSpec(10,10) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_
ax0_0=fig.add_subplot(gs[0:5,0:4])
ax0_1=fig.add_subplot(gs[0:5,4:8])

ax0_0.set_xlabel(r'$g-r$ [mag]',fontsize=12,rotation=0,labelpad=5)
ax0_0.set_ylabel(r'$r$ [mag]',fontsize=12,rotation=90,labelpad=5)
ax0_0.set_xlim([-0.5,1.5])
ax0_0.set_ylim([22.5,15.5])
ax0_0.set_xscale(u'linear')
ax0_0.set_yscale(u'linear')
ax0_0.set_yticks([22,21,20,19,18,17,16])
#ax0_0.set_xticklabels(vticks,rotation=0)
ax0_0.set_xticks([-0.5,0,0.5,1])
#ax0_0.set_yticklabels([4000]+teffticks,rotation=0)
#ax0_0.errorbar(v[tuc4_keep],teff[tuc4_keep],xerr=sigv[tuc4_keep],yerr=sigteff[tuc4_keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax0_0.scatter(tuc4_v_gmag[tuc4_center]-tuc4_v_rmag[tuc4_center],tuc4_v_rmag[tuc4_center],s=2,lw=0,edgecolor='none',alpha=0.99,marker='x',color='0.5',rasterized=True)
ax0_0.scatter(tuc4_v_gmag[tuc4_center]-tuc4_v_rmag[tuc4_center],tuc4_v_rmag[tuc4_center],s=2,lw=0,edgecolor='none',alpha=0.99,marker='o',color='0.5',rasterized=True)
ax0_0.scatter(tuc4_gmag[tuc4_bad]-tuc4_rmag[tuc4_bad],tuc4_rmag[tuc4_bad],s=10,lw=1,edgecolor='k',alpha=0.99,marker='x',color='k',rasterized=True)
ax0_0.scatter(tuc4_gmag[tuc4_nonmem]-tuc4_rmag[tuc4_nonmem],tuc4_rmag[tuc4_nonmem],s=15,lw=1,edgecolor='k',alpha=0.99,marker='o',facecolor='none',rasterized=True)
ax0_0.scatter(tuc4_gmag[tuc4_goodnonmem]-tuc4_rmag[tuc4_goodnonmem],tuc4_rmag[tuc4_goodnonmem],s=15,lw=1,edgecolor='k',alpha=0.99,marker='o',color='k',rasterized=True)
ax0_0.scatter(tuc4_gmag[tuc4_mem]-tuc4_rmag[tuc4_mem],tuc4_rmag[tuc4_mem],s=15,lw=1,edgecolor='r',alpha=0.99,marker='o',facecolor='none',rasterized=True)
ax0_0.scatter(tuc4_gmag[tuc4_goodmem]-tuc4_rmag[tuc4_goodmem],tuc4_rmag[tuc4_goodmem],s=15,lw=1,edgecolor='r',alpha=0.99,marker='o',color='r',rasterized=True)
ax0_0.text(-0.4,16.25,'Tuc 4',fontsize=12)


ax0_0.plot(tuc4_iso_g-tuc4_iso_r,tuc4_iso_r+tuc4_dmodulus,lw=1,color='r')
#ax0_0.plot(vteffx2,vteffy2,color='b',linewidth=0.25)


ax0_1.yaxis.set_major_formatter(plt.NullFormatter())
ax0_1.set_xlabel(r'$g-r$ [mag]',fontsize=12,rotation=0,labelpad=5)
#ax0_1.set_ylabel(r'$r$ [mag]',fontsize=8,rotation=90,labelpad=5)
ax0_1.set_xlim([-0.5,1.5])
ax0_1.set_ylim([22.5,15.5])
ax0_1.set_xscale(u'linear')
ax0_1.set_yscale(u'linear')
ax0_1.set_yticks([22,21,20,19,18,17,16])
ax0_1.set_yticklabels([])
#ax0_1.set_xticklabels(vticks,rotation=0)
ax0_1.set_xticks([0,0.5,1,1.5])
#ax0_1.set_yticklabels([4000]+teffticks,rotation=0)
#ax0_1.errorbar(v[gru1_keep],teff[gru1_keep],xerr=sigv[gru1_keep],yerr=sigteff[gru1_keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax0_1.plot(vteffx2,vteffy2,color='b',linewidth=0.25)


plotfilename='tuc4_cmd.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
