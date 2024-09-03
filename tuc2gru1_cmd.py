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
            
tuc2_dmodulus=18.8
gru1_dmodulus=20.4

tuc2_data1='tuc2_vasilyphot.dat'
tuc2_data2='age12fehm250.iso'
tuc2_data3='tuc2.dat'
tuc2_data4='tuc2_ellipsehalf.res'

with open(tuc2_data2) as g: # read data file
    data=g.readlines()

tuc2_iso_g=[]
tuc2_iso_r=[]
tuc2_iso_i=[]

for line in data: # fill arrays
    p=line.split()
    tuc2_iso_g.append(float(p[6]))
    tuc2_iso_r.append(float(p[7]))
    tuc2_iso_i.append(float(p[8]))

tuc2_iso_g=np.array(tuc2_iso_g)
tuc2_iso_r=np.array(tuc2_iso_r)
tuc2_iso_i=np.array(tuc2_iso_i)

with open(tuc2_data4) as g: # read data file
    data=g.readlines()

tuc2_ellipse_x=[]
tuc2_ellipse_y=[]

for line in data: # fill arrays
    p=line.split()
    tuc2_ellipse_x.append(float(p[0]))
    tuc2_ellipse_y.append(float(p[1]))

tuc2_ellipse_x=np.array(tuc2_ellipse_x)
tuc2_ellipse_y=np.array(tuc2_ellipse_y)

with open(tuc2_data1) as f: # read data file
    data=f.readlines()

tuc2_v_radeg=[]
tuc2_v_decdeg=[]
tuc2_v_gmag=[]
tuc2_v_rmag=[]
tuc2_v_imag=[]
tuc2_v_extg=[]
tuc2_v_extr=[]
tuc2_v_exti=[]
tuc2_v_spreadg=[]
tuc2_v_spreadr=[]
tuc2_v_spreadi=[]

for line in data: # fill arrays
    p=line.split()
    tuc2_v_radeg.append(float(p[0]))
    tuc2_v_decdeg.append(float(p[1]))
    tuc2_v_rmag.append(float(p[2]))
    tuc2_v_gmag.append(float(p[3]))
    tuc2_v_imag.append(float(p[4]))
    tuc2_v_extg.append(float(p[5]))
    tuc2_v_extr.append(float(p[6]))
    tuc2_v_exti.append(float(p[7]))
    tuc2_v_spreadg.append(float(p[8]))
    tuc2_v_spreadr.append(float(p[9]))
    tuc2_v_spreadi.append(float(p[10]))

tuc2_v_radeg=np.array(tuc2_v_radeg)
tuc2_v_decdeg=np.array(tuc2_v_decdeg)
tuc2_v_gmag=np.array(tuc2_v_gmag)
tuc2_v_rmag=np.array(tuc2_v_rmag)
tuc2_v_imag=np.array(tuc2_v_imag)
tuc2_v_extg=np.array(tuc2_v_extg)
tuc2_v_extr=np.array(tuc2_v_extr)
tuc2_v_exti=np.array(tuc2_v_exti)
tuc2_v_spreadg=np.array(tuc2_v_spreadg)
tuc2_v_spreadr=np.array(tuc2_v_spreadr)
tuc2_v_spreadi=np.array(tuc2_v_spreadi)
tuc2_v_gmag=tuc2_v_gmag-tuc2_v_extg
tuc2_v_rmag=tuc2_v_rmag-tuc2_v_extr
tuc2_v_imag=tuc2_v_imag-tuc2_v_exti

tuc2_v_ra=tuc2_v_radeg*np.pi/180.
tuc2_v_dec=tuc2_v_decdeg*np.pi/180.
tuc2_v_radegcenter=342.9796
tuc2_v_decdegcenter=-58.5689
tuc2_v_racenter=np.zeros(len(tuc2_v_ra))+tuc2_v_radegcenter*np.pi/180.
tuc2_v_deccenter=np.zeros(len(tuc2_v_dec))+tuc2_v_decdegcenter*np.pi/180.
tuc2_v_xi,tuc2_v_eta=mycode2.etaxiarr(tuc2_v_ra,tuc2_v_dec,tuc2_v_racenter,tuc2_v_deccenter)
tuc2_v_r=np.sqrt(tuc2_v_xi**2+tuc2_v_eta**2)

tuc2_v_cand=np.zeros(len(tuc2_v_ra),dtype='int')
for i in range(0,len(tuc2_v_cand)):
    tuc2_dist=np.sqrt((tuc2_v_rmag[i]-(tuc2_iso_r+tuc2_dmodulus))**2+((tuc2_v_gmag[i]-tuc2_v_rmag[i])-(tuc2_iso_g-tuc2_iso_r))**2)
    if (min(tuc2_dist) < 0.2) & (tuc2_v_rmag[i] < 22.):
        tuc2_v_cand[i]=1

tuc2_v_cands=np.where(tuc2_v_cand == 1)

tuc2_data1='tuc2.dat'

with open(tuc2_data3) as f: # read data file
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
tuc2_snratio=[]
tuc2_aperture=[]
tuc2_identifier=[]
tuc2_sciname=[]
tuc2_resolution=[]

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
tuc2_gmag=tuc2_gmag-tuc2_extg
tuc2_rmag=tuc2_rmag-tuc2_extr
tuc2_imag=tuc2_imag-tuc2_exti
tuc2_sigr=tuc2_r-tuc2_r
tuc2_snratio=np.array(tuc2_snratio)
tuc2_aperture=np.array(tuc2_aperture)
tuc2_identifier=np.array(tuc2_identifier)
tuc2_sciname=np.array(tuc2_sciname)
tuc2_resolution=np.array(tuc2_resolution)

#tuc2_keep=np.where((tuc2_sigv < 15.) & (tuc2_skewv > -1.) & (tuc2_skewv < 1.) & (tuc2_kurtv > -1.1) & (tuc2_kurtv < 1.1))
tuc2_keep=np.where((tuc2_sigv < 15.))# & (tuc2_skewv > -1.) & (tuc2_skewv < 1.) & (tuc2_kurtv > -1.1) & (tuc2_kurtv < 1.1))
tuc2_mem=np.where((tuc2_sigv < 15.) & (tuc2_pmem > 0.5))
tuc2_goodmem=np.where((tuc2_sigv < 15.) & (tuc2_pmem > 0.9))
tuc2_nonmem=np.where((tuc2_sigv < 15.) & (tuc2_pmem < 0.5))
tuc2_goodnonmem=np.where((tuc2_sigv < 15.) & (tuc2_pmem < 0.5))
tuc2_bad=np.where(tuc2_sigv >= 15.)

tuc2_vmem=[-150.,-110.]
tuc2_loggmem=[0.,4.0]
tuc2_fehmem=[-5.,0.]
tuc2_teffmem=[4000,8000]
tuc2_rmem=[0.,12.]
tuc2_magmem=[21,16]
tuc2_colmem=[-0.2,1]




gru1_data1='gru1_vasilyphot.dat'
gru1_data2='age12fehm250.iso'
gru1_data3='gru1.dat'
gru1_data4='gru1_ellipsehalf.res'

with open(gru1_data2) as g: # read data file
    data=g.readlines()

gru1_iso_g=[]
gru1_iso_r=[]
gru1_iso_i=[]

for line in data: # fill arrays
    p=line.split()
    gru1_iso_g.append(float(p[6]))
    gru1_iso_r.append(float(p[7]))
    gru1_iso_i.append(float(p[8]))

gru1_iso_g=np.array(gru1_iso_g)
gru1_iso_r=np.array(gru1_iso_r)
gru1_iso_i=np.array(gru1_iso_i)

with open(gru1_data4) as g: # read data file
    data=g.readlines()

gru1_ellipse_x=[]
gru1_ellipse_y=[]

for line in data: # fill arrays
    p=line.split()
    gru1_ellipse_x.append(float(p[0]))
    gru1_ellipse_y.append(float(p[1]))

gru1_ellipse_x=np.array(gru1_ellipse_x)
gru1_ellipse_y=np.array(gru1_ellipse_y)

with open(gru1_data1) as f: # read data file
    data=f.readlines()

gru1_v_radeg=[]
gru1_v_decdeg=[]
gru1_v_gmag=[]
gru1_v_rmag=[]
gru1_v_imag=[]
gru1_v_extg=[]
gru1_v_extr=[]
gru1_v_exti=[]
gru1_v_spreadg=[]
gru1_v_spreadr=[]
gru1_v_spreadi=[]

for line in data: # fill arrays
    p=line.split()
    gru1_v_radeg.append(float(p[0]))
    gru1_v_decdeg.append(float(p[1]))
    gru1_v_rmag.append(float(p[2]))
    gru1_v_gmag.append(float(p[3]))
    gru1_v_imag.append(float(p[4]))
    gru1_v_extg.append(float(p[5]))
    gru1_v_extr.append(float(p[6]))
    gru1_v_exti.append(float(p[7]))
    gru1_v_spreadg.append(float(p[8]))
    gru1_v_spreadr.append(float(p[9]))
    gru1_v_spreadi.append(float(p[10]))

gru1_v_radeg=np.array(gru1_v_radeg)
gru1_v_decdeg=np.array(gru1_v_decdeg)
gru1_v_gmag=np.array(gru1_v_gmag)
gru1_v_rmag=np.array(gru1_v_rmag)
gru1_v_imag=np.array(gru1_v_imag)
gru1_v_extg=np.array(gru1_v_extg)
gru1_v_extr=np.array(gru1_v_extr)
gru1_v_exti=np.array(gru1_v_exti)
gru1_v_spreadg=np.array(gru1_v_spreadg)
gru1_v_spreadr=np.array(gru1_v_spreadr)
gru1_v_spreadi=np.array(gru1_v_spreadi)
gru1_v_gmag=gru1_v_gmag-gru1_v_extg
gru1_v_rmag=gru1_v_rmag-gru1_v_extr
gru1_v_imag=gru1_v_imag-gru1_v_exti

gru1_v_ra=gru1_v_radeg*np.pi/180.
gru1_v_dec=gru1_v_decdeg*np.pi/180.
gru1_v_radegcenter=344.1765
gru1_v_decdegcenter=-50.1633
gru1_v_racenter=np.zeros(len(gru1_v_ra))+gru1_v_radegcenter*np.pi/180.
gru1_v_deccenter=np.zeros(len(gru1_v_dec))+gru1_v_decdegcenter*np.pi/180.
gru1_v_xi,gru1_v_eta=mycode2.etaxiarr(gru1_v_ra,gru1_v_dec,gru1_v_racenter,gru1_v_deccenter)
gru1_v_r=np.sqrt(gru1_v_xi**2+gru1_v_eta**2)

gru1_v_cand=np.zeros(len(gru1_v_ra),dtype='int')
for i in range(0,len(gru1_v_cand)):
    gru1_dist=np.sqrt((gru1_v_rmag[i]-(gru1_iso_r+gru1_dmodulus))**2+((gru1_v_gmag[i]-gru1_v_rmag[i])-(gru1_iso_g-gru1_iso_r))**2)
    if (min(gru1_dist) < 0.2) & (gru1_v_rmag[i] < 22.):
        gru1_v_cand[i]=1

gru1_v_cands=np.where(gru1_v_cand == 1)

gru1_data1='gru1.dat'

with open(gru1_data3) as f: # read data file
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
gru1_snratio=[]
gru1_aperture=[]
gru1_identifier=[]
gru1_sciname=[]
gru1_resolution=[]

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
gru1_gmag=gru1_gmag-gru1_extg
gru1_rmag=gru1_rmag-gru1_extr
gru1_imag=gru1_imag-gru1_exti
gru1_sigr=gru1_r-gru1_r
gru1_snratio=np.array(gru1_snratio)
gru1_aperture=np.array(gru1_aperture)
gru1_identifier=np.array(gru1_identifier)
gru1_sciname=np.array(gru1_sciname)
gru1_resolution=np.array(gru1_resolution)

#gru1_keep=np.where((gru1_sigv < 15.) & (gru1_skewv > -1.) & (gru1_skewv < 1.) & (gru1_kurtv > -1.1) & (gru1_kurtv < 1.1))
gru1_keep=np.where((gru1_sigv < 15.))# & (gru1_skewv > -1.) & (gru1_skewv < 1.) & (gru1_kurtv > -1.1) & (gru1_kurtv < 1.1))
gru1_mem=np.where((gru1_sigv < 15.) & (gru1_pmem > 0.5))
gru1_goodmem=np.where((gru1_sigv < 15.) & (gru1_pmem > 0.9))
gru1_nonmem=np.where((gru1_sigv < 15.) & (gru1_pmem < 0.5))
gru1_goodnonmem=np.where((gru1_sigv < 15.) & (gru1_pmem < 0.5))
gru1_bad=np.where(gru1_sigv >= 15.)

gru1_vmem=[-165.,-120.]
gru1_loggmem=[0.,4.0]
gru1_fehmem=[-5.,0.]
gru1_teffmem=[4000,8000]
gru1_rmem=[0.,12.]
gru1_magmem=[21,16]
gru1_colmem=[-0.2,1]



tuc2_center=np.where((tuc2_v_r < 15.))
gru1_center=np.where((gru1_v_r < 15.))

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
#ax0_0.errorbar(v[tuc2_keep],teff[tuc2_keep],xerr=sigv[tuc2_keep],yerr=sigteff[tuc2_keep],elinewidth=0.25,fmt='.',capsize=0,alpha=1,color='k',rasterized=True)
#ax0_0.scatter(tuc2_v_gmag[tuc2_center]-tuc2_v_rmag[tuc2_center],tuc2_v_rmag[tuc2_center],s=2,lw=0,edgecolor='none',alpha=0.99,marker='x',color='0.5',rasterized=True)
ax0_0.scatter(tuc2_v_gmag[tuc2_center]-tuc2_v_rmag[tuc2_center],tuc2_v_rmag[tuc2_center],s=2,lw=0,edgecolor='none',alpha=0.99,marker='o',color='0.5',rasterized=True)
ax0_0.scatter(tuc2_gmag[tuc2_bad]-tuc2_rmag[tuc2_bad],tuc2_rmag[tuc2_bad],s=10,lw=1,edgecolor='k',alpha=0.99,marker='x',color='k',rasterized=True)
ax0_0.scatter(tuc2_gmag[tuc2_nonmem]-tuc2_rmag[tuc2_nonmem],tuc2_rmag[tuc2_nonmem],s=15,lw=1,edgecolor='k',alpha=0.99,marker='o',facecolor='none',rasterized=True)
ax0_0.scatter(tuc2_gmag[tuc2_goodnonmem]-tuc2_rmag[tuc2_goodnonmem],tuc2_rmag[tuc2_goodnonmem],s=15,lw=1,edgecolor='k',alpha=0.99,marker='o',color='k',rasterized=True)
ax0_0.scatter(tuc2_gmag[tuc2_mem]-tuc2_rmag[tuc2_mem],tuc2_rmag[tuc2_mem],s=15,lw=1,edgecolor='r',alpha=0.99,marker='o',facecolor='none',rasterized=True)
ax0_0.scatter(tuc2_gmag[tuc2_goodmem]-tuc2_rmag[tuc2_goodmem],tuc2_rmag[tuc2_goodmem],s=15,lw=1,edgecolor='r',alpha=0.99,marker='o',color='r',rasterized=True)
ax0_0.text(-0.4,16.25,'Tuc 2',fontsize=12)


ax0_0.plot(tuc2_iso_g-tuc2_iso_r,tuc2_iso_r+tuc2_dmodulus,lw=1,color='r')
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
ax0_1.scatter(gru1_v_gmag[gru1_center]-gru1_v_rmag[gru1_center],gru1_v_rmag[gru1_center],s=2,lw=0,edgecolor='none',alpha=0.99,marker='o',color='0.5',rasterized=True)
ax0_1.scatter(gru1_gmag[gru1_bad]-gru1_rmag[gru1_bad],gru1_rmag[gru1_bad],s=10,lw=1,edgecolor='k',alpha=0.99,marker='x',color='k',rasterized=True)
#ax0_1.scatter(gru1_gmag-gru1_rmag,gru1_rmag,s=8,lw=0.75,facecolor='none',edgecolor='k',alpha=0.65,marker='o',color='k',rasterized=True)
ax0_1.scatter(gru1_gmag[gru1_nonmem]-gru1_rmag[gru1_nonmem],gru1_rmag[gru1_nonmem],s=15,lw=1,edgecolor='k',alpha=0.99,marker='o',facecolor='none',rasterized=True)
ax0_1.scatter(gru1_gmag[gru1_goodnonmem]-gru1_rmag[gru1_goodnonmem],gru1_rmag[gru1_goodnonmem],s=15,lw=1,edgecolor='k',alpha=0.99,marker='o',color='k',rasterized=True)
ax0_1.scatter(gru1_gmag[gru1_mem]-gru1_rmag[gru1_mem],gru1_rmag[gru1_mem],s=15,lw=1,edgecolor='r',alpha=0.99,marker='o',facecolor='none',rasterized=True)
ax0_1.scatter(gru1_gmag[gru1_goodmem]-gru1_rmag[gru1_goodmem],gru1_rmag[gru1_goodmem],s=15,lw=1,edgecolor='r',alpha=0.99,marker='o',color='r',rasterized=True)
ax0_1.text(-0.4,16.25,'Gru 1',fontsize=12)


ax0_1.plot(gru1_iso_g-gru1_iso_r,gru1_iso_r+gru1_dmodulus,lw=1,color='r')
#ax0_1.plot(vteffx2,vteffy2,color='b',linewidth=0.25)


plotfilename='tuc2gru1_cmd.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
