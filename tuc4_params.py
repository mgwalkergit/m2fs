import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits 
from scipy import stats
from scipy.stats import gaussian_kde
#import randomp
#import mycode2
from matplotlib.patches import Rectangle
#plt.rc('grid',alpha=0.7) # modify rcparams
#plt.rc('grid',color='white')
#plt.rc('grid',linestyle='-')

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

maxsigv=20.            
data1='tuc4.dat'
out1='tuc4_newcommands.tex'
g1=open(out1,'w')

with open(data1) as f: # read data file
    data=f.readlines()
    
tuc4_radeg=[]
tuc4_decdeg=[]
tuc4_xi=[]
tuc4_eta=[]
tuc4_r=[]
tuc4_hjd=[]
tuc4_v=[]
tuc4_sigv=[]
tuc4_skewv=[]
tuc4_kurtv=[]
tuc4_teff=[]
tuc4_sigteff=[]
tuc4_skewteff=[]
tuc4_kurtteff=[]
tuc4_logg=[]
tuc4_siglogg=[]
tuc4_skewlogg=[]
tuc4_kurtlogg=[]
tuc4_feh=[]
tuc4_sigfeh=[]
tuc4_skewfeh=[]
tuc4_kurtfeh=[]
tuc4_pmem=[]
tuc4_gmag=[]
tuc4_rmag=[]
tuc4_imag=[]
tuc4_extg=[]
tuc4_extr=[]
tuc4_exti=[]
tuc4_snratio=[]
tuc4_aperture=[]
tuc4_identifier=[]
tuc4_filename=[]
tuc4_res=[]
tuc4_id=[]
tuc4_goodnobs=[]

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
    tuc4_pmem.append(float(p[13]))
    tuc4_gmag.append(float(p[14]))
    tuc4_rmag.append(float(p[15]))
    tuc4_imag.append(float(p[16]))
    tuc4_extg.append(float(p[17]))
    tuc4_extr.append(float(p[18]))
    tuc4_exti.append(float(p[19]))
    tuc4_id.append(long(p[20]))
    tuc4_goodnobs.append(long(p[21]))

tuc4_radeg=np.array(tuc4_radeg)
tuc4_decdeg=np.array(tuc4_decdeg)
tuc4_xi=np.array(tuc4_xi)
tuc4_eta=np.array(tuc4_eta)
tuc4_r=np.array(tuc4_r)
tuc4_hjd=np.array(tuc4_hjd)
tuc4_v=np.array(tuc4_v)
tuc4_sigv=np.array(tuc4_sigv)
tuc4_skewv=np.array(tuc4_skewv)
tuc4_kurtv=np.array(tuc4_kurtv)
tuc4_teff=np.array(tuc4_teff)
tuc4_sigteff=np.array(tuc4_sigteff)
tuc4_skewteff=np.array(tuc4_skewteff)
tuc4_kurtteff=np.array(tuc4_kurtteff)
tuc4_logg=np.array(tuc4_logg)
tuc4_siglogg=np.array(tuc4_siglogg)
tuc4_skewlogg=np.array(tuc4_skewlogg)
tuc4_kurtlogg=np.array(tuc4_kurtlogg)
tuc4_feh=np.array(tuc4_feh)
tuc4_sigfeh=np.array(tuc4_sigfeh)
tuc4_skewfeh=np.array(tuc4_skewfeh)
tuc4_kurtfeh=np.array(tuc4_kurtfeh)
tuc4_pmem=np.array(tuc4_pmem)
tuc4_gmag=np.array(tuc4_gmag)
tuc4_rmag=np.array(tuc4_rmag)
tuc4_imag=np.array(tuc4_imag)
tuc4_extg=np.array(tuc4_extg)
tuc4_extr=np.array(tuc4_extr)
tuc4_exti=np.array(tuc4_exti)
tuc4_sigr=tuc4_r-tuc4_r
tuc4_mag=tuc4_rmag-tuc4_extr
tuc4_col=tuc4_gmag-tuc4_extg-(tuc4_rmag-tuc4_extr)
tuc4_sigmag=tuc4_mag-tuc4_mag
tuc4_sigcol=tuc4_col-tuc4_col
tuc4_snratio=np.array(tuc4_snratio)
tuc4_aperture=np.array(tuc4_aperture)
tuc4_identifier=np.array(tuc4_identifier)
tuc4_filename=np.array(tuc4_filename)
tuc4_res=np.array(tuc4_res)
tuc4_goodnobs=np.array(tuc4_goodnobs)

tuc4_keep=np.where((tuc4_sigv < maxsigv))#) & (tuc4_skewv >= -1.) & (tuc4_skewv <= 1.) & (tuc4_kurtv >= -1) & (tuc4_kurtv <= 1))
tuc4_mem=np.where((tuc4_sigv < maxsigv) & (tuc4_pmem > 0.5))
tuc4_goodmem=np.where((tuc4_sigv < maxsigv) & (tuc4_pmem > 0.9))
tuc4_nonmem=np.where((tuc4_sigv < maxsigv) & (tuc4_pmem < 0.5))
tuc4_goodnonmem=np.where((tuc4_sigv < maxsigv) & (tuc4_pmem < 0.5))
tuc4_qc=np.where(tuc4_goodnobs >= 1)
tuc4_bad=np.where(tuc4_sigv >= maxsigv)


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



tuc4_vmem=[-150.,-115.]
tuc4_loggmem=[0.5,4.0]
tuc4_fehmem=[-3.,-1.]
tuc4_teffmem=[4000,6000]
tuc4_rmem=[0.,14.]
tuc4_magmem=[21.5,18]
tuc4_colmem=[0.4,0.75]

gs=plt.GridSpec(7,5) # define multi-panel plot
gs2=plt.GridSpec(15,15) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
gs2.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
#fig.set_

ax10_10=fig.add_subplot(gs2[12:15,12:15])
ax10_9=fig.add_subplot(gs2[12:15,9:12])
ax9_10=fig.add_subplot(gs2[9:12,12:15])

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

ax3_4=fig.add_subplot(gs[3,4])


ax2_0=fig.add_subplot(gs[2,0])
ax2_1=fig.add_subplot(gs[2,1])
ax2_2=fig.add_subplot(gs[2,2])
ax2_3=fig.add_subplot(gs[2,3])

vlim=[-100,250]
tefflim=[4001.,7999.] 
revtefflim=[8000.,4000.] 
logglim=[0.01,4.99]
revlogglim=[5,0]
fehlim=[-4.99,0.99]
rlim=[0.,20.]
collim=[0,1]
maglim=[16.5,20.5]

vlabel=[r'$v_{\rm los}$ [km/s]']
tefflabel=[r'$T_{\rm eff}$ [K]']
logglabel=[r'$\log_{10}[g/(\mathrm{cm/s}^2)]$']
fehlabel=[r'$[\mathrm{Fe/H}]$']
rlabel=[r'$R$ [arcmin]']
collabel=[r'g-r [mag]']
maglabel=[r'r [mag]']

histticks=[3,6,9,12,15]
histlim=[0,15]
vticks=[-150,0,150,300]
vticklabels=['-150','0','150','300']
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
colticks=[0.25,0.5,0.75]
colticklabels=colticks
magticks=[17,18,19,20]
magticklabels=magticks

tuc4_vteffx=[tuc4_vmem[0],tuc4_vmem[1],tuc4_vmem[1],tuc4_vmem[0],tuc4_vmem[0]]
tuc4_vteffy=[tuc4_teffmem[0],tuc4_teffmem[0],tuc4_teffmem[1],tuc4_teffmem[1],tuc4_teffmem[0]]
tuc4_vloggx=[tuc4_vmem[0],tuc4_vmem[1],tuc4_vmem[1],tuc4_vmem[0],tuc4_vmem[0]]
tuc4_vloggy=[tuc4_loggmem[0],tuc4_loggmem[0],tuc4_loggmem[1],tuc4_loggmem[1],tuc4_loggmem[0]]
tuc4_vfehx=[tuc4_vmem[0],tuc4_vmem[1],tuc4_vmem[1],tuc4_vmem[0],tuc4_vmem[0]]
tuc4_vfehy=[tuc4_fehmem[0],tuc4_fehmem[0],tuc4_fehmem[1],tuc4_fehmem[1],tuc4_fehmem[0]]
tuc4_vrx=[tuc4_vmem[0],tuc4_vmem[1],tuc4_vmem[1],tuc4_vmem[0],tuc4_vmem[0]]
tuc4_vry=[tuc4_rmem[0],tuc4_rmem[0],tuc4_rmem[1],tuc4_rmem[1],tuc4_rmem[0]]
tuc4_vcolx=[tuc4_vmem[0],tuc4_vmem[1],tuc4_vmem[1],tuc4_vmem[0],tuc4_vmem[0]]
tuc4_vcoly=[tuc4_colmem[0],tuc4_colmem[0],tuc4_colmem[1],tuc4_colmem[1],tuc4_colmem[0]]
tuc4_vmagx=[tuc4_vmem[0],tuc4_vmem[1],tuc4_vmem[1],tuc4_vmem[0],tuc4_vmem[0]]
tuc4_vmagy=[tuc4_magmem[0],tuc4_magmem[0],tuc4_magmem[1],tuc4_magmem[1],tuc4_magmem[0]]

tuc4_teffloggx=[tuc4_teffmem[0],tuc4_teffmem[1],tuc4_teffmem[1],tuc4_teffmem[0],tuc4_teffmem[0]]
tuc4_teffloggy=[tuc4_loggmem[0],tuc4_loggmem[0],tuc4_loggmem[1],tuc4_loggmem[1],tuc4_loggmem[0]]
tuc4_tefffehx=[tuc4_teffmem[0],tuc4_teffmem[1],tuc4_teffmem[1],tuc4_teffmem[0],tuc4_teffmem[0]]
tuc4_tefffehy=[tuc4_fehmem[0],tuc4_fehmem[0],tuc4_fehmem[1],tuc4_fehmem[1],tuc4_fehmem[0]]
tuc4_teffrx=[tuc4_teffmem[0],tuc4_teffmem[1],tuc4_teffmem[1],tuc4_teffmem[0],tuc4_teffmem[0]]
tuc4_teffry=[tuc4_rmem[0],tuc4_rmem[0],tuc4_rmem[1],tuc4_rmem[1],tuc4_rmem[0]]
tuc4_teffcolx=[tuc4_teffmem[0],tuc4_teffmem[1],tuc4_teffmem[1],tuc4_teffmem[0],tuc4_teffmem[0]]
tuc4_teffcoly=[tuc4_colmem[0],tuc4_colmem[0],tuc4_colmem[1],tuc4_colmem[1],tuc4_colmem[0]]
tuc4_teffmagx=[tuc4_teffmem[0],tuc4_teffmem[1],tuc4_teffmem[1],tuc4_teffmem[0],tuc4_teffmem[0]]
tuc4_teffmagy=[tuc4_magmem[0],tuc4_magmem[0],tuc4_magmem[1],tuc4_magmem[1],tuc4_magmem[0]]

tuc4_loggfehx=[tuc4_loggmem[0],tuc4_loggmem[1],tuc4_loggmem[1],tuc4_loggmem[0],tuc4_loggmem[0]]
tuc4_loggfehy=[tuc4_fehmem[0],tuc4_fehmem[0],tuc4_fehmem[1],tuc4_fehmem[1],tuc4_fehmem[0]]
tuc4_loggrx=[tuc4_loggmem[0],tuc4_loggmem[1],tuc4_loggmem[1],tuc4_loggmem[0],tuc4_loggmem[0]]
tuc4_loggry=[tuc4_rmem[0],tuc4_rmem[0],tuc4_rmem[1],tuc4_rmem[1],tuc4_rmem[0]]
tuc4_loggcolx=[tuc4_loggmem[0],tuc4_loggmem[1],tuc4_loggmem[1],tuc4_loggmem[0],tuc4_loggmem[0]]
tuc4_loggcoly=[tuc4_colmem[0],tuc4_colmem[0],tuc4_colmem[1],tuc4_colmem[1],tuc4_colmem[0]]
tuc4_loggmagx=[tuc4_loggmem[0],tuc4_loggmem[1],tuc4_loggmem[1],tuc4_loggmem[0],tuc4_loggmem[0]]
tuc4_loggmagy=[tuc4_magmem[0],tuc4_magmem[0],tuc4_magmem[1],tuc4_magmem[1],tuc4_magmem[0]]

tuc4_fehrx=[tuc4_fehmem[0],tuc4_fehmem[1],tuc4_fehmem[1],tuc4_fehmem[0],tuc4_fehmem[0]]
tuc4_fehry=[tuc4_rmem[0],tuc4_rmem[0],tuc4_rmem[1],tuc4_rmem[1],tuc4_rmem[0]]
tuc4_fehcolx=[tuc4_fehmem[0],tuc4_fehmem[1],tuc4_fehmem[1],tuc4_fehmem[0],tuc4_fehmem[0]]
tuc4_fehcoly=[tuc4_colmem[0],tuc4_colmem[0],tuc4_colmem[1],tuc4_colmem[1],tuc4_colmem[0]]
tuc4_fehmagx=[tuc4_fehmem[0],tuc4_fehmem[1],tuc4_fehmem[1],tuc4_fehmem[0],tuc4_fehmem[0]]
tuc4_fehmagy=[tuc4_magmem[0],tuc4_magmem[0],tuc4_magmem[1],tuc4_magmem[1],tuc4_magmem[0]]

ax6_0.set_xlabel(vlabel[0],fontsize=8,rotation=0)
ax6_0.set_ylabel(tefflabel[0],fontsize=8,rotation=90,labelpad=5)
ax6_0.set_xlim(vlim)
ax6_0.set_ylim(revtefflim)
ax6_0.set_xscale(u'linear')
ax6_0.set_yscale(u'linear')
ax6_0.set_xticks(vticks)
ax6_0.set_xticklabels(vticks,rotation=0)
ax6_0.set_yticks([8000]+revteffticks)
ax6_0.set_yticklabels([8000]+revteffticks,rotation=0)
#ax6_0.add_patch(Rectangle((tuc4_vmem[0],tuc4_teffmem[0]),tuc4_vmem[1]-tuc4_vmem[0],tuc4_teffmem[1]-tuc4_teffmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax6_0.errorbar(tuc4_v[tuc4_keep],tuc4_teff[tuc4_keep],xerr=tuc4_sigv[tuc4_keep],yerr=tuc4_sigteff[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)

ax5_0.set_ylabel(logglabel[0],fontsize=8,rotation=90,labelpad=5)
ax5_0.set_xlim(vlim)
ax5_0.set_ylim(revlogglim)
ax5_0.set_xscale(u'linear')
ax5_0.set_yscale(u'linear')
ax5_0.set_xticks(vticks)
ax5_0.set_yticks(revloggticks)
ax5_0.set_yticklabels(revloggticks,rotation=0)
ax5_0.xaxis.set_major_formatter(plt.NullFormatter())
#ax5_0.add_patch(Rectangle((tuc4_vmem[0],tuc4_loggmem[0]),tuc4_vmem[1]-tuc4_vmem[0],tuc4_loggmem[1]-tuc4_loggmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax5_0.errorbar(tuc4_v[tuc4_keep],tuc4_logg[tuc4_keep],xerr=tuc4_sigv[tuc4_keep],yerr=tuc4_siglogg[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax5_0.plot(tuc4_vloggx,tuc4_vloggy,color='r',linewidth=0.25)

ax4_0.set_ylabel(fehlabel[0],fontsize=8,rotation=90,labelpad=5)
ax4_0.set_xlim(vlim)
ax4_0.set_ylim(fehlim)
ax4_0.set_xscale(u'linear')
ax4_0.set_yscale(u'linear')
ax4_0.set_xticks(vticks)
ax4_0.set_yticks(fehticks)
ax4_0.set_yticklabels(fehticks,rotation=0)
ax4_0.xaxis.set_major_formatter(plt.NullFormatter())
#ax4_0.add_patch(Rectangle((tuc4_vmem[0],tuc4_fehmem[0]),tuc4_vmem[1]-tuc4_vmem[0],tuc4_fehmem[1]-tuc4_fehmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax4_0.errorbar(tuc4_v[tuc4_keep],tuc4_feh[tuc4_keep],xerr=tuc4_sigv[tuc4_keep],yerr=tuc4_sigfeh[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax4_0.plot(tuc4_vfehx,tuc4_vfehy,color='r',linewidth=0.25)

ax3_0.set_ylabel(rlabel[0],fontsize=8,rotation=90,labelpad=5)
ax3_0.set_xlim(vlim)
ax3_0.set_ylim(rlim)
ax3_0.set_xscale(u'linear')
ax3_0.set_yscale(u'linear')
ax3_0.set_xticks(vticks)
ax3_0.set_yticks(rticks)
ax3_0.set_yticklabels(rticks,rotation=0)
ax3_0.xaxis.set_major_formatter(plt.NullFormatter())
#ax3_0.add_patch(Rectangle((tuc4_vmem[0],tuc4_rmem[0]),tuc4_vmem[1]-tuc4_vmem[0],tuc4_rmem[1]-tuc4_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_0.errorbar(tuc4_v[tuc4_keep],tuc4_r[tuc4_keep],xerr=tuc4_sigv[tuc4_keep],yerr=tuc4_sigr[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax3_0.plot(tuc4_vrx,tuc4_vry,color='r',linewidth=0.25)

ax2_0.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_0.xaxis.set_major_formatter(plt.NullFormatter())
ax2_0.set_xlim(vlim)
ax2_0.set_ylim(histlim)
ax2_0.set_xticks(vticks)
ax2_0.set_yticks(histticks)
ax2_0.hist(tuc4_v[tuc4_keep],bins=30,range=vlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,label='tuc 4')
#ax2_0.hist(tuc4_v[tuc4_mem],bins=30,range=vlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)
ax2_0.legend(loc=2,fontsize=6,handlelength=0.5,numpoints=1,scatterpoints=1,shadow=False,borderpad=0)


ax5_1.set_xlabel(tefflabel[0],fontsize=8,rotation=0)
ax5_1.set_xlim(revtefflim)
ax5_1.set_ylim(revlogglim)
ax5_1.set_xscale(u'linear')
ax5_1.set_yscale(u'linear')
ax5_1.set_xticks([4000]+revteffticks)
ax5_1.set_xticklabels([4000]+revteffticks,rotation=0)
ax5_1.set_yticks(revloggticks)
ax5_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax5_1.add_patch(Rectangle((tuc4_teffmem[0],tuc4_loggmem[0]),tuc4_teffmem[1]-tuc4_teffmem[0],tuc4_loggmem[1]-tuc4_loggmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax5_1.errorbar(tuc4_teff[tuc4_keep],tuc4_logg[tuc4_keep],xerr=tuc4_sigteff[tuc4_keep],yerr=tuc4_siglogg[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)

ax4_1.set_xlim(revtefflim)
ax4_1.set_ylim(fehlim)
ax4_1.set_xscale(u'linear')
ax4_1.set_yscale(u'linear')
ax4_1.set_xticks(revteffticks)
ax4_1.set_yticks(fehticks)
ax4_1.xaxis.set_major_formatter(plt.NullFormatter())
ax4_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax4_1.add_patch(Rectangle((tuc4_teffmem[0],tuc4_fehmem[0]),tuc4_teffmem[1]-tuc4_teffmem[0],tuc4_fehmem[1]-tuc4_fehmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax4_1.errorbar(tuc4_teff[tuc4_keep],tuc4_feh[tuc4_keep],xerr=tuc4_sigteff[tuc4_keep],yerr=tuc4_sigfeh[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)

ax3_1.set_xlim(revtefflim)
ax3_1.set_ylim(rlim)
ax3_1.set_xscale(u'linear')
ax3_1.set_yscale(u'linear')
ax3_1.set_xticks(revteffticks)
ax3_1.set_yticks(rticks)
ax3_1.xaxis.set_major_formatter(plt.NullFormatter())
ax3_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax3_1.add_patch(Rectangle((tuc4_teffmem[0],tuc4_rmem[0]),tuc4_teffmem[1]-tuc4_teffmem[0],tuc4_rmem[1]-tuc4_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_1.errorbar(tuc4_teff[tuc4_keep],tuc4_r[tuc4_keep],xerr=tuc4_sigteff[tuc4_keep],yerr=tuc4_sigr[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)

ax2_1.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_1.xaxis.set_major_formatter(plt.NullFormatter())
ax2_1.yaxis.set_major_formatter(plt.NullFormatter())
ax2_1.set_xlim(revtefflim)
ax2_1.set_ylim(histlim)
ax2_1.set_xticks(revteffticks)
ax2_1.set_yticks(histticks)
ax2_1.hist(tuc4_teff[tuc4_keep],bins=30,range=tefflim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25)
#ax2_1.hist(tuc4_teff[tuc4_mem],bins=30,range=tefflim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)


ax4_2.set_xlabel(logglabel[0],fontsize=8,rotation=0)
ax4_2.set_xlim(revlogglim)
ax4_2.set_ylim(fehlim)
ax4_2.set_xscale(u'linear')
ax4_2.set_yscale(u'linear')
ax4_2.set_xticks([0]+revloggticks)
ax4_2.set_xticklabels([0]+revloggticks,rotation=0)
ax4_2.set_yticks(fehticks)
ax4_2.yaxis.set_major_formatter(plt.NullFormatter())
#ax4_2.add_patch(Rectangle((tuc4_loggmem[0],tuc4_fehmem[0]),tuc4_loggmem[1]-tuc4_loggmem[0],tuc4_fehmem[1]-tuc4_fehmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax4_2.errorbar(tuc4_logg[tuc4_keep],tuc4_r[tuc4_keep],xerr=tuc4_siglogg[tuc4_keep],yerr=tuc4_sigr[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax4_2.errorbar(tuc4_logg[tuc4_keep],tuc4_feh[tuc4_keep],xerr=tuc4_siglogg[tuc4_keep],yerr=tuc4_sigfeh[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax4_2.plot(tuc4_loggfehx,tuc4_loggfehy,color='r',linewidth=0.25)


ax3_2.set_xlim(revlogglim)
ax3_2.set_ylim(rlim)
ax3_2.set_xscale(u'linear')
ax3_2.set_yscale(u'linear')
ax3_2.set_xticks(revloggticks)
ax3_2.set_yticks(rticks)
ax3_2.xaxis.set_major_formatter(plt.NullFormatter())
ax3_2.yaxis.set_major_formatter(plt.NullFormatter())
#ax3_2.add_patch(Rectangle((tuc4_loggmem[0],tuc4_rmem[0]),tuc4_loggmem[1]-tuc4_loggmem[0],tuc4_rmem[1]-tuc4_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_2.errorbar(tuc4_logg[tuc4_keep],tuc4_r[tuc4_keep],xerr=tuc4_siglogg[tuc4_keep],yerr=tuc4_sigr[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax3_2.errorbar(tuc4_logg[tuc4_keep],tuc4_r[tuc4_keep],xerr=tuc4_siglogg[tuc4_keep],yerr=tuc4_sigr[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax3_2.plot(tuc4_loggrx,tuc4_loggry,color='r',linewidth=0.25)

ax2_2.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_2.xaxis.set_major_formatter(plt.NullFormatter())
ax2_2.yaxis.set_major_formatter(plt.NullFormatter())
ax2_2.set_xlim(revlogglim)
ax2_2.set_ylim(histlim)
ax2_2.set_xticks(revloggticks)
ax2_2.set_yticks(histticks)
ax2_2.hist(tuc4_logg[tuc4_keep],bins=30,range=logglim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25)
#ax2_2.hist(tuc4_logg[tuc4_mem],bins=30,range=logglim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)



ax3_3.set_xlabel(fehlabel[0],fontsize=8,rotation=0)
ax3_3.set_xlim(fehlim)
ax3_3.set_ylim(rlim)
ax3_3.set_xscale(u'linear')
ax3_3.set_yscale(u'linear')
ax3_3.set_xticks(fehticks)
ax3_3.set_xticklabels(fehticks,rotation=0)
ax3_3.set_yticks(rticks)
ax3_3.yaxis.set_major_formatter(plt.NullFormatter())
#ax3_3.add_patch(Rectangle((tuc4_fehmem[0],tuc4_rmem[0]),tuc4_fehmem[1]-tuc4_fehmem[0],tuc4_rmem[1]-tuc4_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_3.errorbar(tuc4_feh[tuc4_keep],tuc4_r[tuc4_keep],xerr=tuc4_sigfeh[tuc4_keep],yerr=tuc4_sigr[tuc4_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax3_3.plot(tuc4_fehrx,tuc4_fehry,color='r',linewidth=0.25)

ax2_3.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_3.xaxis.set_major_formatter(plt.NullFormatter())
ax2_3.yaxis.set_major_formatter(plt.NullFormatter())
ax2_3.set_ylim(histlim)
ax2_3.set_xticks(fehticks)
ax2_3.set_yticks(histticks)
ax2_3.hist(tuc4_feh[tuc4_keep],bins=30,range=fehlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25)
#ax2_3.hist(tuc4_feh[tuc4_mem],bins=30,range=fehlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)



#ax3_4.set_xlabel('N',fontsize=8,rotation=0,labelpad=5)
ax3_4.xaxis.set_major_formatter(plt.NullFormatter())
ax3_4.yaxis.set_major_formatter(plt.NullFormatter())
ax3_4.set_xlim(histlim)
ax3_4.set_yticks(rticks)
ax3_4.set_xticks(histticks)
ax3_4.hist(tuc4_r[tuc4_keep],bins=30,range=rlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.25)
#ax3_4.hist(tuc4_r[tuc4_mem],bins=30,range=rlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.25,alpha=0.5)


ax10_10.set_xlabel(r'$T_{\rm eff}$ [K]',fontsize=8,rotation=0,labelpad=5)
ax10_10.set_ylabel(r'$\log g$',fontsize=8,rotation=90,labelpad=5)
ax10_10.set_xlim([8000,4000])
ax10_10.set_ylim([5,0.01])
ax10_10.yaxis.set_label_position('right')
ax10_10.yaxis.tick_right()
ax10_10.set_xscale(u'linear')
ax10_10.set_yscale(u'linear')
ax10_10.set_yticks([5,4,3,2,1])
#ax10_10.set_xticklabels(vticks,rotation=0)
ax10_10.set_xticks([7000,6000,5000,4000])
ax10_10.plot(iso250_teff,iso250_logg,lw=1,color='k',ls='-')
ax10_10.plot(iso100_teff,iso100_logg,lw=1,color='k',ls='--')
#ax10_10.plot(iso000_teff,iso000_logg,lw=1,color='g')
ax10_10.scatter(tuc4_teff[tuc4_keep],tuc4_logg[tuc4_keep],marker='.',s=1,alpha=0.65,color='k',rasterized=True,label='foreground')
#ax10_10.scatter(tuc4_teff[tuc4_mem],tuc4_logg[tuc4_mem],marker='.',s=1,alpha=0.65,color='r',rasterized=True)
#ax10_10.errorbar(tuc4_teff[tuc4_mem],tuc4_logg[tuc4_mem],xerr=tuc4_sigteff[tuc4_mem],yerr=tuc4_siglogg[tuc4_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True,label='tuc 4 member')
#ax10_10.legend(loc=2,fontsize=8,handlelength=0,shadow=False)
#ax2_0.text(-0.4,16.5,'Crater',fontsize=10)
ax10_10.legend(loc=2,fontsize=4,handlelength=0,numpoints=1,scatterpoints=1,shadow=False)


#ax9_10.set_xlabel(r'$T_{\rm eff}$ [K]',fontsize=12,rotation=0,labelpad=5)
ax9_10.set_ylabel(r'$r-(m-M)$',fontsize=7,rotation=90,labelpad=5)
ax9_10.set_xlim([8000,4000])
ax9_10.yaxis.set_label_position('right')
ax9_10.yaxis.tick_right()
ax9_10.set_ylim([2.49,-3.9])
ax9_10.set_xscale(u'linear')
ax9_10.set_yscale(u'linear')
ax9_10.set_yticks([2,1,0,-1,-2,-3])
ax9_10.set_xticks([8000,7000,6000,5000,4000])
ax9_10.xaxis.set_major_formatter(plt.NullFormatter())
ax9_10.plot(iso250_teff,iso250_r,lw=1,color='k',label=r"[Fe/H]=$-2.5$",ls='-')
ax9_10.plot(iso100_teff,iso100_r,lw=1,color='k',label=r"[Fe/H]=$-1.0$",ls='--')
#ax9_10.plot(iso000_teff,iso000_r+dmodulus,lw=1,color='g',label=r"[Fe/H]=$0$")
#ax9_10.errorbar(teff,rmag,xerr=sigteff,yerr=siglogg-siglogg,elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='k',rasterized=True)
#ax9_10.errorbar(tuc4_teff[tuc4_mem],tuc4_mag[tuc4_mem]-tuc4_dmodulus,xerr=tuc4_sigteff[tuc4_mem],yerr=tuc4_siglogg[tuc4_mem]-tuc4_siglogg[tuc4_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax9_10.legend(loc=2,fontsize=4,handlelength=4,shadow=False)
#ax9_10.text(-0.4,16.5,'Crater',fontsize=10)
#ax9_10.text(7500,16.5,'tuc 4',fontsize=12)

ax10_9.set_xlabel(r'$r-(m-M)$',fontsize=8,rotation=0,labelpad=5)
#ax10_9.set_ylabel(r'$\log g$',fontsize=8,rotation=90,labelpad=5)
ax10_9.set_xlim([2.49,-3.9])
ax10_9.set_ylim([5,0.01])
ax10_9.set_xscale(u'linear')
ax10_9.set_yscale(u'linear')
ax10_9.set_yticks([5,4,3,2,1])
ax10_9.yaxis.set_major_formatter(plt.NullFormatter())
#ax10_9.set_xticklabels(vticks,rotation=0)
ax10_9.set_xticks([2,1,0,-1,-2,-3])
ax10_9.plot(iso250_r,iso250_logg,lw=1,color='k',label=r"[Fe/H]=$-2.5$",ls='-')
ax10_9.plot(iso100_r,iso100_logg,lw=1,color='k',label=r"[Fe/H]=$-1.0$",ls='--')
#ax10_9.plot(iso000_r+dmodulus,iso000_logg,lw=1,color='g',label=r"[Fe/H]=$0$")
#ax10_9.errorbar(rmag,logg,xerr=sigteff-sigteff,yerr=siglogg,elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='k',rasterized=True)
#ax10_9.errorbar(tuc4_mag[tuc4_mem]-tuc4_dmodulus,tuc4_logg[tuc4_mem],xerr=tuc4_sigteff[tuc4_mem]-tuc4_sigteff[tuc4_mem],yerr=tuc4_siglogg[tuc4_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax10_9.legend(loc=2,fontsize=12,handlelength=0,shadow=False)
#ax2_0.text(-0.4,16.5,'Crater',fontsize=10)

g1.write(r'\newcommand{\tucnobs}{$'+str(len(tuc4_v))+r'$}'+'\n')
g1.write(r'\newcommand{\tucgoodnobs}{$'+str(len(tuc4_v[tuc4_keep]))+r'$}'+'\n')
g1.write(r'\newcommand{\tucqc}{$'+str(len(tuc4_v[tuc4_qc]))+r'$}'+'\n')
#g1.write(r'\newcommand{\nred}{$'+str(len(tuc4_v[mem]))+r'$}'+'\n')
#g1.write(r'\newcommand{\nblue}{$'+str(len(v[mem3]))+r'$}'+'\n')
g1.write(r'\newcommand{\tucmembers}{$'+str(np.size(tuc4_mem))+r'$}'+'\n')
g1.write(r'\newcommand{\tucgoodmembers}{$'+str(np.size(tuc4_goodmem))+r'$}'+'\n')
g1.close()



plotfilename='tuc4_params.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
