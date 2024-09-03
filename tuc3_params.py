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
data1='tuc3.dat'
out1='tuc3_newcommands.tex'
g1=open(out1,'w')

with open(data1) as f: # read data file
    data=f.readlines()
    
tuc3_radeg=[]
tuc3_decdeg=[]
tuc3_xi=[]
tuc3_eta=[]
tuc3_r=[]
tuc3_hjd=[]
tuc3_v=[]
tuc3_sigv=[]
tuc3_skewv=[]
tuc3_kurtv=[]
tuc3_teff=[]
tuc3_sigteff=[]
tuc3_skewteff=[]
tuc3_kurtteff=[]
tuc3_logg=[]
tuc3_siglogg=[]
tuc3_skewlogg=[]
tuc3_kurtlogg=[]
tuc3_feh=[]
tuc3_sigfeh=[]
tuc3_skewfeh=[]
tuc3_kurtfeh=[]
tuc3_pmem=[]
tuc3_gmag=[]
tuc3_rmag=[]
tuc3_imag=[]
tuc3_extg=[]
tuc3_extr=[]
tuc3_exti=[]
tuc3_snratio=[]
tuc3_aperture=[]
tuc3_identifier=[]
tuc3_filename=[]
tuc3_res=[]
tuc3_id=[]
tuc3_goodnobs=[]

for line in data: # fill arrays
    p=line.split()
    tuc3_radeg.append(float(p[0]))
    tuc3_decdeg.append(float(p[1]))
    tuc3_xi.append(float(p[2]))
    tuc3_eta.append(float(p[3]))
    tuc3_r.append(float(p[4]))
    tuc3_v.append(float(p[5]))
    tuc3_sigv.append(float(p[6]))
    tuc3_teff.append(float(p[7]))
    tuc3_sigteff.append(float(p[8]))
    tuc3_logg.append(float(p[9]))
    tuc3_siglogg.append(float(p[10]))
    tuc3_feh.append(float(p[11]))
    tuc3_sigfeh.append(float(p[12]))
    tuc3_pmem.append(float(p[13]))
    tuc3_gmag.append(float(p[14]))
    tuc3_rmag.append(float(p[15]))
    tuc3_imag.append(float(p[16]))
    tuc3_extg.append(float(p[17]))
    tuc3_extr.append(float(p[18]))
    tuc3_exti.append(float(p[19]))
    tuc3_id.append(long(p[20]))
    tuc3_goodnobs.append(long(p[21]))

tuc3_radeg=np.array(tuc3_radeg)
tuc3_decdeg=np.array(tuc3_decdeg)
tuc3_xi=np.array(tuc3_xi)
tuc3_eta=np.array(tuc3_eta)
tuc3_r=np.array(tuc3_r)
tuc3_hjd=np.array(tuc3_hjd)
tuc3_v=np.array(tuc3_v)
tuc3_sigv=np.array(tuc3_sigv)
tuc3_skewv=np.array(tuc3_skewv)
tuc3_kurtv=np.array(tuc3_kurtv)
tuc3_teff=np.array(tuc3_teff)
tuc3_sigteff=np.array(tuc3_sigteff)
tuc3_skewteff=np.array(tuc3_skewteff)
tuc3_kurtteff=np.array(tuc3_kurtteff)
tuc3_logg=np.array(tuc3_logg)
tuc3_siglogg=np.array(tuc3_siglogg)
tuc3_skewlogg=np.array(tuc3_skewlogg)
tuc3_kurtlogg=np.array(tuc3_kurtlogg)
tuc3_feh=np.array(tuc3_feh)
tuc3_sigfeh=np.array(tuc3_sigfeh)
tuc3_skewfeh=np.array(tuc3_skewfeh)
tuc3_kurtfeh=np.array(tuc3_kurtfeh)
tuc3_pmem=np.array(tuc3_pmem)
tuc3_gmag=np.array(tuc3_gmag)
tuc3_rmag=np.array(tuc3_rmag)
tuc3_imag=np.array(tuc3_imag)
tuc3_extg=np.array(tuc3_extg)
tuc3_extr=np.array(tuc3_extr)
tuc3_exti=np.array(tuc3_exti)
tuc3_sigr=tuc3_r-tuc3_r
tuc3_mag=tuc3_rmag-tuc3_extr
tuc3_col=tuc3_gmag-tuc3_extg-(tuc3_rmag-tuc3_extr)
tuc3_sigmag=tuc3_mag-tuc3_mag
tuc3_sigcol=tuc3_col-tuc3_col
tuc3_snratio=np.array(tuc3_snratio)
tuc3_aperture=np.array(tuc3_aperture)
tuc3_identifier=np.array(tuc3_identifier)
tuc3_filename=np.array(tuc3_filename)
tuc3_res=np.array(tuc3_res)
tuc3_goodnobs=np.array(tuc3_goodnobs)

tuc3_keep=np.where((tuc3_sigv < maxsigv))#) & (tuc3_skewv >= -1.) & (tuc3_skewv <= 1.) & (tuc3_kurtv >= -1) & (tuc3_kurtv <= 1))
tuc3_mem=np.where((tuc3_sigv < maxsigv) & (tuc3_pmem > 0.5))
tuc3_goodmem=np.where((tuc3_sigv < maxsigv) & (tuc3_pmem > 0.9))
tuc3_nonmem=np.where((tuc3_sigv < maxsigv) & (tuc3_pmem < 0.5))
tuc3_goodnonmem=np.where((tuc3_sigv < maxsigv) & (tuc3_pmem < 0.5))
tuc3_qc=np.where(tuc3_goodnobs >= 1)
tuc3_bad=np.where(tuc3_sigv >= maxsigv)


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



tuc3_vmem=[-150.,-115.]
tuc3_loggmem=[0.5,4.0]
tuc3_fehmem=[-3.,-1.]
tuc3_teffmem=[4000,6000]
tuc3_rmem=[0.,14.]
tuc3_magmem=[21.5,18]
tuc3_colmem=[0.4,0.75]

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

vlim=[-100,300]
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

tuc3_vteffx=[tuc3_vmem[0],tuc3_vmem[1],tuc3_vmem[1],tuc3_vmem[0],tuc3_vmem[0]]
tuc3_vteffy=[tuc3_teffmem[0],tuc3_teffmem[0],tuc3_teffmem[1],tuc3_teffmem[1],tuc3_teffmem[0]]
tuc3_vloggx=[tuc3_vmem[0],tuc3_vmem[1],tuc3_vmem[1],tuc3_vmem[0],tuc3_vmem[0]]
tuc3_vloggy=[tuc3_loggmem[0],tuc3_loggmem[0],tuc3_loggmem[1],tuc3_loggmem[1],tuc3_loggmem[0]]
tuc3_vfehx=[tuc3_vmem[0],tuc3_vmem[1],tuc3_vmem[1],tuc3_vmem[0],tuc3_vmem[0]]
tuc3_vfehy=[tuc3_fehmem[0],tuc3_fehmem[0],tuc3_fehmem[1],tuc3_fehmem[1],tuc3_fehmem[0]]
tuc3_vrx=[tuc3_vmem[0],tuc3_vmem[1],tuc3_vmem[1],tuc3_vmem[0],tuc3_vmem[0]]
tuc3_vry=[tuc3_rmem[0],tuc3_rmem[0],tuc3_rmem[1],tuc3_rmem[1],tuc3_rmem[0]]
tuc3_vcolx=[tuc3_vmem[0],tuc3_vmem[1],tuc3_vmem[1],tuc3_vmem[0],tuc3_vmem[0]]
tuc3_vcoly=[tuc3_colmem[0],tuc3_colmem[0],tuc3_colmem[1],tuc3_colmem[1],tuc3_colmem[0]]
tuc3_vmagx=[tuc3_vmem[0],tuc3_vmem[1],tuc3_vmem[1],tuc3_vmem[0],tuc3_vmem[0]]
tuc3_vmagy=[tuc3_magmem[0],tuc3_magmem[0],tuc3_magmem[1],tuc3_magmem[1],tuc3_magmem[0]]

tuc3_teffloggx=[tuc3_teffmem[0],tuc3_teffmem[1],tuc3_teffmem[1],tuc3_teffmem[0],tuc3_teffmem[0]]
tuc3_teffloggy=[tuc3_loggmem[0],tuc3_loggmem[0],tuc3_loggmem[1],tuc3_loggmem[1],tuc3_loggmem[0]]
tuc3_tefffehx=[tuc3_teffmem[0],tuc3_teffmem[1],tuc3_teffmem[1],tuc3_teffmem[0],tuc3_teffmem[0]]
tuc3_tefffehy=[tuc3_fehmem[0],tuc3_fehmem[0],tuc3_fehmem[1],tuc3_fehmem[1],tuc3_fehmem[0]]
tuc3_teffrx=[tuc3_teffmem[0],tuc3_teffmem[1],tuc3_teffmem[1],tuc3_teffmem[0],tuc3_teffmem[0]]
tuc3_teffry=[tuc3_rmem[0],tuc3_rmem[0],tuc3_rmem[1],tuc3_rmem[1],tuc3_rmem[0]]
tuc3_teffcolx=[tuc3_teffmem[0],tuc3_teffmem[1],tuc3_teffmem[1],tuc3_teffmem[0],tuc3_teffmem[0]]
tuc3_teffcoly=[tuc3_colmem[0],tuc3_colmem[0],tuc3_colmem[1],tuc3_colmem[1],tuc3_colmem[0]]
tuc3_teffmagx=[tuc3_teffmem[0],tuc3_teffmem[1],tuc3_teffmem[1],tuc3_teffmem[0],tuc3_teffmem[0]]
tuc3_teffmagy=[tuc3_magmem[0],tuc3_magmem[0],tuc3_magmem[1],tuc3_magmem[1],tuc3_magmem[0]]

tuc3_loggfehx=[tuc3_loggmem[0],tuc3_loggmem[1],tuc3_loggmem[1],tuc3_loggmem[0],tuc3_loggmem[0]]
tuc3_loggfehy=[tuc3_fehmem[0],tuc3_fehmem[0],tuc3_fehmem[1],tuc3_fehmem[1],tuc3_fehmem[0]]
tuc3_loggrx=[tuc3_loggmem[0],tuc3_loggmem[1],tuc3_loggmem[1],tuc3_loggmem[0],tuc3_loggmem[0]]
tuc3_loggry=[tuc3_rmem[0],tuc3_rmem[0],tuc3_rmem[1],tuc3_rmem[1],tuc3_rmem[0]]
tuc3_loggcolx=[tuc3_loggmem[0],tuc3_loggmem[1],tuc3_loggmem[1],tuc3_loggmem[0],tuc3_loggmem[0]]
tuc3_loggcoly=[tuc3_colmem[0],tuc3_colmem[0],tuc3_colmem[1],tuc3_colmem[1],tuc3_colmem[0]]
tuc3_loggmagx=[tuc3_loggmem[0],tuc3_loggmem[1],tuc3_loggmem[1],tuc3_loggmem[0],tuc3_loggmem[0]]
tuc3_loggmagy=[tuc3_magmem[0],tuc3_magmem[0],tuc3_magmem[1],tuc3_magmem[1],tuc3_magmem[0]]

tuc3_fehrx=[tuc3_fehmem[0],tuc3_fehmem[1],tuc3_fehmem[1],tuc3_fehmem[0],tuc3_fehmem[0]]
tuc3_fehry=[tuc3_rmem[0],tuc3_rmem[0],tuc3_rmem[1],tuc3_rmem[1],tuc3_rmem[0]]
tuc3_fehcolx=[tuc3_fehmem[0],tuc3_fehmem[1],tuc3_fehmem[1],tuc3_fehmem[0],tuc3_fehmem[0]]
tuc3_fehcoly=[tuc3_colmem[0],tuc3_colmem[0],tuc3_colmem[1],tuc3_colmem[1],tuc3_colmem[0]]
tuc3_fehmagx=[tuc3_fehmem[0],tuc3_fehmem[1],tuc3_fehmem[1],tuc3_fehmem[0],tuc3_fehmem[0]]
tuc3_fehmagy=[tuc3_magmem[0],tuc3_magmem[0],tuc3_magmem[1],tuc3_magmem[1],tuc3_magmem[0]]

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
#ax6_0.add_patch(Rectangle((tuc3_vmem[0],tuc3_teffmem[0]),tuc3_vmem[1]-tuc3_vmem[0],tuc3_teffmem[1]-tuc3_teffmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax6_0.errorbar(tuc3_v[tuc3_keep],tuc3_teff[tuc3_keep],xerr=tuc3_sigv[tuc3_keep],yerr=tuc3_sigteff[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)

ax5_0.set_ylabel(logglabel[0],fontsize=8,rotation=90,labelpad=5)
ax5_0.set_xlim(vlim)
ax5_0.set_ylim(revlogglim)
ax5_0.set_xscale(u'linear')
ax5_0.set_yscale(u'linear')
ax5_0.set_xticks(vticks)
ax5_0.set_yticks(revloggticks)
ax5_0.set_yticklabels(revloggticks,rotation=0)
ax5_0.xaxis.set_major_formatter(plt.NullFormatter())
#ax5_0.add_patch(Rectangle((tuc3_vmem[0],tuc3_loggmem[0]),tuc3_vmem[1]-tuc3_vmem[0],tuc3_loggmem[1]-tuc3_loggmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax5_0.errorbar(tuc3_v[tuc3_keep],tuc3_logg[tuc3_keep],xerr=tuc3_sigv[tuc3_keep],yerr=tuc3_siglogg[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax5_0.plot(tuc3_vloggx,tuc3_vloggy,color='r',linewidth=0.25)

ax4_0.set_ylabel(fehlabel[0],fontsize=8,rotation=90,labelpad=5)
ax4_0.set_xlim(vlim)
ax4_0.set_ylim(fehlim)
ax4_0.set_xscale(u'linear')
ax4_0.set_yscale(u'linear')
ax4_0.set_xticks(vticks)
ax4_0.set_yticks(fehticks)
ax4_0.set_yticklabels(fehticks,rotation=0)
ax4_0.xaxis.set_major_formatter(plt.NullFormatter())
#ax4_0.add_patch(Rectangle((tuc3_vmem[0],tuc3_fehmem[0]),tuc3_vmem[1]-tuc3_vmem[0],tuc3_fehmem[1]-tuc3_fehmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax4_0.errorbar(tuc3_v[tuc3_keep],tuc3_feh[tuc3_keep],xerr=tuc3_sigv[tuc3_keep],yerr=tuc3_sigfeh[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax4_0.plot(tuc3_vfehx,tuc3_vfehy,color='r',linewidth=0.25)

ax3_0.set_ylabel(rlabel[0],fontsize=8,rotation=90,labelpad=5)
ax3_0.set_xlim(vlim)
ax3_0.set_ylim(rlim)
ax3_0.set_xscale(u'linear')
ax3_0.set_yscale(u'linear')
ax3_0.set_xticks(vticks)
ax3_0.set_yticks(rticks)
ax3_0.set_yticklabels(rticks,rotation=0)
ax3_0.xaxis.set_major_formatter(plt.NullFormatter())
#ax3_0.add_patch(Rectangle((tuc3_vmem[0],tuc3_rmem[0]),tuc3_vmem[1]-tuc3_vmem[0],tuc3_rmem[1]-tuc3_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_0.errorbar(tuc3_v[tuc3_keep],tuc3_r[tuc3_keep],xerr=tuc3_sigv[tuc3_keep],yerr=tuc3_sigr[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax3_0.plot(tuc3_vrx,tuc3_vry,color='r',linewidth=0.25)

ax2_0.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_0.xaxis.set_major_formatter(plt.NullFormatter())
ax2_0.set_ylim(histlim)
ax2_0.set_xlim(vlim)
ax2_0.set_xticks(vticks)
ax2_0.set_yticks(histticks)
ax2_0.hist(tuc3_v[tuc3_keep],bins=30,range=vlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,label='Tuc 3')
#ax2_0.hist(tuc3_v[tuc3_mem],bins=30,range=vlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)
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
#ax5_1.add_patch(Rectangle((tuc3_teffmem[0],tuc3_loggmem[0]),tuc3_teffmem[1]-tuc3_teffmem[0],tuc3_loggmem[1]-tuc3_loggmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax5_1.errorbar(tuc3_teff[tuc3_keep],tuc3_logg[tuc3_keep],xerr=tuc3_sigteff[tuc3_keep],yerr=tuc3_siglogg[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)

ax4_1.set_xlim(revtefflim)
ax4_1.set_ylim(fehlim)
ax4_1.set_xscale(u'linear')
ax4_1.set_yscale(u'linear')
ax4_1.set_xticks(revteffticks)
ax4_1.set_yticks(fehticks)
ax4_1.xaxis.set_major_formatter(plt.NullFormatter())
ax4_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax4_1.add_patch(Rectangle((tuc3_teffmem[0],tuc3_fehmem[0]),tuc3_teffmem[1]-tuc3_teffmem[0],tuc3_fehmem[1]-tuc3_fehmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax4_1.errorbar(tuc3_teff[tuc3_keep],tuc3_feh[tuc3_keep],xerr=tuc3_sigteff[tuc3_keep],yerr=tuc3_sigfeh[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)

ax3_1.set_xlim(revtefflim)
ax3_1.set_ylim(rlim)
ax3_1.set_xscale(u'linear')
ax3_1.set_yscale(u'linear')
ax3_1.set_xticks(revteffticks)
ax3_1.set_yticks(rticks)
ax3_1.xaxis.set_major_formatter(plt.NullFormatter())
ax3_1.yaxis.set_major_formatter(plt.NullFormatter())
#ax3_1.add_patch(Rectangle((tuc3_teffmem[0],tuc3_rmem[0]),tuc3_teffmem[1]-tuc3_teffmem[0],tuc3_rmem[1]-tuc3_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_1.errorbar(tuc3_teff[tuc3_keep],tuc3_r[tuc3_keep],xerr=tuc3_sigteff[tuc3_keep],yerr=tuc3_sigr[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)

ax2_1.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_1.xaxis.set_major_formatter(plt.NullFormatter())
ax2_1.yaxis.set_major_formatter(plt.NullFormatter())
ax2_1.set_xlim(revtefflim)
ax2_1.set_ylim(histlim)
ax2_1.set_xticks(revteffticks)
ax2_1.set_yticks(histticks)
ax2_1.hist(tuc3_teff[tuc3_keep],bins=30,range=tefflim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25)
#ax2_1.hist(tuc3_teff[tuc3_mem],bins=30,range=tefflim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)


ax4_2.set_xlabel(logglabel[0],fontsize=8,rotation=0)
ax4_2.set_xlim(revlogglim)
ax4_2.set_ylim(fehlim)
ax4_2.set_xscale(u'linear')
ax4_2.set_yscale(u'linear')
ax4_2.set_xticks([0]+revloggticks)
ax4_2.set_xticklabels([0]+revloggticks,rotation=0)
ax4_2.set_yticks(fehticks)
ax4_2.yaxis.set_major_formatter(plt.NullFormatter())
#ax4_2.add_patch(Rectangle((tuc3_loggmem[0],tuc3_fehmem[0]),tuc3_loggmem[1]-tuc3_loggmem[0],tuc3_fehmem[1]-tuc3_fehmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax4_2.errorbar(tuc3_logg[tuc3_keep],tuc3_r[tuc3_keep],xerr=tuc3_siglogg[tuc3_keep],yerr=tuc3_sigr[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax4_2.errorbar(tuc3_logg[tuc3_keep],tuc3_feh[tuc3_keep],xerr=tuc3_siglogg[tuc3_keep],yerr=tuc3_sigfeh[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax4_2.plot(tuc3_loggfehx,tuc3_loggfehy,color='r',linewidth=0.25)


ax3_2.set_xlim(revlogglim)
ax3_2.set_ylim(rlim)
ax3_2.set_xscale(u'linear')
ax3_2.set_yscale(u'linear')
ax3_2.set_xticks(revloggticks)
ax3_2.set_yticks(rticks)
ax3_2.xaxis.set_major_formatter(plt.NullFormatter())
ax3_2.yaxis.set_major_formatter(plt.NullFormatter())
#ax3_2.add_patch(Rectangle((tuc3_loggmem[0],tuc3_rmem[0]),tuc3_loggmem[1]-tuc3_loggmem[0],tuc3_rmem[1]-tuc3_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_2.errorbar(tuc3_logg[tuc3_keep],tuc3_r[tuc3_keep],xerr=tuc3_siglogg[tuc3_keep],yerr=tuc3_sigr[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax3_2.errorbar(tuc3_logg[tuc3_keep],tuc3_r[tuc3_keep],xerr=tuc3_siglogg[tuc3_keep],yerr=tuc3_sigr[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax3_2.plot(tuc3_loggrx,tuc3_loggry,color='r',linewidth=0.25)

ax2_2.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_2.xaxis.set_major_formatter(plt.NullFormatter())
ax2_2.yaxis.set_major_formatter(plt.NullFormatter())
ax2_2.set_xlim(revlogglim)
ax2_2.set_ylim(histlim)
ax2_2.set_xticks(revloggticks)
ax2_2.set_yticks(histticks)
ax2_2.hist(tuc3_logg[tuc3_keep],bins=30,range=logglim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25)
#ax2_2.hist(tuc3_logg[tuc3_mem],bins=30,range=logglim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)



ax3_3.set_xlabel(fehlabel[0],fontsize=8,rotation=0)
ax3_3.set_xlim(fehlim)
ax3_3.set_ylim(rlim)
ax3_3.set_xscale(u'linear')
ax3_3.set_yscale(u'linear')
ax3_3.set_xticks(fehticks)
ax3_3.set_xticklabels(fehticks,rotation=0)
ax3_3.set_yticks(rticks)
ax3_3.yaxis.set_major_formatter(plt.NullFormatter())
#ax3_3.add_patch(Rectangle((tuc3_fehmem[0],tuc3_rmem[0]),tuc3_fehmem[1]-tuc3_fehmem[0],tuc3_rmem[1]-tuc3_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_3.errorbar(tuc3_feh[tuc3_keep],tuc3_r[tuc3_keep],xerr=tuc3_sigfeh[tuc3_keep],yerr=tuc3_sigr[tuc3_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax3_3.plot(tuc3_fehrx,tuc3_fehry,color='r',linewidth=0.25)

ax2_3.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_3.xaxis.set_major_formatter(plt.NullFormatter())
ax2_3.yaxis.set_major_formatter(plt.NullFormatter())
ax2_3.set_ylim(histlim)
ax2_3.set_xticks(fehticks)
ax2_3.set_yticks(histticks)
ax2_3.hist(tuc3_feh[tuc3_keep],bins=30,range=fehlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25)
#ax2_3.hist(tuc3_feh[tuc3_mem],bins=30,range=fehlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)



#ax3_4.set_xlabel('N',fontsize=8,rotation=0,labelpad=5)
ax3_4.xaxis.set_major_formatter(plt.NullFormatter())
ax3_4.yaxis.set_major_formatter(plt.NullFormatter())
ax3_4.set_xlim(histlim)
ax3_4.set_yticks(rticks)
ax3_4.set_xticks(histticks)
ax3_4.hist(tuc3_r[tuc3_keep],bins=30,range=rlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.25)
#ax3_4.hist(tuc3_r[tuc3_mem],bins=30,range=rlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.25,alpha=0.5)


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
ax10_10.scatter(tuc3_teff[tuc3_keep],tuc3_logg[tuc3_keep],marker='.',s=1,alpha=0.65,color='k',rasterized=True,label='foreground')
#ax10_10.scatter(tuc3_teff[tuc3_mem],tuc3_logg[tuc3_mem],marker='.',s=1,alpha=0.65,color='r',rasterized=True)
#ax10_10.errorbar(tuc3_teff[tuc3_mem],tuc3_logg[tuc3_mem],xerr=tuc3_sigteff[tuc3_mem],yerr=tuc3_siglogg[tuc3_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True,label='tuc 3 member')
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
#ax9_10.errorbar(tuc3_teff[tuc3_mem],tuc3_mag[tuc3_mem]-tuc3_dmodulus,xerr=tuc3_sigteff[tuc3_mem],yerr=tuc3_siglogg[tuc3_mem]-tuc3_siglogg[tuc3_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax9_10.legend(loc=2,fontsize=4,handlelength=4,shadow=False)
#ax9_10.text(-0.4,16.5,'Crater',fontsize=10)
#ax9_10.text(7500,16.5,'tuc 3',fontsize=12)

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
#ax10_9.errorbar(tuc3_mag[tuc3_mem]-tuc3_dmodulus,tuc3_logg[tuc3_mem],xerr=tuc3_sigteff[tuc3_mem]-tuc3_sigteff[tuc3_mem],yerr=tuc3_siglogg[tuc3_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax10_9.legend(loc=2,fontsize=12,handlelength=0,shadow=False)
#ax2_0.text(-0.4,16.5,'Crater',fontsize=10)

g1.write(r'\newcommand{\tucnobs}{$'+str(len(tuc3_v))+r'$}'+'\n')
g1.write(r'\newcommand{\tucgoodnobs}{$'+str(len(tuc3_v[tuc3_keep]))+r'$}'+'\n')
g1.write(r'\newcommand{\tucqc}{$'+str(len(tuc3_v[tuc3_qc]))+r'$}'+'\n')
#g1.write(r'\newcommand{\nred}{$'+str(len(tuc3_v[mem]))+r'$}'+'\n')
#g1.write(r'\newcommand{\nblue}{$'+str(len(v[mem3]))+r'$}'+'\n')
g1.write(r'\newcommand{\tucmembers}{$'+str(np.size(tuc3_mem))+r'$}'+'\n')
g1.write(r'\newcommand{\tucgoodmembers}{$'+str(np.size(tuc3_goodmem))+r'$}'+'\n')
g1.close()



plotfilename='tuc3_params.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
