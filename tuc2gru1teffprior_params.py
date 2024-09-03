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
            
data1='tuc2teffprior.dat'
data2='gru1teffprior.dat'
out1='tuc2gru1teffprior_newcommands.tex'
g1=open(out1,'w')

tuc2_dmodulus=18.8
gru1_dmodulus=20.4            
maxsigv=20.

with open('tuc2_besancon.dat') as f: # read besancon model
    data=f.readlines()[90:]

tuc2_besancon_logteff=[]
tuc2_besancon_logg=[]
tuc2_besancon_gminusr=[]
tuc2_besancon_rminusi=[]
tuc2_besancon_r=[]
tuc2_besancon_v=[]
tuc2_besancon_feh=[]

for line in data: # fill arrays
    p=line.split()
    tuc2_besancon_logteff.append(float(p[4]))
    tuc2_besancon_logg.append(float(p[5]))
    tuc2_besancon_gminusr.append(float(p[9]))
    tuc2_besancon_rminusi.append(float(p[10]))
    tuc2_besancon_r.append(float(p[12]))
    tuc2_besancon_v.append(float(p[15]))
    tuc2_besancon_feh.append(float(p[19]))

tuc2_besancon_logteff=np.array(tuc2_besancon_logteff)
tuc2_besancon_logg=np.array(tuc2_besancon_logg)
tuc2_besancon_gminusr=np.array(tuc2_besancon_gminusr)
tuc2_besancon_rminusi=np.array(tuc2_besancon_rminusi)
tuc2_besancon_r=np.array(tuc2_besancon_r)
tuc2_besancon_v=np.array(tuc2_besancon_v)
tuc2_besancon_feh=np.array(tuc2_besancon_feh)
tuc2_besancon_g=tuc2_besancon_gminusr+tuc2_besancon_r
tuc2_besancon_teff=10.**tuc2_besancon_logteff

with open(data1) as f: # read data file
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
tuc2_filename=[]
tuc2_res=[]
tuc2_id=[]
tuc2_goodnobs=[]

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
    tuc2_id.append(long(p[20]))
    tuc2_goodnobs.append(long(p[21]))

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
tuc2_sigr=tuc2_r-tuc2_r
tuc2_mag=tuc2_rmag-tuc2_extr
tuc2_col=tuc2_gmag-tuc2_extg-(tuc2_rmag-tuc2_extr)
tuc2_sigmag=tuc2_mag-tuc2_mag
tuc2_sigcol=tuc2_col-tuc2_col
tuc2_snratio=np.array(tuc2_snratio)
tuc2_aperture=np.array(tuc2_aperture)
tuc2_identifier=np.array(tuc2_identifier)
tuc2_filename=np.array(tuc2_filename)
tuc2_res=np.array(tuc2_res)
tuc2_goodnobs=np.array(tuc2_goodnobs)

#tuc2_keep=np.where((tuc2_sigv < maxsigv) & (tuc2_skewv >= -1.) & (tuc2_skewv <= 1.) & (tuc2_kurtv >= -1) & (tuc2_kurtv <= 1))
tuc2_keep=np.where((tuc2_sigv < maxsigv))#) & (tuc2_skewv >= -1.) & (tuc2_skewv <= 1.) & (tuc2_kurtv >= -1) & (tuc2_kurtv <= 1))
#tuc2_mem=np.where((tuc2_sigv < maxsigv) & (tuc2_skewv > -1.) & (tuc2_skewv < 1.) & (tuc2_kurtv > -1.) & (tuc2_kurtv < 1.) & (tuc2_pmem > 0.9))
tuc2_mem=np.where((tuc2_sigv < maxsigv) & (tuc2_pmem > 0.5))
tuc2_goodmem=np.where((tuc2_sigv < maxsigv) & (tuc2_pmem > 0.9))
tuc2_nonmem=np.where((tuc2_sigv < maxsigv) & (tuc2_pmem < 0.5))
tuc2_goodnonmem=np.where((tuc2_sigv < maxsigv) & (tuc2_pmem < 0.5))
tuc2_qc=np.where(tuc2_goodnobs >= 1)
tuc2_bad=np.where(tuc2_sigv >= maxsigv)

tuc2_vmem=[-150.,-115.]
tuc2_loggmem=[0.5,4.0]
tuc2_fehmem=[-3.,-1.]
tuc2_teffmem=[4000,6000]
tuc2_rmem=[0.,14.]
tuc2_magmem=[21.5,18]
tuc2_colmem=[0.4,0.75]

with open('gru1_besancon.dat') as f: # read besancon model
    data=f.readlines()[91:]

gru1_besancon_logteff=[]
gru1_besancon_logg=[]
gru1_besancon_gminusr=[]
gru1_besancon_rminusi=[]
gru1_besancon_r=[]
gru1_besancon_v=[]
gru1_besancon_feh=[]

for line in data: # fill arrays
    p=line.split()
    gru1_besancon_logteff.append(float(p[4]))
    gru1_besancon_logg.append(float(p[5]))
    gru1_besancon_gminusr.append(float(p[9]))
    gru1_besancon_rminusi.append(float(p[10]))
    gru1_besancon_r.append(float(p[12]))
    gru1_besancon_v.append(float(p[15]))
    gru1_besancon_feh.append(float(p[19]))

gru1_besancon_logteff=np.array(gru1_besancon_logteff)
gru1_besancon_logg=np.array(gru1_besancon_logg)
gru1_besancon_gminusr=np.array(gru1_besancon_gminusr)
gru1_besancon_rminusi=np.array(gru1_besancon_rminusi)
gru1_besancon_r=np.array(gru1_besancon_r)
gru1_besancon_v=np.array(gru1_besancon_v)
gru1_besancon_feh=np.array(gru1_besancon_feh)
gru1_besancon_g=gru1_besancon_gminusr+gru1_besancon_r
gru1_besancon_teff=10.**gru1_besancon_logteff

with open(data2) as f: # read data file
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
gru1_filename=[]
gru1_res=[]
gru1_id=[]
gru1_goodnobs=[]

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
    gru1_id.append(long(p[20]))
    gru1_goodnobs.append(long(p[21]))

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
gru1_sigr=gru1_r-gru1_r
gru1_mag=gru1_rmag-gru1_extr
gru1_col=gru1_gmag-gru1_extg-(gru1_rmag-gru1_extr)
gru1_sigmag=gru1_mag-gru1_mag
gru1_sigcol=gru1_col-gru1_col
gru1_snratio=np.array(gru1_snratio)
gru1_aperture=np.array(gru1_aperture)
gru1_identifier=np.array(gru1_identifier)
gru1_filename=np.array(gru1_filename)
gru1_res=np.array(gru1_res)
gru1_id=np.array(gru1_id)
gru1_goodnobs=np.array(gru1_goodnobs)

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




#gru1_keep=np.where((gru1_sigv < maxsigv) & (gru1_skewv >= -1.) & (gru1_skewv <= 1.) & (gru1_kurtv >= -1) & (gru1_kurtv <= 1))
gru1_keep=np.where((gru1_sigv < maxsigv))# & (gru1_skewv >= -1.) & (gru1_skewv <= 1.) & (gru1_kurtv >= -1) & (gru1_kurtv <= 1))
#gru1_mem=np.where((gru1_sigv < maxsigv) & (gru1_skewv > -1.) & (gru1_skewv < 1.) & (gru1_kurtv > -1.) & (gru1_kurtv < 1.) & (gru1_pmem > 0.9))
gru1_mem=np.where((gru1_sigv < maxsigv) & (gru1_pmem > 0.5))
gru1_goodmem=np.where((gru1_sigv < maxsigv) & (gru1_pmem > 0.9))
gru1_nonmem=np.where((gru1_sigv < maxsigv) & (gru1_pmem < 0.5))
gru1_goodnonmem=np.where((gru1_sigv < maxsigv) & (gru1_pmem < 0.5))
gru1_qc=np.where(gru1_goodnobs >= 1)
gru1_bad=np.where(gru1_sigv >= maxsigv)

gru1_vmem=[-155.,-125.]
gru1_loggmem=[0.5,4.0]
gru1_fehmem=[-3.,-1.]
gru1_teffmem=[4000,6000]
gru1_rmem=[0.,6.]
gru1_magmem=[21.7,18.4]
gru1_colmem=[0.4,1.1]

tuc2_besancon_cand=np.zeros(len(tuc2_besancon_v),dtype='int')
for i in range(0,len(tuc2_besancon_cand)):
    tuc2_besancon_dist=np.sqrt((tuc2_besancon_r[i]-(iso250_r+tuc2_dmodulus))**2+((tuc2_besancon_g[i]-tuc2_besancon_r[i])-(iso250_g-iso250_r))**2)
    if (min(tuc2_besancon_dist) < 0.2) & (tuc2_besancon_r[i] < 21.):
        tuc2_besancon_cand[i]=1
tuc2_besancon_cands=np.where(tuc2_besancon_cand == 1)

gru1_besancon_cand=np.zeros(len(gru1_besancon_v),dtype='int')
for i in range(0,len(gru1_besancon_cand)):
    gru1_besancon_dist=np.sqrt((gru1_besancon_r[i]-(iso250_r+gru1_dmodulus))**2+((gru1_besancon_g[i]-gru1_besancon_r[i])-(iso250_g-iso250_r))**2)
    if (min(gru1_besancon_dist) < 0.2) & (gru1_besancon_r[i] < 21.):
        gru1_besancon_cand[i]=1
gru1_besancon_cands=np.where(gru1_besancon_cand == 1)

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

vlim=[-200,300]
tefflim=[4001.,7999.] 
revtefflim=[7999.,4001.] 
logglim=[0.001,4.99]
revlogglim=[4.99,0.001]
fehlim=[-4.99,0.99]
rlim=[0.,15.]
collim=[0.25,1]
maglim=[16.5,20.5]
revmaglim=[20.5,16.5]

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
revloggticks=[4,3,2,1]
logglabels=['1','2','3','4']
revlogglabels=['4','3','2','1']
fehticks=[-4,-3,-2,-1,0]
fehlabels=['-4','-3','-2','-1','0']
teffticks=[5000,6000,7000]
revteffticks=[7000,6000,5000]
teffticklabels=['5000','6000','7000']
revteffticklabels=['7000','6000','5000']
colmap=plt.get_cmap("Greys")
rticks=[5,10]
rticklabels=['5','10']
colticks=[0.5,0.75]
colticklabels=colticks
magticks=[17,18,19,20]
revmagticks=[20,19,18,17]
magticklabels=magticks
revmagticklabels=revmagticks

tuc2_vteffx=[tuc2_vmem[0],tuc2_vmem[1],tuc2_vmem[1],tuc2_vmem[0],tuc2_vmem[0]]
tuc2_vteffy=[tuc2_teffmem[0],tuc2_teffmem[0],tuc2_teffmem[1],tuc2_teffmem[1],tuc2_teffmem[0]]
tuc2_vloggx=[tuc2_vmem[0],tuc2_vmem[1],tuc2_vmem[1],tuc2_vmem[0],tuc2_vmem[0]]
tuc2_vloggy=[tuc2_loggmem[0],tuc2_loggmem[0],tuc2_loggmem[1],tuc2_loggmem[1],tuc2_loggmem[0]]
tuc2_vfehx=[tuc2_vmem[0],tuc2_vmem[1],tuc2_vmem[1],tuc2_vmem[0],tuc2_vmem[0]]
tuc2_vfehy=[tuc2_fehmem[0],tuc2_fehmem[0],tuc2_fehmem[1],tuc2_fehmem[1],tuc2_fehmem[0]]
tuc2_vrx=[tuc2_vmem[0],tuc2_vmem[1],tuc2_vmem[1],tuc2_vmem[0],tuc2_vmem[0]]
tuc2_vry=[tuc2_rmem[0],tuc2_rmem[0],tuc2_rmem[1],tuc2_rmem[1],tuc2_rmem[0]]
tuc2_vcolx=[tuc2_vmem[0],tuc2_vmem[1],tuc2_vmem[1],tuc2_vmem[0],tuc2_vmem[0]]
tuc2_vcoly=[tuc2_colmem[0],tuc2_colmem[0],tuc2_colmem[1],tuc2_colmem[1],tuc2_colmem[0]]
tuc2_vmagx=[tuc2_vmem[0],tuc2_vmem[1],tuc2_vmem[1],tuc2_vmem[0],tuc2_vmem[0]]
tuc2_vmagy=[tuc2_magmem[0],tuc2_magmem[0],tuc2_magmem[1],tuc2_magmem[1],tuc2_magmem[0]]

tuc2_teffloggx=[tuc2_teffmem[0],tuc2_teffmem[1],tuc2_teffmem[1],tuc2_teffmem[0],tuc2_teffmem[0]]
tuc2_teffloggy=[tuc2_loggmem[0],tuc2_loggmem[0],tuc2_loggmem[1],tuc2_loggmem[1],tuc2_loggmem[0]]
tuc2_tefffehx=[tuc2_teffmem[0],tuc2_teffmem[1],tuc2_teffmem[1],tuc2_teffmem[0],tuc2_teffmem[0]]
tuc2_tefffehy=[tuc2_fehmem[0],tuc2_fehmem[0],tuc2_fehmem[1],tuc2_fehmem[1],tuc2_fehmem[0]]
tuc2_teffrx=[tuc2_teffmem[0],tuc2_teffmem[1],tuc2_teffmem[1],tuc2_teffmem[0],tuc2_teffmem[0]]
tuc2_teffry=[tuc2_rmem[0],tuc2_rmem[0],tuc2_rmem[1],tuc2_rmem[1],tuc2_rmem[0]]
tuc2_teffcolx=[tuc2_teffmem[0],tuc2_teffmem[1],tuc2_teffmem[1],tuc2_teffmem[0],tuc2_teffmem[0]]
tuc2_teffcoly=[tuc2_colmem[0],tuc2_colmem[0],tuc2_colmem[1],tuc2_colmem[1],tuc2_colmem[0]]
tuc2_teffmagx=[tuc2_teffmem[0],tuc2_teffmem[1],tuc2_teffmem[1],tuc2_teffmem[0],tuc2_teffmem[0]]
tuc2_teffmagy=[tuc2_magmem[0],tuc2_magmem[0],tuc2_magmem[1],tuc2_magmem[1],tuc2_magmem[0]]

tuc2_loggfehx=[tuc2_loggmem[0],tuc2_loggmem[1],tuc2_loggmem[1],tuc2_loggmem[0],tuc2_loggmem[0]]
tuc2_loggfehy=[tuc2_fehmem[0],tuc2_fehmem[0],tuc2_fehmem[1],tuc2_fehmem[1],tuc2_fehmem[0]]
tuc2_loggrx=[tuc2_loggmem[0],tuc2_loggmem[1],tuc2_loggmem[1],tuc2_loggmem[0],tuc2_loggmem[0]]
tuc2_loggry=[tuc2_rmem[0],tuc2_rmem[0],tuc2_rmem[1],tuc2_rmem[1],tuc2_rmem[0]]
tuc2_loggcolx=[tuc2_loggmem[0],tuc2_loggmem[1],tuc2_loggmem[1],tuc2_loggmem[0],tuc2_loggmem[0]]
tuc2_loggcoly=[tuc2_colmem[0],tuc2_colmem[0],tuc2_colmem[1],tuc2_colmem[1],tuc2_colmem[0]]
tuc2_loggmagx=[tuc2_loggmem[0],tuc2_loggmem[1],tuc2_loggmem[1],tuc2_loggmem[0],tuc2_loggmem[0]]
tuc2_loggmagy=[tuc2_magmem[0],tuc2_magmem[0],tuc2_magmem[1],tuc2_magmem[1],tuc2_magmem[0]]

tuc2_fehrx=[tuc2_fehmem[0],tuc2_fehmem[1],tuc2_fehmem[1],tuc2_fehmem[0],tuc2_fehmem[0]]
tuc2_fehry=[tuc2_rmem[0],tuc2_rmem[0],tuc2_rmem[1],tuc2_rmem[1],tuc2_rmem[0]]
tuc2_fehcolx=[tuc2_fehmem[0],tuc2_fehmem[1],tuc2_fehmem[1],tuc2_fehmem[0],tuc2_fehmem[0]]
tuc2_fehcoly=[tuc2_colmem[0],tuc2_colmem[0],tuc2_colmem[1],tuc2_colmem[1],tuc2_colmem[0]]
tuc2_fehmagx=[tuc2_fehmem[0],tuc2_fehmem[1],tuc2_fehmem[1],tuc2_fehmem[0],tuc2_fehmem[0]]
tuc2_fehmagy=[tuc2_magmem[0],tuc2_magmem[0],tuc2_magmem[1],tuc2_magmem[1],tuc2_magmem[0]]

gru1_vteffx=[gru1_vmem[0],gru1_vmem[1],gru1_vmem[1],gru1_vmem[0],gru1_vmem[0]]
gru1_vteffy=[gru1_teffmem[0],gru1_teffmem[0],gru1_teffmem[1],gru1_teffmem[1],gru1_teffmem[0]]
gru1_vloggx=[gru1_vmem[0],gru1_vmem[1],gru1_vmem[1],gru1_vmem[0],gru1_vmem[0]]
gru1_vloggy=[gru1_loggmem[0],gru1_loggmem[0],gru1_loggmem[1],gru1_loggmem[1],gru1_loggmem[0]]
gru1_vfehx=[gru1_vmem[0],gru1_vmem[1],gru1_vmem[1],gru1_vmem[0],gru1_vmem[0]]
gru1_vfehy=[gru1_fehmem[0],gru1_fehmem[0],gru1_fehmem[1],gru1_fehmem[1],gru1_fehmem[0]]
gru1_vrx=[gru1_vmem[0],gru1_vmem[1],gru1_vmem[1],gru1_vmem[0],gru1_vmem[0]]
gru1_vry=[gru1_rmem[0],gru1_rmem[0],gru1_rmem[1],gru1_rmem[1],gru1_rmem[0]]
gru1_vcolx=[gru1_vmem[0],gru1_vmem[1],gru1_vmem[1],gru1_vmem[0],gru1_vmem[0]]
gru1_vcoly=[gru1_colmem[0],gru1_colmem[0],gru1_colmem[1],gru1_colmem[1],gru1_colmem[0]]
gru1_vmagx=[gru1_vmem[0],gru1_vmem[1],gru1_vmem[1],gru1_vmem[0],gru1_vmem[0]]
gru1_vmagy=[gru1_magmem[0],gru1_magmem[0],gru1_magmem[1],gru1_magmem[1],gru1_magmem[0]]

gru1_teffloggx=[gru1_teffmem[0],gru1_teffmem[1],gru1_teffmem[1],gru1_teffmem[0],gru1_teffmem[0]]
gru1_teffloggy=[gru1_loggmem[0],gru1_loggmem[0],gru1_loggmem[1],gru1_loggmem[1],gru1_loggmem[0]]
gru1_tefffehx=[gru1_teffmem[0],gru1_teffmem[1],gru1_teffmem[1],gru1_teffmem[0],gru1_teffmem[0]]
gru1_tefffehy=[gru1_fehmem[0],gru1_fehmem[0],gru1_fehmem[1],gru1_fehmem[1],gru1_fehmem[0]]
gru1_teffrx=[gru1_teffmem[0],gru1_teffmem[1],gru1_teffmem[1],gru1_teffmem[0],gru1_teffmem[0]]
gru1_teffry=[gru1_rmem[0],gru1_rmem[0],gru1_rmem[1],gru1_rmem[1],gru1_rmem[0]]
gru1_teffcolx=[gru1_teffmem[0],gru1_teffmem[1],gru1_teffmem[1],gru1_teffmem[0],gru1_teffmem[0]]
gru1_teffcoly=[gru1_colmem[0],gru1_colmem[0],gru1_colmem[1],gru1_colmem[1],gru1_colmem[0]]
gru1_teffmagx=[gru1_teffmem[0],gru1_teffmem[1],gru1_teffmem[1],gru1_teffmem[0],gru1_teffmem[0]]
gru1_teffmagy=[gru1_magmem[0],gru1_magmem[0],gru1_magmem[1],gru1_magmem[1],gru1_magmem[0]]

gru1_loggfehx=[gru1_loggmem[0],gru1_loggmem[1],gru1_loggmem[1],gru1_loggmem[0],gru1_loggmem[0]]
gru1_loggfehy=[gru1_fehmem[0],gru1_fehmem[0],gru1_fehmem[1],gru1_fehmem[1],gru1_fehmem[0]]
gru1_loggrx=[gru1_loggmem[0],gru1_loggmem[1],gru1_loggmem[1],gru1_loggmem[0],gru1_loggmem[0]]
gru1_loggry=[gru1_rmem[0],gru1_rmem[0],gru1_rmem[1],gru1_rmem[1],gru1_rmem[0]]
gru1_loggcolx=[gru1_loggmem[0],gru1_loggmem[1],gru1_loggmem[1],gru1_loggmem[0],gru1_loggmem[0]]
gru1_loggcoly=[gru1_colmem[0],gru1_colmem[0],gru1_colmem[1],gru1_colmem[1],gru1_colmem[0]]
gru1_loggmagx=[gru1_loggmem[0],gru1_loggmem[1],gru1_loggmem[1],gru1_loggmem[0],gru1_loggmem[0]]
gru1_loggmagy=[gru1_magmem[0],gru1_magmem[0],gru1_magmem[1],gru1_magmem[1],gru1_magmem[0]]

gru1_fehrx=[gru1_fehmem[0],gru1_fehmem[1],gru1_fehmem[1],gru1_fehmem[0],gru1_fehmem[0]]
gru1_fehry=[gru1_rmem[0],gru1_rmem[0],gru1_rmem[1],gru1_rmem[1],gru1_rmem[0]]
gru1_fehcolx=[gru1_fehmem[0],gru1_fehmem[1],gru1_fehmem[1],gru1_fehmem[0],gru1_fehmem[0]]
gru1_fehcoly=[gru1_colmem[0],gru1_colmem[0],gru1_colmem[1],gru1_colmem[1],gru1_colmem[0]]
gru1_fehmagx=[gru1_fehmem[0],gru1_fehmem[1],gru1_fehmem[1],gru1_fehmem[0],gru1_fehmem[0]]
gru1_fehmagy=[gru1_magmem[0],gru1_magmem[0],gru1_magmem[1],gru1_magmem[1],gru1_magmem[0]]

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
ax6_0.add_patch(Rectangle((tuc2_vmem[0],tuc2_teffmem[0]),tuc2_vmem[1]-tuc2_vmem[0],tuc2_teffmem[1]-tuc2_teffmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax6_0.add_patch(Rectangle((gru1_vmem[0],gru1_teffmem[0]),gru1_vmem[1]-gru1_vmem[0],gru1_teffmem[1]-gru1_teffmem[0],facecolor='b',alpha=0.2,edgecolor='None'))
ax6_0.errorbar(tuc2_v[tuc2_keep],tuc2_teff[tuc2_keep],xerr=tuc2_sigv[tuc2_keep],yerr=tuc2_sigteff[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax6_0.errorbar(gru1_v[gru1_keep],gru1_teff[gru1_keep],xerr=gru1_sigv[gru1_keep],yerr=gru1_sigteff[gru1_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)

ax5_0.set_ylabel(logglabel[0],fontsize=8,rotation=90,labelpad=5)
ax5_0.set_xlim(vlim)
ax5_0.set_ylim(revlogglim)
ax5_0.set_xscale(u'linear')
ax5_0.set_yscale(u'linear')
ax5_0.set_xticks(vticks)
ax5_0.set_yticks(revloggticks)
ax5_0.set_yticklabels(revloggticks,rotation=0)
ax5_0.xaxis.set_major_formatter(plt.NullFormatter())
ax5_0.add_patch(Rectangle((tuc2_vmem[0],tuc2_loggmem[0]),tuc2_vmem[1]-tuc2_vmem[0],tuc2_loggmem[1]-tuc2_loggmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax5_0.add_patch(Rectangle((gru1_vmem[0],gru1_loggmem[0]),gru1_vmem[1]-gru1_vmem[0],gru1_loggmem[1]-gru1_loggmem[0],facecolor='b',alpha=0.2,edgecolor='None'))
ax5_0.errorbar(tuc2_v[tuc2_keep],tuc2_logg[tuc2_keep],xerr=tuc2_sigv[tuc2_keep],yerr=tuc2_siglogg[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax5_0.plot(tuc2_vloggx,tuc2_vloggy,color='r',linewidth=0.25)
ax5_0.errorbar(gru1_v[gru1_keep],gru1_logg[gru1_keep],xerr=gru1_sigv[gru1_keep],yerr=gru1_siglogg[gru1_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)
#ax5_0.plot(gru1_vloggx,gru1_vloggy,color='b',linewidth=0.25)

ax4_0.set_ylabel(fehlabel[0],fontsize=8,rotation=90,labelpad=5)
ax4_0.set_xlim(vlim)
ax4_0.set_ylim(fehlim)
ax4_0.set_xscale(u'linear')
ax4_0.set_yscale(u'linear')
ax4_0.set_xticks(vticks)
ax4_0.set_yticks(fehticks)
ax4_0.set_yticklabels(fehticks,rotation=0)
ax4_0.xaxis.set_major_formatter(plt.NullFormatter())
ax4_0.add_patch(Rectangle((tuc2_vmem[0],tuc2_fehmem[0]),tuc2_vmem[1]-tuc2_vmem[0],tuc2_fehmem[1]-tuc2_fehmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax4_0.add_patch(Rectangle((gru1_vmem[0],gru1_fehmem[0]),gru1_vmem[1]-gru1_vmem[0],gru1_fehmem[1]-gru1_fehmem[0],facecolor='b',alpha=0.2,edgecolor='None'))
ax4_0.errorbar(tuc2_v[tuc2_keep],tuc2_feh[tuc2_keep],xerr=tuc2_sigv[tuc2_keep],yerr=tuc2_sigfeh[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax4_0.plot(tuc2_vfehx,tuc2_vfehy,color='r',linewidth=0.25)
ax4_0.errorbar(gru1_v[gru1_keep],gru1_feh[gru1_keep],xerr=gru1_sigv[gru1_keep],yerr=gru1_sigfeh[gru1_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)
#ax4_0.plot(gru1_vfehx,gru1_vfehy,color='b',linewidth=0.25)

ax3_0.set_ylabel(rlabel[0],fontsize=8,rotation=90,labelpad=5)
ax3_0.set_xlim(vlim)
ax3_0.set_ylim(rlim)
ax3_0.set_xscale(u'linear')
ax3_0.set_yscale(u'linear')
ax3_0.set_xticks(vticks)
ax3_0.set_yticks(rticks)
ax3_0.set_yticklabels(rticks,rotation=0)
ax3_0.xaxis.set_major_formatter(plt.NullFormatter())
ax3_0.add_patch(Rectangle((tuc2_vmem[0],tuc2_rmem[0]),tuc2_vmem[1]-tuc2_vmem[0],tuc2_rmem[1]-tuc2_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_0.add_patch(Rectangle((gru1_vmem[0],gru1_rmem[0]),gru1_vmem[1]-gru1_vmem[0],gru1_rmem[1]-gru1_rmem[0],facecolor='b',alpha=0.2,edgecolor='None'))
ax3_0.errorbar(tuc2_v[tuc2_keep],tuc2_r[tuc2_keep],xerr=tuc2_sigv[tuc2_keep],yerr=tuc2_sigr[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax3_0.plot(tuc2_vrx,tuc2_vry,color='r',linewidth=0.25)
ax3_0.errorbar(gru1_v[gru1_keep],gru1_r[gru1_keep],xerr=gru1_sigv[gru1_keep],yerr=gru1_sigr[gru1_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)
#ax3_0.plot(gru1_vrx,gru1_vry,color='b',linewidth=0.25)
ax3_0.plot([-135.45,-135.45],[0,20],linestyle=':',lw=0.25)

ax2_0.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_0.xaxis.set_major_formatter(plt.NullFormatter())
ax2_0.set_ylim(histlim)
ax2_0.set_xticks(vticks)
ax2_0.set_yticks(histticks)
ax2_0.hist(tuc2_v[tuc2_keep],bins=30,range=vlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,label='Tuc 2')
ax2_0.hist(tuc2_v[tuc2_mem],bins=30,range=vlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)
ax2_0.hist(gru1_v[gru1_keep],bins=30,range=vlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25,label='Gru 1')
ax2_0.hist(gru1_v[gru1_mem],bins=30,range=vlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25,alpha=0.5)
ax2_0.legend(loc=2,fontsize=6,handlelength=0.5,numpoints=1,scatterpoints=1,shadow=False,borderpad=0)

piss=gaussian_kde(gru1_besancon_v[gru1_besancon_cands])
gru1_xs=np.arange(min(vlim),max(vlim),1)
gru1_ys=piss(gru1_xs)
gru1_area2=np.trapz(gru1_ys,x=gru1_xs)
factor=np.size(gru1_keep)*(max(vlim)-min(vlim))/30.
ax2_0.plot(gru1_xs,gru1_ys*factor,color='b',linestyle=':',lw=0.5)

piss=gaussian_kde(tuc2_besancon_v[tuc2_besancon_cands])
tuc2_xs=np.arange(min(vlim),max(vlim),1)
tuc2_ys=piss(tuc2_xs)
tuc2_area2=np.trapz(tuc2_ys,x=tuc2_xs)
factor=np.size(tuc2_keep)*(max(vlim)-min(vlim))/30.
ax2_0.plot(tuc2_xs,tuc2_ys*factor,color='r',linestyle=':',lw=0.5)


ax5_1.set_xlabel(tefflabel[0],fontsize=8,rotation=0)
ax5_1.set_xlim(revtefflim)
ax5_1.set_ylim(revlogglim)
ax5_1.set_xscale(u'linear')
ax5_1.set_yscale(u'linear')
ax5_1.set_xticks([4000]+revteffticks)
ax5_1.set_xticklabels([4000]+revteffticks,rotation=0)
ax5_1.set_yticks(revloggticks)
ax5_1.yaxis.set_major_formatter(plt.NullFormatter())
ax5_1.add_patch(Rectangle((tuc2_teffmem[0],tuc2_loggmem[0]),tuc2_teffmem[1]-tuc2_teffmem[0],tuc2_loggmem[1]-tuc2_loggmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax5_1.add_patch(Rectangle((gru1_teffmem[0],gru1_loggmem[0]),gru1_teffmem[1]-gru1_teffmem[0],gru1_loggmem[1]-gru1_loggmem[0],facecolor='b',alpha=0.2,edgecolor='None'))
ax5_1.errorbar(tuc2_teff[tuc2_keep],tuc2_logg[tuc2_keep],xerr=tuc2_sigteff[tuc2_keep],yerr=tuc2_siglogg[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax5_1.errorbar(gru1_teff[gru1_keep],gru1_logg[gru1_keep],xerr=gru1_sigteff[gru1_keep],yerr=gru1_siglogg[gru1_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)

ax4_1.set_xlim(revtefflim)
ax4_1.set_ylim(fehlim)
ax4_1.set_xscale(u'linear')
ax4_1.set_yscale(u'linear')
ax4_1.set_xticks(revteffticks)
ax4_1.set_yticks(fehticks)
ax4_1.xaxis.set_major_formatter(plt.NullFormatter())
ax4_1.yaxis.set_major_formatter(plt.NullFormatter())
ax4_1.add_patch(Rectangle((tuc2_teffmem[0],tuc2_fehmem[0]),tuc2_teffmem[1]-tuc2_teffmem[0],tuc2_fehmem[1]-tuc2_fehmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax4_1.add_patch(Rectangle((gru1_teffmem[0],gru1_fehmem[0]),gru1_teffmem[1]-gru1_teffmem[0],gru1_fehmem[1]-gru1_fehmem[0],facecolor='b',alpha=0.2,edgecolor='None'))
ax4_1.errorbar(tuc2_teff[tuc2_keep],tuc2_feh[tuc2_keep],xerr=tuc2_sigteff[tuc2_keep],yerr=tuc2_sigfeh[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax4_1.errorbar(gru1_teff[gru1_keep],gru1_feh[gru1_keep],xerr=gru1_sigteff[gru1_keep],yerr=gru1_sigfeh[gru1_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)

ax3_1.set_xlim(revtefflim)
ax3_1.set_ylim(rlim)
ax3_1.set_xscale(u'linear')
ax3_1.set_yscale(u'linear')
ax3_1.set_xticks(revteffticks)
ax3_1.set_yticks(rticks)
ax3_1.xaxis.set_major_formatter(plt.NullFormatter())
ax3_1.yaxis.set_major_formatter(plt.NullFormatter())
ax3_1.add_patch(Rectangle((tuc2_teffmem[0],tuc2_rmem[0]),tuc2_teffmem[1]-tuc2_teffmem[0],tuc2_rmem[1]-tuc2_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_1.add_patch(Rectangle((gru1_teffmem[0],gru1_rmem[0]),gru1_teffmem[1]-gru1_teffmem[0],gru1_rmem[1]-gru1_rmem[0],facecolor='b',alpha=0.2,edgecolor='None'))
ax3_1.errorbar(tuc2_teff[tuc2_keep],tuc2_r[tuc2_keep],xerr=tuc2_sigteff[tuc2_keep],yerr=tuc2_sigr[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax3_1.errorbar(gru1_teff[gru1_keep],gru1_r[gru1_keep],xerr=gru1_sigteff[gru1_keep],yerr=gru1_sigr[gru1_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)

ax2_1.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_1.xaxis.set_major_formatter(plt.NullFormatter())
ax2_1.yaxis.set_major_formatter(plt.NullFormatter())
ax2_1.set_xlim(revtefflim)
ax2_1.set_ylim(histlim)
ax2_1.set_xticks(revteffticks)
ax2_1.set_yticks(histticks)
ax2_1.hist(tuc2_teff[tuc2_keep],bins=30,range=tefflim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25)
ax2_1.hist(tuc2_teff[tuc2_mem],bins=30,range=tefflim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)
ax2_1.hist(gru1_teff[gru1_keep],bins=30,range=tefflim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25)
ax2_1.hist(gru1_teff[gru1_mem],bins=30,range=tefflim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25,alpha=0.5)

piss=gaussian_kde(gru1_besancon_teff[gru1_besancon_cands])
gru1_xs=np.arange(min(tefflim),max(tefflim),1)
gru1_ys=piss(gru1_xs)
gru1_area2=np.trapz(gru1_ys,x=gru1_xs)
factor=np.size(gru1_keep)*(max(tefflim)-min(tefflim))/30.
ax2_1.plot(gru1_xs,gru1_ys*factor,color='b',linestyle=':',lw=0.5)

piss=gaussian_kde(tuc2_besancon_teff[tuc2_besancon_cands])
tuc2_xs=np.arange(min(tefflim),max(tefflim),1)
tuc2_ys=piss(tuc2_xs)
tuc2_area2=np.trapz(tuc2_ys,x=tuc2_xs)
factor=np.size(tuc2_keep)*(max(tefflim)-min(tefflim))/30.
ax2_1.plot(tuc2_xs,tuc2_ys*factor,color='r',linestyle=':',lw=0.5)


ax4_2.set_xlabel(logglabel[0],fontsize=8,rotation=0)
ax4_2.set_xlim(revlogglim)
ax4_2.set_ylim(fehlim)
ax4_2.set_xscale(u'linear')
ax4_2.set_yscale(u'linear')
ax4_2.set_xticks([0]+revloggticks)
ax4_2.set_xticklabels([0]+revloggticks,rotation=0)
ax4_2.set_yticks(fehticks)
ax4_2.yaxis.set_major_formatter(plt.NullFormatter())
ax4_2.add_patch(Rectangle((gru1_loggmem[0],gru1_fehmem[0]),gru1_loggmem[1]-gru1_loggmem[0],gru1_fehmem[1]-gru1_fehmem[0],facecolor='b',alpha=0.2,edgecolor='None'))
ax4_2.add_patch(Rectangle((tuc2_loggmem[0],tuc2_fehmem[0]),tuc2_loggmem[1]-tuc2_loggmem[0],tuc2_fehmem[1]-tuc2_fehmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax4_2.errorbar(tuc2_logg[tuc2_keep],tuc2_r[tuc2_keep],xerr=tuc2_siglogg[tuc2_keep],yerr=tuc2_sigr[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax4_2.errorbar(tuc2_logg[tuc2_keep],tuc2_feh[tuc2_keep],xerr=tuc2_siglogg[tuc2_keep],yerr=tuc2_sigfeh[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax4_2.plot(tuc2_loggfehx,tuc2_loggfehy,color='r',linewidth=0.25)
ax4_2.errorbar(gru1_logg[gru1_keep],gru1_feh[gru1_keep],xerr=gru1_siglogg[gru1_keep],yerr=gru1_sigfeh[gru1_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)
#ax4_2.plot(gru1_loggfehx,gru1_loggfehy,color='b',linewidth=0.25)


ax3_2.set_xlim(revlogglim)
ax3_2.set_ylim(rlim)
ax3_2.set_xscale(u'linear')
ax3_2.set_yscale(u'linear')
ax3_2.set_xticks(revloggticks)
ax3_2.set_yticks(rticks)
ax3_2.xaxis.set_major_formatter(plt.NullFormatter())
ax3_2.yaxis.set_major_formatter(plt.NullFormatter())
ax3_2.add_patch(Rectangle((gru1_loggmem[0],gru1_rmem[0]),gru1_loggmem[1]-gru1_loggmem[0],gru1_rmem[1]-gru1_rmem[0],facecolor='b',alpha=0.2,edgecolor='None'))
ax3_2.add_patch(Rectangle((tuc2_loggmem[0],tuc2_rmem[0]),tuc2_loggmem[1]-tuc2_loggmem[0],tuc2_rmem[1]-tuc2_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_2.errorbar(tuc2_logg[tuc2_keep],tuc2_r[tuc2_keep],xerr=tuc2_siglogg[tuc2_keep],yerr=tuc2_sigr[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax3_2.errorbar(tuc2_logg[tuc2_keep],tuc2_r[tuc2_keep],xerr=tuc2_siglogg[tuc2_keep],yerr=tuc2_sigr[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax3_2.plot(tuc2_loggrx,tuc2_loggry,color='r',linewidth=0.25)
ax3_2.errorbar(gru1_logg[gru1_keep],gru1_r[gru1_keep],xerr=gru1_siglogg[gru1_keep],yerr=gru1_sigr[gru1_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)
#ax3_2.plot(gru1_loggrx,gru1_loggry,color='b',linewidth=0.25)

ax2_2.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_2.xaxis.set_major_formatter(plt.NullFormatter())
ax2_2.yaxis.set_major_formatter(plt.NullFormatter())
ax2_2.set_xlim(revlogglim)
ax2_2.set_ylim(histlim)
ax2_2.set_xticks(revloggticks)
ax2_2.set_yticks(histticks)
ax2_2.hist(tuc2_logg[tuc2_keep],bins=30,range=logglim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25)
ax2_2.hist(tuc2_logg[tuc2_mem],bins=30,range=logglim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)
ax2_2.hist(gru1_logg[gru1_keep],bins=30,range=logglim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25)
ax2_2.hist(gru1_logg[gru1_mem],bins=30,range=logglim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25,alpha=0.5)

piss=gaussian_kde(gru1_besancon_logg[gru1_besancon_cands])
gru1_xs=np.arange(min(logglim),max(logglim),0.1)
gru1_ys=piss(gru1_xs)
gru1_area2=np.trapz(gru1_ys,x=gru1_xs)
factor=np.size(gru1_keep)*(max(logglim)-min(logglim))/30.
ax2_2.plot(gru1_xs,gru1_ys*factor,color='b',linestyle=':',lw=0.5)

piss=gaussian_kde(tuc2_besancon_logg[tuc2_besancon_cands])
tuc2_xs=np.arange(min(logglim),max(logglim),0.1)
tuc2_ys=piss(tuc2_xs)
tuc2_area2=np.trapz(tuc2_ys,x=tuc2_xs)
factor=np.size(tuc2_keep)*(max(logglim)-min(logglim))/30.
ax2_2.plot(tuc2_xs,tuc2_ys*factor,color='r',linestyle=':',lw=0.5)



ax3_3.set_xlabel(fehlabel[0],fontsize=8,rotation=0)
ax3_3.set_xlim(fehlim)
ax3_3.set_ylim(rlim)
ax3_3.set_xscale(u'linear')
ax3_3.set_yscale(u'linear')
ax3_3.set_xticks(fehticks)
ax3_3.set_xticklabels(fehticks,rotation=0)
ax3_3.set_yticks(rticks)
ax3_3.yaxis.set_major_formatter(plt.NullFormatter())
ax3_3.add_patch(Rectangle((gru1_fehmem[0],gru1_rmem[0]),gru1_fehmem[1]-gru1_fehmem[0],gru1_rmem[1]-gru1_rmem[0],facecolor='b',alpha=0.2,edgecolor='None'))
ax3_3.add_patch(Rectangle((tuc2_fehmem[0],tuc2_rmem[0]),tuc2_fehmem[1]-tuc2_fehmem[0],tuc2_rmem[1]-tuc2_rmem[0],facecolor='r',alpha=0.2,edgecolor='None'))
ax3_3.errorbar(tuc2_feh[tuc2_keep],tuc2_r[tuc2_keep],xerr=tuc2_sigfeh[tuc2_keep],yerr=tuc2_sigr[tuc2_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
#ax3_3.plot(tuc2_fehrx,tuc2_fehry,color='r',linewidth=0.25)
ax3_3.errorbar(gru1_feh[gru1_keep],gru1_r[gru1_keep],xerr=gru1_sigfeh[gru1_keep],yerr=gru1_sigr[gru1_keep],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)
#ax3_3.plot(gru1_fehrx,gru1_fehry,color='b',linewidth=0.25)

ax2_3.set_ylabel('N',fontsize=8,rotation=90,labelpad=5)
ax2_3.xaxis.set_major_formatter(plt.NullFormatter())
ax2_3.yaxis.set_major_formatter(plt.NullFormatter())
ax2_3.set_ylim(histlim)
ax2_3.set_xticks(fehticks)
ax2_3.set_yticks(histticks)
ax2_3.hist(tuc2_feh[tuc2_keep],bins=30,range=fehlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25)
ax2_3.hist(tuc2_feh[tuc2_mem],bins=30,range=fehlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='r',linewidth=0.25,alpha=0.5)
ax2_3.hist(gru1_feh[gru1_keep],bins=30,range=fehlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25)
ax2_3.hist(gru1_feh[gru1_mem],bins=30,range=fehlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='vertical',color='b',linewidth=0.25,alpha=0.5)

piss=gaussian_kde(gru1_besancon_feh[gru1_besancon_cands])
gru1_xs=np.arange(min(fehlim),max(fehlim),0.1)
gru1_ys=piss(gru1_xs)
gru1_area2=np.trapz(gru1_ys,x=gru1_xs)
factor=np.size(gru1_keep)*(max(fehlim)-min(fehlim))/30.
ax2_3.plot(gru1_xs,gru1_ys*factor,color='b',linestyle=':',lw=0.5)

piss=gaussian_kde(tuc2_besancon_feh[tuc2_besancon_cands])
tuc2_xs=np.arange(min(fehlim),max(fehlim),0.1)
tuc2_ys=piss(tuc2_xs)
tuc2_area2=np.trapz(tuc2_ys,x=tuc2_xs)
factor=np.size(tuc2_keep)*(max(fehlim)-min(fehlim))/30.
ax2_3.plot(tuc2_xs,tuc2_ys*factor,color='r',linestyle=':',lw=0.5)


#ax3_4.set_xlabel('N',fontsize=8,rotation=0,labelpad=5)
ax3_4.xaxis.set_major_formatter(plt.NullFormatter())
ax3_4.yaxis.set_major_formatter(plt.NullFormatter())
ax3_4.set_xlim(histlim)
ax3_4.set_yticks(rticks)
ax3_4.set_xticks(histticks)
ax3_4.hist(tuc2_r[tuc2_keep],bins=30,range=rlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.25)
ax3_4.hist(tuc2_r[tuc2_mem],bins=30,range=rlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='horizontal',color='r',linewidth=0.25,alpha=0.5)
ax3_4.hist(gru1_r[gru1_keep],bins=30,range=rlim,normed=False,histtype='step',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.25)
ax3_4.hist(gru1_r[gru1_mem],bins=30,range=rlim,normed=False,histtype='stepfilled',align='mid',rwidth=0,orientation='horizontal',color='b',linewidth=0.25,alpha=0.5)


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
ax10_10.scatter(tuc2_teff[tuc2_keep],tuc2_logg[tuc2_keep],marker='.',s=1,alpha=0.65,color='k',rasterized=True,label='foreground')
ax10_10.scatter(tuc2_teff[tuc2_mem],tuc2_logg[tuc2_mem],marker='.',s=1,alpha=0.65,color='r',rasterized=True)
ax10_10.errorbar(tuc2_teff[tuc2_mem],tuc2_logg[tuc2_mem],xerr=tuc2_sigteff[tuc2_mem],yerr=tuc2_siglogg[tuc2_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True,label='Tuc 2 member')
ax10_10.scatter(gru1_teff[gru1_keep],gru1_logg[gru1_keep],marker='.',s=1,alpha=0.65,color='k',rasterized=True)
ax10_10.scatter(gru1_teff[gru1_mem],gru1_logg[gru1_mem],marker='.',s=1,alpha=0.65,color='r',rasterized=True)
ax10_10.errorbar(gru1_teff[gru1_mem],gru1_logg[gru1_mem],xerr=gru1_sigteff[gru1_mem],yerr=gru1_siglogg[gru1_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True,label='Gru 1 member')
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
ax9_10.errorbar(tuc2_teff[tuc2_mem],tuc2_mag[tuc2_mem]-tuc2_dmodulus,xerr=tuc2_sigteff[tuc2_mem],yerr=tuc2_siglogg[tuc2_mem]-tuc2_siglogg[tuc2_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax9_10.errorbar(gru1_teff[gru1_mem],gru1_mag[gru1_mem]-gru1_dmodulus,xerr=gru1_sigteff[gru1_mem],yerr=gru1_siglogg[gru1_mem]-gru1_siglogg[gru1_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)
ax9_10.legend(loc=2,fontsize=4,handlelength=4,shadow=False)
#ax9_10.text(-0.4,16.5,'Crater',fontsize=10)
#ax9_10.text(7500,16.5,'Tuc 2',fontsize=12)

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
ax10_9.errorbar(tuc2_mag[tuc2_mem]-tuc2_dmodulus,tuc2_logg[tuc2_mem],xerr=tuc2_sigteff[tuc2_mem]-tuc2_sigteff[tuc2_mem],yerr=tuc2_siglogg[tuc2_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='r',rasterized=True)
ax10_9.errorbar(gru1_mag[gru1_mem]-gru1_dmodulus,gru1_logg[gru1_mem],xerr=gru1_sigteff[gru1_mem]-gru1_sigteff[gru1_mem],yerr=gru1_siglogg[gru1_mem],elinewidth=0.5,fmt='.',capsize=0,alpha=0.65,color='b',rasterized=True)
#ax10_9.legend(loc=2,fontsize=12,handlelength=0,shadow=False)
#ax2_0.text(-0.4,16.5,'Crater',fontsize=10)

g1.write(r'\newcommand{\tucnobs}{$'+str(len(tuc2_v))+r'$}'+'\n')
g1.write(r'\newcommand{\tucgoodnobs}{$'+str(len(tuc2_v[tuc2_keep]))+r'$}'+'\n')
g1.write(r'\newcommand{\grunobs}{$'+str(len(gru1_v))+r'$}'+'\n')
g1.write(r'\newcommand{\grugoodnobs}{$'+str(len(gru1_v[gru1_keep]))+r'$}'+'\n')
g1.write(r'\newcommand{\tucgrunobs}{$'+str(len(tuc2_v)+len(gru1_v))+r'$}'+'\n')
g1.write(r'\newcommand{\tucgrugoodnobs}{$'+str(len(tuc2_v[tuc2_keep])+len(gru1_v[gru1_keep]))+r'$}'+'\n')
g1.write(r'\newcommand{\tucqc}{$'+str(len(tuc2_v[tuc2_qc]))+r'$}'+'\n')
g1.write(r'\newcommand{\gruqc}{$'+str(len(gru1_v[gru1_qc]))+r'$}'+'\n')
#g1.write(r'\newcommand{\nred}{$'+str(len(tuc2_v[mem]))+r'$}'+'\n')
#g1.write(r'\newcommand{\nblue}{$'+str(len(v[mem3]))+r'$}'+'\n')
g1.write(r'\newcommand{\medsigv}{$'+str(round(np.median(np.concatenate((tuc2_sigv[tuc2_keep],gru1_sigv[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.write(r'\newcommand{\medsigteff}{$'+str(int(np.median(np.concatenate((tuc2_sigteff[tuc2_keep],gru1_sigteff[gru1_keep]),axis=0))))+r'$}'+'\n')
g1.write(r'\newcommand{\medsiglogg}{$'+str(round(np.median(np.concatenate((tuc2_siglogg[tuc2_keep],gru1_siglogg[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.write(r'\newcommand{\medsigfeh}{$'+str(round(np.median(np.concatenate((tuc2_sigfeh[tuc2_keep],gru1_sigfeh[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.write(r'\newcommand{\tucmembers}{$'+str(np.size(tuc2_mem))+r'$}'+'\n')
g1.write(r'\newcommand{\grumembers}{$'+str(np.size(gru1_mem))+r'$}'+'\n')
g1.write(r'\newcommand{\tucgoodmembers}{$'+str(np.size(tuc2_goodmem))+r'$}'+'\n')
g1.write(r'\newcommand{\grugoodmembers}{$'+str(np.size(gru1_goodmem))+r'$}'+'\n')
g1.write(r'\newcommand{\tucgrugoodmembers}{$'+str(np.size(tuc2_goodmem)+np.size(gru1_goodmem))+r'$}'+'\n')
g1.write(r'\newcommand{\minsigv}{$'+str(round(min(np.concatenate((tuc2_sigv[tuc2_keep],gru1_sigv[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.write(r'\newcommand{\maxsigv}{$'+str(round(max(np.concatenate((tuc2_sigv[tuc2_keep],gru1_sigv[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.write(r'\newcommand{\minsigteff}{$'+str(round(min(np.concatenate((tuc2_sigteff[tuc2_keep],gru1_sigteff[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.write(r'\newcommand{\maxsigteff}{$'+str(round(max(np.concatenate((tuc2_sigteff[tuc2_keep],gru1_sigteff[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.write(r'\newcommand{\minsiglogg}{$'+str(round(min(np.concatenate((tuc2_siglogg[tuc2_keep],gru1_siglogg[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.write(r'\newcommand{\maxsiglogg}{$'+str(round(max(np.concatenate((tuc2_siglogg[tuc2_keep],gru1_siglogg[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.write(r'\newcommand{\minsigfeh}{$'+str(round(min(np.concatenate((tuc2_sigfeh[tuc2_keep],gru1_sigfeh[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.write(r'\newcommand{\maxsigfeh}{$'+str(round(max(np.concatenate((tuc2_sigfeh[tuc2_keep],gru1_sigfeh[gru1_keep]),axis=0)),2))+r'$}'+'\n')
g1.close()

plotfilename='tuc2gru1teffprior_params.pdf'
plt.savefig(plotfilename,dpi=400)
plt.show()
plt.close()
