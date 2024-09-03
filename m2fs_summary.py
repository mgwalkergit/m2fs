import numpy as np
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import mycode
from pymultinest.solve import solve
matplotlib.use('TkAgg')
#matplotlib.use('pdf')

gal='scl'

with open('dsph.dat') as f:
    data=f.readlines()
dsph_name=[]
dsph_name2=[]
dsph_radeg=[]
dsph_decdeg=[]
dsph_distance=[]
dsph_rhalf=[]
dsph_ellipticity=[]
dsph_pa=[]
for line in data:
    p=line.split()
    dsph_name.append(p[0])
    dsph_name2.append(p[1])
    dsph_radeg.append(float(p[2]))
    dsph_decdeg.append(float(p[3]))
    dsph_distance.append(float(p[4]))
    dsph_rhalf.append(float(p[9]))
    dsph_ellipticity.append(float(p[12]))
    dsph_pa.append(float(p[15]))
dsph_name=np.array(dsph_name)
dsph_name2=np.array(dsph_name2)
dsph_radeg=np.array(dsph_radeg)
dsph_decdeg=np.array(dsph_decdeg)
dsph_distance=np.array(dsph_distance)
dsph_rhalf=np.array(dsph_rhalf)
dsph_ellipticity=np.array(dsph_ellipticity)
dsph_pa=np.array(dsph_pa)
fix=np.where(dsph_pa>180.)[0]
dsph_pa[fix]-=180.

this_dsph=np.where(dsph_name==gal)[0][0]
ra_deg_center,dec_deg_center=dsph_radeg[this_dsph],dsph_decdeg[this_dsph]

poop=fits.open('all_m2fshiresian.fits')
#poop=fits.open('../nelson/nelson_all_ian.fits')
keep0=np.where(poop[1].data['object']==gal)[0]
keep=np.where((poop[1].data['vlos_error']<5.)&(poop[1].data['vlos_error']>0.)&(np.abs(poop[1].data['vlos_skew'])<=1.)&(np.abs(poop[1].data['vlos_kurtosis'])<=1.))[0]
keep2=np.where((np.abs(poop[1].data['vlos'])<500.)&(poop[1].data['vlos_error']<5.)&(poop[1].data['vlos_error']>0.)&(np.abs(poop[1].data['vlos_skew'])<=1.)&(np.abs(poop[1].data['vlos_kurtosis'])<=1.)&(poop[1].data['object']==gal))[0]

obs=np.zeros(len(keep2))
vmean=[]
zmean=[]
sigvmean=[]
sigzmean=[]
ra_deg=[]
dec_deg=[]
for i in range(0,len(keep2)):
    if obs[i]==0:
        dist=np.sqrt((poop[1].data['ra_deg'][keep2][i]-poop[1].data['ra_deg'][keep2])**2+(poop[1].data['dec_deg'][keep2][i]-poop[1].data['dec_deg'][keep2])**2)*3600.
        this=np.where(dist<0.5)[0]
        sum1=np.sum(poop[1].data['teffprior_vlos'][keep2][this]/poop[1].data['teffprior_vlos_error'][keep2][this]**2)
        sum2=np.sum(1./poop[1].data['teffprior_vlos_error'][keep2][this]**2)
        sum3=np.sum(poop[1].data['teffprior_z'][keep2][this]/poop[1].data['teffprior_z_error'][keep2][this]**2)
        sum4=np.sum(1./poop[1].data['teffprior_z_error'][keep2][this]**2)
        ra_deg.append(poop[1].data['ra_deg'][keep2][i])
        dec_deg.append(poop[1].data['dec_deg'][keep2][i])
        vmean.append(sum1/sum2)
        zmean.append(sum3/sum4)
        if np.abs(sum1)>1.e+30:
            np.pause()
        sigvmean.append(1./np.sqrt(sum2))
        sigzmean.append(1./np.sqrt(sum4))
        for j in range(0,len(this)):
            obs[this[j]]=j+1
stars=np.where(obs==1)[0]
vmean=np.array(vmean)
zmean=np.array(zmean)
sigvmean=np.array(sigvmean)
sigzmean=np.array(sigzmean)
ra_deg=np.array(ra_deg)
dec_deg=np.array(dec_deg)

coords_center=SkyCoord(ra_deg_center*u.deg,dec_deg_center*u.deg)
xi,eta=mycode.etaxiarr(ra_deg*np.pi/180.,dec_deg*np.pi/180.,coords_center.ra.rad,coords_center.dec.rad)
r=np.sqrt(xi**2+eta**2)
rpc=dsph_distance[this_dsph]*r/60.*np.pi/180.

rplummer=dsph_rhalf[this_dsph]
v=vmean
z=zmean
sigv=sigvmean
sigz=sigzmean

def myloglike0(cube):
#    sigma_mem0=nsample*fmem/np.pi/rplummer**2
#    sigma_non0=nsample*(1-fmem)/np.pi/rmax**2
    sigma_mem0=10.**cube[8]
    sigma_non0=10.**cube[9]
    sigma_mem=sigma_mem0/(1.+(r/rplummer**2))**2
    sigma_non=sigma_non0
    vmean_mem=cube[0]
    vdisp_mem=cube[1]
    vmean_non1=cube[2]
    vdisp_non1=cube[3]
    zmean_mem=cube[4]+cube[6]
    zdisp_mem=cube[5]
    zmean_non1=cube[6]
    zdisp_non1=cube[7]
    vmean_non2=cube[2]+cube[11]
    vdisp_non2=cube[12]
    zmean_non2=cube[6]+cube[14]
    zdisp_non2=cube[15]
    p1=1./np.sqrt(2.*np.pi*(sigv**2+vdisp_mem**2))*np.exp(-0.5*(v-vmean_mem)**2/(sigv**2+vdisp_mem**2)) \
        *1./np.sqrt(2.*np.pi*(sigz**2+zdisp_mem**2))*np.exp(-0.5*(z-zmean_mem)**2/(sigz**2+zdisp_mem**2))
    p2=(cube[10]/np.sqrt(2.*np.pi*(sigv**2+vdisp_non1**2))*np.exp(-0.5*(v-vmean_non1)**2/(sigv**2+vdisp_non1**2)) \
        +(1-cube[10])/np.sqrt(2.*np.pi*(sigv**2+vdisp_non2**2))*np.exp(-0.5*(v-vmean_non2)**2/(sigv**2+vdisp_non2**2))) \
        *(cube[13]/np.sqrt(2.*np.pi*(sigz**2+zdisp_non1**2))*np.exp(-0.5*(z-zmean_non1)**2/(sigz**2+zdisp_non1**2)) \
          +(1.-cube[13])/np.sqrt(2.*np.pi*(sigz**2+zdisp_non2**2))*np.exp(-0.5*(z-zmean_non2)**2/(sigz**2+zdisp_non2**2)))
          
    loglike=np.sum(np.log((sigma_mem*p1+sigma_non*p2)/(sigma_mem+sigma_non)))
    pmem=sigma_mem*p1/(sigma_mem*p1+sigma_non*p2)
    return loglike,pmem

def myloglike2(cube):
    shite=myloglike0(cube)
    return shite[0]

def myprior(cube):
    prior=[]
    prior.append([-300,300])#vmean_mem
    prior.append([0.,50.])#vdisp_mem / vdisp_non
    prior.append([-500,500])#vmean_non1
    prior.append([50.,300.])#vdisp_non1
    prior.append([-6,0.])#zmean_mem-zmean_non
    prior.append([0.,3.])#zdisp_mem
    prior.append([-2,1])#zmean_non
    prior.append([0.,3.])#zdisp_non
    prior.append([-15,15])#Sigma0_mem
    prior.append([-15,15])#Sigma0_non
    prior.append([0.,1.])#mixing fraction for 2-gaussian foreground velocities
    prior.append([-500.,0.])#vmean_non2-zmean_non1
    prior.append([200.,1000.])#vdisp_non2
    prior.append([0.,1.])#mixing fraction for 2-gaussian foreground metallicities
    prior.append([-5.,0.])#zmean_non2-zmean_non1
    prior.append([0.,5.])#zdisp_non2
    prior=np.array(prior)
    x=np.array(cube)

    for i in range(0,len(x)):
        x[i]=prior[i][0]+(prior[i][1]-prior[i][0])*cube[i]
    return x

prefix='pymultinest_test'
n_params=16
result=solve(LogLikelihood=myloglike2,Prior=myprior,n_dims=n_params,outputfiles_basename=prefix,verbose=True,resume=False,n_live_points=1000)

last=len(result['samples'])-1
last_model=result['samples'][last]
crap,pmem=myloglike0(last_model)


plt.hist(result['samples'].T[1])
plt.show()
plt.close()

plt.scatter(v,z,s=1,color='k',alpha=0.3)
mem=np.where((pmem>0.5)&(sigz<0.5))[0]
plt.scatter(v[mem],z[mem],s=5,color='r',alpha=0.9)
plt.show()
plt.close()

plt.scatter(rpc[mem],v[mem],c=z[mem],s=10,cmap='rainbow',rasterized=True)
plt.colorbar(label='[Fe/H]')
plt.xlabel('R [kpc]')
plt.ylabel(r'$V_{\rm los}$ [km/s]')
plt.text(0,96,'Crater 2')
plt.savefig('cra2_vogelsberger.pdf',dpi=200)
plt.show()
plt.close()
print(len(keep0),len(keep2),len(stars),np.sum(pmem))
