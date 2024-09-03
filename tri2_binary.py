import numpy as np
import astropy
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
import scipy
import mycode as mc
import dill as pickle
from pymultinest.solve import solve
from pymultinest import Analyzer
from astropy.io import fits
import xymass

directory='/hildafs/projects/phy200028p/mgwalker/m2fs/'
prefix=directory+'chains/tri2binary'
#prior=np.array([[-500,500],[np.log10(0.01),np.log10(10000)],[0.,1.],[0.7,0.9],[0.,1.],[0.,2.*np.pi],[0.,1.],[0.,1.]])#vcm,log10(period/years),eccentricity,m_primary/msun,mass_ratio,longitude/radians,cos(inclination/radians,phase0)
prior=np.array([[-500,500],[np.log10(0.001),np.log10(100)],[0.,1.],[0.77,0.78],[np.log10(0.1),np.log10(10.)],[0.,2.*np.pi],[0.,1.],[0.,1.]])#vcm,log10(period/years),eccentricity,m1/msun,log10(mtot/msun),longitude/radians,cos(inclination/radians,phase0)
resume=True
sampler='multinest'

hjd=np.array([2457374.80,2457683.72,2457688.77,2457711.69,2457303.10,2457416.70,2457569.10,2457639.10,2457284.07])
vlos=np.array([-400.4,-405.9,-405.1,-386.4,-375.8,-386.7,-376.3,-384.6,-373.8])
vlos_error=np.array([0.7,0.8,1.0,2.1,1.7,1.6,4.0,1.6,2.4])

result,bestfit=mc.fit_binary_orbit((hjd*u.day).to(u.year).value,vlos,vlos_error,prefix=prefix,prior=prior,resume=resume,sampler=sampler)

x=np.linspace(np.min(hjd),np.max(hjd),10000)*u.day

for i in np.random.permutation(np.arange(len(result['samples'])))[0:100]:
    vcm=result['samples'].T[0][i]
    period=10.**(result['samples'].T[1][i])*u.year
    eccentricity=result['samples'].T[2][i]
    mass_1=result['samples'].T[3][i]*u.M_sun
    mass_tot=(10.**result['samples'].T[4][i])*u.M_sun
    longitude=result['samples'].T[5][i]*u.rad
    inclination=np.arccos(result['samples'].T[6][i])*u.rad
    phase0=result['samples'].T[7][i]

    case1=True
    mass_2=mass_tot-mass_1
    
    if mass_1.value>mass_2.value:
        mass_ratio=mass_2.value/mass_1.value
    else:
        mass_ratio=mass_1.value/mass_2.value
        case1=False
        
    f_period=phase0+(x-np.min(x)).to(u.year).value/period.value
    mass_primary=np.max([mass_1.value,mass_2.value])*u.M_sun
    
    sample_orbit=xymass.sample_orbit_2body(f_period,period=period,eccentricity=eccentricity,mass_primary=mass_primary,mass_ratio=mass_ratio,longitude=longitude,inclination=inclination)
    
    if case1==True:
        vlos_model=vcm+sample_orbit.v1_obs_xyz.T[2].value*4.74047
    else:
        vlos_model=vcm+sample_orbit.v2_obs_xyz.T[2].value*4.74047        
    print(i,period,eccentricity,mass_1,mass_2,mass_tot,longitude,inclination,sample_orbit.semimajor_axis[0],np.max(vlos_model))
    plt.plot(x,vlos_model,alpha=0.3,color='0.5',rasterized=True)
    
vcm=bestfit['parameters'][0]
period=10.**(bestfit['parameters'][1])*u.year
eccentricity=bestfit['parameters'][2]
mass_1=bestfit['parameters'][3]*u.M_sun
mass_tot=(10.**bestfit['parameters'][4])*u.M_sun
longitude=bestfit['parameters'][5]*u.rad
inclination=np.arccos(bestfit['parameters'][6])*u.rad
phase0=bestfit['parameters'][7]

case1=True
mass_2=mass_tot-mass_1

if mass_1.value>=mass_2.value:
    mass_ratio=mass_2.value/mass_1.value
else:
    mass_ratio=mass_1.value/mass_2.value
    case1=False
    
mass_primary=np.max([mass_1.value,mass_2.value])*u.M_sun
f_period=phase0+(x-np.min(x)).to(u.year).value/period.value

sample_orbit=xymass.sample_orbit_2body(f_period,period=period,eccentricity=eccentricity,mass_primary=mass_primary,mass_ratio=mass_ratio,longitude=longitude,inclination=inclination)

if case1==True:
    vlos_model=vcm+sample_orbit.v1_obs_xyz.T[2].value*4.74047
else:
    vlos_model=vcm+sample_orbit.v2_obs_xyz.T[2].value*4.74047        

plt.plot(x,vlos_model,color='r')
    
plt.errorbar(hjd,vlos,yerr=vlos_error,fmt='.',color='k')
plt.xlabel('HJD [days]')
plt.ylabel(r'$v_{\rm LOS}$ [km/s]')
#plt.ylim([-150,-100])
plt.savefig('tuc2_binary.pdf',dpi=200)
plt.show()
plt.close()

plt.hist(result['samples'].T[4],bins=30)
plt.xlabel(r'$\log_{10}[M_{\rm tot}/M_{\odot}]$')
plt.ylabel('posterior prob.')
plt.savefig('tuc2_binary_mass.pdf',dpi=200)
plt.show()
plt.close()


