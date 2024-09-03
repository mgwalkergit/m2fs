import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import StdDevUncertainty
from astropy.nddata import CCDData
import dustmaps.sfd
import astropy.units as u
from astropy.modeling import models
import os
from os import path
import scipy
import mycode as mc
import m2fs_process as m2fs
import crossmatcher
#matplotlib.use('TkAgg')
import dill as pickle
from pymultinest.solve import solve
from isochrones.mist import MIST_Isochrone
from isochrones.mist import MIST_EvolutionTrack
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from dynesty import DynamicNestedSampler,NestedSampler

imf_alpha1=0.3
imf_alpha2=1.3
imf_alpha3=2.3
imf_break1=0.08
imf_break2=0.5
imf_min_mass=0.03
imf_max_mass=np.inf
imf_part1=(imf_break1**(1.-imf_alpha1)-imf_min_mass**(1.-imf_alpha1))/(1.-imf_alpha1)
imf_part2=imf_break1**(imf_alpha2-imf_alpha1)*(imf_break2**(1.-imf_alpha2)-imf_break1**(1.-imf_alpha2))/(1.-imf_alpha2)
imf_part3=imf_break1**(imf_alpha2-imf_alpha1)*imf_break2**(imf_alpha3-imf_alpha2)*(imf_max_mass**(1.-imf_alpha3)-imf_break2**(1.-imf_alpha3))/(1.-imf_alpha3)
imf_const=imf_part1+imf_part2+imf_part3

mass=mc.imf_sample(size=100000,model='kroupa',alpha1=0.3,alpha2=1.3,alpha3=2.3,break1=0.08,break2=0.5,min_mass=0.05,max_mass=10.)
plt.hist(np.log10(mass),bins=100,density=True,color='b',alpha=0.2)
plt.show()
np.pause()
mass=[]
for i in range(0,1000000):
    mass.append(mc.imf_sample(model='kroupa',alpha1=0.3,alpha2=1.3,alpha3=2.3,break1=0.08,break2=0.5,min_mass=0.03,max_mass=10.))
mass=np.array(mass)

np.pause()
#matplotlib.use('pdf')

mist=MIST_Isochrone()
mist_track=MIST_EvolutionTrack()
bc_grid=MISTBolometricCorrectionGrid(['G','BP','RP','DECam_g','DECam_r','DECam_i','DECam_z','DECam_Y','SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z','PS_g','PS_r','PS_i','PS_z'])

data_directory='/hildafs/projects/phy200028p/mgwalker/m2fs_data/'
chains_directory='/hildafs/projects/phy200028p/mgwalker/m2fs_chains/'
fit_directory='/hildafs/projects/phy200028p/mgwalker/m2fs_chains/'

fits_list0='all_m2fshiresian_files'
fits_list='/hildafs/projects/phy200028p/mgwalker/m2fs/'+fits_list0
fits_table_filename=fits_list.split('_files')[0]+'.fits'
data_out=fits_list.split('_files')[0]+'.dat'
data_filename='/hildafs/projects/phy200028p/mgwalker/MultiNest_v2.17/'+fits_list0.split('_files')[0]+'_data'
chains_filename='/hildafs/projects/phy200028p/mgwalker/MultiNest_v2.17/'+fits_list0.split('_files')[0]+'_chains'

with open(fits_list) as f:
    data=f.readlines()
fitsfile=[]
obj=[]
for line in data:
    p=line.split()
    fitsfile.append(p[0])
    obj.append(p[1])
fitsfile=np.array(fitsfile)
obj=np.array(obj)

for i in range(0,len(fitsfile)):
    hdul=fits.open(fitsfile[i])
    fitsobject=m2fs.m2fs_getfromfits(hdul)
    hdul.close()

    ebv=[]
    for j in range(0,len(fitsobject.radeg)):
        if fitsobject.radeg[j]>-998.:
            ebv.append(dustmaps.sfd.SFDQuery().query_equ(fitsobject.radeg[j],fitsobject.decdeg[j]))
        else:
            ebv.append(-999.)
    ebv=np.array(ebv)

    gaia_objid,gaia_gmag,gaia_bpmag,gaia_rpmag,gaia_gflux,gaia_bpflux,gaia_rpflux,gaia_siggflux,gaia_sigbpflux,gaia_sigrpflux,parallax=crossmatcher.doit('gaia_edr3.gaia_source',fitsobject.radeg,fitsobject.decdeg,'source_id,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,phot_g_mean_flux,phot_bp_mean_flux,phot_rp_mean_flux,phot_g_mean_flux_error,phot_bp_mean_flux_error,phot_rp_mean_flux_error,parallax',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')
    des_gmag,des_siggmag,des_rmag,des_sigrmag,des_imag,des_sigimag,des_zmag,des_sigzmag,des_ymag,des_sigymag,des_ebv,des_gspread,des_rspread,des_ispread,des_zspread,des_yspread=crossmatcher.doit('des_dr2.main',fitsobject.radeg,fitsobject.decdeg,'wavg_mag_psf_g,wavg_magerr_psf_g,wavg_mag_psf_r,wavg_magerr_psf_r,wavg_mag_psf_i,wavg_magerr_psf_i,wavg_mag_psf_z,wavg_magerr_psf_z,wavg_mag_psf_y,wavg_magerr_psf_y,ebv_sfd98,wavg_spread_model_g,wavg_spread_model_r,wavg_spread_model_i,wavg_spread_model_z,wavg_spread_model_y',extra='wavg_magerr_psf_g>=0.',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')
    decals_gflux,decals_ivargflux,decals_rflux,decals_ivarrflux,decals_zflux,decals_ivarzflux,decals_type=crossmatcher.doit('decals_dr9.main',fitsobject.radeg,fitsobject.decdeg,'flux_g,flux_ivar_g,flux_r,flux_ivar_r,flux_z,flux_ivar_z,type',extra='flux_g>=0',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')
    decals_gmag=22.5-2.5*np.log10(decals_gflux)
    decals_rmag=22.5-2.5*np.log10(decals_rflux)
    decals_zmag=22.5-2.5*np.log10(decals_zflux)
    decals_siggmag=2.5/decals_gflux/np.log(10.)/np.sqrt(decals_ivargflux)
    decals_sigrmag=2.5/decals_rflux/np.log(10.)/np.sqrt(decals_ivarrflux)
    decals_sigzmag=2.5/decals_zflux/np.log(10.)/np.sqrt(decals_ivarzflux)    
    ps1_objid,ps1_gmag,ps1_rmag,ps1_imag,ps1_zmag,ps1_siggmag,ps1_sigrmag,ps1_sigimag,ps1_sigzmag,ps1_ebv,ps1_rkronmag=crossmatcher.doit('panstarrs_dr1.stackobjectthin',fitsobject.radeg,fitsobject.decdeg,'objid,gpsfmag,rpsfmag,ipsfmag,zpsfmag,gpsfmagerr,rpsfmagerr,ipsfmagerr,zpsfmagerr,ebv,rkronmag',extra='(rinfoflag3& panstarrs_dr1.detectionflags3(\'STACK_PRIMARY\'))>0 and gpsfmag=gpsfmag',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')
    sdss_objid,sdss_umag,sdss_gmag,sdss_rmag,sdss_imag,sdss_zmag,sdss_sigumag,sdss_siggmag,sdss_sigrmag,sdss_sigimag,sdss_sigzmag,sdss_extinction_u,sdss_extinction_g,sdss_extinction_r,sdss_extinction_i,sdss_extinction_z,sdss_type,sdss_mode=crossmatcher.doit('sdssdr9.phototag',fitsobject.radeg,fitsobject.decdeg,'objid,psfmag_u,psfmag_g,psfmag_r,psfmag_i,psfmag_z,psfmagerr_u,psfmagerr_g,psfmagerr_r,psfmagerr_i,psfmagerr_z,extinction_u,extinction_g,extinction_r,extinction_i,extinction_z,type,mode',extra='psfmag_g>-9998.',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')

    sdss_uext=ebv*4.239#de-reddened mag will be the psfmag minus this value
    sdss_gext=ebv*3.303
    sdss_rext=ebv*2.285
    sdss_iext=ebv*1.698
    sdss_zext=ebv*1.263
    sdss=m2fs.survey_phot_sdss(sdss_objid,sdss_umag,sdss_gmag,sdss_rmag,sdss_imag,sdss_zmag,sdss_sigumag,sdss_siggmag,sdss_sigrmag,sdss_sigimag,sdss_sigzmag,sdss_type,sdss_mode,ebv,sdss_uext,sdss_gext,sdss_rext,sdss_iext,sdss_zext)
    
    des_gext=ebv*3.237#3.214
    des_rext=ebv*2.176#2.165
    des_iext=ebv*1.595#1.592
    des_zext=ebv*1.217#1.211
    des_yext=ebv*1.058#1.064#de-reddened mag will be the psfmag minus this value
    des=m2fs.survey_phot_des(des_gmag,des_rmag,des_imag,des_zmag,des_ymag,des_siggmag,des_sigrmag,des_sigimag,des_sigzmag,des_sigymag,des_gspread,des_rspread,des_ispread,des_zspread,des_yspread,ebv,des_gext,des_rext,des_iext,des_zext,des_yext)

    decals_gext=ebv*3.237#3.214
    decals_rext=ebv*2.176#2.165
    decals_zext=ebv*1.217#1.211
    decals=m2fs.survey_phot_decals(decals_gmag,decals_rmag,decals_zmag,decals_siggmag,decals_sigrmag,decals_sigzmag,decals_type,ebv,decals_gext,decals_rext,decals_zext)
    
    ps1_gext=ebv*3.172
    ps1_rext=ebv*2.271
    ps1_iext=ebv*1.682
    ps1_zext=ebv*1.322
    ps1=m2fs.survey_phot_ps1(ps1_objid,ps1_gmag,ps1_rmag,ps1_imag,ps1_zmag,ps1_siggmag,ps1_sigrmag,ps1_sigimag,ps1_sigzmag,ps1_rkronmag,ebv,ps1_gext,ps1_rext,ps1_iext,ps1_zext)

    gaia_siggmag=(2.5/gaia_gflux/2.30258)**2*gaia_siggflux**2
    gaia_sigbpmag=(2.5/gaia_bpflux/2.30258)**2*gaia_sigbpflux**2
    gaia_sigrpmag=(2.5/gaia_rpflux/2.30258)**2*gaia_sigrpflux**2
    a0=3.1*ebv#use Eq. 1 from Babusiaux et al 2018 (1804.09378) to get extinction coefficients, apply extinction to update BP-RP, then interate again to get final estimate of extinction coefficient
    kg_c=np.array([0.9761,-0.1704,0.0086,0.0011,-0.0438,0.0013,0.0099])
    kbp_c=np.array([1.1517,-0.0871,-0.0333,0.0173,-0.0230,0.0006,0.0043])
    krp_c=np.array([0.6104,-0.0170,-0.0026,-0.0017,-0.0078,0.00005,0.0006])
    col0=gaia_bpmag-gaia_rpmag
    kg=kg_c[0]+kg_c[1]*col0+kg_c[2]*col0**2+kg_c[3]*col0**3+kg_c[4]*a0+kg_c[5]*a0**2+kg_c[6]*col0*a0
    kbp=kbp_c[0]+kbp_c[1]*col0+kbp_c[2]*col0**2+kbp_c[3]*col0**3+kbp_c[4]*a0+kbp_c[5]*a0**2+kbp_c[6]*col0*a0
    krp=krp_c[0]+krp_c[1]*col0+krp_c[2]*col0**2+krp_c[3]*col0**3+krp_c[4]*a0+krp_c[5]*a0**2+krp_c[6]*col0*a0
    ag=kg*a0
    abp=kbp*a0
    arp=krp*a0
    col1=gaia_bpmag-abp-(gaia_rpmag-arp)
    kg=kg_c[0]+kg_c[1]*col1+kg_c[2]*col1**2+kg_c[3]*col1**3+kg_c[4]*a0+kg_c[5]*a0**2+kg_c[6]*col1*a0
    kbp=kbp_c[0]+kbp_c[1]*col1+kbp_c[2]*col1**2+kbp_c[3]*col1**3+kbp_c[4]*a0+kbp_c[5]*a0**2+kbp_c[6]*col1*a0
    krp=krp_c[0]+krp_c[1]*col1+krp_c[2]*col1**2+krp_c[3]*col1**3+krp_c[4]*a0+krp_c[5]*a0**2+krp_c[6]*col1*a0
    gaia_ag=kg*a0
    gaia_abp=kbp*a0
    gaia_arp=krp*a0
    gaia_gmag_dered=gaia_gmag-gaia_ag
    gaia_bpmag_dered=gaia_bpmag-gaia_abp
    gaia_rpmag_dered=gaia_rpmag-gaia_arp

    gaia=m2fs.survey_phot_gaia(gaia_objid,gaia_gmag,gaia_bpmag,gaia_rpmag,gaia_siggmag,gaia_sigbpmag,gaia_sigrpmag,parallax,ebv,gaia_ag,gaia_abp,gaia_arp)
    survey_phot=m2fs.survey_phot(gaia,des,decals,sdss,ps1)

    for j in range(0,len(survey_phot.gaia.gmag)):
        mags=np.array([survey_phot.gaia.gmag[j],survey_phot.gaia.bpmag[j],survey_phot.gaia.rpmag[j],survey_phot.des.gmag[j],survey_phot.des.rmag[j],survey_phot.des.imag[j],survey_phot.des.zmag[j],survey_phot.des.ymag[j],survey_phot.decals.gmag[j],survey_phot.decals.rmag[j],survey_phot.decals.zmag[j],survey_phot.sdss.umag[j],survey_phot.sdss.gmag[j],survey_phot.sdss.rmag[j],survey_phot.sdss.imag[j],survey_phot.sdss.zmag[j],survey_phot.ps1.gmag[j],survey_phot.ps1.rmag[j],survey_phot.ps1.imag[j],survey_phot.ps1.zmag[j]])
        sigmags=np.array([survey_phot.gaia.siggmag[j],survey_phot.gaia.sigbpmag[j],survey_phot.gaia.sigrpmag[j],survey_phot.des.siggmag[j],survey_phot.des.sigrmag[j],survey_phot.des.sigimag[j],survey_phot.des.sigzmag[j],survey_phot.des.sigymag[j],survey_phot.decals.siggmag[j],survey_phot.decals.sigrmag[j],survey_phot.decals.sigzmag[j],survey_phot.sdss.sigumag[j],survey_phot.sdss.siggmag[j],survey_phot.sdss.sigrmag[j],survey_phot.sdss.sigimag[j],survey_phot.sdss.sigzmag[j],survey_phot.ps1.siggmag[j],survey_phot.ps1.sigrmag[j],survey_phot.ps1.sigimag[j],survey_phot.ps1.sigzmag[j]])
        keep=np.where((mags==mags)&(sigmags==sigmags))[0]
        if len(keep)>=1.:
            plt.scatter(gaia_bpmag_dered-gaia_rpmag_dered,gaia_gmag_dered,s=3,color='k')
            plt.scatter(gaia_bpmag_dered[j]-gaia_rpmag_dered[j],gaia_gmag_dered[j],s=10,color='r')
            def myprior(cube):
                from scipy.special import erfinv
                from statistics import NormalDist
                prior=[]
                prior.append([0.,0.])#stellar mass
                prior.append([200,808])#EEP
                prior.append([-4.,+0.5])#[Fe/H]
                prior.append([0.,300.])# distance / kpc
                prior.append([0.,0.])# reddening Av
                prior=np.array(prior)
                x=np.array(cube)
                #draw stellar mass from Kroupa IMF
                if cube[0]<=imf_part1/imf_const:
                    x[0]=((1.-imf_alpha1)*imf_const*cube[0]+imf_min_mass**(1.-imf_alpha1))**(1./(1.-imf_alpha1))
                elif ((cube[0]>imf_part1/imf_const)&(cube[0]<=(imf_part1+imf_part2)/imf_const)):
                    x[0]=((1.-imf_alpha2)*(imf_const*cube[0]-imf_part1)*imf_break1**(imf_alpha1-imf_alpha2)+imf_break1**(1.-imf_alpha2))**(1./(1.-imf_alpha2))
                elif cube[0]>(imf_part1+imf_part2)/imf_const:
                    x[0]=((1.-imf_alpha3)*(imf_const*cube[0]-imf_part1-imf_part2)*imf_break1**(imf_alpha1-imf_alpha2)*imf_break2**(imf_alpha2-imf_alpha3)+imf_break2**(1.-imf_alpha3))**(1./(1.-imf_alpha3))
                else:
                    raise ValueError('something wrong in sampling Kroupa IMF')
                #Gaussian prior on A_V, centered on dustmap value, with sigma = 15% of dustmap value
                av_mean=ebv[j]*2.742#reddening at R_V=3.1 value from Schlafly & Finkbeiner 2011
                av_sigma=0.15*av_mean
                x[4]=np.max(np.array([NormalDist(mu=av_mean,sigma=av_sigma).inv_cdf(cube[4])]))
                #for the rest of parameters, use flat priors
                for i in range(1,len(x)-1):
                    x[i]=prior[i][0]+(prior[i][1]-prior[i][0])*cube[i]
                return x
            
            def myloglike(cube):
                mass=cube[0]
                eep=cube[1]
                feh=cube[2]
                distance=cube[3]*1000.
                av=cube[4]
                pars=[mass,eep,feh]
                Mbol,teff,logg,feh=mist_track.interp_value(pars,['Mbol','Teff','logg','feh'])
                bolometric_correction=bc_grid.interp([teff,logg,feh,av],['G','BP','RP','DECam_g','DECam_r','DECam_i','DECam_z','DECam_Y','DECam_g','DECam_r','DECam_z','SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z','PS_g','PS_r','PS_i','PS_z'])
                dmodulus=5.*np.log10(distance)-5.
                absmags=Mbol-bolometric_correction
                iso_model_mags=absmags+dmodulus
                sum1=-0.5*np.log(2.*np.pi)*len(keep)
                sum2=np.sum(-0.5*np.log(sigmags[keep]**2))
                sum3=np.sum(-0.5*(mags[keep]-iso_model_mags[keep])**2/(sigmags[keep]**2))
                if sum1+sum2+sum3!=sum1+sum2+sum3:
                    return -1.e+30
                else:
                    return sum1+sum2+sum3
        
            parameters=['mass','eep','feh','distance','av']
            n_params=len(parameters)
            prefix="chains/poop-"
#            result=solve(LogLikelihood=myloglike,Prior=myprior,n_dims=n_params,outputfiles_basename=prefix,n_live_points=1000,verbose=True,resume=False)
            sampler=DynamicNestedSampler(myloglike,myprior,n_params,bound='multi')
#            sampler=NestedSampler(myloglike,myprior,n_params,bound='multi',nlive=400)
            sampler.run_nested(dlogz_init=0.5,nlive_init=500,wt_kwargs={'pfrac':1.0})#dlogz=0.5,maxiter=10000,maxcall=50000)
            res=sampler.results
            pickle.dump(res,open('dynesty.res','wb'))
            mass=result['samples'].T[0]
            eep=result['samples'].T[1]
            distance=result['samples'].T[3]
            teff=[]
            logg=[]
            feh=[]
            for j in range(0,len(result['samples'].T[0])):
                teff0,logg0,feh0=mist_track.interp_value([result['samples'].T[0][j],result['samples'].T[1][j],result['samples'].T[2][j]],['Teff','logg','feh'])
                teff.append(teff0)
                logg.append(logg0)
                feh.append(feh0)
            teff=np.array(teff)
            logg=np.array(logg)
            feh=np.array(feh)
            print(np.median(mass),np.std(mass))
            print(np.median(teff),np.std(teff))
            print(np.median(logg),np.std(logg))
            print(np.median(feh),np.std(feh))
            print(np.median(distance),np.std(distance))
            print(np.median(eep),np.std(eep))
#            pars[np.median(mass),np.median(]
#            shite=mist_track.interp_mag(pars,['G','BP','RP'])
            plt.ylim([23,16])
            plt.xlim([-1,3])
            plt.show()
            plt.close()

            
