import numpy as np
import astropy
import sqlutil
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import StdDevUncertainty
from astropy.nddata import CCDData
from astropy.coordinates import SkyCoord
import dustmaps.sfd
import astropy.units as u
from astropy.modeling import models
import os
import csv
from os import path
import scipy
import m2fs_process as m2fs
import mycode
import crossmatcher
matplotlib.use('TkAgg')
from matplotlib.patches import Ellipse
from copy import deepcopy
from specutils.spectra import Spectrum1D
import dill as pickle
from isochrones.mist import MIST_Isochrone
from isochrones.mist import MIST_EvolutionTrack
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from pymultinest.solve import solve
from pymultinest import Analyzer
#matplotlib.use('pdf')

get_surveyphot=False#pull photometry for each spectroscopic target from various sky surveys
get_obstable=False#write observation log in latex table
fit_continuum=False
make_skysub=False#generate 1d sky-subtracted spectra?
get_fits=True#fit is done, do post-processing of multinest output
get_catalog_raw=True#catalog post-processed data
get_errors=True#fit error models using repeat observations
get_catalog_final=True#update catalog for calibrated errors
check_badfield=False

data_directory='/hildafs/projects/phy200028p/mgwalker/m2fs_data/'
chains_directory='/hildafs/projects/phy200028p/mgwalker/m2fs_chains/'
fit_directory='/hildafs/projects/phy200028p/mgwalker/m2fs_chains/'
lambdamin=5127.
lambdamax=5190.
bluelambdamin=5160.
bluelambdamax=5167.
redlambdamin=5176.
redlambdamax=5183.

fits_list0='all_m2fs_files15'
fits_list='/hildafs/projects/phy200028p/mgwalker/m2fs/'+fits_list0
data_out=fits_list.split('_files')[0]+'.dat'
data_filename='/hildafs/projects/phy200028p/mgwalker/MultiNest_v2.17/'+fits_list0.split('_files')[0]+'_data'
chains_filename='/hildafs/projects/phy200028p/mgwalker/MultiNest_v2.17/'+fits_list0.split('_files')[0]+'_chains'

mist=MIST_Isochrone()
mist_track=MIST_EvolutionTrack()
bc_grid=MISTBolometricCorrectionGrid(['G','BP','RP','DECam_g','DECam_r','DECam_i','DECam_z','DECam_Y','SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z','PS_g','PS_r','PS_i','PS_z'])

des_est50,des_est16,des_est84=pickle.load(open('/hildafs/projects/phy200028p/mgwalker/gaia_dwarfs/photometry/des_ensemble.pickle','rb'))
decals_est50,decals_est16,decals_est84=pickle.load(open('/hildafs/projects/phy200028p/mgwalker/gaia_dwarfs/photometry/decals_ensemble.pickle','rb'))
gaia_est50,gaia_est16,gaia_est84=pickle.load(open('/hildafs/projects/phy200028p/mgwalker/gaia_dwarfs/photometry/gaia_ensemble.pickle','rb'))
ps1_est50,ps1_est16,ps1_est84=pickle.load(open('/hildafs/projects/phy200028p/mgwalker/gaia_dwarfs/photometry/ps1_ensemble.pickle','rb'))
sdss_est50,sdss_est16,sdss_est84=pickle.load(open('/hildafs/projects/phy200028p/mgwalker/gaia_dwarfs/photometry/sdss_ensemble.pickle','rb'))

thing=['vlos','teffprior_vlos','teff','teffprior_teff','logg','teffprior_logg','feh','teffprior_feh','mgfe','teffprior_mgfe','smooth','teffprior_smooth']
thing_symbol=[r'$v_{\rm los}$',r'$v_{\rm los}$',r'$T_{\rm eff}$',r'$T_{\rm eff}$',r'$\log_{10}[g]$',r'$\log_{10}[g]$',r'[Fe/H]',r'[Fe/H]',r'[Mg/Fe]',r'[Mg/Fe]',r'$\sigma_{\rm LSF}$',r'$\sigma_{\rm LSF}$']

class teffpriorstuff:
    def __init__(self,gaia_teffprior=None,des_teffprior=None,decals_teffprior=None,sdss_teffprior=None,ps1_teffprior=None):
        self.gaia_teffprior=gaia_teffprior
        self.des_teffprior=des_teffprior
        self.decals_teffprior=decals_teffprior
        self.sdss_teffprior=sdss_teffprior
        self.ps1_teffprior=ps1_teffprior

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
    this=np.where(fitsfile==fitsfile[i])[0]
    if len(this)>1:
        print('ERROR: multiple listings of same observation in input file!!!')
        print(this)
        np.pause()

if get_surveyphot:

    rrl_id=sqlutil.get('''select source_id from gaia_dr3.vari_rrlyrae''',db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')[0]#pull gaia source IDs for all stars in RRL catalog
    cepheid_id=sqlutil.get('''select source_id from gaia_dr3.vari_cepheid''',db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')[0]#pull gaia source IDs for all stars in RRL catalog
    agn_id=sqlutil.get('''select source_id from gaia_dr3.vari_agn''',db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')[0]#pull gaia source IDs for all stars in RRL catalog
    rvvariable_id=sqlutil.get('''select source_id from gaia_dr3.vari_epoch_radial_velocity''',db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')[0]#pull gaia source IDs for all stars in RRL catalog
    compact_id=sqlutil.get('''select source_id from gaia_dr3.vari_compact_companion''',db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')[0]#pull gaia source IDs for all stars in RRL catalog

    for i in range(0,len(fitsfile)):
        print(i,' of ',len(fitsfile))
        hdul=fits.open(fitsfile[i])
        fitsobject=m2fs.m2fs_getfromfits(hdul)
        filtername=hdul[0].header['filtername']
        m2fsrun=hdul[0].header['m2fsrun']
        field_name=hdul[0].header['field_name']
        temperature=hdul[0].header['CCD_temp']
        hdul.close()
        surveyphot_filename=fitsfile[i].split('_stackskysub.fits')[0]+'_surveyphot.pkl'
    
        ebv=[]
        for j in range(0,len(fitsobject.radeg)):
            if fitsobject.radeg[j]>-998.:
                ebv.append(dustmaps.sfd.SFDQuery().query_equ(fitsobject.radeg[j],fitsobject.decdeg[j]))
            else:
                ebv.append(-999.)
        ebv=np.array(ebv)

        gaia_objid,gaia_ra,gaia_dec,gaia_gmag,gaia_bpmag,gaia_rpmag,gaia_gflux,gaia_bpflux,gaia_rpflux,gaia_siggflux,gaia_sigbpflux,gaia_sigrpflux,gaia_parallax,gaia_sigparallax,gaia_pmra,gaia_sigpmra,gaia_pmdec,gaia_sigpmdec,gaia_rv,gaia_sigrv,gaia_phot_variable,gaia_in_qso,gaia_in_galaxy,gaia_non_single_star,gaia_classprob_quasar,gaia_classprob_galaxy,gaia_classprob_star,gaia_teff,gaia_teff_lower,gaia_teff_upper,gaia_logg,gaia_logg_lower,gaia_logg_upper,gaia_feh,gaia_feh_lower,gaia_feh_upper,gaia_libname=crossmatcher.doit('gaia_dr3.gaia_source',fitsobject.radeg,fitsobject.decdeg,'source_id,ra,dec,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,phot_g_mean_flux,phot_bp_mean_flux,phot_rp_mean_flux,phot_g_mean_flux_error,phot_bp_mean_flux_error,phot_rp_mean_flux_error,parallax,parallax_error,pmra,pmra_error,pmdec,pmdec_error,radial_velocity,radial_velocity_error,phot_variable_flag,in_qso_candidates,in_galaxy_candidates,non_single_star,classprob_dsc_combmod_quasar,classprob_dsc_combmod_galaxy,classprob_dsc_combmod_star,teff_gspphot,teff_gspphot_lower,teff_gspphot_upper,logg_gspphot,logg_gspphot_lower,logg_gspphot_upper,mh_gspphot,mh_gspphot_lower,mh_gspphot_upper,libname_gspphot',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')

        gaia_rrl=np.zeros(len(gaia_objid),dtype='int')
        gaia_cepheid=np.zeros(len(gaia_objid),dtype='int')
        gaia_agn=np.zeros(len(gaia_objid),dtype='int')
        gaia_rvvariable=np.zeros(len(gaia_objid),dtype='int')
        gaia_compact=np.zeros(len(gaia_objid),dtype='int')
        for j in range(0,len(gaia_objid)):
            this=np.where(rrl_id==gaia_objid[j])[0]
            if len(this)>0:
                gaia_rrl[j]=1
            this=np.where(cepheid_id==gaia_objid[j])[0]
            if len(this)>0:
                gaia_cepheid[j]=1
            this=np.where(agn_id==gaia_objid[j])[0]
            if len(this)>0:
                gaia_agn[j]=1
            this=np.where(rvvariable_id==gaia_objid[j])[0]
            if len(this)>0:
                gaia_rvvariable[j]=1
            this=np.where(compact_id==gaia_objid[j])[0]
            if len(this)>0:
                gaia_compact[j]=1
        des_ra,des_dec,des_gmag,des_siggmag,des_rmag,des_sigrmag,des_imag,des_sigimag,des_zmag,des_sigzmag,des_ymag,des_sigymag,des_ebv,des_gspread,des_rspread,des_ispread,des_zspread,des_yspread=crossmatcher.doit('des_dr2.main',fitsobject.radeg,fitsobject.decdeg,'ra,dec,wavg_mag_psf_g,wavg_magerr_psf_g,wavg_mag_psf_r,wavg_magerr_psf_r,wavg_mag_psf_i,wavg_magerr_psf_i,wavg_mag_psf_z,wavg_magerr_psf_z,wavg_mag_psf_y,wavg_magerr_psf_y,ebv_sfd98,wavg_spread_model_g,wavg_spread_model_r,wavg_spread_model_i,wavg_spread_model_z,wavg_spread_model_y',extra='wavg_magerr_psf_g>=0.',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')
        decals_ra,decals_dec,decals_gflux,decals_ivargflux,decals_rflux,decals_ivarrflux,decals_zflux,decals_ivarzflux,decals_type=crossmatcher.doit('decals_dr9.main',fitsobject.radeg,fitsobject.decdeg,'ra,dec,flux_g,flux_ivar_g,flux_r,flux_ivar_r,flux_z,flux_ivar_z,type',extra='flux_g>=0',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')
        decals_gmag=22.5-2.5*np.log10(decals_gflux)
        decals_rmag=22.5-2.5*np.log10(decals_rflux)
        decals_zmag=22.5-2.5*np.log10(decals_zflux)
        decals_siggmag=2.5/decals_gflux/np.log(10.)/np.sqrt(decals_ivargflux)
        decals_sigrmag=2.5/decals_rflux/np.log(10.)/np.sqrt(decals_ivarrflux)
        decals_sigzmag=2.5/decals_zflux/np.log(10.)/np.sqrt(decals_ivarzflux)    
        ps1_ra,ps1_dec,ps1_objid,ps1_gmag,ps1_rmag,ps1_imag,ps1_zmag,ps1_siggmag,ps1_sigrmag,ps1_sigimag,ps1_sigzmag,ps1_ebv,ps1_rkronmag=crossmatcher.doit('panstarrs_dr1.stackobjectthin',fitsobject.radeg,fitsobject.decdeg,'ra,dec,objid,gpsfmag,rpsfmag,ipsfmag,zpsfmag,gpsfmagerr,rpsfmagerr,ipsfmagerr,zpsfmagerr,ebv,rkronmag',extra='(rinfoflag3& panstarrs_dr1.detectionflags3(\'STACK_PRIMARY\'))>0 and gpsfmag=gpsfmag',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')
        sdss_ra,sdss_dec,sdss_objid,sdss_umag,sdss_gmag,sdss_rmag,sdss_imag,sdss_zmag,sdss_sigumag,sdss_siggmag,sdss_sigrmag,sdss_sigimag,sdss_sigzmag,sdss_extinction_u,sdss_extinction_g,sdss_extinction_r,sdss_extinction_i,sdss_extinction_z,sdss_type,sdss_mode=crossmatcher.doit('sdssdr9.phototag',fitsobject.radeg,fitsobject.decdeg,'ra,dec,objid,psfmag_u,psfmag_g,psfmag_r,psfmag_i,psfmag_z,psfmagerr_u,psfmagerr_g,psfmagerr_r,psfmagerr_i,psfmagerr_z,extinction_u,extinction_g,extinction_r,extinction_i,extinction_z,type,mode',extra='psfmag_g>-9998.',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')

        sdss_uext=ebv*4.239#de-reddened mag will be the psfmag minus this value
        sdss_gext=ebv*3.303
        sdss_rext=ebv*2.285
        sdss_iext=ebv*1.698
        sdss_zext=ebv*1.263
        sdss=m2fs.survey_phot_sdss(sdss_ra,sdss_dec,sdss_objid,sdss_umag,sdss_gmag,sdss_rmag,sdss_imag,sdss_zmag,sdss_sigumag,sdss_siggmag,sdss_sigrmag,sdss_sigimag,sdss_sigzmag,sdss_type,sdss_mode,ebv,sdss_uext,sdss_gext,sdss_rext,sdss_iext,sdss_zext)
    
        des_gext=ebv*3.237#3.214
        des_rext=ebv*2.176#2.165
        des_iext=ebv*1.595#1.592
        des_zext=ebv*1.217#1.211
        des_yext=ebv*1.058#1.064#de-reddened mag will be the psfmag minus this value
        des=m2fs.survey_phot_des(des_ra,des_dec,des_gmag,des_rmag,des_imag,des_zmag,des_ymag,des_siggmag,des_sigrmag,des_sigimag,des_sigzmag,des_sigymag,des_gspread,des_rspread,des_ispread,des_zspread,des_yspread,ebv,des_gext,des_rext,des_iext,des_zext,des_yext)
        
        decals_gext=ebv*3.237#3.214
        decals_rext=ebv*2.176#2.165
        decals_zext=ebv*1.217#1.211
        decals=m2fs.survey_phot_decals(decals_ra,decals_dec,decals_gmag,decals_rmag,decals_zmag,decals_siggmag,decals_sigrmag,decals_sigzmag,decals_type,ebv,decals_gext,decals_rext,decals_zext)
    
        ps1_gext=ebv*3.172
        ps1_rext=ebv*2.271
        ps1_iext=ebv*1.682
        ps1_zext=ebv*1.322
        ps1=m2fs.survey_phot_ps1(ps1_ra,ps1_dec,ps1_objid,ps1_gmag,ps1_rmag,ps1_imag,ps1_zmag,ps1_siggmag,ps1_sigrmag,ps1_sigimag,ps1_sigzmag,ps1_rkronmag,ebv,ps1_gext,ps1_rext,ps1_iext,ps1_zext)

        gaia_siggmag=(2.5/gaia_gflux/2.30258)**2*gaia_siggflux**2
        gaia_sigbpmag=(2.5/gaia_bpflux/2.30258)**2*gaia_sigbpflux**2
        gaia_sigrpmag=(2.5/gaia_rpflux/2.30258)**2*gaia_sigrpflux**2
        a0=3.1*ebv*0.86#use Eq. 1 from Babusiaux et al 2018 (1804.09378) to get extinction coefficients, apply extinction to update BP-RP, then interate again to get final estimate of extinction coefficient.  The factor of 0.86 comes from Sergey, as explained to me in email from Andrew on July 11, 2022: "I've recently realized that Babusieux's formula was written to be applied to "True" EBV. But the SFD EBV is overestimated according to Schlafly11. So if you use SFD EBV's for the getDust, then the ebv needs to be multiplied by 0.86. (when you work with other surveys it's usually not a problem, because the coefficients are already given wrt SFD, but that I learned doesn't apply for gaia's code."
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
        
        gaia=m2fs.survey_phot_gaia(gaia_objid,gaia_ra,gaia_dec,gaia_gmag,gaia_bpmag,gaia_rpmag,gaia_siggmag,gaia_sigbpmag,gaia_sigrpmag,ebv,gaia_ag,gaia_abp,gaia_arp,gaia_pmra,gaia_sigpmra,gaia_pmdec,gaia_sigpmdec,gaia_parallax,gaia_sigparallax,gaia_rrl,gaia_cepheid,gaia_agn,gaia_rvvariable,gaia_compact,gaia_rv,gaia_sigrv,gaia_phot_variable,gaia_in_qso,gaia_in_galaxy,gaia_non_single_star,gaia_classprob_quasar,gaia_classprob_galaxy,gaia_classprob_star,gaia_teff,gaia_teff_lower,gaia_teff_upper,gaia_logg,gaia_logg_lower,gaia_logg_upper,gaia_feh,gaia_feh_lower,gaia_feh_upper,gaia_libname)
        survey_phot=m2fs.survey_phot(gaia,des,decals,sdss,ps1)
        print(i,surveyphot_filename)
        pickle.dump(survey_phot,open(surveyphot_filename,'wb'))

if get_obstable:

    field_center0=[]
    field_center=[]
    field_instrument=[]
    field_id=[]
    field_utdate=[]
    field_uttime=[]
    field_exptime=[]
    field_object=[]
    field_ntarget=[]
    field_nsky=[]
    field_subframes=[]

    exptime_total=0.
    nfields_total=0
    nfields_hires=0
    nfields_medres=0
    nfields_mix=0
    nnights=0
    
    for i in range(0,len(fitsfile)):
        print(i,' of ',len(fitsfile))
        hdul=fits.open(fitsfile[i])
        fitsobject=m2fs.m2fs_getfromfits(hdul)
        filtername=hdul[0].header['filtername']
        m2fsrun=hdul[0].header['m2fsrun']
        field_name=hdul[0].header['field_name']
        temperature=hdul[0].header['CCD_temp']
        filtername=hdul[0].header['filtername']
        hdul.close()
        surveyphot_filename=fitsfile[i].split('_stackskysub.fits')[0]+'_surveyphot.pkl'
    
        skies=np.where(fitsobject.obj=='SKY')[0]
        targets=np.where(fitsobject.icode>0)[0]
        if (('HiRes' in filtername)&(len(skies)>0)):
            field_center.append(SkyCoord(hdul[0].header['RA']+hdul[0].header['DEC'],unit=(u.hourangle,u.deg)))
            field_id.append(hdul[0].header['field_name'])
            field_utdate.append(hdul[0].header['UT-DATE'])
            field_uttime.append(hdul[0].header['UT-TIME'])
            field_exptime.append(hdul[0].header['exp_times'])
            ttt=hdul[0].header['exp_times'].split(',')
#            for j in ttt:
#                exptime_total+=np.float(j)
            field_object.append(hdul[0].header['object'])
            field_ntarget.append(len(targets))
            field_nsky.append(len(skies))
            field_subframes.append(hdul[0].header['sub_frames'])
            field_instrument.append('M2FS HiRes')

        elif (('MedRes' in filtername)&(len(skies)>0)):
            field_center.append(SkyCoord(hdul[0].header['RA']+hdul[0].header['DEC'],unit=(u.hourangle,u.deg)))
            field_id.append(hdul[0].header['field_name'])
            field_utdate.append(hdul[0].header['UT-DATE'])
            field_uttime.append(hdul[0].header['UT-TIME'])
            field_exptime.append(hdul[0].header['exp_times'])
            ttt=hdul[0].header['exp_times'].split(',')
#            for j in ttt:
#                exptime_total+=np.float(j)
            field_object.append(hdul[0].header['object'])
            field_ntarget.append(len(targets))
            field_nsky.append(len(skies))
            field_subframes.append(hdul[0].header['sub_frames'])
            field_instrument.append('M2FS MedRes')
        else:
            if len(skies)>0:
                print('ERROR: unidentified filtername')
                np.pause()

    field_center=np.array(field_center)
    field_id=np.array(field_id)
    field_utdate=np.array(field_utdate)
    field_uttime=np.array(field_uttime)
    field_exptime=np.array(field_exptime)
    field_object=np.array(field_object)
    field_ntarget=np.array(field_ntarget)
    field_nsky=np.array(field_nsky)
    field_subframes=np.array(field_subframes)
    field_instrument=np.array(field_instrument)

    order=np.lexsort((field_uttime,field_utdate))
    field_center=field_center[order]
    field_id=field_id[order]
    field_utdate=field_utdate[order]
    field_uttime=field_uttime[order]
    field_exptime=field_exptime[order]
    field_object=field_object[order]
    field_ntarget=field_ntarget[order]
    field_nsky=field_nsky[order]
    field_subframes=field_subframes[order]
    field_instrument=field_instrument[order]

    field_object_formal=np.empty(len(field_object),dtype='object')
    dsph=np.where(field_object=='ant2')[0]
    field_object_formal[dsph]='Antlia II'
    dsph=np.where(field_object=='car')[0]
    field_object_formal[dsph]='Carina'
    dsph=np.where(field_object=='cra')[0]
    field_object_formal[dsph]='Crater'
    dsph=np.where(field_object=='cra2')[0]
    field_object_formal[dsph]='Crater II'
    dsph=np.where(field_object=='for')[0]
    field_object_formal[dsph]='Fornax'
    dsph=np.where(field_object=='gru1')[0]
    field_object_formal[dsph]='Grus I'
    dsph=np.where(field_object=='gru2')[0]
    field_object_formal[dsph]='Grus II'
    dsph=np.where(field_object=='hor2')[0]
    field_object_formal[dsph]='Horologium II'
    dsph=np.where(field_object=='hyd1')[0]
    field_object_formal[dsph]='Hyd 1'
    dsph=np.where(field_object=='ind1')[0]
    field_object_formal[dsph]='Indus I'
    dsph=np.where(field_object=='ind2')[0]
    field_object_formal[dsph]='Indus II'
    dsph=np.where(field_object=='kgo2')[0]
    field_object_formal[dsph]='Gran 3'
    dsph=np.where(field_object=='kgo4')[0]
    field_object_formal[dsph]='Gran 4'
    dsph=np.where(field_object=='kgo7')[0]
    field_object_formal[dsph]='Gaia 9'
    dsph=np.where(field_object=='kgo8')[0]
    field_object_formal[dsph]='Gaia 11'
    dsph=np.where(field_object=='kgo10')[0]
    field_object_formal[dsph]='Gaia 10'
    dsph=np.where(field_object=='kgo13')[0]
    field_object_formal[dsph]='KGO 13'
    dsph=np.where(field_object=='kgo22')[0]
    field_object_formal[dsph]='Garro-01'
    dsph=np.where(field_object=='kop2')[0]
    field_object_formal[dsph]='Koposov 2'    
    dsph=np.where(field_object=='pal5')[0]
    field_object_formal[dsph]='Palomar 5'
    dsph=np.where(field_object=='pho2')[0]
    field_object_formal[dsph]='Phoenix II'
    dsph=np.where(field_object=='ret2')[0]
    field_object_formal[dsph]='Reticulum II'
    dsph=np.where(field_object=='sgr2')[0]
    field_object_formal[dsph]='Sagittarius 2'
    dsph=np.where(field_object=='scl')[0]
    field_object_formal[dsph]='Sculptor'
    dsph=np.where(field_object=='sex')[0]
    field_object_formal[dsph]='Sextans'
    dsph=np.where(field_object=='tuc2')[0]
    field_object_formal[dsph]='Tucana II'
    dsph=np.where(field_object=='tuc3')[0]
    field_object_formal[dsph]='Tucana III'
    dsph=np.where(field_object=='tuc4')[0]
    field_object_formal[dsph]='Tucana IV'
    dsph=np.where(field_object=='tuc5')[0]
    field_object_formal[dsph]='Tucana V'

    used=np.zeros(len(field_utdate),dtype='int')
    for i in range(0,len(field_utdate)):
        if used[i]==0:
            nnights+=1
            this=np.where(field_utdate==field_utdate[i])[0]
            used[this]=1
        
    out_table=open('m2fs_obs_table.tex','w')
    out_table_short=open('m2fs_obs_table_short.tex','w')
    used=np.zeros(len(field_center),dtype='long')
    fudge_uttime=SkyCoord(field_uttime,np.full(len(field_uttime),'+00:00:00'),unit=(u.hourangle,u.deg)).ra.hour

    out_table.write(r'\begin{deluxetable*}{lllllcccl}'+' \n')
    out_table.write(r'\tablewidth{0pt}'+' \n')
    out_table.write(r'\tablecaption{Log of M2FS Observations of Galactic Halo Objects \label{tab:m2fs_obs_table}} \n')
    out_table.write(r'\tablehead{\colhead{Instrument}&\multicolumn{2}{l}{\colhead{Field Center}}&\colhead{UT date\tablenotemark{a}}&\colhead{UT start\tablenotemark{b}}&\colhead{Exp. Time}&\colhead{$N_{\rm exp}$}&\colhead{$N_{\rm target}$}&\colhead{Object}\\'+' \n')
    out_table.write(r'&\colhead{$\alpha_{2000}$ [deg.]}&\colhead{$\delta_{2000}$ [deg.]}&&&\colhead{[sec.]}&}'+' \n')
    out_table.write(r'\startdata'+' \n')

    out_table_short.write(r'\begin{deluxetable*}{lllllcccl}'+' \n')
    out_table_short.write(r'\tablewidth{0pt}'+' \n')
    out_table_short.write(r'\tablecaption{Log of M2FS Observations of Galactic Halo Objects (abbreviated---see electronic version for full table) \label{tab:m2fs_obs_table}} \n')
    out_table_short.write(r'\tablehead{\colhead{Instrument}&\multicolumn{2}{l}{\colhead{Field Center}}&\colhead{UT date\tablenotemark{a}}&\colhead{UT start\tablenotemark{b}}&\colhead{Exp. Time}&\colhead{$N_{\rm exp}$}&\colhead{$N_{\rm target}$}&\colhead{Object}\\'+' \n')
    out_table_short.write(r'&\colhead{$\alpha_{2000}$ [deg.]}&\colhead{$\delta_{2000}$ [deg.]}&&&\colhead{[sec.]}&}'+' \n')
    out_table_short.write(r'\startdata'+' \n')

    for i in range(0,len(order)):
        if used[i]==0:
            this=np.where((field_id==field_id[i])&(field_object==field_object[i])&(field_utdate==field_utdate[i])&(np.abs(fudge_uttime-fudge_uttime[i])<0.5))[0]
            used[this]=1
            hires_used=False
            medres_used=False
            ttt=field_exptime[i].split(',')
            for j in ttt:
                exptime_total+=np.float(j)
            for j in range(0,len(this)):
                if 'HiRes' in field_instrument[this[j]]:
                    hires_used=True
                if 'MedRes' in field_instrument[this[j]]:
                    medres_used=True
            nfields_total+=1
            field_center0.append(field_center[i])
            if ((hires_used)&(not medres_used)):
                config='HiRes'
                nfields_hires+=1
            if ((hires_used)&(medres_used)):
                config='HiRes/MedRes'
                nfields_mix+=1
            if ((not hires_used)&(medres_used)):
                config='MedRes'
                nfields_medres+=1
            if field_center[i].ra.deg<10.:
                rastring='$00'+f'{field_center[i].ra.deg:.6f}'
            elif field_center[i].ra.deg<100.:
                rastring='$0'+f'{field_center[i].ra.deg:.6f}'
            else:
                rastring='$'+f'{field_center[i].ra.deg:.6f}'
            if field_center[i].dec.deg<0:
                if np.abs(field_center[i].dec.deg)<10.:
                    decstring='$-00$'+f'{np.abs(field_center[i].dec.deg):.6f}'
                elif np.abs(field_center[i].dec.deg)<100.:
                    decstring='$-0$'+f'{np.abs(field_center[i].dec.deg):.6f}'
                else:
                    decstring='$-$'+f'{np.abs(field_center[i].dec.deg):.6f}'
            else:
                if np.abs(field_center[i].dec.deg)<10.:
                    decstring='$+00$'+f'{np.abs(field_center[i].dec.deg):.6f}'
                elif np.abs(field_center[i].dec.deg)<100.:
                    decstring='$+0$'+f'{np.abs(field_center[i].dec.deg):.6f}'
                else:
                    decstring='$+$'+f'{np.abs(field_center[i].dec.deg):.6f}'
                
            string='M2FS '+config+' & '+rastring+'$ & $'+decstring+'$ & '+field_utdate[i]+' & '+field_uttime[i]+' &$'+str(int(np.sum(np.array(field_exptime[i].split(','),dtype=float))))+'$ & $'+str(len(np.array(field_exptime[i].split(','),dtype=float)))+'$ & $'+str(int(np.sum(field_ntarget[this])))+'$ & '+str(field_object_formal[i])+'\\\ \n'
            out_table.write(string)
            if i<25:
                out_table_short.write(string)
            print(this,field_subframes[this])

    obs_out=open('m2fs_obs_newcommands.tex','w')
    obs_out.write('\\newcommand{\mtwofsnnights}{'+str(nnights)+'} \n')
    obs_out.write('\\newcommand{\mtwofsnfieldstot}{'+str(nfields_total)+'} \n')
    obs_out.write('\\newcommand{\mtwofsnfieldshires}{'+str(nfields_hires)+'} \n')
    obs_out.write('\\newcommand{\mtwofsnfieldsmedres}{'+str(nfields_medres)+'} \n')
    obs_out.write('\\newcommand{\mtwofsnfieldsmix}{'+str(nfields_mix)+'} \n')
    obs_out.write('\\newcommand{\mtwofsexptimetot}{'+str(round(exptime_total/1.e6,2))+'} \n')
    obs_out.close()
                        
    out_table.write(r'\enddata'+' \n')
    out_table.write(r'\tablenotetext{a}{YYYY-MM-DD format}'+' \n')
    out_table.write(r'\tablenotetext{b}{Universal time at start of first exposure; HH:MM:SS format}'+' \n')
    out_table.write(r'\end{deluxetable*}'+' \n')

    out_table_short.write(r'\enddata'+' \n')
    out_table_short.write(r'\tablenotetext{a}{YYYY-MM-DD format}'+' \n')
    out_table_short.write(r'\tablenotetext{b}{Universal time at start of first exposure; HH:MM:SS format}'+' \n')
    out_table_short.write(r'\end{deluxetable*}'+' \n')

    out_table.close()
    out_table_short.close()
    pickle.dump(field_center0,open('m2fs_field_center.pkl','wb'))

if fit_continuum:

    continuum_rejection_low=-1.5
    continuum_rejection_high=3.
    continuum_rejection_iterations=10
    continuum_rejection_order=6

    for i in range(0,len(fitsfile)):
        print(i,' of ',len(fitsfile))
        continuum_file=fitsfile[i].split('.fits')[0]+'_continuum_func_array.pkl'
        hdul=fits.open(fitsfile[i])
        fitsobject=m2fs.m2fs_getfromfits(hdul)
        hdul.close()

        continuum_func_array=[]
        for j in range(0,len(fitsobject.radeg)):
            spec=deepcopy(fitsobject.spec[j])
            varspec=deepcopy(fitsobject.var[j])
            inf=np.where(varspec>1.e+30)[0]
            varspec[inf]=1.e+30
            nan=np.where(spec!=spec)[0]
            mask=np.full(len(spec),False)
            mask[inf]=True
            mask[nan]=True
            spec1d0=Spectrum1D(spectral_axis=np.arange(len(spec))*u.AA,flux=spec*u.electron,uncertainty=StdDevUncertainty(np.sqrt(varspec)),mask=mask)
            continuum0,rms0=m2fs.get_continuum(spec1d0,continuum_rejection_low,continuum_rejection_high,continuum_rejection_iterations,continuum_rejection_order)
            continuum_func_array.append(continuum0)
        pickle.dump(continuum_func_array,open(continuum_file,'wb'))
    
if make_skysub:

    skysub_list='/hildafs/projects/phy200028p/mgwalker/scripts/'+fits_list0.split('_files')[0]
    g00=open(skysub_list,'w')
    g11=open(data_filename,'w')
    g12=open(chains_filename,'w')

    for i in range(0,len(fitsfile)):
        print(i,' of ',len(fitsfile))
        continuum_file=fitsfile[i].split('.fits')[0]+'_continuum_func_array.pkl'
        continuum_func_array=pickle.load(open(continuum_file,'rb'))
        hdul=fits.open(fitsfile[i])
        fitsobject=m2fs.m2fs_getfromfits(hdul)
        filtername=hdul[0].header['filtername']
        m2fsrun=hdul[0].header['m2fsrun']
        field_name=hdul[0].header['field_name']
        temperature=hdul[0].header['CCD_temp']
        hdul.close()
        surveyphot_filename=fitsfile[i].split('_stackskysub.fits')[0]+'_surveyphot.pkl'
    
        root=[]
        root2=[]
        for j in range(0,len(fitsobject.obj)):
            root.append('m2fs_ian_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3))
            root2.append('m2fs_ian_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3))
        root=np.array(root)
        root2=np.array(root2)

        skies=np.where(fitsobject.obj=='SKY')[0]
        targets=np.where(fitsobject.icode>0)[0]

        if len(targets)==0:
            print('WARNING: NO TARGETS')
        for j in targets:
            out=data_directory+root[j]+'_skysub.dat'
            if len(np.where(fitsobject.mask[j]==False)[0])>0:#write skysub.dat file only if >100 good pixels in spectrum
                print('writing .skysub.dat file for frame ',i,root[j],j)
                g1=open(out,'w')
                g00.write(out+' '+chains_directory+root2[j]+' \n')
                g11.write(out+' \n')
                g12.write(chains_directory+root2[j]+'post_equal_weights.dat'+' \n')

                spec=deepcopy(fitsobject.spec[j])
                varspec=deepcopy(fitsobject.var[j])
                continuum0=continuum_func_array[j]
                continuum=continuum0(np.arange(len(spec)))
                normalization=continuum/np.median(continuum)
                spec2=spec/normalization
                varspec2=1./(normalization**2)*varspec
                
                for k in range(0,len(fitsobject.wav[j])):
#                    if fitsobject.mask[j][k]==False:
                    string=str(round(fitsobject.wav[j][k],10))+' '+str(round(fitsobject.spec[j][k],3))+' '+str(round(fitsobject.var[j][k],5))+' '+str(round(spec2[k],3))+' '+str(round(varspec2[k],5))+' \n'
                    g1.write(string)
                g1.close()
            else:
                print('NOT writing .skysub.dat file for frame ',i,root[j],j)
    g00.close()
    g11.close()
    g12.close()

if get_fits:

    for i in range(0,len(fitsfile)):
        print(i,' of ',len(fitsfile))
        continuum_file=fitsfile[i].split('.fits')[0]+'_continuum_func_array.pkl'
        continuum_func_array=pickle.load(open(continuum_file,'rb'))
        hdul=fits.open(fitsfile[i])
        fitsobject=m2fs.m2fs_getfromfits(hdul)
        filtername=hdul[0].header['filtername']
        m2fsrun=hdul[0].header['m2fsrun']
        field_name=hdul[0].header['field_name']
        temperature=hdul[0].header['CCD_temp']
        hdul.close()
        surveyphot_filename=fitsfile[i].split('_stackskysub.fits')[0]+'_surveyphot.pkl'

        root=[]
        root2=[]
        for j in range(0,len(fitsobject.obj)):
            root.append('m2fs_ian_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3))
            root2.append('m2fs_ian_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3))
        root=np.array(root)
        root2=np.array(root2)

        skies=np.where(fitsobject.obj=='SKY')[0]
        targets=np.where(fitsobject.icode>0)[0]
        ra_dec_source=[]
        for j in range(0,len(fitsobject.radeg)):
            ra_dec_source.append('m2fs_targeting')
        ra_dec_source=np.array(ra_dec_source)
        
        survey_phot=pickle.load(open(surveyphot_filename,'rb'))
        gaia_ra=survey_phot.gaia.ra
        gaia_dec=survey_phot.gaia.dec
        gaia_gmag_dered=survey_phot.gaia.gmag-survey_phot.gaia.gext
        gaia_bpmag_dered=survey_phot.gaia.bpmag-survey_phot.gaia.bpext
        gaia_rpmag_dered=survey_phot.gaia.rpmag-survey_phot.gaia.rpext
        des_ra=survey_phot.des.ra
        des_dec=survey_phot.des.dec
        des_gmag_dered=survey_phot.des.gmag-survey_phot.des.gext
        des_rmag_dered=survey_phot.des.rmag-survey_phot.des.rext
        des_zmag_dered=survey_phot.des.zmag-survey_phot.des.zext
        decals_ra=survey_phot.decals.ra
        decals_dec=survey_phot.decals.dec
        decals_gmag_dered=survey_phot.decals.gmag-survey_phot.decals.gext
        decals_rmag_dered=survey_phot.decals.rmag-survey_phot.decals.gext
        decals_zmag_dered=survey_phot.decals.zmag-survey_phot.decals.gext
        sdss_ra=survey_phot.sdss.ra
        sdss_dec=survey_phot.sdss.dec
        sdss_gmag_dered=survey_phot.sdss.gmag-survey_phot.sdss.gext
        sdss_rmag_dered=survey_phot.sdss.rmag-survey_phot.sdss.gext
        sdss_zmag_dered=survey_phot.sdss.zmag-survey_phot.sdss.gext
        ps1_ra=survey_phot.ps1.ra
        ps1_dec=survey_phot.ps1.dec
        ps1_gmag_dered=survey_phot.ps1.gmag-survey_phot.ps1.gext
        ps1_rmag_dered=survey_phot.ps1.rmag-survey_phot.ps1.gext
        ps1_zmag_dered=survey_phot.ps1.zmag-survey_phot.ps1.gext
        
        gaia_teffprior=((gaia_est50,gaia_est16,gaia_est84),gaia_gmag_dered,gaia_bpmag_dered,gaia_rpmag_dered)
        des_teffprior=((des_est50,des_est16,des_est84),des_gmag_dered,des_rmag_dered,des_zmag_dered)
        decals_teffprior=((decals_est50,decals_est16,decals_est84),decals_gmag_dered,decals_rmag_dered,decals_zmag_dered)
        sdss_teffprior=((sdss_est50,sdss_est16,sdss_est84),sdss_gmag_dered,sdss_rmag_dered,sdss_zmag_dered)
        ps1_teffprior=((ps1_est50,ps1_est16,ps1_est84),ps1_gmag_dered,ps1_rmag_dered,ps1_zmag_dered)

        print(filtername)
#        if ((len(fitsobject.obj)>0)&(filtername=='Mgb_HiRes')):
        if ((len(fitsobject.obj)>0)):
            teffpriorstuff0=teffpriorstuff(gaia_teffprior,des_teffprior,decals_teffprior,sdss_teffprior,ps1_teffprior)
            multinest=m2fs.m2fs_multinest_ian(fit_directory,root2,targets,fitsobject,len(fitsobject.wav[0]),teffpriorstuff0)#object containing sample of posterior, moments thereof, bestfit wavelength and bestfit spectrum
            hdr=fits.Header(fitsobject.header)
            primary_hdu=fits.PrimaryHDU(header=hdr)
            new_hdul=fits.HDUList([primary_hdu])

            update_coords_to_gaia=np.where((gaia_ra==gaia_ra)&(gaia_dec==gaia_dec))[0]#if found gaia match, update coords to gaia DR3
            fitsobject.radeg[update_coords_to_gaia]=gaia_ra[update_coords_to_gaia]
            fitsobject.decdeg[update_coords_to_gaia]=gaia_dec[update_coords_to_gaia]
            ra_dec_source[update_coords_to_gaia]='gaia_dr3'

            update_coords_to_des=np.where((ra_dec_source=='m2fs_targeting')&(des_ra==des_ra)&(des_dec==des_dec))[0]
            fitsobject.radeg[update_coords_to_des]=des_ra[update_coords_to_des]
            fitsobject.decdeg[update_coords_to_des]=des_dec[update_coords_to_des]
            ra_dec_source[update_coords_to_des]='des_dr2'

            update_coords_to_decals=np.where((ra_dec_source=='m2fs_targeting')&(decals_ra==decals_ra)&(decals_dec==decals_dec))[0]
            fitsobject.radeg[update_coords_to_decals]=decals_ra[update_coords_to_decals]
            fitsobject.decdeg[update_coords_to_decals]=decals_dec[update_coords_to_decals]
            ra_dec_source[update_coords_to_decals]='decals_dr9'

            update_coords_to_sdss=np.where((ra_dec_source=='m2fs_targeting')&(sdss_ra==sdss_ra)&(sdss_dec==sdss_dec))[0]
            fitsobject.radeg[update_coords_to_sdss]=sdss_ra[update_coords_to_sdss]
            fitsobject.decdeg[update_coords_to_sdss]=sdss_dec[update_coords_to_sdss]
            ra_dec_source[update_coords_to_sdss]='sdss_dr9'

            update_coords_to_ps1=np.where((ra_dec_source=='m2fs_targeting')&(ps1_ra==ps1_ra)&(ps1_dec==ps1_dec))[0]
            fitsobject.radeg[update_coords_to_ps1]=ps1_ra[update_coords_to_ps1]
            fitsobject.decdeg[update_coords_to_ps1]=ps1_dec[update_coords_to_ps1]
            ra_dec_source[update_coords_to_ps1]='ps1_dr1'

            col1=fits.Column(name='ra',format='D',array=fitsobject.radeg,unit='degree')
            col2=fits.Column(name='dec',format='D',array=fitsobject.decdeg,unit='degree')
            col3=fits.Column(name='ra_dec_source',format='A30',array=ra_dec_source,unit='')
            col4=fits.Column(name='mjd',format='D',array=fitsobject.mjd,unit='day')
            col5=fits.Column(name='hjd',format='D',array=fitsobject.hjd,unit='day')
            col6=fits.Column(name='snratio',format='D',array=fitsobject.snratio,unit='')
            col7=fits.Column(name='vhelio_correction',format='D',array=fitsobject.vheliocorr,unit='km/s')
            col8=fits.Column(name='posterior_moments',format='68D',dim='(17,4)',array=multinest.moments)
            col9=fits.Column(name='posterior_1000',format='17000D',dim='(17,1000)',array=multinest.posterior_1000)
            col10=fits.Column(name='teffprior_posterior_moments',format='68D',dim='(17,4)',array=multinest.teffprior_moments)
            col11=fits.Column(name='teffprior_posterior_1000',format='17000D',dim='(17,1000)',array=multinest.teffprior_posterior_1000)
            col12=fits.Column(name='teffprior_survey',format='A10',array=multinest.teffprior_survey,unit='')
            col13=fits.Column(name='objtype',format='A6',array=fitsobject.obj,unit='')
            col14=fits.Column(name='run_id',format='A100',array=fitsobject.run_id,unit='')
            col15=fits.Column(name='field_name',format='A100',array=fitsobject.field_name,unit='')
            col16=fits.Column(name='filter_name',format='A100',array=fitsobject.filtername,unit='')
            col17=fits.Column(name='wav_npoints',format='PI()',array=fitsobject.wav_npoints,unit='')
            col18=fits.Column(name='wav_rms',format='PD()',array=fitsobject.wav_rms,unit='Angstrom')
            col19=fits.Column(name='wav_resolution',format='PD()',array=fitsobject.wav_resolution,unit='Angstrom')
            col20=fits.Column(name='wav_min',format='PD()',array=fitsobject.wav_min,unit='Angstrom')
            col21=fits.Column(name='wav_max',format='PD()',array=fitsobject.wav_max,unit='Angstrom')
            col22=fits.Column(name='row',format='D',array=fitsobject.row,unit='')
            col23=fits.Column(name='temperature',format='PD()',array=fitsobject.temperature,unit='degree Celcius')
            id_int=[]
            for xxx in range(0,len(survey_phot.gaia.objid)):
                id_int.append(np.int64(survey_phot.gaia.objid[xxx]))
            id_int=np.array(id_int)
            col24=fits.Column(name='gaia_source_id',format='K',array=id_int,unit='')
            col25=fits.Column(name='gaia_gmag',format='D',array=survey_phot.gaia.gmag,unit='mag')
            col26=fits.Column(name='gaia_siggmag',format='D',array=survey_phot.gaia.siggmag,unit='mag')
            col27=fits.Column(name='gaia_gmag_dered',format='D',array=gaia_gmag_dered,unit='mag')
            col28=fits.Column(name='gaia_bpmag',format='D',array=survey_phot.gaia.bpmag,unit='mag')
            col29=fits.Column(name='gaia_sigbpmag',format='D',array=survey_phot.gaia.sigbpmag,unit='mag')
            col30=fits.Column(name='gaia_bpmag_dered',format='D',array=gaia_bpmag_dered,unit='mag')
            col31=fits.Column(name='gaia_rpmag',format='D',array=survey_phot.gaia.rpmag,unit='mag')
            col32=fits.Column(name='gaia_sigrpmag',format='D',array=survey_phot.gaia.sigrpmag,unit='mag')
            col33=fits.Column(name='gaia_rpmag_dered',format='D',array=gaia_rpmag_dered,unit='mag')
            col34=fits.Column(name='gaia_pmra',format='D',array=survey_phot.gaia.pmra,unit='mas/yr')
            col35=fits.Column(name='gaia_sigpmra',format='D',array=survey_phot.gaia.sigpmra,unit='mas/yr')
            col36=fits.Column(name='gaia_pmdec',format='D',array=survey_phot.gaia.pmdec,unit='mas/yr')
            col37=fits.Column(name='gaia_sigpmdec',format='D',array=survey_phot.gaia.sigpmdec,unit='mas/yr')
            col38=fits.Column(name='gaia_parallax',format='D',array=survey_phot.gaia.parallax,unit='mas')
            col39=fits.Column(name='gaia_sigparallax',format='D',array=survey_phot.gaia.sigparallax,unit='mas')
            col40=fits.Column(name='gaia_rrl',format='I',array=survey_phot.gaia.rrl,unit='')
            col41=fits.Column(name='gaia_cepheid',format='I',array=survey_phot.gaia.cepheid,unit='')
            col42=fits.Column(name='gaia_agn',format='I',array=survey_phot.gaia.agn,unit='')
            col43=fits.Column(name='gaia_rvvariable',format='I',array=survey_phot.gaia.rvvariable,unit='')
            col44=fits.Column(name='gaia_compact',format='I',array=survey_phot.gaia.compact,unit='')
            col45=fits.Column(name='gaia_rv',format='D',array=survey_phot.gaia.rv,unit='km/s')
            col46=fits.Column(name='gaia_sigrv',format='D',array=survey_phot.gaia.sigrv,unit='km/s')
            col47=fits.Column(name='gaia_phot_variable',format='A20',array=survey_phot.gaia.variable,unit='')
            col48=fits.Column(name='gaia_in_qso',format='L',array=survey_phot.gaia.in_qso,unit='')
            col49=fits.Column(name='gaia_in_galaxy',format='L',array=survey_phot.gaia.in_galaxy,unit='')
            col50=fits.Column(name='gaia_in_blend',format='I',array=survey_phot.gaia.in_blend,unit='')
            col51=fits.Column(name='gaia_teff',format='D',array=survey_phot.gaia.teff,unit='Kelvin')
            col52=fits.Column(name='gaia_teff_lower',format='D',array=survey_phot.gaia.teff_lower,unit='Kelvin')
            col53=fits.Column(name='gaia_teff_upper',format='D',array=survey_phot.gaia.teff_upper,unit='Kelvin')
            col54=fits.Column(name='gaia_logg',format='D',array=survey_phot.gaia.logg,unit='dex (cgs)')
            col55=fits.Column(name='gaia_logg_lower',format='D',array=survey_phot.gaia.logg_lower,unit='dex (cgs)')
            col56=fits.Column(name='gaia_logg_upper',format='D',array=survey_phot.gaia.logg_upper,unit='dex (cgs)')
            col57=fits.Column(name='gaia_feh',format='D',array=survey_phot.gaia.feh,unit='dex')
            col58=fits.Column(name='gaia_feh_lower',format='D',array=survey_phot.gaia.feh_lower,unit='dex')
            col59=fits.Column(name='gaia_feh_upper',format='D',array=survey_phot.gaia.feh_upper,unit='dex')
            col60=fits.Column(name='gaia_libname',format='A30',array=survey_phot.gaia.libname,unit='')
            cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41,col42,col43,col44,col45,col46,col47,col48,col49,col50,col51,col52,col53,col54,col55,col56,col57,col58,col59,col60])
            cols_with_posterior=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41,col42,col43,col44,col45,col46,col47,col48,col49,col50,col51,col52,col53,col54,col55,col56,col57,col58,col59,col60])
            table_hdu=fits.FITS_rec.from_columns(cols)#BinTableHDU.from_columns(cols)
            table_hdu_with_posterior=fits.FITS_rec.from_columns(cols_with_posterior)#BinTableHDU.from_columns(cols)

            primary_hdu=fits.PrimaryHDU([],header=hdr)
            new_hdul=fits.HDUList([primary_hdu])
            new_hdul_with_posterior=fits.HDUList([primary_hdu])
            new_hdul.append(fits.ImageHDU(fitsobject.wav,name='wavlength'))
            new_hdul.append(fits.ImageHDU(fitsobject.spec,name='sky_subtracted'))
            new_hdul.append(fits.ImageHDU(fitsobject.var,name='variance'))
            new_hdul.append(fits.ImageHDU(fitsobject.mask,name='mask'))
            new_hdul_with_posterior.append(fits.ImageHDU(fitsobject.wav,name='wavlength'))
            new_hdul_with_posterior.append(fits.ImageHDU(fitsobject.spec,name='sky_subtracted'))
            new_hdul_with_posterior.append(fits.ImageHDU(fitsobject.var,name='variance'))
            new_hdul_with_posterior.append(fits.ImageHDU(fitsobject.mask,name='mask'))
            bestfit_array=[]
            for j in range(0,len(multinest.bestfit_fit)):
                continuum0=continuum_func_array[j]
                continuum=continuum0(np.arange(len(fitsobject.spec[j])))
                normalization=continuum/np.median(continuum)                
                bestfit_array.append(multinest.bestfit_fit[j]*normalization)
            bestfit_array=np.array(bestfit_array)
            new_hdul.append(fits.ImageHDU(bestfit_array,name='bestfit'))
            new_hdul.append(fits.BinTableHDU(table_hdu,name='data_table'))
            new_hdul.append(fits.ImageHDU(fitsobject.sky_spec,name='sky'))
            new_hdul_with_posterior.append(fits.ImageHDU(bestfit_array,name='bestfit'))
            new_hdul_with_posterior.append(fits.BinTableHDU(table_hdu_with_posterior,name='data_table_with_posterior'))
            new_hdul_with_posterior.append(fits.ImageHDU(fitsobject.sky_spec,name='sky'))
            
#            outfile=fitsfile[i].split('.fits')[0]+'_postfit_ian.fits'
            outfile=fitsfile[i].split('_stackskysub.fits')[0]+'_final.fits'
            outfile_with_posterior=fitsfile[i].split('_stackskysub.fits')[0]+'_final_with_posterior.fits'
            print('writing fit results to '+outfile)
            new_hdul.writeto(outfile,overwrite=True)
            new_hdul_with_posterior.writeto(outfile_with_posterior,overwrite=True)

            for k in range(0,len(new_hdul[4].data)):
                rastring=str('{0:.7f}'.format(new_hdul[6].data['ra'][k]))
                decstring=str('{0:.7f}'.format(new_hdul[6].data['dec'][k]))
                hjdstring=str('{0:.3f}'.format(new_hdul[6].data['hjd'][k]))
                if new_hdul[6].data['dec'][k]>=0.:
                    radechjdstring=rastring+'_+'+decstring+'_'+hjdstring
                else:
                    radechjdstring=rastring+'_'+decstring+'_'+hjdstring

if get_catalog_raw:

    for filtername0 in ['HiRes','MedRes']:
        if filtername0=='HiRes':
            fits_table_raw_filename=fits_list.split('_files')[0]+'_HiRes_raw.fits'
        if filtername0=='MedRes':
            fits_table_raw_filename=fits_list.split('_files')[0]+'_MedRes_raw.fits'
        
        postfit_object=[]
        postfit_obsid=[]
        postfit_index=[]
        postfit_objtype=[]
        postfit_radeg=[]
        postfit_decdeg=[]
        postfit_ra_dec_source=[]
        postfit_mjd=[]
        postfit_hjd=[]
        postfit_snratio=[]
        postfit_gaiaid=[]
        postfit_gaiagmag=[]
        postfit_siggaiagmag=[]
        postfit_gaiagmag_dered=[]
        postfit_gaiabpmag=[]
        postfit_siggaiabpmag=[]
        postfit_gaiabpmag_dered=[]
        postfit_gaiarpmag=[]
        postfit_siggaiarpmag=[]
        postfit_gaiarpmag_dered=[]
        postfit_gaiapmra=[]
        postfit_siggaiapmra=[]
        postfit_gaiapmdec=[]
        postfit_siggaiapmdec=[]
        postfit_gaiaparallax=[]
        postfit_siggaiaparallax=[]
        postfit_gaia_rrl=[]
        postfit_gaia_rv=[]
        postfit_gaia_sigrv=[]
        postfit_gaia_phot_variable=[]
        postfit_gaia_cepheid=[]
        postfit_gaia_agn=[]
        postfit_gaia_rvvariable=[]
        postfit_gaia_compact=[]
        postfit_gaia_in_qso=[]
        postfit_gaia_in_galaxy=[]
        postfit_gaia_in_blend=[]
        postfit_gaia_teff=[]
        postfit_gaia_teff_lower=[]
        postfit_gaia_teff_upper=[]
        postfit_gaia_logg=[]
        postfit_gaia_logg_lower=[]
        postfit_gaia_logg_upper=[]
        postfit_gaia_feh=[]
        postfit_gaia_feh_lower=[]
        postfit_gaia_feh_upper=[]
        postfit_gaia_libname=[]
        postfit_vhelio_correction=[]
        postfit_mean=[]
        postfit_std=[]
        postfit_skew=[]
        postfit_kurtosis=[]        
        postfit_teffprior_mean=[]
        postfit_teffprior_std=[]
        postfit_teffprior_skew=[]
        postfit_teffprior_kurtosis=[]
        postfit_teffprior_survey=[]
        postfit_mediansky=[]
        postfit_medianskysub=[]
        postfit_stdsky=[]
        postfit_run_id=[]
        postfit_field_name=[]
        postfit_filtername=[]
        postfit_chi2=[]
        postfit_bluechi2=[]
        postfit_redchi2=[]
        postfit_chi22=[]
        postfit_bluespec=[]
        postfit_redspec=[]
        postfit_n=[]
        postfit_bluen=[]
        postfit_redn=[]
        postfit_wav_npoints=[]
        postfit_wav_rms=[]
        postfit_wav_resolution=[]
        postfit_wav_min=[]
        postfit_wav_max=[]
        postfit_row=[]
        postfit_temperature=[]
        postfit_filename=[]
        postfit_allmasked=[]
        postfit_exptime=[]
        postfit_nsubexp=[]
        postfit_airmass=[]
    
        for i in range(0,len(fitsfile)):
            print(i,' of ',len(fitsfile))
            hdul=fits.open(fitsfile[i])
            fitsobject=m2fs.m2fs_getfromfits(hdul)
            filtername=hdul[0].header['filtername']
            m2fsrun=hdul[0].header['m2fsrun']
            field_name=hdul[0].header['field_name']
            temperature=hdul[0].header['CCD_temp']
            exptime_total=0.
            field_exptime=hdul[0].header['exp_times']
            ttt=field_exptime.split(',')
            for j in ttt:
                exptime_total+=np.float(j)
            nsubexp=len(ttt)
            airmass=hdul[0].header['airmass']
            hdul.close()
            surveyphot_filename=fitsfile[i].split('_stackskysub.fits')[0]+'_surveyphot.pkl'
    
            root=[]
            root2=[]
            for j in range(0,len(fitsobject.obj)):
                root.append('m2fs_ian_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3))
                root2.append('m2fs_ian_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3))
            root=np.array(root)
            root2=np.array(root2)

            all_masked=np.zeros(len(fitsobject.mask),dtype='int')
            for j in range(0,len(fitsobject.mask)):
                if np.min(fitsobject.mask[j])>0:
                    all_masked[j]=1
                
            skies=np.where((fitsobject.obj=='SKY')&(all_masked==0))[0]
            targets=np.where((fitsobject.icode>0)&(all_masked==0))[0]

            if ((len(fitsobject.obj)>0)&(filtername=='Mgb_'+filtername0)&(len(skies)>0)):
#                infile=fitsfile[i].split('.fits')[0]+'_postfit_ian.fits'
                infile=fitsfile[i].split('_stackskysub.fits')[0]+'_final_with_posterior.fits'
                postfit=fits.open(infile)
                mediansky=[]
                medianskysub=[]
                for j in range(0,len(postfit[6].data)):
                    rastring=str('{0:.7f}'.format(postfit[6].data['ra'][j]))
                    decstring=str('{0:.7f}'.format(postfit[6].data['dec'][j]))
                    hjdstring=str('{0:.3f}'.format(postfit[6].data['hjd'][j]))
                    if postfit[6].data['dec'][j]>=0.:
                        radechjdstring=rastring+'_+'+decstring+'_'+hjdstring
                    else:
                        radechjdstring=rastring+'_'+decstring+'_'+hjdstring
                    postfit_temperature.append(postfit[6].data['temperature'][j])
                    postfit_obsid.append(radechjdstring)
                    postfit_index.append(j)
                    postfit_objtype.append(postfit[6].data['objtype'][j])
                    postfit_radeg.append(postfit[6].data['ra'][j])
                    postfit_decdeg.append(postfit[6].data['dec'][j])
                    postfit_ra_dec_source.append(postfit[6].data['ra_dec_source'][j])
                    postfit_mjd.append(postfit[6].data['mjd'][j])
                    postfit_hjd.append(postfit[6].data['hjd'][j])
                    postfit_snratio.append(postfit[6].data['snratio'][j])
                    postfit_gaiaid.append(postfit[6].data['gaia_source_id'][j])
                    postfit_gaiagmag.append(postfit[6].data['gaia_gmag'][j])
                    postfit_siggaiagmag.append(postfit[6].data['gaia_siggmag'][j])
                    postfit_gaiagmag_dered.append(postfit[6].data['gaia_gmag_dered'][j])
                    postfit_gaiabpmag.append(postfit[6].data['gaia_bpmag'][j])
                    postfit_siggaiabpmag.append(postfit[6].data['gaia_sigbpmag'][j])
                    postfit_gaiabpmag_dered.append(postfit[6].data['gaia_bpmag_dered'][j])
                    postfit_gaiarpmag.append(postfit[6].data['gaia_rpmag'][j])
                    postfit_siggaiarpmag.append(postfit[6].data['gaia_sigrpmag'][j])
                    postfit_gaiarpmag_dered.append(postfit[6].data['gaia_rpmag_dered'][j])
                    postfit_gaiapmra.append(postfit[6].data['gaia_pmra'][j])
                    postfit_siggaiapmra.append(postfit[6].data['gaia_sigpmra'][j])
                    postfit_gaiapmdec.append(postfit[6].data['gaia_pmdec'][j])
                    postfit_siggaiapmdec.append(postfit[6].data['gaia_sigpmdec'][j])
                    postfit_gaiaparallax.append(postfit[6].data['gaia_parallax'][j])
                    postfit_siggaiaparallax.append(postfit[6].data['gaia_sigparallax'][j])
                    postfit_gaia_rrl.append(postfit[6].data['gaia_rrl'][j])
                    postfit_gaia_rv.append(postfit[6].data['gaia_rv'][j])
                    postfit_gaia_sigrv.append(postfit[6].data['gaia_sigrv'][j])
                    postfit_gaia_phot_variable.append(postfit[6].data['gaia_phot_variable'][j])
                    postfit_gaia_cepheid.append(postfit[6].data['gaia_cepheid'][j])
                    postfit_gaia_agn.append(postfit[6].data['gaia_agn'][j])
                    postfit_gaia_rvvariable.append(postfit[6].data['gaia_rvvariable'][j])
                    postfit_gaia_compact.append(postfit[6].data['gaia_compact'][j])
                    postfit_gaia_in_qso.append(postfit[6].data['gaia_in_qso'][j])
                    postfit_gaia_in_galaxy.append(postfit[6].data['gaia_in_galaxy'][j])
                    postfit_gaia_in_blend.append(postfit[6].data['gaia_in_blend'][j])
                    postfit_gaia_teff.append(postfit[6].data['gaia_teff'][j])
                    postfit_gaia_teff_lower.append(postfit[6].data['gaia_teff_lower'][j])
                    postfit_gaia_teff_upper.append(postfit[6].data['gaia_teff_upper'][j])
                    postfit_gaia_logg.append(postfit[6].data['gaia_logg'][j])
                    postfit_gaia_logg_lower.append(postfit[6].data['gaia_logg_lower'][j])
                    postfit_gaia_logg_upper.append(postfit[6].data['gaia_logg_upper'][j])
                    postfit_gaia_feh.append(postfit[6].data['gaia_feh_upper'][j])
                    postfit_gaia_feh_lower.append(postfit[6].data['gaia_feh_lower'][j])
                    postfit_gaia_feh_upper.append(postfit[6].data['gaia_feh_upper'][j])
                    postfit_gaia_libname.append(postfit[6].data['gaia_libname'][j])
                    postfit_vhelio_correction.append(postfit[6].data['vhelio_correction'][j])
                    postfit_mean.append(postfit[6].data['posterior_moments'][j][0])
                    postfit_std.append(postfit[6].data['posterior_moments'][j][1])
                    postfit_skew.append(postfit[6].data['posterior_moments'][j][2])
                    postfit_kurtosis.append(postfit[6].data['posterior_moments'][j][3])
                    postfit_teffprior_mean.append(postfit[6].data['teffprior_posterior_moments'][j][0])
                    postfit_teffprior_std.append(postfit[6].data['teffprior_posterior_moments'][j][1])
                    postfit_teffprior_skew.append(postfit[6].data['teffprior_posterior_moments'][j][2])
                    postfit_teffprior_kurtosis.append(postfit[6].data['teffprior_posterior_moments'][j][3])
                    postfit_teffprior_survey.append(postfit[6].data['teffprior_survey'][j])
                    keep=np.where((postfit[4].data[j]==0)&(postfit[3].data[j]<1.e+5))[0]
                    bluekeep=np.where((postfit[4].data[j]==0)&(postfit[3].data[j]<1.e+5)&(postfit[1].data[j]>bluelambdamin)&(postfit[1].data[j]<bluelambdamax))[0]
                    redkeep=np.where((postfit[4].data[j]==0)&(postfit[3].data[j]<1.e+5)&(postfit[1].data[j]<redlambdamax)&(postfit[1].data[j]>=redlambdamin))[0]
                    mediansky.append(np.median(postfit[7].data[j][keep]))
                    medianskysub.append(np.median(postfit[2].data[j][keep]))
                    postfit_mediansky.append(np.median(postfit[7].data[j][keep]))
                    postfit_medianskysub.append(np.median(postfit[2].data[j][keep]))
                    postfit_object.append(obj[i])
                    postfit_run_id.append(postfit[6].data['run_id'][j])
                    postfit_field_name.append(postfit[6].data['field_name'][j])
                    postfit_filtername.append(postfit[6].data['filter_name'][j])
                    postfit_wav_npoints.append(postfit[6].data['wav_npoints'][j])
                    postfit_wav_rms.append(postfit[6].data['wav_rms'][j])
                    postfit_wav_resolution.append(postfit[6].data['wav_resolution'][j])
                    postfit_wav_min.append(postfit[6].data['wav_min'][j])
                    postfit_wav_max.append(postfit[6].data['wav_max'][j])
                    postfit_row.append(postfit[6].data['row'][j])
                    postfit_temperature.append(postfit[6].data['temperature'][j])
                    short_filename=infile.split('/ut2')[1][8:]
                    postfit_filename.append(short_filename)
                    postfit_allmasked.append(all_masked[j])
                    postfit_exptime.append(exptime_total)
                    postfit_nsubexp.append(nsubexp)
                    postfit_airmass.append(airmass)
                    spec=postfit[2].data[j]
                    mask=postfit[4].data[j]
                    bestfit=postfit[5].data[j]
                    var=postfit[3].data[j]
                    
                    chi2=np.sum((spec[keep]-bestfit[keep])**2/var[keep])/len(keep)
                    bluespec=np.median(spec[bluekeep])
                    redspec=np.median(spec[redkeep])
                    bluechi2=np.sum((spec[bluekeep]-bestfit[bluekeep])**2/var[bluekeep])/len(bluekeep)
                    redchi2=np.sum((spec[redkeep]-bestfit[redkeep])**2/var[redkeep])/len(redkeep)
                
                    postfit_chi2.append(chi2)
                    postfit_n.append(len(keep))
                    postfit_bluechi2.append(bluechi2)
                    postfit_bluen.append(len(bluekeep))
                    postfit_redchi2.append(redchi2)
                    postfit_redn.append(len(redkeep))
                    postfit_bluespec.append(bluespec)
                    postfit_redspec.append(redspec)
                
                    sigma0=postfit[6].data['posterior_moments'][j][0][14]
                    sigma1=postfit[6].data['posterior_moments'][j][0][15]
                    var2=10.**sigma1*var+(10.**sigma0)**2
                    chi22=np.sum((spec[keep]-bestfit[keep])**2/var2[keep])/len(keep)
                    postfit_chi22.append(chi22)
                mediansky=np.array(mediansky)
                medianskysub=np.array(medianskysub)
                stdsky=np.std(mediansky[skies])
                for j in range(0,len(postfit[6].data)):
                    postfit_stdsky.append(stdsky)

        postfit_allmasked=np.array(postfit_allmasked)
        postfit_object=np.array(postfit_object)[np.where(postfit_allmasked==0)[0]]
        postfit_obsid=np.array(postfit_obsid)[np.where(postfit_allmasked==0)[0]]
        postfit_index=np.array(postfit_index)[np.where(postfit_allmasked==0)[0]]
        postfit_objtype=np.array(postfit_objtype)[np.where(postfit_allmasked==0)[0]]
        postfit_radeg=np.array(postfit_radeg)[np.where(postfit_allmasked==0)[0]]
        postfit_decdeg=np.array(postfit_decdeg)[np.where(postfit_allmasked==0)[0]]
        postfit_ra_dec_source=np.array(postfit_ra_dec_source)[np.where(postfit_allmasked==0)[0]]
        postfit_mjd=np.array(postfit_mjd)[np.where(postfit_allmasked==0)[0]]
        postfit_hjd=np.array(postfit_hjd)[np.where(postfit_allmasked==0)[0]]
        postfit_snratio=np.array(postfit_snratio)[np.where(postfit_allmasked==0)[0]]
        postfit_gaiaid=np.array(postfit_gaiaid)[np.where(postfit_allmasked==0)[0]]
        postfit_gaiagmag=np.array(postfit_gaiagmag)[np.where(postfit_allmasked==0)[0]]
        postfit_siggaiagmag=np.array(postfit_siggaiagmag)[np.where(postfit_allmasked==0)[0]]
        postfit_gaiagmag_dered=np.array(postfit_gaiagmag_dered)[np.where(postfit_allmasked==0)[0]]
        postfit_gaiabpmag=np.array(postfit_gaiabpmag)[np.where(postfit_allmasked==0)[0]]
        postfit_siggaiabpmag=np.array(postfit_siggaiabpmag)[np.where(postfit_allmasked==0)[0]]
        postfit_gaiabpmag_dered=np.array(postfit_gaiabpmag_dered)[np.where(postfit_allmasked==0)[0]]
        postfit_gaiarpmag=np.array(postfit_gaiarpmag)[np.where(postfit_allmasked==0)[0]]
        postfit_siggaiarpmag=np.array(postfit_siggaiarpmag)[np.where(postfit_allmasked==0)[0]]
        postfit_gaiarpmag_dered=np.array(postfit_gaiarpmag_dered)[np.where(postfit_allmasked==0)[0]]
        postfit_gaiapmra=np.array(postfit_gaiapmra)[np.where(postfit_allmasked==0)[0]]
        postfit_siggaiapmra=np.array(postfit_siggaiapmra)[np.where(postfit_allmasked==0)[0]]
        postfit_gaiapmdec=np.array(postfit_gaiapmdec)[np.where(postfit_allmasked==0)[0]]
        postfit_siggaiapmdec=np.array(postfit_siggaiapmdec)[np.where(postfit_allmasked==0)[0]]
        postfit_gaiaparallax=np.array(postfit_gaiaparallax)[np.where(postfit_allmasked==0)[0]]
        postfit_siggaiaparallax=np.array(postfit_siggaiaparallax)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_rrl=np.array(postfit_gaia_rrl)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_rv=np.array(postfit_gaia_rv)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_sigrv=np.array(postfit_gaia_sigrv)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_phot_variable=np.array(postfit_gaia_phot_variable)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_cepheid=np.array(postfit_gaia_cepheid)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_agn=np.array(postfit_gaia_agn)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_rvvariable=np.array(postfit_gaia_rvvariable)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_compact=np.array(postfit_gaia_compact)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_in_qso=np.array(postfit_gaia_in_qso)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_in_galaxy=np.array(postfit_gaia_in_galaxy)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_in_blend=np.array(postfit_gaia_in_blend)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_teff=np.array(postfit_gaia_teff)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_teff_lower=np.array(postfit_gaia_teff_lower)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_teff_upper=np.array(postfit_gaia_teff_upper)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_logg=np.array(postfit_gaia_logg)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_logg_lower=np.array(postfit_gaia_logg_lower)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_logg_upper=np.array(postfit_gaia_logg_upper)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_feh=np.array(postfit_gaia_feh)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_feh_lower=np.array(postfit_gaia_feh_lower)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_feh_upper=np.array(postfit_gaia_feh_upper)[np.where(postfit_allmasked==0)[0]]
        postfit_gaia_libname=np.array(postfit_gaia_libname)[np.where(postfit_allmasked==0)[0]]
        postfit_vhelio_correction=np.array(postfit_vhelio_correction)[np.where(postfit_allmasked==0)[0]]
        postfit_mean=np.array(postfit_mean)[np.where(postfit_allmasked==0)[0]]
        postfit_std=np.array(postfit_std)[np.where(postfit_allmasked==0)[0]]
        postfit_skew=np.array(postfit_skew)[np.where(postfit_allmasked==0)[0]]
        postfit_kurtosis=np.array(postfit_kurtosis)[np.where(postfit_allmasked==0)[0]]
        postfit_teffprior_mean=np.array(postfit_teffprior_mean)[np.where(postfit_allmasked==0)[0]]
        postfit_teffprior_std=np.array(postfit_teffprior_std)[np.where(postfit_allmasked==0)[0]]
        postfit_teffprior_skew=np.array(postfit_teffprior_skew)[np.where(postfit_allmasked==0)[0]]
        postfit_teffprior_kurtosis=np.array(postfit_teffprior_kurtosis)[np.where(postfit_allmasked==0)[0]]
        postfit_teffprior_survey=np.array(postfit_teffprior_survey)[np.where(postfit_allmasked==0)[0]]
        postfit_mediansky=np.array(postfit_mediansky)[np.where(postfit_allmasked==0)[0]]
        postfit_medianskysub=np.array(postfit_medianskysub)[np.where(postfit_allmasked==0)[0]]
        postfit_stdsky=np.array(postfit_stdsky)[np.where(postfit_allmasked==0)[0]]
        postfit_run_id=np.array(postfit_run_id)[np.where(postfit_allmasked==0)[0]]
        postfit_field_name=np.array(postfit_field_name)[np.where(postfit_allmasked==0)[0]]
        postfit_filtername=np.array(postfit_filtername)[np.where(postfit_allmasked==0)[0]]
        postfit_chi2=np.array(postfit_chi2)[np.where(postfit_allmasked==0)[0]]
        postfit_bluechi2=np.array(postfit_bluechi2)[np.where(postfit_allmasked==0)[0]]
        postfit_redchi2=np.array(postfit_redchi2)[np.where(postfit_allmasked==0)[0]]
        postfit_chi22=np.array(postfit_chi22)[np.where(postfit_allmasked==0)[0]]
        postfit_n=np.array(postfit_n)[np.where(postfit_allmasked==0)[0]]
        postfit_bluen=np.array(postfit_bluen)[np.where(postfit_allmasked==0)[0]]
        postfit_redn=np.array(postfit_redn)[np.where(postfit_allmasked==0)[0]]
        postfit_bluespec=np.array(postfit_bluespec)[np.where(postfit_allmasked==0)[0]]
        postfit_redspec=np.array(postfit_redspec)[np.where(postfit_allmasked==0)[0]]
        postfit_wav_npoints=np.array(postfit_wav_npoints)[np.where(postfit_allmasked==0)[0]]
        postfit_wav_rms=np.array(postfit_wav_rms)[np.where(postfit_allmasked==0)[0]]
        postfit_wav_resolution=np.array(postfit_wav_resolution)[np.where(postfit_allmasked==0)[0]]
        postfit_wav_min=np.array(postfit_wav_min)[np.where(postfit_allmasked==0)[0]]
        postfit_wav_max=np.array(postfit_wav_max)[np.where(postfit_allmasked==0)[0]]
        postfit_row=np.array(postfit_row)[np.where(postfit_allmasked==0)[0]]
        postfit_temperature=np.array(postfit_temperature)[np.where(postfit_allmasked==0)[0]]
        postfit_filename=np.array(postfit_filename)[np.where(postfit_allmasked==0)[0]]
        postfit_exptime=np.array(postfit_exptime)[np.where(postfit_allmasked==0)[0]]
        postfit_nsubexp=np.array(postfit_nsubexp)[np.where(postfit_allmasked==0)[0]]
        postfit_airmass=np.array(postfit_airmass)[np.where(postfit_allmasked==0)[0]]
        
        filtername_fix=np.where(postfit_filtername=='HiRes?')[0]
        if len(filtername_fix)>0:
            postfit_filtername[filtername_fix]='HiRes'

        filtername_fix=np.where(postfit_filtername=='LoRes')[0]
        if len(filtername_fix)>0:
            postfit_filtername=np.full(len(postfit_filtername),'MedRes',dtype='U20')
    
        v=np.array([postfit_mean[q][0] for q in range(0,len(postfit_mean))])
        v_sig=np.array([postfit_std[q][0] for q in range(0,len(postfit_std))])
        v_skew=np.array([postfit_skew[q][0] for q in range(0,len(postfit_skew))])
        v_kurt=np.array([postfit_kurtosis[q][0] for q in range(0,len(postfit_kurtosis))])
        teff=np.array([postfit_mean[q][1] for q in range(0,len(postfit_mean))])
        teff_sig=np.array([postfit_std[q][1] for q in range(0,len(postfit_std))])
        teff_skew=np.array([postfit_skew[q][1] for q in range(0,len(postfit_skew))])
        teff_kurt=np.array([postfit_kurtosis[q][1] for q in range(0,len(postfit_kurtosis))])
        logg=np.array([postfit_mean[q][2] for q in range(0,len(postfit_mean))])
        logg_sig=np.array([postfit_std[q][2] for q in range(0,len(postfit_std))])
        logg_skew=np.array([postfit_skew[q][2] for q in range(0,len(postfit_skew))])
        logg_kurt=np.array([postfit_kurtosis[q][2] for q in range(0,len(postfit_kurtosis))])
        z=np.array([postfit_mean[q][3] for q in range(0,len(postfit_mean))])
        z_sig=np.array([postfit_std[q][3] for q in range(0,len(postfit_std))])
        z_skew=np.array([postfit_skew[q][3] for q in range(0,len(postfit_skew))])
        z_kurt=np.array([postfit_kurtosis[q][3] for q in range(0,len(postfit_kurtosis))])
        mgfe=np.array([postfit_mean[q][4] for q in range(0,len(postfit_mean))])
        mgfe_sig=np.array([postfit_std[q][4] for q in range(0,len(postfit_std))])
        mgfe_skew=np.array([postfit_skew[q][4] for q in range(0,len(postfit_skew))])
        mgfe_kurt=np.array([postfit_kurtosis[q][4] for q in range(0,len(postfit_kurtosis))])
        smooth=np.array([postfit_mean[q][11] for q in range(0,len(postfit_mean))])
        smooth_sig=np.array([postfit_std[q][11] for q in range(0,len(postfit_std))])
        smooth_skew=np.array([postfit_skew[q][11] for q in range(0,len(postfit_skew))])
        smooth_kurt=np.array([postfit_kurtosis[q][11] for q in range(0,len(postfit_kurtosis))])
        phantom1=np.array([postfit_mean[q][14] for q in range(0,len(postfit_mean))])
        phantom1_sig=np.array([postfit_std[q][14] for q in range(0,len(postfit_std))])
        phantom1_skew=np.array([postfit_skew[q][14] for q in range(0,len(postfit_skew))])
        phantom1_kurt=np.array([postfit_kurtosis[q][14] for q in range(0,len(postfit_kurtosis))])
        phantom2=np.array([postfit_mean[q][15] for q in range(0,len(postfit_mean))])
        phantom2_sig=np.array([postfit_std[q][15] for q in range(0,len(postfit_std))])
        phantom2_skew=np.array([postfit_skew[q][15] for q in range(0,len(postfit_skew))])
        phantom2_kurt=np.array([postfit_kurtosis[q][15] for q in range(0,len(postfit_kurtosis))])
    
        teffprior_v=np.array([postfit_teffprior_mean[q][0] for q in range(0,len(postfit_teffprior_mean))])
        teffprior_v_sig=np.array([postfit_teffprior_std[q][0] for q in range(0,len(postfit_teffprior_std))])
        teffprior_v_skew=np.array([postfit_teffprior_skew[q][0] for q in range(0,len(postfit_teffprior_skew))])
        teffprior_v_kurt=np.array([postfit_teffprior_kurtosis[q][0] for q in range(0,len(postfit_teffprior_kurtosis))])
        teffprior_teff=np.array([postfit_teffprior_mean[q][1] for q in range(0,len(postfit_teffprior_mean))])
        teffprior_teff_sig=np.array([postfit_teffprior_std[q][1] for q in range(0,len(postfit_teffprior_std))])
        teffprior_teff_skew=np.array([postfit_teffprior_skew[q][1] for q in range(0,len(postfit_teffprior_skew))])
        teffprior_teff_kurt=np.array([postfit_teffprior_kurtosis[q][1] for q in range(0,len(postfit_teffprior_kurtosis))])
        teffprior_logg=np.array([postfit_teffprior_mean[q][2] for q in range(0,len(postfit_teffprior_mean))])
        teffprior_logg_sig=np.array([postfit_teffprior_std[q][2] for q in range(0,len(postfit_teffprior_std))])
        teffprior_logg_skew=np.array([postfit_teffprior_skew[q][2] for q in range(0,len(postfit_teffprior_skew))])
        teffprior_logg_kurt=np.array([postfit_teffprior_kurtosis[q][2] for q in range(0,len(postfit_teffprior_kurtosis))])
        teffprior_z=np.array([postfit_teffprior_mean[q][3] for q in range(0,len(postfit_teffprior_mean))])
        teffprior_z_sig=np.array([postfit_teffprior_std[q][3] for q in range(0,len(postfit_teffprior_std))])
        teffprior_z_skew=np.array([postfit_teffprior_skew[q][3] for q in range(0,len(postfit_teffprior_skew))])
        teffprior_z_kurt=np.array([postfit_teffprior_kurtosis[q][3] for q in range(0,len(postfit_teffprior_kurtosis))])
        teffprior_mgfe=np.array([postfit_teffprior_mean[q][4] for q in range(0,len(postfit_teffprior_mean))])
        teffprior_mgfe_sig=np.array([postfit_teffprior_std[q][4] for q in range(0,len(postfit_teffprior_std))])
        teffprior_mgfe_skew=np.array([postfit_teffprior_skew[q][4] for q in range(0,len(postfit_teffprior_skew))])
        teffprior_mgfe_kurt=np.array([postfit_teffprior_kurtosis[q][4] for q in range(0,len(postfit_teffprior_kurtosis))])
        teffprior_smooth=np.array([postfit_teffprior_mean[q][11] for q in range(0,len(postfit_teffprior_mean))])
        teffprior_smooth_sig=np.array([postfit_teffprior_std[q][11] for q in range(0,len(postfit_teffprior_std))])
        teffprior_smooth_skew=np.array([postfit_teffprior_skew[q][11] for q in range(0,len(postfit_teffprior_skew))])
        teffprior_smooth_kurt=np.array([postfit_teffprior_kurtosis[q][11] for q in range(0,len(postfit_teffprior_kurtosis))])
        teffprior_phantom1=np.array([postfit_teffprior_mean[q][14] for q in range(0,len(postfit_teffprior_mean))])
        teffprior_phantom1_sig=np.array([postfit_teffprior_std[q][14] for q in range(0,len(postfit_teffprior_std))])
        teffprior_phantom1_skew=np.array([postfit_teffprior_skew[q][14] for q in range(0,len(postfit_teffprior_skew))])
        teffprior_phantom1_kurt=np.array([postfit_teffprior_kurtosis[q][14] for q in range(0,len(postfit_teffprior_kurtosis))])
        teffprior_phantom2=np.array([postfit_teffprior_mean[q][15] for q in range(0,len(postfit_teffprior_mean))])
        teffprior_phantom2_sig=np.array([postfit_teffprior_std[q][15] for q in range(0,len(postfit_teffprior_std))])
        teffprior_phantom2_skew=np.array([postfit_teffprior_skew[q][15] for q in range(0,len(postfit_teffprior_skew))])
        teffprior_phantom2_kurt=np.array([postfit_teffprior_kurtosis[q][15] for q in range(0,len(postfit_teffprior_kurtosis))])
        teffprior_survey=np.array([postfit_teffprior_survey[q] for q in range(0,len(postfit_teffprior_survey))])
    
        v=v+postfit_vhelio_correction#apply heliocentrcic correction!!!
        teffprior_v=teffprior_v+postfit_vhelio_correction#apply heliocentric correction!!!
    
        g1=open(data_out,'w')
        g1.write('#Object # RA [deg] # Dec [deg] # HJD [days] # median S/N/pix # vhelio [km/s] # err_vhelio [km/s] # skew_vhelio # kurt_vhelio # Teff [K] # err_Teff [K] # skew_Teff # kurt_Teff # logg # err_logg # skew_logg # kurt_logg # Fe/H # err_Fe/H # skew_Fe/H # kurt_Fe/H # [Mg/Fe] # err_[Mg/Fe] # skew_[Mg/Fe] # kurt_[Mg/Fe] # resolution \n ')
        for i in range(0,len(postfit_radeg)):
            if postfit_objtype[i]=='TARGET':
                string=postfit_object[i]+' '+str.format('{0:.7f}',round(postfit_radeg[i],7)).zfill(7)+' '+str.format('{0:.7f}',round(postfit_decdeg[i],7)).zfill(7)+' '+str.format('{0:.3f}',round(postfit_hjd[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(postfit_snratio[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(v[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(v_sig[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(v_skew[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(v_kurt[i],3)).zfill(3)+' '+str.format('{0:.1f}',round(teff[i],1)).zfill(1)+' '+str.format('{0:.1f}',round(teff_sig[i],1)).zfill(1)+' '+str.format('{0:.1f}',round(teff_skew[i],1)).zfill(1)+' '+str.format('{0:.1f}',round(teff_kurt[i],1)).zfill(1)+' '+str.format('{0:.3f}',round(logg[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(logg_sig[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(logg_skew[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(logg_kurt[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(z[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(z_sig[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(z_skew[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(z_kurt[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(mgfe[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(mgfe_sig[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(mgfe_skew[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(mgfe_kurt[i],3)).zfill(3)+' '+postfit_filtername[i]+' \n'
                print(string)
                if ((postfit_snratio[i]==postfit_snratio[i])&(not((v_kurt[i]==0.)&(v_kurt[i]==0.)&(z_kurt[i]==0.)&(z_skew[i]==0.)&(logg_kurt[i]==0.)&(logg_skew[i]==0.)))):
                    g1.write(string)
        g1.close()

        targets=np.where((postfit_objtype=='TARGET')&(v_sig>1.e-10)&(teff_sig>1.e-10)&(logg_sig>1.e-10)&(z_sig>1.e-10))[0]
        col1=fits.Column(name='instrument',format='A20',array=np.full(len(targets),'M2FS_'+filtername0,dtype='U20'),unit='')
        col2=fits.Column(name='target_system',format='A100',array=postfit_object[targets],unit='')
        col3=fits.Column(name='obj_id',format='A100',array=postfit_obsid[targets],unit='')
        col4=fits.Column(name='ra',format='D',array=postfit_radeg[targets],unit='degree')
        col5=fits.Column(name='dec',format='D',array=postfit_decdeg[targets],unit='degree')
        col6=fits.Column(name='ra_dec_source',format='A30',array=postfit_ra_dec_source[targets],unit='')
        col7=fits.Column(name='hjd',format='D',array=postfit_hjd[targets],unit='day')
        col8=fits.Column(name='sn_ratio',format='D',array=postfit_snratio[targets],unit='')

        col9=fits.Column(name='vlos_raw',format='D',array=v[targets],unit='km/s')
        col10=fits.Column(name='vlos_raw_error',format='D',array=v_sig[targets],unit='km/s')
        col11=fits.Column(name='vlos_raw_skew',format='D',array=v_skew[targets],unit='')
        col12=fits.Column(name='vlos_raw_kurtosis',format='D',array=v_kurt[targets],unit='')
        col13=fits.Column(name='vlos_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='km/s')
        col14=fits.Column(name='teff_raw',format='D',array=teff[targets],unit='Kelvin')
        col15=fits.Column(name='teff_raw_error',format='D',array=teff_sig[targets],unit='Kelvin')
        col16=fits.Column(name='teff_raw_skew',format='D',array=teff_skew[targets],unit='')
        col17=fits.Column(name='teff_raw_kurtosis',format='D',array=teff_kurt[targets],unit='')
        col18=fits.Column(name='teff_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='Kelvin')
        col19=fits.Column(name='logg_raw',format='D',array=logg[targets],unit='dex (cgs)')
        col20=fits.Column(name='logg_raw_error',format='D',array=logg_sig[targets],unit='dex (cgs)')
        col21=fits.Column(name='logg_raw_skew',format='D',array=logg_skew[targets],unit='')
        col22=fits.Column(name='logg_raw_kurtosis',format='D',array=logg_kurt[targets],unit='')
        col23=fits.Column(name='logg_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='dex (cgs)')
        col24=fits.Column(name='feh_raw',format='D',array=z[targets],unit='dex')
        col25=fits.Column(name='feh_raw_error',format='D',array=z_sig[targets],unit='dex')
        col26=fits.Column(name='feh_raw_skew',format='D',array=z_skew[targets],unit='')
        col27=fits.Column(name='feh_raw_kurtosis',format='D',array=z_kurt[targets],unit='')
        col28=fits.Column(name='feh_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='dex')
        col29=fits.Column(name='mgfe_raw',format='D',array=mgfe[targets],unit='dex')
        col30=fits.Column(name='mgfe_raw_error',format='D',array=mgfe_sig[targets],unit='dex')
        col31=fits.Column(name='mgfe_raw_skew',format='D',array=mgfe_skew[targets],unit='')
        col32=fits.Column(name='mgfe_raw_kurtosis',format='D',array=mgfe_kurt[targets],unit='')
        col33=fits.Column(name='mgfe_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='dex')
        col34=fits.Column(name='smooth_raw',format='D',array=smooth[targets],unit='Angstrom')
        col35=fits.Column(name='smooth_raw_error',format='D',array=smooth_sig[targets],unit='Angstrom')
        col36=fits.Column(name='smooth_raw_skew',format='D',array=smooth_skew[targets],unit='')
        col37=fits.Column(name='smooth_raw_kurtosis',format='D',array=smooth_kurt[targets],unit='')
        col38=fits.Column(name='smooth_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='Angstrom')

        col39=fits.Column(name='teffprior_vlos_raw',format='D',array=teffprior_v[targets],unit='km/s')
        col40=fits.Column(name='teffprior_vlos_raw_error',format='D',array=teffprior_v_sig[targets],unit='km/s')
        col41=fits.Column(name='teffprior_vlos_raw_skew',format='D',array=teffprior_v_skew[targets],unit='')
        col42=fits.Column(name='teffprior_vlos_raw_kurtosis',format='D',array=teffprior_v_kurt[targets],unit='')
        col43=fits.Column(name='teffprior_vlos_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='km/s')
        col44=fits.Column(name='teffprior_teff_raw',format='D',array=teffprior_teff[targets],unit='Kelvin')
        col45=fits.Column(name='teffprior_teff_raw_error',format='D',array=teffprior_teff_sig[targets],unit='Kelvin')
        col46=fits.Column(name='teffprior_teff_raw_skew',format='D',array=teffprior_teff_skew[targets],unit='')
        col47=fits.Column(name='teffprior_teff_raw_kurtosis',format='D',array=teffprior_teff_kurt[targets],unit='')
        col48=fits.Column(name='teffprior_teff_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='Kelvin')
        col49=fits.Column(name='teffprior_logg_raw',format='D',array=teffprior_logg[targets],unit='dex (cgs)')
        col50=fits.Column(name='teffprior_logg_raw_error',format='D',array=teffprior_logg_sig[targets],unit='dex (cgs)')
        col51=fits.Column(name='teffprior_logg_raw_skew',format='D',array=teffprior_logg_skew[targets],unit='')
        col52=fits.Column(name='teffprior_logg_raw_kurtosis',format='D',array=teffprior_logg_kurt[targets],unit='')
        col53=fits.Column(name='teffprior_logg_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='dex (cgs)')
        col54=fits.Column(name='teffprior_feh_raw',format='D',array=teffprior_z[targets],unit='dex')
        col55=fits.Column(name='teffprior_feh_raw_error',format='D',array=teffprior_z_sig[targets],unit='dex')
        col56=fits.Column(name='teffprior_feh_raw_skew',format='D',array=teffprior_z_skew[targets],unit='')
        col57=fits.Column(name='teffprior_feh_raw_kurtosis',format='D',array=teffprior_z_kurt[targets],unit='')
        col58=fits.Column(name='teffprior_feh_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='dex')
        col59=fits.Column(name='teffprior_mgfe_raw',format='D',array=teffprior_mgfe[targets],unit='dex')
        col60=fits.Column(name='teffprior_mgfe_raw_error',format='D',array=teffprior_mgfe_sig[targets],unit='dex')
        col61=fits.Column(name='teffprior_mgfe_raw_skew',format='D',array=teffprior_mgfe_skew[targets],unit='')
        col62=fits.Column(name='teffprior_mgfe_raw_kurtosis',format='D',array=teffprior_mgfe_kurt[targets],unit='')
        col63=fits.Column(name='teffprior_mgfe_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='dex')
        col64=fits.Column(name='teffprior_smooth_raw',format='D',array=teffprior_smooth[targets],unit='Angstrom')
        col65=fits.Column(name='teffprior_smooth_raw_error',format='D',array=teffprior_smooth_sig[targets],unit='Angstrom')
        col66=fits.Column(name='teffprior_smooth_raw_skew',format='D',array=teffprior_smooth_skew[targets],unit='')
        col67=fits.Column(name='teffprior_smooth_raw_kurtosis',format='D',array=teffprior_smooth_kurt[targets],unit='')
        col68=fits.Column(name='teffprior_smooth_error',format='D',array=np.full(len(targets),np.nan,dtype='float'),unit='Angstrom')
        col69=fits.Column(name='teffprior_survey',format='A10',array=teffprior_survey[targets],unit='')

        col70=fits.Column(name='median_sky',format='D',array=postfit_mediansky[targets],unit='electron')
        col71=fits.Column(name='median_skysub',format='D',array=postfit_medianskysub[targets],unit='electron')
        col72=fits.Column(name='standard_deviation_median_sky',format='D',array=postfit_stdsky[targets],unit='electron')
        col73=fits.Column(name='filter_name',format='A100',array=postfit_filtername[targets],unit='')
        col74=fits.Column(name='chi2',format='D',array=postfit_chi2[targets],unit='')
        col75=fits.Column(name='chi2_rescaled',format='D',array=postfit_chi22[targets],unit='')
        col76=fits.Column(name='npix',format='D',array=postfit_n[targets],unit='')
        col77=fits.Column(name='w5163',format='D',array=postfit_bluespec[targets],unit='electron')
        col78=fits.Column(name='w5180',format='D',array=postfit_redspec[targets],unit='electron')
        col79=fits.Column(name='vhelio_correction',format='D',array=postfit_vhelio_correction[targets],unit='km/s')
        col80=fits.Column(name='fits_filename',format='A200',array=postfit_filename[targets],unit='')
        col81=fits.Column(name='fits_index',format='I',array=postfit_index[targets],unit='')

        col82=fits.Column(name='gaia_source_id',format='K',array=postfit_gaiaid[targets],unit='')
        col83=fits.Column(name='gaia_gmag',format='D',array=postfit_gaiagmag[targets],unit='mag')
        col84=fits.Column(name='gaia_siggmag',format='D',array=postfit_siggaiagmag[targets],unit='mag')
        col85=fits.Column(name='gaia_gmag_dered',format='D',array=postfit_gaiagmag_dered[targets],unit='mag')
        col86=fits.Column(name='gaia_bpmag',format='D',array=postfit_gaiabpmag[targets],unit='mag')
        col87=fits.Column(name='gaia_sigbpmag',format='D',array=postfit_siggaiabpmag[targets],unit='mag')
        col88=fits.Column(name='gaia_bpmag_dered',format='D',array=postfit_gaiabpmag_dered[targets],unit='mag')
        col89=fits.Column(name='gaia_rpmag',format='D',array=postfit_gaiarpmag[targets],unit='mag')
        col90=fits.Column(name='gaia_sigrpmag',format='D',array=postfit_siggaiarpmag[targets],unit='mag')
        col91=fits.Column(name='gaia_rpmag_dered',format='D',array=postfit_gaiarpmag_dered[targets],unit='mag')
        col92=fits.Column(name='gaia_pmra',format='D',array=postfit_gaiapmra[targets],unit='mas/yr')
        col93=fits.Column(name='gaia_sigpmra',format='D',array=postfit_siggaiapmra[targets],unit='mas/yr')
        col94=fits.Column(name='gaia_pmdec',format='D',array=postfit_gaiapmdec[targets],unit='mas/yr')
        col95=fits.Column(name='gaia_sigpmdec',format='D',array=postfit_siggaiapmdec[targets],unit='mas/yr')
        col96=fits.Column(name='gaia_parallax',format='D',array=postfit_gaiaparallax[targets],unit='mas')
        col97=fits.Column(name='gaia_sigparallax',format='D',array=postfit_siggaiaparallax[targets],unit='mas')
        col98=fits.Column(name='gaia_rrl',format='I',array=postfit_gaia_rrl[targets],unit='')
        col99=fits.Column(name='gaia_cepheid',format='I',array=postfit_gaia_cepheid[targets],unit='')
        col100=fits.Column(name='gaia_agn',format='I',array=postfit_gaia_agn[targets],unit='')
        col101=fits.Column(name='gaia_rv_variable',format='I',array=postfit_gaia_rvvariable[targets],unit='')
        col102=fits.Column(name='gaia_compact_companion',format='I',array=postfit_gaia_compact[targets],unit='')
        col103=fits.Column(name='gaia_radial_velocity',format='D',array=postfit_gaia_rv[targets],unit='km/s')
        col104=fits.Column(name='gaia_radial_velocity_error',format='D',array=postfit_gaia_sigrv[targets],unit='km/s')
        col105=fits.Column(name='gaia_phot_variable_flag',format='A20',array=postfit_gaia_phot_variable[targets],unit='')
        col106=fits.Column(name='gaia_in_qso_candidates',format='L',array=postfit_gaia_in_qso[targets],unit='')
        col107=fits.Column(name='gaia_in_galaxy_candidates',format='L',array=postfit_gaia_in_galaxy[targets],unit='')
        col108=fits.Column(name='gaia_non_single_star',format='I',array=postfit_gaia_in_blend[targets],unit='')
        col109=fits.Column(name='gaia_teff_gspphot',format='D',array=postfit_gaia_teff[targets],unit='Kelvin')
        col110=fits.Column(name='gaia_teff_gspphot_lower',format='D',array=postfit_gaia_teff_lower[targets],unit='Kelvin')
        col111=fits.Column(name='gaia_teff_gspphot_upper',format='D',array=postfit_gaia_teff_upper[targets],unit='Kelvin')
        col112=fits.Column(name='gaia_logg_gspphot',format='D',array=postfit_gaia_logg[targets],unit='dex (cgs)')
        col113=fits.Column(name='gaia_logg_gspphot_lower',format='D',array=postfit_gaia_logg_lower[targets],unit='dex (cgs)')
        col114=fits.Column(name='gaia_logg_gspphot_upper',format='D',array=postfit_gaia_logg_upper[targets],unit='dex (cgs)')
        col115=fits.Column(name='gaia_mh_gspphot',format='D',array=postfit_gaia_feh[targets],unit='dex')
        col116=fits.Column(name='gaia_mh_gspphot_lower',format='D',array=postfit_gaia_feh_lower[targets],unit='dex')
        col117=fits.Column(name='gaia_mh_gspphot_upper',format='D',array=postfit_gaia_feh_upper[targets],unit='dex')
        col118=fits.Column(name='gaia_libname_gspphot',format='A30',array=postfit_gaia_libname[targets],unit='')
        col119=fits.Column(name='exptime',format='D',array=postfit_exptime[targets],unit='second')
        col120=fits.Column(name='n_subexp',format='D',array=postfit_nsubexp[targets],unit='')
        col121=fits.Column(name='airmass',format='D',array=postfit_airmass[targets],unit='')
        col122=fits.Column(name='run_id',format='A100',array=postfit_run_id[targets],unit='')
        col123=fits.Column(name='wav_npoints',format='PI()',array=postfit_wav_npoints[targets],unit='')
        col124=fits.Column(name='wav_rms',format='PD()',array=postfit_wav_rms[targets],unit='Angstrom')
        col125=fits.Column(name='wav_resolution',format='PD()',array=postfit_wav_resolution[targets],unit='Angstrom')
        col126=fits.Column(name='wav_min',format='PD()',array=postfit_wav_min[targets],unit='Angstrom')
        col127=fits.Column(name='wav_max',format='PD()',array=postfit_wav_max[targets],unit='Angstrom')
        col128=fits.Column(name='row',format='D',array=postfit_row[targets],unit='')
        col129=fits.Column(name='temperature',format='PD()',array=postfit_temperature[targets],unit='degree Celcius')
        cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41,col42,col43,col44,col45,col46,col47,col48,col49,col50,col51,col52,col53,col54,col55,col56,col57,col58,col59,col60,col61,col62,col63,col64,col65,col66,col67,col68,col69,col70,col71,col72,col73,col74,col75,col76,col77,col78,col79,col80,col81,col82,col83,col84,col85,col86,col87,col88,col89,col90,col91,col92,col93,col94,col95,col96,col97,col98,col99,col100,col101,col102,col103,col104,col105,col106,col107,col108,col109,col110,col111,col112,col113,col114,col115,col116,col117,col118,col119,col120,col121,col122,col123,col124,col125,col126,col127,col128,col129])

        t=fits.BinTableHDU.from_columns(cols)
        t.writeto(fits_table_raw_filename,overwrite=True)

if get_errors:

    g0=open('m2fs_error_adjust.tex','w')
    
    for filtername0 in ['HiRes','MedRes']:
        if filtername0=='HiRes':
            fits_table_raw_filename=fits_list.split('_files')[0]+'_HiRes_raw.fits'
            vlos_error_max=5.
        if filtername0=='MedRes':
            fits_table_raw_filename=fits_list.split('_files')[0]+'_MedRes_raw.fits'
            vlos_error_max=5.
            
        catalog=fits.open(fits_table_raw_filename)
        order=np.argsort(catalog[1].data['hjd'])
        catalog_sorted=catalog[1].data[order]
        gaiaid=np.array(catalog_sorted['gaia_source_id'],dtype='int')
        keep=np.where((catalog_sorted['vlos_raw']==catalog_sorted['vlos_raw'])&(catalog_sorted['vlos_raw_error']<vlos_error_max)&(catalog_sorted['filter_name']==filtername0)&(catalog_sorted['sn_ratio']>0.)&(gaiaid>0.)&(catalog_sorted['gaia_rrl']==0))[0]
#    keep=np.where((catalog_sorted['vlos_raw']==catalog_sorted['vlos_raw'])&(catalog_sorted['vlos_raw_error']<5.)&(catalog_sorted['filter_name']=='HiRes')&(catalog_sorted['sn_ratio']>0.)&(gaiaid>0.)&(catalog_sorted['gaia_rrl']==0))[0]
#    keep=np.where((catalog_sorted['vlos_raw']==catalog_sorted['vlos_raw'])&(catalog_sorted['vlos_raw_error']<5.)&(catalog_sorted['filtername']=='HiRes')&(catalog_sorted['sn_ratio']>0.)&(gaiaid>0.))[0]
        catalog_keep=catalog_sorted[keep]
        used=np.zeros(len(catalog_keep),dtype='int')
        count=np.zeros(len(catalog_keep),dtype='int')
        obs=np.zeros(len(catalog_keep),dtype='int')
        nobs=np.zeros(len(catalog_keep),dtype='int')

        wav_rms=[]
        for i in range(0,len(catalog[1].data)):
            for k in range(0,len(catalog[1].data['wav_rms'][i])):
                wav_rms.append(catalog[1].data['wav_rms'][i][k])
        wav_rms=np.array(wav_rms)
        keep=np.where((wav_rms>np.percentile(wav_rms,1))&(wav_rms<np.percentile(wav_rms,99)))[0]
        string='\\newcommand{\mtwofs'+filtername0+'wavrmsn}{$'+str(len(wav_rms))+'$} \n'
        g0.write(string)
        string='\\newcommand{\mtwofs'+filtername0+'wavrmsmean}{$'+str(round(np.mean(wav_rms[keep]),3))+'$} \n'
        g0.write(string)
        string='\\newcommand{\mtwofs'+filtername0+'wavrmsstd}{$'+str(round(np.std(wav_rms[keep]),3))+'$} \n'
        g0.write(string)

        count0=0
        for i in range(0,len(catalog_keep)):
            if used[i]==0:
                count0+=1
                dist=np.sqrt((1./np.cos(catalog_keep['dec']*np.pi/180.)*(catalog_keep['ra']-catalog_keep['ra'][i]))**2+(catalog_keep['dec']-catalog_keep['dec'][i])**2)*3600.
                this=np.where(dist<1.)[0]
                used[this]=1
                nobs[this]=len(this)
                count[this]=count0
                for j in range(0,len(this)):
                    obs[this[j]]=j+1

        repeat_ind1=[]
        repeat_ind2=[]
        for i in range(0,len(catalog_keep)):
            if ((obs[i]==1)&(nobs[i]>1)):
                this=np.where(count==count[i])[0]
                for j in range(0,len(this)):
                    for k in range(j+1,len(this)):
                        repeat_ind1.append(this[j])
                        repeat_ind2.append(this[k])
        repeat_ind1=np.array(repeat_ind1)
        repeat_ind2=np.array(repeat_ind2)
        string='\\newcommand{\mtwofsnrepeatv}{'+str(len(repeat_ind1))+'} \n'
        g0.write(string)
    
        thing_newcommand1=['mtwofssigvfloor','mtwofsteffsigvfloor','mtwofssigtefffloor','mtwofsteffsigtefffloor','mtwofssigloggfloor','mtwofsteffsigloggfloor','mtwofssigfehfloor','mtwofsteffsigfehfloor','mtwofssigmgfefloor','mtwofsteffsigmgfefloor','mtwofssigresolutionfloor','mtwofsteffsigresolutionfloor']
        thing_newcommand2=['mtwofssigvscale','mtwofsteffsigvscale','mtwofssigteffscale','mtwofsteffsigteffscale','mtwofssigloggscale','mtwofsteffsigloggscale','mtwofssigfehscale','mtwofsteffsigfehscale','mtwofssigmgfescale','mtwofsteffsigmgfescale','mtwofssigresolutionscale','mtwofsteffsigresolutionscale']
        thing_newcommand3=['mtwofssigvmixfrac','mtwofsteffsigvmixfrac','mtwofssigteffmixfrac','mtwofsteffsigteffmixfrac','mtwofssigloggmixfrac','mtwofsteffsigloggmixfrac','mtwofssigfehmixfrac','mtwofsteffsigfehmixfrac','mtwofssigmgfemixfrac','mtwofsteffsigmgfemixfrac','mtwofssigresolutionmixfrac','mtwofsteffsigresolutionmixfrac']
        thing_newcommand4=['mtwofssigvsigout','mtwofsteffsigvsigout','mtwofssigteffsigout','mtwofsteffsigteffsigout','mtwofssigloggsigout','mtwofsteffsigloggsigout','mtwofssigfehsigout','mtwofsteffsigfehsigout','mtwofssigmgfesigout','mtwofsteffsigmgfesigout','mtwofssigresolutionsigout','mtwofsteffsigresolutionsigout']
        thing_newcommand5=['mtwofsvnpairs','mtwofsteffvnpairs','mtwofsteffnpairs','mtwofsteffteffnpairs','mtwofsloggnpairs','mtwofsteffloggnpairs','mtwofsfehnpairs','mtwofstefffehnpairs','mtwofsmgfenpairs','mtwofsteffmgfenpairs','mtwofsresolutionnpairs','mtwofsteffresolutionnpairs']
        thing_dmax=np.array([100.,100.,2000.,2000.,2.5,2.5,2.5,2.5,1.,1.,0.03,0.03])
        floor=np.array([[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.]])
        scale=np.array([[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.]])
        frac=np.array([[0.51,1.],[0.51,1.],[0.51,1.],[0.51,1.],[0.51,1.],[0.51,1.],[0.51,1.],[0.51,1.],[0.51,1.],[0.51,1.],[0.51,1.],[0.51,1.]])
        sigmaout=np.array([[-1.,3.],[-1.,3.],[-1.,4.],[-1.,4.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.]])
        for k in range(0,len(thing)):
            x=catalog_keep[thing[k]+'_raw']
            sigx=catalog_keep[thing[k]+'_raw_error']
            dx=x[repeat_ind1]-x[repeat_ind2]
            keep=np.where(np.abs(dx)<=thing_dmax[k])[0]

            result,bestfit=m2fs.get_error_adjust(x,sigx,floor[k],scale[k],frac[k],sigmaout[k],repeat_ind1[keep],repeat_ind2[keep],thing[k],thing_symbol[k],thing_dmax[k])

            pickle.dump((x,sigx,repeat_ind1,repeat_ind2,thing[k],thing_symbol[k],thing_dmax[k],result,bestfit),open('m2fs_ian_'+thing[k]+'_'+filtername0+'_adjust.pkl','wb'))
            string='\\newcommand{\\'+thing_newcommand1[k]+filtername0+'}{$'+str('{0:.2f}'.format(np.mean(10.**result['samples'].T[0])))+' \pm '+str('{0:.2f}'.format(np.std(10.**result['samples'].T[0])))+'$} \n'
            g0.write(string)
            string='\\newcommand{\\'+thing_newcommand2[k]+filtername0+'}{$'+str('{0:.2f}'.format(np.mean(10.**result['samples'].T[1])))+' \pm '+str('{0:.2f}'.format(np.std(10.**result['samples'].T[1])))+'$} \n'
            g0.write(string)
            string='\\newcommand{\\'+thing_newcommand3[k]+filtername0+'}{$'+str('{0:.2f}'.format(1.-np.mean(result['samples'].T[2])))+' \pm '+str('{0:.2f}'.format(np.std(result['samples'].T[2])))+'$} \n'
            g0.write(string)
            string='\\newcommand{\\'+thing_newcommand4[k]+filtername0+'}{$'+str('{0:.2f}'.format(np.mean(10.**result['samples'].T[3])))+' \pm '+str('{0:.2f}'.format(np.std(10.**result['samples'].T[3])))+'$} \n'
            g0.write(string)
            string='\\newcommand{\\'+thing_newcommand5[k]+filtername0+'}{$'+str(len(keep))+'$} \n'
            g0.write(string)
            
    g0.close()

    
if get_catalog_final:

    m2fs_rho0=1.2
    m2fs_gamma=0.
    m2fs_beta=3.
    m2fs_rs=25.
    m2fs_alpha=1.

    for filtername0 in ['MedRes','HiRes']:
        if filtername0=='HiRes':
            fits_table_raw_filename=fits_list.split('_files')[0]+'_HiRes_raw.fits'
            fits_table_final_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_HiRes_table.fits'

        if filtername0=='MedRes':
            fits_table_raw_filename=fits_list.split('_files')[0]+'_MedRes_raw.fits'
            fits_table_final_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_MedRes_table.fits'    
    
        catalog_raw=fits.open(fits_table_raw_filename)
        catalog=catalog_raw[1].data
        
        m2fs_chi2max=m2fs_rho0/(catalog['sn_ratio']/m2fs_rs)**m2fs_gamma/(1.+(catalog['sn_ratio']/m2fs_rs)**m2fs_alpha)**((m2fs_gamma-m2fs_beta)/m2fs_alpha)

        m2fs_ratiomin=1.8*10.**(-0.7*(1./catalog['sn_ratio'])**0.434)
        m2fs_ratiomin[m2fs_ratiomin>0.8]=0.8

        m2fs_chi2_outlier=np.where((catalog['chi2']>m2fs_chi2max))[0]
        m2fs_carbon=np.where((catalog['w5163']/catalog['w5180']<m2fs_ratiomin))[0]
        
        for k in range(0,len(thing)):
            print(thing[k])
            crap,crap,crap,crap,crap,crap,crap,samples,bestfit=pickle.load(open('m2fs_ian_'+thing[k]+'_'+filtername0+'_adjust.pkl','rb'))
            print(thing[k],10.**bestfit['parameters'][0],10.**bestfit['parameters'][1])
            catalog[thing[k]+'_error']=10.**bestfit['parameters'][0]+10.**bestfit['parameters'][1]*catalog_raw[1].data[thing[k]+'_raw_error']
        used=np.zeros(len(catalog),dtype='int')
        count=np.zeros(len(catalog),dtype='int')
        obs=np.zeros(len(catalog),dtype='int')
        goodobs=np.zeros(len(catalog),dtype='int')
        goodvobs=np.zeros(len(catalog),dtype='int')
        nobs=np.zeros(len(catalog),dtype='int')
        goodnobs=np.zeros(len(catalog),dtype='int')
        count0=0

        v_mean=np.zeros(len(catalog))
        teff_mean=np.zeros(len(catalog))
        logg_mean=np.zeros(len(catalog))
        z_mean=np.zeros(len(catalog))
        mgfe_mean=np.zeros(len(catalog))
        sigv_mean=np.zeros(len(catalog))
        sigteff_mean=np.zeros(len(catalog))
        siglogg_mean=np.zeros(len(catalog))
        sigfeh_mean=np.zeros(len(catalog))
        sigmgfe_mean=np.zeros(len(catalog))
        stdv_mean=np.zeros(len(catalog))
        stdteff_mean=np.zeros(len(catalog))
        stdlogg_mean=np.zeros(len(catalog))
        stdfeh_mean=np.zeros(len(catalog))
        stdmgfe_mean=np.zeros(len(catalog))
        
        teffprior_v_mean=np.zeros(len(catalog))
        teffprior_teff_mean=np.zeros(len(catalog))
        teffprior_logg_mean=np.zeros(len(catalog))
        teffprior_z_mean=np.zeros(len(catalog))
        teffprior_mgfe_mean=np.zeros(len(catalog))
        teffprior_sigv_mean=np.zeros(len(catalog))
        teffprior_sigteff_mean=np.zeros(len(catalog))
        teffprior_siglogg_mean=np.zeros(len(catalog))
        teffprior_sigfeh_mean=np.zeros(len(catalog))
        teffprior_sigmgfe_mean=np.zeros(len(catalog))
        teffprior_stdv_mean=np.zeros(len(catalog))
        teffprior_stdteff_mean=np.zeros(len(catalog))
        teffprior_stdlogg_mean=np.zeros(len(catalog))
        teffprior_stdfeh_mean=np.zeros(len(catalog))
        teffprior_stdmgfe_mean=np.zeros(len(catalog))

        any_chi2_flag=np.zeros(len(catalog))
        any_carbon_flag=np.zeros(len(catalog))
        vlos_variable_flag=np.zeros(len(catalog))
        teff_variable_flag=np.zeros(len(catalog))
        logg_variable_flag=np.zeros(len(catalog))
        feh_variable_flag=np.zeros(len(catalog))
        mgfe_variable_flag=np.zeros(len(catalog))
        
        for i in range(0,len(catalog)):
            if used[i]==0:
                count0+=1
                dist=np.sqrt((1./np.cos(catalog['dec']*np.pi/180.)*(catalog['ra']-catalog['ra'][i]))**2+(catalog['dec']-catalog['dec'][i])**2)*3600.
                this=np.where(dist<1.)[0]
                this_goodv=np.where((catalog['sn_ratio']>0.)&(catalog['vlos_raw']==catalog['vlos_raw'])&(dist<1.)&(catalog['vlos_raw_error']<5.))[0]
                this_goodteff=np.where((catalog['sn_ratio']>0.)&(catalog['vlos_raw']==catalog['vlos_raw'])&(dist<1.)&(catalog['vlos_raw_error']<5.)&(catalog['teff_raw_error']<100000.))[0]
                this_goodlogg=np.where((catalog['sn_ratio']>0.)&(catalog['vlos_raw']==catalog['vlos_raw'])&(dist<1.)&(catalog['vlos_raw_error']<5.)&(catalog['logg_raw_error']<1000.5))[0]
                this_goodz=np.where((catalog['sn_ratio']>0.)&(catalog['vlos_raw']==catalog['vlos_raw'])&(dist<1.)&(catalog['vlos_raw_error']<5.)&(catalog['logg_raw_error']<1000.5)&(catalog['feh_raw_error']<1000.5))[0]
                this_goodmgfe=np.where((catalog['sn_ratio']>0.)&(catalog['vlos_raw']==catalog['vlos_raw'])&(dist<1.)&(catalog['vlos_raw_error']<5.)&(catalog['logg_raw_error']<1000.5)&(catalog['feh_raw_error']<1000.5)&(catalog['mgfe_raw_error']<1000.3))[0]
                used[this]=1
                nobs[this]=len(this)
                goodnobs[this]=len(this_goodv)
                count[this]=count0
                for j in range(0,len(this)):
                    obs[this[j]]=j+1

                any_chi2=False
                any_carbon=False
            
                for j in range(0,len(this_goodv)):
                    goodobs[this_goodv[j]]=j+1
                    if this_goodv[j] in m2fs_chi2_outlier:
                        any_chi2=True
                    if this_goodv[j] in m2fs_carbon:
                        any_carbon=True
                    
                if any_chi2:
                    any_chi2_flag[this_goodv]=True
                if any_carbon:
                    any_carbon_flag[this_goodv]=True
                
                if len(this_goodv)>0:
                    mean,sigmean,std=mycode.weightedmean(catalog['vlos_raw'][this_goodv],catalog['vlos_error'][this_goodv])
                    v_mean[this]=mean
                    sigv_mean[this]=sigmean
                    stdv_mean[this]=std
                    mean,sigmean,std=mycode.weightedmean(catalog['teffprior_vlos_raw'][this_goodv],catalog['teffprior_vlos_error'][this_goodv])
                    teffprior_v_mean[this]=mean
                    teffprior_sigv_mean[this]=sigmean
                    teffprior_stdv_mean[this]=std
                else:
                    v_mean[this]=np.nan
                    sigv_mean[this]=np.nan
                    stdv_mean[this]=np.nan
                    teffprior_v_mean[this]=np.nan
                    teffprior_sigv_mean[this]=np.nan
                    teffprior_stdv_mean[this]=np.nan
                
                if len(this_goodteff)>0:
                    mean,sigmean,std=mycode.weightedmean(catalog['teff_raw'][this_goodteff],catalog['teff_error'][this_goodteff])
                    teff_mean[this]=mean
                    sigteff_mean[this]=sigmean
                    stdteff_mean[this]=std
                    mean,sigmean,std=mycode.weightedmean(catalog['teffprior_teff_raw'][this_goodteff],catalog['teffprior_teff_error'][this_goodteff])
                    teffprior_teff_mean[this]=mean
                    teffprior_sigteff_mean[this]=sigmean
                    teffprior_stdteff_mean[this]=std
                else:
                    teff_mean[this]=np.nan
                    sigteff_mean[this]=np.nan
                    stdteff_mean[this]=np.nan
                    teffprior_teff_mean[this]=np.nan
                    teffprior_sigteff_mean[this]=np.nan
                    teffprior_stdteff_mean[this]=np.nan

                if len(this_goodlogg)>0:
                    mean,sigmean,std=mycode.weightedmean(catalog['logg_raw'][this_goodlogg],catalog['logg_error'][this_goodlogg])
                    logg_mean[this]=mean
                    siglogg_mean[this]=sigmean
                    stdlogg_mean[this]=std
                    mean,sigmean,std=mycode.weightedmean(catalog['teffprior_logg_raw'][this_goodlogg],catalog['teffprior_logg_error'][this_goodlogg])
                    teffprior_logg_mean[this]=mean
                    teffprior_siglogg_mean[this]=sigmean
                    teffprior_stdlogg_mean[this]=std
                else:
                    logg_mean[this]=np.nan
                    siglogg_mean[this]=np.nan
                    stdlogg_mean[this]=np.nan
                    teffprior_logg_mean[this]=np.nan
                    teffprior_siglogg_mean[this]=np.nan
                    teffprior_stdlogg_mean[this]=np.nan

                if len(this_goodz)>0:
                    mean,sigmean,std=mycode.weightedmean(catalog['feh_raw'][this_goodz],catalog['feh_error'][this_goodz])
                    z_mean[this]=mean
                    sigfeh_mean[this]=sigmean
                    stdfeh_mean[this]=std
                    mean,sigmean,std=mycode.weightedmean(catalog['teffprior_feh_raw'][this_goodz],catalog['teffprior_feh_error'][this_goodz])
                    teffprior_z_mean[this]=mean
                    teffprior_sigfeh_mean[this]=sigmean
                    teffprior_stdfeh_mean[this]=std
                else:
                    z_mean[this]=np.nan
                    sigfeh_mean[this]=np.nan
                    stdfeh_mean[this]=np.nan
                    teffprior_z_mean[this]=np.nan
                    teffprior_sigfeh_mean[this]=np.nan
                    teffprior_stdfeh_mean[this]=np.nan

                if len(this_goodmgfe)>0:
                    mean,sigmean,std=mycode.weightedmean(catalog['mgfe_raw'][this_goodmgfe],catalog['mgfe_error'][this_goodmgfe])
                    mgfe_mean[this]=mean
                    sigmgfe_mean[this]=sigmean
                    stdmgfe_mean[this]=std
                    mean,sigmean,std=mycode.weightedmean(catalog['teffprior_mgfe_raw'][this_goodmgfe],catalog['teffprior_mgfe_error'][this_goodmgfe])
                    teffprior_mgfe_mean[this]=mean
                    teffprior_sigmgfe_mean[this]=sigmean
                    teffprior_stdmgfe_mean[this]=std
                else:
                    mgfe_mean[this]=np.nan
                    sigmgfe_mean[this]=np.nan
                    stdmgfe_mean[this]=np.nan
                    teffprior_mgfe_mean[this]=np.nan
                    teffprior_sigmgfe_mean[this]=np.nan
                    teffprior_stdmgfe_mean[this]=np.nan

        vlos_variable=np.where(stdv_mean>3.*sigv_mean)[0]
        teff_variable=np.where(stdteff_mean>3.*sigteff_mean)[0]
        logg_variable=np.where(stdlogg_mean>3.*siglogg_mean)[0]
        feh_variable=np.where(stdfeh_mean>3.*sigfeh_mean)[0]
        mgfe_variable=np.where(stdmgfe_mean>3.*sigmgfe_mean)[0]

        vlos_variable_flag[vlos_variable]=True
        teff_variable_flag[teff_variable]=True
        logg_variable_flag[logg_variable]=True
        feh_variable_flag[feh_variable]=True
        mgfe_variable_flag[mgfe_variable]=True
    
        col1=fits.Column(name='vlos_raw_mean',format='D',array=v_mean)
        col2=fits.Column(name='vlos_mean_error',format='D',array=sigv_mean)
        col3=fits.Column(name='vlos_mean_scatter',format='D',array=stdv_mean)
        col4=fits.Column(name='teff_raw_mean',format='D',array=teff_mean)
        col5=fits.Column(name='teff_mean_error',format='D',array=sigteff_mean)
        col6=fits.Column(name='teff_mean_scatter',format='D',array=stdteff_mean)
        col7=fits.Column(name='logg_raw_mean',format='D',array=logg_mean)
        col8=fits.Column(name='logg_mean_error',format='D',array=siglogg_mean)
        col9=fits.Column(name='logg_mean_scatter',format='D',array=stdlogg_mean)
        col10=fits.Column(name='feh_raw_mean',format='D',array=z_mean)
        col11=fits.Column(name='feh_mean_error',format='D',array=sigfeh_mean)
        col12=fits.Column(name='feh_mean_scatter',format='D',array=stdfeh_mean)
        col13=fits.Column(name='mgfe_raw_mean',format='D',array=mgfe_mean)
        col14=fits.Column(name='mgfe_mean_error',format='D',array=sigmgfe_mean)
        col15=fits.Column(name='mgfe_mean_scatter',format='D',array=stdmgfe_mean)

        col16=fits.Column(name='teffprior_vlos_raw_mean',format='D',array=teffprior_v_mean)
        col17=fits.Column(name='teffprior_vlos_mean_error',format='D',array=teffprior_sigv_mean)
        col18=fits.Column(name='teffprior_vlos_mean_scatter',format='D',array=teffprior_stdv_mean)
        col19=fits.Column(name='teffprior_teff_raw_mean',format='D',array=teffprior_teff_mean)
        col20=fits.Column(name='teffprior_teff_mean_error',format='D',array=teffprior_sigteff_mean)
        col21=fits.Column(name='teffprior_teff_mean_scatter',format='D',array=teffprior_stdteff_mean)
        col22=fits.Column(name='teffprior_logg_raw_mean',format='D',array=teffprior_logg_mean)
        col23=fits.Column(name='teffprior_logg_mean_error',format='D',array=teffprior_siglogg_mean)
        col24=fits.Column(name='teffprior_logg_mean_scatter',format='D',array=teffprior_stdlogg_mean)
        col25=fits.Column(name='teffprior_feh_raw_mean',format='D',array=teffprior_z_mean)
        col26=fits.Column(name='teffprior_feh_mean_error',format='D',array=teffprior_sigfeh_mean)
        col27=fits.Column(name='teffprior_feh_mean_scatter',format='D',array=teffprior_stdfeh_mean)
        col28=fits.Column(name='teffprior_mgfe_raw_mean',format='D',array=teffprior_mgfe_mean)
        col29=fits.Column(name='teffprior_mgfe_mean_error',format='D',array=teffprior_sigmgfe_mean)
        col30=fits.Column(name='teffprior_mgfe_mean_scatter',format='D',array=teffprior_stdmgfe_mean)

        col31=fits.Column(name='obs',format='I',array=obs)
        col32=fits.Column(name='n_obs',format='I',array=nobs)
        col33=fits.Column(name='good_obs',format='I',array=goodobs)
        col34=fits.Column(name='good_n_obs',format='I',array=goodnobs)

        nthar=np.zeros(len(v_mean),dtype='int')
        temp_min=np.zeros(len(v_mean),dtype='float')
        temp_max=np.zeros(len(v_mean),dtype='float')
        wav_cal_flag=np.full(len(v_mean),False,dtype='bool')
        chi2_flag=np.full(len(v_mean),False,dtype='bool')
        carbon_flag=np.full(len(v_mean),False,dtype='bool')
        for i in range(0,len(catalog)):
            nthar[i]=len(catalog['wav_rms'][i])
            temp_min[i]=np.min(catalog['temperature'][i])
            temp_max[i]=np.max(catalog['temperature'][i])
        flag=np.where((temp_max-temp_min>1.)&(nthar==1))[0]
        wav_cal_flag[flag]=True
        chi2_flag[m2fs_chi2_outlier]=True
        carbon_flag[m2fs_carbon]=True    
        col35=fits.Column(name='n_wav_cal',format='I',array=nthar)
        col36=fits.Column(name='temp_min',format='D',array=temp_min)
        col37=fits.Column(name='temp_max',format='D',array=temp_max)
        col38=fits.Column(name='wav_cal_flag',format='L',array=wav_cal_flag)
        col39=fits.Column(name='chi2_flag',format='L',array=chi2_flag)
        col40=fits.Column(name='carbon_flag',format='L',array=carbon_flag)
        col41=fits.Column(name='any_chi2_flag',format='L',array=any_chi2_flag)
        col42=fits.Column(name='any_carbon_flag',format='L',array=any_carbon_flag)
        col43=fits.Column(name='vlos_variable_flag',format='L',array=vlos_variable_flag)
        col44=fits.Column(name='teff_variable_flag',format='L',array=teff_variable_flag)
        col45=fits.Column(name='logg_variable_flag',format='L',array=logg_variable_flag)
        col46=fits.Column(name='feh_variable_flag',format='L',array=feh_variable_flag)
        col47=fits.Column(name='mgfe_variable_flag',format='L',array=mgfe_variable_flag)

        raw_cols=catalog.columns[0:121]
        new_cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41,col42,col43,col44,col45,col46,col47])
        cols=raw_cols+new_cols    
        table_hdu=fits.BinTableHDU.from_columns(cols)#BinTableHDU.from_columns(cols)
        
        table_hdu.writeto(fits_table_final_filename,overwrite=True)
        shite=fits.open(fits_table_final_filename)[1].data

if check_badfield:

    fits_table_final_filename='/hildafs/projects/phy200028p/mgwalker/m2fs/m2fs_HiRes_table.fits'
    m2fs=fits.open(fits_table_final_filename)[1].data

    used=np.zeros(len(m2fs),dtype='int')
    for i in range(0,len(m2fs)):
        if ((m2fs['good_obs'][i]>0)&(used[i]==0)):
            dist=np.sqrt((m2fs['ra'][i]-m2fs['ra'])**2+(m2fs['dec'][i]-m2fs['dec'])**2)*3600.
            this=np.where((dist<0.5)&(m2fs['good_obs']>0))[0]
            used[this]=1
            dv=m2fs['vlos_raw'][this]-m2fs['vlos_raw_mean'][this]
            plt.scatter(m2fs['hjd'][this],dv,s=1,color='k')
#    plt.yscale('log')
#    plt.ylim([0.1,1000])
    plt.show()
    plt.close()
#    obs=np.zeros(len(m2fs),dtype='int')
#    count=0
#    rate=[]
#    for i in range(0,len(m2fs)):
#        if obs[i]>-1:
#            this=np.where((m2fs['fits_filename']==m2fs['fits_filename'][i]))[0]
#            if len(this)>0:
#                obs[this]=1
#                keep=np.where(m2fs['good_obs'][this]==1)[0]
#                if len(keep)>0:
#                    count+=1
#                    for j in range(0,len(keep)):
#                        stat=m2fs['vlos_mean_scatter'][this][keep]/m2fs['vlos_mean_error'][this][keep]
#                    bad=np.where(stat>10.)[0]
#                    rate0=len(bad)/len(keep)
#                    rate.append(rate0)
#                    if rate0>0.15:
#                        print(i,count,m2fs['fits_filename'][i],m2fs['temp_min'][i],m2fs['temp_max'][i],m2fs['obj_id'][i],rate0)
#    rate=np.array(rate)
    
