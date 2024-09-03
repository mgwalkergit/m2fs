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
import dill as pickle
from isochrones.mist import MIST_Isochrone
from isochrones.mist import MIST_EvolutionTrack
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from isochrones import get_ichrone
from pymultinest.solve import solve
from pymultinest import Analyzer

def bit_solve(n,k):
    temp=n>>(k-1)
    if temp&1:
        return True
    return False

apokasc=fits.open('apokasc.fits')[1].data

gaiadr3_id,gaiadr3_gmag,gaiadr3_siggmag,gaiadr3_bpmag,gaiadr3_sigbpmag,gaiadr3_rpmag,gaiadr3_sigrpmag,gaiadr3_parallax,gaiadr3_sigparallax,gaiadr3_rv,gaiadr3_sigrv,gaiadr3_teff,gaiadr3_teff_lower,gaiadr3_teff_upper,gaiadr3_logg,gaiadr3_logg_lower,gaiadr3_logg_upper,gaiadr3_feh,gaiadr3_feh_lower,gaiadr3_feh_upper=crossmatcher.doit('gaia_dr3.gaia_source',apokasc['_ra'],apokasc['_de'],'source_id,phot_g_mean_mag,phot_g_mean_flux_error,phot_bp_mean_mag,phot_bp_mean_flux_error,phot_rp_mean_mag,phot_rp_mean_flux_error,parallax,parallax_error,radial_velocity,radial_velocity_error,teff_gspphot,teff_gspphot_lower,teff_gspphot_upper,logg_gspphot,logg_gspphot_lower,logg_gspphot_upper,mh_gspphot,mh_gspphot_lower,mh_gspphot_upper',rad=1.,db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')

#apogee_objid,apogee_ra,apogee_dec,apogee_vlos,apogee_sigvlos,apogee_stdvlos,apogee_teff,apogee_sigteff,apogee_logg,apogee_siglogg,apogee_z,apogee_sigz,apogee_flagz,apogee_mg,apogee_sigmg,apogee_flagmg,apogee_alpha,apogee_sigalpha,apogee_param,apogee_sigparam,apogee_flag,apogee_nvisits,apogee_rv_flag,apogee_aspcap_flag=sqlutil.get('''select apogee_id,ra,dec,vhelio_avg,verr,vscatter,teff,teff_err,logg,logg_err,fe_h,fe_h_err,fe_h_flag,mg_fe,mg_fe_err,mg_fe_flag,alpha_m,alpha_m_err,param,param_cov,rv_flag,aspcapflag,nvisits,aspcapflag from apogee_dr17.allstar where ra!= \'nan\' and dec!=\'nan\' and teff!= \'nan\' and logg!=\'nan\' and fe_h!=\'nan\' and m_h!=\'nan\' and mg_fe!=\'nan\' limit 1000000''',db='wsdb',host='wsdb.hpc1.cs.cmu.edu',user='matt_walker',password='5UN*ur8Y')
binary_repr_v=np.vectorize(np.binary_repr)
bit_solve_v=np.vectorize(bit_solve)

