import numpy as np
import astropy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import StdDevUncertainty
from astropy.nddata import CCDData
import ccdproc
import astropy.units as u
from astropy.modeling import models
from ccdproc import Combiner
import os
from os import path
import scipy
import m2fs_process as m2fs
import mycode
matplotlib.use('TkAgg')
#matplotlib.use('pdf')

make_skysub=False#generate 1d sky-subtracted spectra?
compile_fits=True#fit is done, do post-processing of multinest output
analyze=True#analyze post-processed data

data_directory='/hildafs/projects/phy200028p/mgwalker/m2fs_data/'
fit_directory='/hildafs/projects/phy200028p/mgwalker/m2fs_chains/'

data_out='scl_m2fs.dat'
fits_table_filename='scl_m2fs.fits'

with open('scl_m2fs_files') as f:

    data=f.readlines()
fitsfile=[]
obj=[]
for line in data:
    p=line.split()
    fitsfile.append(p[0])
    obj.append(p[1])
fitsfile=np.array(fitsfile)
obj=np.array(obj)

postfit_object=[]
postfit_obsid=[]
postfit_objtype=[]
postfit_radeg=[]
postfit_decdeg=[]
postfit_mjd=[]
postfit_hjd=[]
postfit_snratio=[]
postfit_vhelio_correction=[]
postfit_mean=[]
postfit_std=[]
postfit_skew=[]
postfit_kurtosis=[]        
postfit_mediansky=[]
postfit_stdsky=[]
postfit_run_id=[]
postfit_field_name=[]
postfit_filtername=[]
postfit_chi2=[]
postfit_chi22=[]
postfit_n=[]
postfit_wav_npoints=[]
postfit_wav_rms=[]
postfit_wav_resolution=[]
postfit_wav_min=[]
postfit_wav_max=[]
postfit_row=[]
postfit_temperature=[]

if make_skysub:
    g0=open('bad','w')

dev=[]
for i in range(0,len(fitsfile)):
    hdul=fits.open(fitsfile[i])
    fitsobject=m2fs.m2fs_getfromfits(hdul)
    filtername=hdul[0].header['filtername']
    m2fsrun=hdul[0].header['m2fsrun']
    field_name=hdul[0].header['field_name']
    temperature=hdul[0].header['dome_temp']
    hdul.close()

    root=[]
    for j in range(0,len(fitsobject.obj)):
        root.append('m2fs_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3))
    root=np.array(root)

    skies=np.where(fitsobject.obj=='SKY')[0]
    targets=np.where(fitsobject.icode>0)[0]

    if make_skysub:#write skysub.dat file
        if len(targets)==0:
            print('WARNING: NO TARGETS')
        for j in targets:
            out=data_directory+root[j]+'_skysub.dat'
            if len(np.where(fitsobject.mask[j]==False)[0])>0:#write skysub.dat file only if >100 good pixels in spectrum
                print('writing .skysub.dat file for frame ',i,root[j],j)
                g1=open(out,'w')
                for k in range(0,len(fitsobject.wav[j])):
#                    if fitsobject.mask[j][k]==False:
                    string=str(round(fitsobject.wav[j][k],10))+' '+str(round(fitsobject.spec[j][k],3))+' '+str(round(fitsobject.var[j][k],5))+' \n'
                    g1.write(string)
                g1.close()
            else:
                print('NOT writing .skysub.dat file for frame ',i,root[j],j)
                g0.write(str(i)+root[j]+str(j)+'\n')
#                np.pause()
    if compile_fits:#process posterior files and put results into new fits file

        print(i,len(fitsobject.obj))
        if len(fitsobject.obj)>0:
            root2=[]
            for j in range(0,len(root)):
                shite=root[j].split('m2fs_')
                root2.append('m2fs_'+shite[1])
            root2=np.array(root2)
            multinest=m2fs.m2fs_multinest(fit_directory,root,targets,fitsobject,len(fitsobject.wav[0]))#object containing sample of posterior, moments thereof, bestfit wavelength and bestfit spectrum
            hdr=fits.Header(fitsobject.header)
            primary_hdu=fits.PrimaryHDU(header=hdr)
            new_hdul=fits.HDUList([primary_hdu])

            col1=fits.Column(name='ra_deg',format='D',array=fitsobject.radeg)
            col2=fits.Column(name='dec_deg',format='D',array=fitsobject.decdeg)
            col3=fits.Column(name='mjd',format='D',array=fitsobject.mjd)
            col4=fits.Column(name='hjd',format='D',array=fitsobject.hjd)
            col5=fits.Column(name='snratio',format='D',array=fitsobject.snratio)
            col6=fits.Column(name='vhelio_correction',format='D',array=fitsobject.vheliocorr)
            col7=fits.Column(name='posterior_moments',format='64D',dim='(16,4)',array=multinest.moments)
            col8=fits.Column(name='posterior_1000',format='16000D',dim='(16,1000)',array=multinest.posterior_1000)
            col9=fits.Column(name='objtype',format='A6',array=fitsobject.obj)
            col10=fits.Column(name='run_id',format='A100',array=fitsobject.run_id)
            col11=fits.Column(name='field_name',format='A100',array=fitsobject.field_name)
            col12=fits.Column(name='filtername',format='A100',array=fitsobject.filtername)
            col13=fits.Column(name='wav_npoints',format='PI()',array=fitsobject.wav_npoints)
            col14=fits.Column(name='wav_rms',format='PD()',array=fitsobject.wav_rms)
            col15=fits.Column(name='wav_resolution',format='PD()',array=fitsobject.wav_resolution)
            col16=fits.Column(name='wav_min',format='PD()',array=fitsobject.wav_min)
            col17=fits.Column(name='wav_max',format='PD()',array=fitsobject.wav_max)
            col18=fits.Column(name='row',format='D',array=fitsobject.row)
            col19=fits.Column(name='temperature',format='PD()',array=fitsobject.temperature)
#            cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19])
            cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19])
            table_hdu=fits.FITS_rec.from_columns(cols)#BinTableHDU.from_columns(cols)

            primary_hdu=fits.PrimaryHDU([],header=hdr)
            new_hdul=fits.HDUList([primary_hdu])
            new_hdul.append(fits.ImageHDU(fitsobject.wav,name='wavlength'))
            new_hdul.append(fits.ImageHDU(fitsobject.spec,name='sky_subtracted'))
            new_hdul.append(fits.ImageHDU(fitsobject.var,name='variance'))
            new_hdul.append(fits.ImageHDU(fitsobject.mask,name='mask'))
            new_hdul.append(fits.ImageHDU(multinest.bestfit_fit,name='bestfit'))
            new_hdul.append(fits.BinTableHDU(table_hdu,name='data_table'))
            new_hdul.append(fits.ImageHDU(fitsobject.sky_spec,name='sky'))

#            for j in range(0,len(postfit[5].data)):
#                mean=postfit[5].data['posterior_moments'][j][0][0]
#                std=postfit[5].data['posterior_moments'][j][1][0]
#                skew=postfit[5].data['posterior_moments'][j][2][0]
#                kurt=postfit[5].data['posterior_moments'][j][3][0]
#                print(mean,std,skew,kurt)
#                if ((std<5.)&(std>0.)&(np.abs(skew)<1.)&(np.abs(kurt)<1.)&(mean<-900.)):
#                    plt.plot(fitsobject.wav[j],fitsobject.spec[j],color='k',lw=0.3)
#                    plt.plot(fitsobject.wav[j],multinest.bestfit_fit[j],color='r',lw=0.5)
#                    plt.show()
#                    plt.close()

            if fitsobject.filtername[0]=='HiRes':
                for k in range(0,len(multinest.bestfit_fit)):
                    for q in range(0,len(fitsobject.spec[k])):
                        if fitsobject.var[k][q]<1.e9:
                            dev.append((fitsobject.spec[k][q]-multinest.bestfit_fit[k][q])/np.sqrt(fitsobject.var[k][q]))
#                    plt.hist((fitsobject.spec[k][keep]-multinest.bestfit_fit[k][keep])/np.sqrt(fitsobject.var[k][keep]),bins=100,range=[-5,5])
#                    plt.show()
#                    plt.close()
#                    plt.plot(fitsobject.wav[k],(multinest.bestfit_fit[k]-fitsobject.spec[k])/np.sqrt(fitsobject.var[k]),lw=0.1)
#                    plt.xlim([5130,5190])
#                    plt.ylim([-5,5])
                    
#                plt.show()
#                plt.close()
#        for j in range(0,len(targets)):
#            diff=np.where(fitsobject.wav[targets[j]]!=multinest.bestfit_wav[j])
#            if len(diff)>0: 
#                print('ERROR: wavelengths do not match in input fits file and best-fit model filt')
#                np.pause()

            outfile=fitsfile[i].split('.fits')[0]+'_postfit.fits'
            print('writing fit results to '+outfile)
            new_hdul.writeto(outfile,overwrite=True)

            for k in range(0,len(new_hdul[4].data)):
                rastring=str('{0:.7f}'.format(new_hdul[6].data['ra_deg'][k]))
                decstring=str('{0:.7f}'.format(new_hdul[6].data['dec_deg'][k]))
                hjdstring=str('{0:.3f}'.format(new_hdul[6].data['hjd'][k]))
                if new_hdul[6].data['dec_deg'][k]>=0.:
                    radechjdstring=rastring+'_+'+decstring+'_'+hjdstring
                else:
                    radechjdstring=rastring+'_'+decstring+'_'+hjdstring
#                plt.plot(new_hdul[0].data[k],new_hdul[1].data[k],color='k',lw=0.5)
#                plt.plot(new_hdul[0].data[k],new_hdul[4].data[k],color='r',lw=0.5)
#                plt.xlim([5150,5300])
#                plt.ylim([-30,200])
#                plt.savefig('/hildafs/projects/phy200028p/mgwalker//m2fs/pdf/'+radechjdstring+'_bestfit.pdf',dpi=200)
#            plt.show()
#                plt.close()

    if analyze:

        if len(fitsobject.obj)>0:
            infile=fitsfile[i].split('.fits')[0]+'_postfit.fits'
            postfit=fits.open(infile)
            mediansky=[]
            for j in range(0,len(postfit[6].data)):
                rastring=str('{0:.7f}'.format(postfit[6].data['ra_deg'][j]))
                decstring=str('{0:.7f}'.format(postfit[6].data['dec_deg'][j]))
                hjdstring=str('{0:.3f}'.format(postfit[6].data['hjd'][j]))
                if postfit[6].data['dec_deg'][j]>=0.:
                    radechjdstring=rastring+'_+'+decstring+'_'+hjdstring
                else:
                    radechjdstring=rastring+'_'+decstring+'_'+hjdstring

#                postfit_temperature.append(temperature)

                postfit_temperature.append(postfit[6].data['temperature'][j])
                postfit_obsid.append(radechjdstring)
                postfit_objtype.append(postfit[6].data['objtype'][j])
                postfit_radeg.append(postfit[6].data['ra_deg'][j])
                postfit_decdeg.append(postfit[6].data['dec_deg'][j])
                postfit_mjd.append(postfit[6].data['mjd'][j])
                postfit_hjd.append(postfit[6].data['hjd'][j])
                postfit_snratio.append(postfit[6].data['snratio'][j])
                postfit_vhelio_correction.append(postfit[6].data['vhelio_correction'][j])
                postfit_mean.append(postfit[6].data['posterior_moments'][j][0])
                postfit_std.append(postfit[6].data['posterior_moments'][j][1])
                postfit_skew.append(postfit[6].data['posterior_moments'][j][2])
                postfit_kurtosis.append(postfit[6].data['posterior_moments'][j][3])
                keep=np.where((postfit[4].data[j]==0)&(postfit[3].data[j]<1.e+5))[0]
                mediansky.append(np.median(postfit[7].data[j][keep]))
                postfit_mediansky.append(np.median(postfit[7].data[j][keep]))
                postfit_object.append(obj[i])
                postfit_run_id.append(postfit[6].data['run_id'][j])
                postfit_field_name.append(postfit[6].data['field_name'][j])
                postfit_filtername.append(postfit[6].data['filtername'][j])
                postfit_wav_npoints.append(postfit[6].data['wav_npoints'][j])
                postfit_wav_rms.append(postfit[6].data['wav_rms'][j])
                postfit_wav_resolution.append(postfit[6].data['wav_resolution'][j])
                postfit_wav_min.append(postfit[6].data['wav_min'][j])
                postfit_wav_max.append(postfit[6].data['wav_max'][j])
                postfit_row.append(postfit[6].data['row'][j])
                postfit_temperature.append(postfit[6].data['temperature'][j])

                spec=postfit[2].data[j]
                mask=postfit[4].data[j]
                bestfit=postfit[5].data[j]
                var=postfit[3].data[j]
#            fitkeep=np.where(mask==False)[0]
                chi2=np.sum((spec[keep]-bestfit[keep])**2/var[keep])

                postfit_chi2.append(chi2)
                postfit_n.append(len(keep))

                sigma0=postfit[6].data['posterior_moments'][j][0][13]
                sigma1=postfit[6].data['posterior_moments'][j][0][14]
                var2=10.**sigma1*var+(10.**sigma0)**2
                chi22=np.sum((spec[keep]-bestfit[keep])**2/var2[keep])
                postfit_chi22.append(chi22)

            mediansky=np.array(mediansky)
            stdsky=np.std(mediansky[skies])
            for j in range(0,len(postfit[6].data)):
                postfit_stdsky.append(stdsky)
                

postfit_object=np.array(postfit_object)
postfit_obsid=np.array(postfit_obsid)
postfit_objtype=np.array(postfit_objtype)
postfit_radeg=np.array(postfit_radeg)
postfit_decdeg=np.array(postfit_decdeg)
postfit_mjd=np.array(postfit_mjd)
postfit_hjd=np.array(postfit_hjd)
postfit_snratio=np.array(postfit_snratio)
postfit_vhelio_correction=np.array(postfit_vhelio_correction)
postfit_mean=np.array(postfit_mean)
postfit_std=np.array(postfit_std)
postfit_skew=np.array(postfit_skew)
postfit_kurtosis=np.array(postfit_kurtosis)
postfit_mediansky=np.array(postfit_mediansky)
postfit_stdsky=np.array(postfit_stdsky)
postfit_run_id=np.array(postfit_run_id)
postfit_field_name=np.array(postfit_field_name)
postfit_filtername=np.array(postfit_filtername)
postfit_chi2=np.array(postfit_chi2)
postfit_chi22=np.array(postfit_chi22)
postfit_n=np.array(postfit_n)
postfit_wav_npoints=np.array(postfit_wav_npoints)
postfit_wav_rms=np.array(postfit_wav_rms)
postfit_wav_resolution=np.array(postfit_wav_resolution)
postfit_wav_min=np.array(postfit_wav_min)
postfit_wav_max=np.array(postfit_wav_max)
postfit_row=np.array(postfit_row)
postfit_temperature=np.array(postfit_temperature)

#                plt.plot(postfit[0].data[j],postfit[1].data[j],color='k')
#                plt.plot(postfit[0].data[j],postfit[4].data[j],color='r')
#                plt.plot(postfit[0].data[j],(postfit[1].data[j]-postfit[4].data[j])/np.sqrt(postfit[2].data[j]))
#                plt.xlim([5130,5190])
#                plt.ylim([-10,10])
#                plt.show()
#                plt.close()

if make_skysub:
    g0.close()

x=np.linspace(-10.,10.,1000)
y=1./np.sqrt(2.*np.pi)*np.exp(-0.5*x**2)
y2=1./np.sqrt(2.*np.pi*0.95**2)*np.exp(-0.5*x**2/0.95**2)
y3=1./np.sqrt(2.*np.pi*0.90**2)*np.exp(-0.5*x**2/0.90**2)
y4=1./np.sqrt(2.*np.pi*0.85**2)*np.exp(-0.5*x**2/0.85**2)

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
resolution_sigma=np.array([postfit_mean[q][10] for q in range(0,len(postfit_mean))])
resolution_sigma_sig=np.array([postfit_std[q][10] for q in range(0,len(postfit_std))])
resolution_sigma_skew=np.array([postfit_skew[q][10] for q in range(0,len(postfit_skew))])
resolution_sigma_kurt=np.array([postfit_kurtosis[q][10] for q in range(0,len(postfit_kurtosis))])
phantom1=np.array([postfit_mean[q][13] for q in range(0,len(postfit_mean))])
phantom1_sig=np.array([postfit_std[q][13] for q in range(0,len(postfit_std))])
phantom1_skew=np.array([postfit_skew[q][13] for q in range(0,len(postfit_skew))])
phantom1_kurt=np.array([postfit_kurtosis[q][13] for q in range(0,len(postfit_kurtosis))])
phantom2=np.array([postfit_mean[q][14] for q in range(0,len(postfit_mean))])
phantom2_sig=np.array([postfit_std[q][14] for q in range(0,len(postfit_std))])
phantom2_skew=np.array([postfit_skew[q][14] for q in range(0,len(postfit_skew))])
phantom2_kurt=np.array([postfit_kurtosis[q][14] for q in range(0,len(postfit_kurtosis))])

v=v+postfit_vhelio_correction#apply heliocentrcic correction!!!

g1=open(data_out,'w')
g1.write('#Object # RA [deg] # Dec [deg] # HJD [days] # median S/N/pix # vhelio [km/s] # err_vhelio [km/s] # skew_vhelio # kurt_vhelio # Teff [K] # err_Teff [K] # skew_Teff # kurt_Teff # logg # err_logg # skew_logg # kurt_logg # Fe/H # err_Fe/H # skew_Fe/H # kurt_Fe/H # resolution \n ')
for i in range(0,len(postfit_radeg)):
    if postfit_objtype[i]=='TARGET':
        string=postfit_object[i]+' '+str.format('{0:.7f}',round(postfit_radeg[i],7)).zfill(7)+' '+str.format('{0:.7f}',round(postfit_decdeg[i],7)).zfill(7)+' '+str.format('{0:.3f}',round(postfit_hjd[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(postfit_snratio[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(v[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(v_sig[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(v_skew[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(v_kurt[i],3)).zfill(3)+' '+str.format('{0:.1f}',round(teff[i],1)).zfill(1)+' '+str.format('{0:.1f}',round(teff_sig[i],1)).zfill(1)+' '+str.format('{0:.1f}',round(teff_skew[i],1)).zfill(1)+' '+str.format('{0:.1f}',round(teff_kurt[i],1)).zfill(1)+' '+str.format('{0:.3f}',round(logg[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(logg_sig[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(logg_skew[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(logg_kurt[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(z[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(z_sig[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(z_skew[i],3)).zfill(3)+' '+str.format('{0:.3f}',round(z_kurt[i],3)).zfill(3)+' '+postfit_filtername[i]+' \n'
        print(string)
        if ((postfit_snratio[i]==postfit_snratio[i])&(not((v_kurt[i]==0.)&(v_kurt[i]==0.)&(z_kurt[i]==0.)&(z_skew[i]==0.)&(logg_kurt[i]==0.)&(logg_skew[i]==0.)))):
            g1.write(string)
g1.close()

targets=np.where(postfit_objtype=='TARGET')[0]
col1=fits.Column(name='object',format='A100',array=postfit_object[targets])
col2=fits.Column(name='obs_id',format='A100',array=postfit_obsid[targets])
col3=fits.Column(name='ra_deg',format='D',array=postfit_radeg[targets])
col4=fits.Column(name='dec_deg',format='D',array=postfit_decdeg[targets])
col5=fits.Column(name='mjd',format='D',array=postfit_mjd[targets])
col6=fits.Column(name='hjd',format='D',array=postfit_hjd[targets])
col7=fits.Column(name='sn_ratio',format='D',array=postfit_snratio[targets])
col8=fits.Column(name='vlos',format='D',array=v[targets])
col9=fits.Column(name='vlos_error',format='D',array=v_sig[targets])
col10=fits.Column(name='vlos_skew',format='D',array=v_skew[targets])
col11=fits.Column(name='vlos_kurtosis',format='D',array=v_kurt[targets])
col12=fits.Column(name='teff',format='D',array=teff[targets])
col13=fits.Column(name='teff_error',format='D',array=teff_sig[targets])
col14=fits.Column(name='teff_skew',format='D',array=teff_skew[targets])
col15=fits.Column(name='teff_kurtosis',format='D',array=teff_kurt[targets])
col16=fits.Column(name='logg',format='D',array=logg[targets])
col17=fits.Column(name='logg_error',format='D',array=logg_sig[targets])
col18=fits.Column(name='logg_skew',format='D',array=logg_skew[targets])
col19=fits.Column(name='logg_kurtosis',format='D',array=logg_kurt[targets])
col20=fits.Column(name='z',format='D',array=z[targets])
col21=fits.Column(name='z_error',format='D',array=z_sig[targets])
col22=fits.Column(name='z_skew',format='D',array=z_skew[targets])
col23=fits.Column(name='z_kurtosis',format='D',array=z_kurt[targets])
col24=fits.Column(name='resolution_sigma',format='D',array=resolution_sigma[targets])
col25=fits.Column(name='resolution_sigma_error',format='D',array=resolution_sigma_sig[targets])
col26=fits.Column(name='resolution_sigma_skew',format='D',array=resolution_sigma_skew[targets])
col27=fits.Column(name='resolution_sigma_kurtosis',format='D',array=resolution_sigma_kurt[targets])
col28=fits.Column(name='median_sky',format='D',array=postfit_mediansky[targets])
col29=fits.Column(name='standard_deviation_median_sky',format='D',array=postfit_stdsky[targets])
col30=fits.Column(name='filtername',format='A100',array=postfit_filtername[targets])
col31=fits.Column(name='chi2',format='D',array=postfit_chi2[targets])
col32=fits.Column(name='chi2_modified',format='D',array=postfit_chi22[targets])
col33=fits.Column(name='n',format='D',array=postfit_n[targets])
col34=fits.Column(name='run_id',format='A100',array=postfit_run_id[targets])
col35=fits.Column(name='field_name',format='A100',array=postfit_field_name[targets])
col36=fits.Column(name='wav_npoints',format='PI()',array=postfit_wav_npoints[targets])
col37=fits.Column(name='wav_rms',format='PD()',array=postfit_wav_rms[targets])
col38=fits.Column(name='wav_resolution',format='PD()',array=postfit_wav_resolution[targets])
col39=fits.Column(name='wav_min',format='PD()',array=postfit_wav_min[targets])
col40=fits.Column(name='wav_max',format='PD()',array=postfit_wav_max[targets])
col41=fits.Column(name='row',format='D',array=postfit_row[targets])
col42=fits.Column(name='temperature',format='PD()',array=postfit_temperature[targets])
col43=fits.Column(name='vhelio_correction',format='D',array=postfit_vhelio_correction[targets])

cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41,col42,col43])
t=fits.BinTableHDU.from_columns(cols)
#table_hdu=fits.FITS_rec.from_columns(cols)
t.writeto(fits_table_filename,overwrite=True)

ra=postfit_radeg*np.pi/180.
dec=postfit_decdeg*np.pi/180.

#center=SkyCoord('11:31:09.6 +02:13:12.0',unit=(u.hourangle,u.deg))
#center=SkyCoord('10:13:02.9 -01:36:53.0',unit=(u.hourangle,u.deg))
#center=SkyCoord('15:09:08.5 +67:13:21.0',unit=(u.hourangle,u.deg))
#center=SkyCoord('17:20:12.4 +57:54:55.0',unit=(u.hourangle,u.deg))

#racenter=center.ra.rad
#deccenter=center.dec.rad
#x=(ra-racenter)*np.cos(deccenter)*180./np.pi
#y=(dec-deccenter)*180./np.pi
#r=np.sqrt(x**2+y**2)

keep0=np.where((v_sig<5.))
keep=np.where((v_sig<5.)&(v_sig>0.)&(np.abs(v_kurt)<1.)&(np.abs(v_skew)<1.))[0]
memkeep=np.where((z<-1.)&(v_sig<5.)&(np.abs(v_kurt)<1.)&(np.abs(v-55.)<30.))[0]
keepgaia=np.where((v_sig<5.)&(np.abs(v_kurt)<1.)&(postfit_hjd>2458540.))[0]
keepnogaia=np.where((v_sig<5.)&(np.abs(v_kurt)<1.)&(postfit_hjd<2458540.))[0]
keepgaiamem=np.where((z<-1.)&(v_sig<5.)&(np.abs(v_kurt)<1.)&(np.abs(v-224.)<30.))[0]
keepnogaiamem=np.where((z<-2.)&(v_sig<5.)&(np.abs(v_kurt)<1.)&(np.abs(v-224.)<30.))[0]

#keep=np.where((v_sig<5.)&(np.abs(v_kurt)<1.))[0]
#memkeep=np.where((z<-1.5)&(v_sig<5.)&(np.abs(v_kurt)<1.)&(np.abs(v-222.)<20.))[0]
#keepgaia=np.where((v_sig<5.)&(np.abs(v_kurt)<1.)&(postfit_hjd>2458510.))[0]
#keepnogaia=np.where((v_sig<5.)&(np.abs(v_kurt)<1.)&(postfit_hjd<2458510.))[0]
#keepgaiamem=np.where((z<-1.5)&(v_sig<5.)&(np.abs(v_kurt)<1.)&(postfit_hjd>2458510.)&(np.abs(v-222.)<20.))[0]
#keepnogaiamem=np.where((z<-1.5)&(v_sig<5.)&(np.abs(v_kurt)<1.)&(postfit_hjd<2458510.)&(np.abs(v-222.)<20.))[0]

#keep=np.where((v_sig<5.)&(np.abs(v_kurt)<1.)&(np.abs(v_skew)<1.))[0]
#memkeep=np.where((logg<3.)&(v_sig<5.)&(np.abs(v_kurt)<1.)&(np.abs(v-172.)<20.))[0]
#keepgaia=np.where((v_sig<5.)&(np.abs(v_kurt)<1.)&(postfit_hjd>2458510.))[0]
#keepnogaia=np.where((v_sig<5.)&(np.abs(v_kurt)<1.)&(postfit_hjd<2458510.))[0]
#keepgaiamem=np.where((logg<3.)&(v_sig<5.)&(np.abs(v_kurt)<1.)&(postfit_hjd>2458510.)&(np.abs(v-172.)<20.))[0]
#keepnogaiamem=np.where((logg<3.)&(v_sig<5.)&(np.abs(v_kurt)<1.)&(postfit_hjd<2458510.)&(np.abs(v-172.)<20.))[0]

rtide=50.6/60.
ellipticity=0.56
width=rtide*2.
height=width*(1.-ellipticity)
angle=53.

#        for j in range(0,len(new_hdul[0].data)):
#            plt.plot(new_hdul[0].data[j],new_hdul[1].data[j],color='k',lw=0.3)
#            plt.plot(new_hdul[0].data[j],new_hdul[5].data[j],color='r',lw=0.3)
#            plt.xlim([5160,5280])
#            plt.show()
#            time.sleep(3)
#            plt.close('new')

keep2=np.where(postfit_snratio==postfit_snratio)[0]
