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

data_directory='/nfs/nas-0-9/mgwalker.proj/m2fs_data/'
fit_directory='/nfs/nas-0-9/mgwalker.proj/m2fs_chains/'

data_out='car_m2fs_noflat.dat'
fits_table_filename='car_m2fs_noflat.fits'

with open('car_m2fs_noflat_files') as f:

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
postfit_n=[]

if make_skysub:
    g0=open('bad_noflat','w')

dev=[]
for i in range(0,len(fitsfile)):
    hdul=fits.open(fitsfile[i])
    fitsobject=m2fs.m2fs_getfromfits(hdul)
    filtername=hdul[0].header['filtername']
    m2fsrun=hdul[0].header['m2fsrun']
    field_name=hdul[0].header['field_name']
    hdul.close()

    root=[]
    for j in range(0,len(fitsobject.obj)):
        root.append('m2fs_'+m2fsrun+'_'+obj[i]+'_'+field_name+'_'+filtername+'_ra'+str.format('{0:.6f}',round(fitsobject.radeg[j],6)).zfill(6)+'_dec'+str.format('{0:.6f}',round(fitsobject.decdeg[j],6)).zfill(6)+'_hjd'+str.format('{0:.3f}',round(fitsobject.hjd[j],3))+'_ap'+fitsobject.channel[j]+str(int(fitsobject.aperture[j])).zfill(3)+'_noflat')
    root=np.array(root)

    skies=np.where(fitsobject.obj=='SKY')[0]
    targets=np.where(fitsobject.icode>0)[0]

    if make_skysub:#write skysub.dat file
        for j in targets:
            out=data_directory+root[j]+'_skysub_noflat.dat'
            if len(np.where(fitsobject.mask[j]==False)[0])>0:#write skysub.dat file only if >100 good pixels in spectrum
                print('writing _skysub_noflat.dat file for frame ',i,root[j],j)
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

        print(len(fitsobject.obj))
        if len(fitsobject.obj)>0:
            multinest=m2fs.m2fs_multinest_noflat(fit_directory,root,targets,fitsobject,len(fitsobject.wav[0]))#object containing sample of posterior, moments thereof, bestfit wavelength and bestfit spectrum
            hdr=fits.Header(fitsobject.header)
            primary_hdu=fits.PrimaryHDU(header=hdr)
            new_hdul=fits.HDUList([primary_hdu])

            col1=fits.Column(name='ra_deg',format='D',array=fitsobject.radeg)
            col2=fits.Column(name='dec_deg',format='D',array=fitsobject.decdeg)
            col3=fits.Column(name='hjd',format='D',array=fitsobject.hjd)
            col4=fits.Column(name='snratio',format='D',array=fitsobject.snratio)
            col5=fits.Column(name='vhelio_correction',format='D',array=fitsobject.vheliocorr)
            col6=fits.Column(name='posterior_moments',format='64D',dim='(16,4)',array=multinest.moments)
            col7=fits.Column(name='posterior_1000',format='16000D',dim='(16,1000)',array=multinest.posterior_1000)
            col8=fits.Column(name='objtype',format='A6',array=fitsobject.obj)
            col9=fits.Column(name='run_id',format='A100',array=fitsobject.run_id)
            col10=fits.Column(name='field_name',format='A100',array=fitsobject.field_name)
            col11=fits.Column(name='filtername',format='A100',array=fitsobject.filtername)
            cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11])
            table_hdu=fits.FITS_rec.from_columns(cols)#BinTableHDU.from_columns(cols)

            primary_hdu=fits.PrimaryHDU(fitsobject.wav,header=hdr)
            new_hdul=fits.HDUList([primary_hdu])
            new_hdul.append(fits.ImageHDU(fitsobject.spec))
            new_hdul.append(fits.ImageHDU(fitsobject.var))
            new_hdul.append(fits.ImageHDU(fitsobject.mask))
            new_hdul.append(fits.ImageHDU(multinest.bestfit_fit))
            new_hdul.append(fits.BinTableHDU(table_hdu))
            new_hdul.append(fits.ImageHDU(fitsobject.sky_spec))

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
                rastring=str('{0:.7f}'.format(new_hdul[5].data['ra_deg'][k]))
                decstring=str('{0:.7f}'.format(new_hdul[5].data['dec_deg'][k]))
                hjdstring=str('{0:.3f}'.format(new_hdul[5].data['hjd'][k]))
                if new_hdul[5].data['dec_deg'][k]>=0.:
                    radechjdstring=rastring+'_+'+decstring+'_'+hjdstring
                else:
                    radechjdstring=rastring+'_'+decstring+'_'+hjdstring
#                plt.plot(new_hdul[0].data[k],new_hdul[1].data[k],color='k',lw=0.5)
#                plt.plot(new_hdul[0].data[k],new_hdul[4].data[k],color='r',lw=0.5)
#                plt.xlim([5150,5300])
#                plt.ylim([-30,200])
#                plt.savefig('/nfs/nas-0-9/mgwalker.proj/m2fs/pdf/'+radechjdstring+'_bestfit.pdf',dpi=200)
#            plt.show()
#                plt.close()

    if analyze:

        if len(fitsobject.obj)>0:
            infile=fitsfile[i].split('.fits')[0]+'_postfit.fits'
            postfit=fits.open(infile)
            mediansky=[]
            for j in range(0,len(postfit[5].data)):

                rastring=str('{0:.7f}'.format(postfit[5].data['ra_deg'][j]))
                decstring=str('{0:.7f}'.format(postfit[5].data['dec_deg'][j]))
                hjdstring=str('{0:.3f}'.format(postfit[5].data['hjd'][j]))
                if postfit[5].data['dec_deg'][j]>=0.:
                    radechjdstring=rastring+'_+'+decstring+'_'+hjdstring
                else:
                    radechjdstring=rastring+'_'+decstring+'_'+hjdstring

                postfit_obsid.append(radechjdstring)
                postfit_objtype.append(postfit[5].data['objtype'][j])
                postfit_radeg.append(postfit[5].data['ra_deg'][j])
                postfit_decdeg.append(postfit[5].data['dec_deg'][j])
                postfit_hjd.append(postfit[5].data['hjd'][j])
                postfit_snratio.append(postfit[5].data['snratio'][j])
                postfit_vhelio_correction.append(postfit[5].data['vhelio_correction'][j])
                postfit_mean.append(postfit[5].data['posterior_moments'][j][0])
                postfit_std.append(postfit[5].data['posterior_moments'][j][1])
                postfit_skew.append(postfit[5].data['posterior_moments'][j][2])
                postfit_kurtosis.append(postfit[5].data['posterior_moments'][j][3])
                keep=np.where((postfit[3].data[j]==0)&(postfit[2].data[j]<1.e+5))[0]
                mediansky.append(np.median(postfit[6].data[j][keep]))
                postfit_mediansky.append(np.median(postfit[6].data[j][keep]))
                postfit_object.append(obj[i])
                postfit_run_id.append(postfit[5].data['run_id'][j])
                postfit_field_name.append(postfit[5].data['field_name'][j])
                postfit_filtername.append(postfit[5].data['filtername'][j])

                spec=postfit[1].data[j]
                mask=postfit[3].data[j]
                bestfit=postfit[4].data[j]
                var=postfit[2].data[j]
#            fitkeep=np.where(mask==False)[0]
                chi2=np.sum((spec[keep]-bestfit[keep])**2/var[keep])

                postfit_chi2.append(chi2)
                postfit_n.append(len(keep))

            mediansky=np.array(mediansky)
            stdsky=np.std(mediansky[skies])
            for j in range(0,len(postfit[5].data)):
                postfit_stdsky.append(stdsky)

postfit_object=np.array(postfit_object)
postfit_obsid=np.array(postfit_obsid)
postfit_objtype=np.array(postfit_objtype)
postfit_radeg=np.array(postfit_radeg)
postfit_decdeg=np.array(postfit_decdeg)
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
postfit_n=np.array(postfit_n)


#                plt.plot(postfit[0].data[j],postfit[1].data[j],color='k')
#                plt.plot(postfit[0].data[j],postfit[4].data[j],color='r')
#                plt.plot(postfit[0].data[j],(postfit[1].data[j]-postfit[4].data[j])/np.sqrt(postfit[2].data[j]))
#                plt.xlim([5130,5190])
#                plt.ylim([-10,10])
#                plt.show()
#                plt.close()

if make_skysub:
    g0.close()

dev=np.array(dev)
plt.hist(dev,range=[-5,5],bins=200,color='k',normed=True)
x=np.linspace(-10.,10.,1000)
y=1./np.sqrt(2.*np.pi)*np.exp(-0.5*x**2)
y2=1./np.sqrt(2.*np.pi*0.95**2)*np.exp(-0.5*x**2/0.95**2)
y3=1./np.sqrt(2.*np.pi*0.90**2)*np.exp(-0.5*x**2/0.90**2)
y4=1./np.sqrt(2.*np.pi*0.85**2)*np.exp(-0.5*x**2/0.85**2)
plt.plot(x,y,color='r',linestyle=':')
plt.plot(x,y2,color='g',linestyle=':')
plt.plot(x,y3,color='b',linestyle=':')
plt.plot(x,y4,color='orange',linestyle=':')
plt.xlim([-5,5])
plt.show()
plt.close()

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
        if ((v_sig[i]<5.)&(postfit_snratio[i]==postfit_snratio[i])&(not((v_kurt[i]==0.)&(v_kurt[i]==0.)&(z_kurt[i]==0.)&(z_skew[i]==0.)&(logg_kurt[i]==0.)&(logg_skew[i]==0.)))):
            g1.write(string)
g1.close()

targets=np.where(postfit_objtype=='TARGET')[0]
col1=fits.Column(name='object',format='A100',array=postfit_object[targets])
col2=fits.Column(name='obs_id',format='A100',array=postfit_obsid[targets])
col3=fits.Column(name='ra_deg',format='D',array=postfit_radeg[targets])
col4=fits.Column(name='dec_deg',format='D',array=postfit_decdeg[targets])
col5=fits.Column(name='hjd',format='D',array=postfit_hjd[targets])
col6=fits.Column(name='sn_ratio',format='D',array=postfit_snratio[targets])
col7=fits.Column(name='vlos',format='D',array=v[targets])
col8=fits.Column(name='vlos_error',format='D',array=v_sig[targets])
col9=fits.Column(name='vlos_skew',format='D',array=v_skew[targets])
col10=fits.Column(name='vlos_kurtosis',format='D',array=v_kurt[targets])
col11=fits.Column(name='teff',format='D',array=teff[targets])
col12=fits.Column(name='teff_error',format='D',array=teff_sig[targets])
col13=fits.Column(name='teff_skew',format='D',array=teff_skew[targets])
col14=fits.Column(name='teff_kurtosis',format='D',array=teff_kurt[targets])
col15=fits.Column(name='logg',format='D',array=logg[targets])
col16=fits.Column(name='logg_error',format='D',array=logg_sig[targets])
col17=fits.Column(name='logg_skew',format='D',array=logg_skew[targets])
col18=fits.Column(name='logg_kurtosis',format='D',array=logg_kurt[targets])
col19=fits.Column(name='z',format='D',array=z[targets])
col20=fits.Column(name='z_error',format='D',array=z_sig[targets])
col21=fits.Column(name='z_skew',format='D',array=z_skew[targets])
col22=fits.Column(name='z_kurtosis',format='D',array=z_kurt[targets])
col23=fits.Column(name='resolution_sigma',format='D',array=resolution_sigma[targets])
col24=fits.Column(name='resolution_sigma_error',format='D',array=resolution_sigma_sig[targets])
col25=fits.Column(name='resolution_sigma_skew',format='D',array=resolution_sigma_skew[targets])
col26=fits.Column(name='resolution_sigma_kurtosis',format='D',array=resolution_sigma_kurt[targets])
col27=fits.Column(name='median_sky',format='D',array=postfit_mediansky[targets])
col28=fits.Column(name='standard_deviation_median_sky',format='D',array=postfit_stdsky[targets])
col29=fits.Column(name='filtername',format='A100',array=postfit_filtername[targets])
col30=fits.Column(name='chi2',format='D',array=postfit_chi2[targets])
col31=fits.Column(name='n',format='D',array=postfit_n[targets])
col32=fits.Column(name='run_id',format='A100',array=postfit_run_id[targets])
col33=fits.Column(name='field_name',format='A100',array=postfit_field_name[targets])

cols=fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33])
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

gs=plt.GridSpec(10,10) # define multi-panel plot
gs.update(wspace=0,hspace=0) # specify inter-panel spacing
fig=plt.figure(figsize=(6,6)) # define plot size
ax1=fig.add_subplot(gs[0:10,0:10])

rtide=50.6/60.
ellipticity=0.56
width=rtide*2.
height=width*(1.-ellipticity)
angle=53.

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.scatter((10.**phantom1)**2,(10.**phantom2),s=1,color='k',alpha=0.3)
ax1.scatter((10.**phantom1[keep])**2,(10.**phantom2[keep]),s=1,color='r',alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')
#ax1.set_xlabel(r'$v_{\rm helio}$ [km/s')
#ax1.set_ylabel('[Fe/H]')
ax1.set_xlabel(r'$p_0$')
ax1.set_ylabel(r'$p_1$')
#ellipse1=matplotlib.patches.Ellipse((0,0),width,height,angle,fill=False)
#ax1.add_artist(ellipse1)
#ax1.legend(loc=3,fontsize=8)

plt.savefig('car_m2fs_noflat.pdf',dpi=200)
#plt.savefig('nelson_leo5.pdf',dpi=200)
plt.show()
plt.close()


#        for j in range(0,len(new_hdul[0].data)):
#            plt.plot(new_hdul[0].data[j],new_hdul[1].data[j],color='k',lw=0.3)
#            plt.plot(new_hdul[0].data[j],new_hdul[5].data[j],color='r',lw=0.3)
#            plt.xlim([5160,5280])
#            plt.show()
#            time.sleep(3)
#            plt.close('all')

