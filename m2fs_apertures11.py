import astropy
import dill as pickle
from astropy import units
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from astropy.nddata import NDDataRef
import astropy.units as u
from astropy.modeling import models
import m2fs_process as m2fs
import os
from os import path
import mycode
import specutils
from specutils import Spectrum1D
import scipy
from astropy.nddata import StdDevUncertainty
from astropy.visualization import quantity_support
import numpy as np
from specutils.fitting import fit_lines
from specutils.analysis import centroid
from specutils.analysis import fwhm
from specutils.analysis import line_flux
from specutils.analysis import equivalent_width
from specutils import SpectralRegion
from specutils.manipulation import extract_region
from specutils.fitting import fit_generic_continuum
from astropy.modeling import models,fitting
from specutils.fitting import estimate_line_parameters
from ccdproc import Combiner
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
#matplotlib.use('pdf')
matplotlib.use('TkAgg')

shite=False
trim=False#define boundaries of useful spectral region
initialize=False#find peaks in 'spectrum' along columns, representing apertures
find=False#find apertures in image
trace_all=False#trace all apertures
trace_edit=False#edit traces of individual apertures
make_image=False#generate PDF of data frame with aperture traces overlaid
apmask=False#mask apertures to create 2d data frame containing only extra-aperture light
apflat=False
apflatcorr=False
scatteredlightcorr=False#fit 2d function to extra-aperture light and subtract from data frame
extract1d_flat=False#extract 1d spectra for flat frames
extract1d_thar=True#extract 1d spectra for thar frames
extract1d_sci=True#extract 1d spectra for science frames
id_lines_template=False#identify lines in thar template and fit wavelength solution
id_lines_translate=True#translate line IDs from template to new spectrum, fit new wavelenth solution
id_lines_check=False
throughputcorr=True#perform throughput correction (wavelength-dependent)
plugmap=False
skysubtract=False
pickthar=False
overwrite=False#overwrite previous results

#linelist_file='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs_config1b_thar_list'
threshold_factor=25.#multiple of continuum residual rms threshold to impose for aperture detection in find_lines_derivative
n_lines=20#columns to combine when scanning across rows to identify apertures (as 'emission lines')
columnspec_continuum_rejection_low=-5.
columnspec_continuum_rejection_high=1.
columnspec_continuum_rejection_iterations=10#number of iterations of outlier rejection for fitting "continuum"
profile_rejection_iterations=10#number of iterations of outlier rejection for fitting profile amplitude and sigma
profile_nsample=50#number of points along spectral- (x-) axis at which to measure profile amplitude and sigma before performing fit of amplitude(x) and sigma(x)
profile_order=4#order of polynomial used to fit profile amplitude and sigma as functions of pixel along dispersion direction
window=10#pixels, width of aperture window for fitting (gaussian) aperture profiles (perpendicular to spectral axis)
trace_step=n_lines#tracing step
trace_nlost_max=2
trace_shift_max=1.5
trace_order=4
trace_rejection_iterations=10
trace_rejection_sigma=3.#largest rms deviation to accept in fit to aperture trace
trace_rejection_iterations=10
id_lines_continuum_rejection_low=-5.
id_lines_continuum_rejection_high=1.
id_lines_continuum_rejection_sigma=3.
id_lines_continuum_rejection_iterations=10#number of iterations of outlier rejection for fitting "continuum"
id_lines_threshold_factor=10.#multiple of continuum residual rms threshold to impose for aperture detection in find_lines_derivative
id_lines_window=5.#pixels, width of aperture window for fitting (gaussian) line profiles in arc spectra
id_lines_order=5#order of wavelength solution
id_lines_tol_angs=0.05#tolerance for finding new lines to add from linelist (Angstroms)
id_lines_tol_pix=2.#tolerance for matching lines between template and new spectrum (pixels)
id_lines_minlines=30#mininum number of ID'd lines for acceptable wavelength solution (less than this, and cannot fit reliable throughput correction and beyond)
resolution_order=1
resolution_rejection_iterations=10
scatteredlightcorr_order=4
scatteredlightcorr_rejection_iterations=10
scatteredlightcorr_rejection_sigma=3.
id_lines_translate_add_lines_iterations=5
extract1d_aperture_width=3.#maximum (half-)width of aperture for extraction (too large and we get weird edge effects)
throughputcorr_continuum_rejection_low=-3.
throughputcorr_continuum_rejection_high=3.
throughputcorr_continuum_rejection_iterations=10#number of iterations of outlier rejection for fitting "continuum"

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'

m2fsrun='nov18'
#m2fsrun='may19'
#m2fsrun='aug18'
#m2fsrun='may17'
#m2fsrun='nov16'
datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/NovDec2018/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2019/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Aug2018/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/MayJun2017/'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/NovDec2016/'
#m2fsrun='jul15'
#datadir='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs.astro.lsa.umich.edu/data/Jul2015/'

utdate=[]
file1=[]
file2=[]
flatfile=[]
tharfile=[]
scifile=[]
fibermap_file=[]
with open(directory+m2fsrun+'_science_raw11') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        utdate.append(str(p[0]))
        file1.append(int(p[1]))
        file2.append(int(p[2]))
        flatfile.append(p[3])
        tharfile.append(p[4])
        scifile.append(p[6])
        fibermap_file.append(p[7])
utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flatfile=np.array(flatfile)
tharfile=np.array(tharfile)
scifile=np.array(scifile)
fibermap_file=np.array(fibermap_file)

flatfile0=[]
tharfile0=[]
scifile0=[]
allfile0=[]
for i in range(0,len(tharfile)):
    flatfile0.append(flatfile[i].split('-'))
    tharfile0.append(tharfile[i].split('-'))
    scifile0.append(scifile[i].split('-'))
    allfile0.append(flatfile[i].split('-')+tharfile[i].split('-')+scifile[i].split('-'))
flatfile0=np.array(flatfile0,dtype='object')
tharfile0=np.array(tharfile0,dtype='object')
scifile0=np.array(scifile0,dtype='object')
allfile0=np.array(allfile0,dtype='object')

if id_lines_template:
    with open(directory+'arc_templates') as f:
        data=f.readlines()
    arcfilename=[]
    arcfiltername=[]
    for line in data:
        p=line.split()
        arcfilename.append(p[0])
        arcfiltername.append(p[1])

    for i in range(0,len(arcfilename)):
        root=str(arcfilename[i])
        extract1d_array_file=root+'_extract1d_array.pickle'
        id_lines_template_file=root+'_id_lines_template.pickle'
        id_lines_template_exist=path.exists(id_lines_template_file)

        extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
        print('identify line template for '+root)

        linelist_file=directory+'m2fs_'+arcfiltername[i]+'_thar_linelist'
        linelist=m2fs.get_linelist(linelist_file)
        if id_lines_template_exist:
            id_lines_template0=pickle.load(open(id_lines_template_file,'rb'))
            id_lines_template_fiddle=True#says that we already have a template, and will let us fiddle with that
            print('will overwrite existing version')
        else:
            id_lines_template_fiddle=False#start template from scratch
            id_lines_template0=m2fs.id_lines()#initialize values to zero-arrays
        id_lines_template0=m2fs.get_id_lines_template(extract1d_array[0],linelist,id_lines_continuum_rejection_low,id_lines_continuum_rejection_high,id_lines_continuum_rejection_iterations,id_lines_threshold_factor,id_lines_window,id_lines_order,id_lines_continuum_rejection_iterations,id_lines_continuum_rejection_sigma,id_lines_tol_angs,id_lines_template_fiddle,id_lines_template0,resolution_order,resolution_rejection_iterations)
        pickle.dump(id_lines_template0,open(id_lines_template_file,'wb'))

if id_lines_check:
    with open(directory+'id_lines_check') as f:
        data=f.readlines()
    arcfilename=[]
    arcfiltername=[]
    for line in data:
        p=line.split()
        arcfilename.append(p[0])
        arcfiltername.append(p[1])

    for i in range(0,len(arcfilename)):
        root=str(arcfilename[i])
        extract1d_array_file=root+'_extract1d_array.pickle'
        id_lines_array_file=root+'_id_lines_array.pickle'

        extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
        print('identify line template for '+root)

        linelist_file=directory+'m2fs_'+arcfiltername[i]+'_thar_linelist'
        linelist=m2fs.get_linelist(linelist_file)
        id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
        id_lines_template_fiddle=True#says that we already have a template, and will let us fiddle with that
        id_lines_check0=m2fs.get_id_lines_template(extract1d_array[0],linelist,id_lines_continuum_rejection_low,id_lines_continuum_rejection_high,id_lines_continuum_rejection_iterations,id_lines_threshold_factor,id_lines_window,id_lines_order,id_lines_continuum_rejection_iterations,id_lines_continuum_rejection_sigma,id_lines_tol_angs,id_lines_template_fiddle,id_lines_array[0],resolution_order,resolution_rejection_iterations)

for i in range(0,len(utdate)):
    for ccd in ('r','b'):
        root=datadir+utdate[i]+'/'+ccd+str(flatfile[i]).zfill(4)
        data_file=root+'_stitched.fits'
        image_boundary_file=root+'_image_boundary.pickle'
        columnspec_array_file=root+'_columnspec_array.pickle'
        apertures_profile_middle_file=root+'_apertures_profile_middle.pickle'
        aperture_array_file=root+'_aperture_array.pickle'
        image_file=root+'_apertures.pdf'
        apmask_file=root+'_apmask.pickle'
        apflat_file=root+'_apflat.pickle'
        apflat_residual_file=root+'_apflat_residual.pickle'
        extract1d_array_flat_file=root+'_extract1d_array.pickle'

        image_boundary_exists=path.exists(image_boundary_file)
        columnspec_array_exists=path.exists(columnspec_array_file)
        apertures_profile_middle_exists=path.exists(apertures_profile_middle_file)
        aperture_array_exists=path.exists(aperture_array_file)
        apmask_exists=path.exists(apmask_file)
        apflat_exists=path.exists(apflat_file)
        apflat_residual_exists=path.exists(apflat_residual_file)

#        columnspec_array=pickle.load(open(columnspec_array_file,'rb'))
#        aperture_array=pickle.load(open(aperture_array_file,'rb'))
        
        if shite:
            root0=datadir+utdate[i]+'/'+ccd+str(allfile0[i][0]).zfill(4)
            data=pickle.load(open(root0+'_apflatcorr.pickle','rb'))
            vmin=np.quantile(data.data.flatten(),0.35)
            vmax=np.quantile(data.data.flatten(),0.65)
            plt.imshow(data.data,cmap='gray',vmin=vmin,vmax=vmax)
            for j in range(0,len(aperture_array)):
                if aperture_array[j].trace_npoints>40:
                    x=np.linspace(aperture_array[j].trace_pixel_min,aperture_array[j].trace_pixel_max,100)
                    y=aperture_array[j].trace_func(x)
                    plt.plot(x,y,color='r',alpha=1,lw=0.2)
            plt.savefig('crap.pdf',dpi=300)
            plt.show()
            plt.close()

            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
            vmin=np.quantile(data.data.flatten(),0.05)
            vmax=np.quantile(data.data.flatten(),0.95)
            plt.imshow(data.data,cmap='gray',vmin=vmin,vmax=vmax)
            for j in range(0,len(aperture_array)):
                if aperture_array[j].trace_npoints>40:
                    x=np.linspace(aperture_array[j].trace_pixel_min,aperture_array[j].trace_pixel_max,100)
                    y=aperture_array[j].trace_func(x)
                    plt.plot(x,y,color='r',alpha=1,lw=0.2)
            plt.savefig('crap2.pdf',dpi=300)
            plt.show()
            plt.close()

            data=pickle.load(open(apflat_file,'rb'))#format is such that data.data[:,0] has column-0 value in all rows
            vmin=0.
            vmax=2.
            plt.imshow(data.data,cmap='gray',vmin=vmin,vmax=vmax)
            for j in range(0,len(aperture_array)):
                if aperture_array[j].trace_npoints>40:
                    x=np.linspace(aperture_array[j].trace_pixel_min,aperture_array[j].trace_pixel_max,100)
                    y=aperture_array[j].trace_func(x)
                    plt.plot(x,y,color='r',alpha=1,lw=0.2)
            plt.savefig('crap3.pdf',dpi=300)
            plt.show()
            plt.close()

            np.pause()

        if trim:
            print('displaying '+root)
            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
            if image_boundary_exists:
                image_boundary0=pickle.load(open(image_boundary_file,'rb'))
                image_boundary_fiddle=True
            else:
                image_boundary0=m2fs.image_boundary()
                image_boundary_fiddle=False
            image_boundary=m2fs.get_image_boundary(data,image_boundary_fiddle,image_boundary0)
            pickle.dump(image_boundary,open(image_boundary_file,'wb'))#save pickle to file

        if initialize:#find apertures in each column stack and save to pickle files
            print('initializing '+root)
            if columnspec_array_exists:
                print('will overwrite existing '+columnspec_array_file)
            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
            columnspec_array=m2fs.get_columnspec(data,trace_step,n_lines,columnspec_continuum_rejection_low,columnspec_continuum_rejection_high,columnspec_continuum_rejection_iterations,threshold_factor,window)
            '''
            columnspec_array is array of columnspec objects.
            columnspec object contains: 
            columns: numbers of columns in original data frame that are stacked
            spec1d: 'spectrum' across stacked column, where each aperture appears as an 'emission line'
            pixel: value of pixel across stacked column 'spectrum', artificially given units of AA in order to comply with specutils requirements for fitting spectra
            continuum: parameters of 1dGaussian continuum fit to stacked column 'spectrum'
            rms: rms residuals around continuum fit (using only regions between fiber bundles)
            apertures_initial: initial aperture centers returned by specutils find_lines_derivative
            apertures_profile: contains parameters of fits to apertures detected in apertures_initial
            '''
            pickle.dump(columnspec_array,open(columnspec_array_file,'wb'))#save pickle to file

        if find:
            print('finding apertures for '+root)
            columnspec_array=pickle.load(open(columnspec_array_file,'rb'))
            
            '''
            Make initial array of more precise aperture location fits in middle column stack
            '''
            
            if apertures_profile_middle_exists:
                print('loading existing '+apertures_profile_middle_file+', will overwrite')
                apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            else:
                middle_column=np.long(len(columnspec_array)/2)
                apertures_profile_middle=columnspec_array[middle_column].apertures_profile
                
            '''
            apertures_profile_middle object contains: 
            fit: parameters of fit to cross-aperture profile ('line' profile) *for middle column*
            subregion: pixel range of cross-aperture 'spectrum' used for fit *for middle column*
            initial: value of True means aperture was found in automatic procedure, value of False means it was inserted by hand
            realvirtual: value of True means aperture corresponds to real spectrum, value of False means virtual aperture used as place-holder (due to bad or unplugged fiber)
            '''

            '''
            display apertures initially identified in central column stack and allow user to add/delete apertures with screen input
            '''
            apertures_profile_middle=m2fs.fiddle_apertures(columnspec_array,middle_column,window,apertures_profile_middle)
            pickle.dump([apertures_profile_middle,middle_column],open(apertures_profile_middle_file,'wb'))#save pickle to file

        if trace_all:
            print('tracing all apertures for '+root)
            if (not(aperture_array_exists))|(aperture_array_exists & overwrite):
                if aperture_array_exists:
                    print('will overwrite existing '+aperture_array_file)
                image_boundary=pickle.load(open(image_boundary_file,'rb'))
                columnspec_array=pickle.load(open(columnspec_array_file,'rb'))
                apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            
                aperture_array=[]
                for j in range(0,len(apertures_profile_middle.fit)):
#            if apertures_profile_middle.realvirtual[j]:
                    aperture_array.append(m2fs.get_aperture(j,columnspec_array,apertures_profile_middle,middle_column,trace_order,trace_rejection_sigma,trace_rejection_iterations,image_boundary,trace_shift_max,trace_nlost_max,profile_rejection_iterations,profile_nsample,profile_order,window))

                pickle.dump(aperture_array,open(aperture_array_file,'wb'))

        if trace_edit:
            image_boundary=pickle.load(open(image_boundary_file,'rb'))
            columnspec_array=pickle.load(open(columnspec_array_file,'rb'))
            apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
            aperture_array=pickle.load(open(aperture_array_file,'rb'))

            command=input('enter number of aperture to edit (integer)')
            j=np.long(command)-1
            aperture_array[j]=m2fs.get_aperture(j,columnspec_array,apertures_profile_middle,middle_column,trace_order,trace_rejection_sigma,trace_rejection_iterations,image_boundary,trace_shift_max,trace_nlost_max,profile_rejection_iterations,profile_nsample,profile_order,window)

            pickle.dump(aperture_array,open(aperture_array_file,'wb'))

        if make_image:
            print('writing '+image_file)
            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
            columnspec_array=pickle.load(open(columnspec_array_file,'rb'))
            aperture_array=pickle.load(open(aperture_array_file,'rb'))
            
            vmin=np.quantile(data.data.flatten(),0.05)
            vmax=np.quantile(data.data.flatten(),0.95)
            plt.imshow(data.data,cmap='gray',vmin=vmin,vmax=vmax)
            for j in range(0,len(aperture_array)):
                if aperture_array[j].trace_npoints>40:
                    x=np.linspace(aperture_array[j].trace_pixel_min,aperture_array[j].trace_pixel_max,100)
                    y=aperture_array[j].trace_func(x)
                    plt.plot(x,y,color='r',alpha=1,lw=0.2)
            plt.savefig(image_file,dpi=300)
            plt.show()
            plt.close()

        if apmask:

            print('making aperture mask for ',root)
            if (not(apmask_exists))|(apmask_exists & overwrite):
                if overwrite:
                    print('will overwrite existing version')
                    
                apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
                aperture_array=pickle.load(open(aperture_array_file,'rb'))
                aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]
                image_boundary=pickle.load(open(image_boundary_file,'rb'))
                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows

                apmask0=m2fs.get_apmask(data,aperture_array,apertures_profile_middle,aperture_peak,image_boundary)

                pickle.dump(apmask0,open(apmask_file,'wb'))            

        if apflat:

            print('making apflat for ',root)
            if (not(apflat_exists))|(apflat_exists & overwrite):
                if overwrite:
                    print('will overwrite existing version')
                apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
                aperture_array=pickle.load(open(aperture_array_file,'rb'))
                aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]
                image_boundary=pickle.load(open(image_boundary_file,'rb'))
                apmask0=pickle.load(open(apmask_file,'rb'))
                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows

                apflat0,apflat_residual=m2fs.get_apflat(data,aperture_array,apertures_profile_middle,aperture_peak,image_boundary,apmask0)
                pickle.dump(apflat0,open(apflat_file,'wb'))            
                pickle.dump(apflat_residual,open(apflat_residual_file,'wb'))            

        if apflatcorr:

            for j in range(0,len(allfile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(allfile0[i][j]).zfill(4)
                data_file=root0+'_stitched.fits'
                apflatcorr_file=root0+'_apflatcorr.pickle'
                apflatcorr_exists=path.exists(apflatcorr_file)
                print('creating aperture-flat-corrected frame: \n'+apflatcorr_file)
                if (not(apflatcorr_exists))|(apflatcorr_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')
                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    apflat0=pickle.load(open(apflat_file,'rb'))
                    apflatcorr0=data.divide(apflat0)
                    print(data_file,apflat_file,apflatcorr_file)
                    pickle.dump(apflatcorr0,open(apflatcorr_file,'wb'))            

        if scatteredlightcorr:

            for j in range(0,len(allfile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(allfile0[i][j]).zfill(4)
                data_file=root0+'_stitched.fits'
                apflatcorr_file=root0+'_apflatcorr.pickle'
                scatteredlightcorr_file=root0+'_scatteredlightcorr.pickle'
                scatteredlightcorr_exists=path.exists(scatteredlightcorr_file)
                print('creating scattered-light-corrected frame: \n'+scatteredlightcorr_file)
                if (not(scatteredlightcorr_exists))|(scatteredlightcorr_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')
                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    apmask0=pickle.load(open(apmask_file,'rb'))
                    apflatcorr0=pickle.load(open(apflatcorr_file,'rb'))

                    scatteredlightfunc=m2fs.get_scatteredlightfunc(data,apmask0,scatteredlightcorr_order,scatteredlightcorr_rejection_iterations,scatteredlightcorr_rejection_sigma)
                    y,x=np.mgrid[:len(data.data),:len(data.data[0])]
                    scattered_model=CCDData(scatteredlightfunc.func(x,y)*u.electron,mask=np.full((len(data.data),len(data.data[0])),False,dtype=bool),uncertainty=StdDevUncertainty(np.full((len(data.data),len(data.data[0])),scatteredlightfunc.rms*u.electron,dtype='float')))
                    scatteredlightcorr0=apflatcorr0.subtract(scattered_model)

                    pickle.dump(scatteredlightcorr0,open(scatteredlightcorr_file,'wb'))            

        if extract1d_flat:

            for j in range(0,len(flatfile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(flatfile0[i][j]).zfill(4)
                data_file=root0+'_stitched.fits'
                apflat_file=root0+'_apflat.pickle'
                apflatcorr_file=root0+'_apflatcorr.pickle'
                apflat_residual_file=root0+'_apflat_residual.pickle'
                scatteredlightcorr_file=root0+'_scatteredlightcorr.pickle'
                extract1d_array_file=root0+'_extract1d_array.pickle'
                extract1d_array_exists=path.exists(extract1d_array_file)

                print('extracting to '+extract1d_array_file)
                if (not(extract1d_array_exists))|(extract1d_array_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')
                    apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
                    aperture_array=pickle.load(open(aperture_array_file,'rb'))
                    aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]
                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    apflatcorr0=pickle.load(open(apflatcorr_file,'rb'))
                    apflat0=pickle.load(open(apflat_file,'rb'))
                    apflat_residual=pickle.load(open(apflat_residual_file,'rb'))
                    scatteredlightcorr0=pickle.load(open(scatteredlightcorr_file,'rb'))
                    pix=np.arange(len(scatteredlightcorr0.data[0]))

                    extract1d_array=[]
                    for k in range(0,len(aperture_array)):
#                    for k in range(0,5):
                        print('   extracting aperture ',aperture_array[k].trace_aperture,' of ',len(aperture_array))
                        if apertures_profile_middle.realvirtual[k]:
#                        if aperture_array[k].trace_rms>0.:
                            extract1d_array.append(m2fs.get_extract1d(k,scatteredlightcorr0,apertures_profile_middle,aperture_array,aperture_peak,pix,extract1d_aperture_width))
#                            plt.plot(shite.spec1d_flux)
#                            plt.imshow(shite,cmap='gray')
#                            for j in range(0,len(aperture_array)):
#                                if aperture_array[j].trace_npoints>40:
#                                    x=np.linspace(aperture_array[j].trace_pixel_min,aperture_array[j].trace_pixel_max,100)
#                                    y=aperture_array[j].trace_func(x)
#                                    plt.plot(x,y,color='r',alpha=1,lw=0.2)
                    pickle.dump(extract1d_array,open(extract1d_array_file,'wb'))
#                    np.pause()

        if extract1d_thar:
            
            for j in range(0,len(tharfile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)
                data_file=root0+'_stitched.fits'
                scatteredlightcorr_file=root0+'_scatteredlightcorr.pickle'
                extract1d_array_file=root0+'_extract1d_array.pickle'
                extract1d_array_exists=path.exists(extract1d_array_file)
                print('extracting to '+extract1d_array_file)
                if (not(extract1d_array_exists))|(extract1d_array_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')
                    
                    apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
                    aperture_array=pickle.load(open(aperture_array_file,'rb'))
                    aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]
                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    scatteredlightcorr0=pickle.load(open(scatteredlightcorr_file,'rb'))

                    pix=np.arange(len(scatteredlightcorr0.data[0]))

                    extract1d_array=[]
                    for k in range(0,len(aperture_array)):
                        print('   extracting aperture ',aperture_array[k].trace_aperture,' of ',len(aperture_array))
                        if apertures_profile_middle.realvirtual[k]:
#                        if aperture_array[k].trace_rms>0.:
                            extract1d_array.append(m2fs.get_extract1d(k,scatteredlightcorr0,apertures_profile_middle,aperture_array,aperture_peak,pix,extract1d_aperture_width))
#                            plt.plot(extract1d_array[len(extract1d_array)-1].spec1d_flux)
#                            plt.show()
#                            plt.close()
#                            np.pause()
                    pickle.dump(extract1d_array,open(extract1d_array_file,'wb'))

        if extract1d_sci:
            
            for j in range(0,len(scifile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(scifile0[i][j]).zfill(4)
                data_file=root0+'_stitched.fits'
                scatteredlightcorr_file=root0+'_scatteredlightcorr.pickle'
                extract1d_array_file=root0+'_extract1d_array.pickle'
                extract1d_array_exists=path.exists(extract1d_array_file)
                print('extracting to '+extract1d_array_file)
                if (not(extract1d_array_exists))|(extract1d_array_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')

                    apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
                    aperture_array=pickle.load(open(aperture_array_file,'rb'))
                    aperture_peak=[apertures_profile_middle.fit[q].mean.value for q in range(0,len(apertures_profile_middle.fit))]
#                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    scatteredlightcorr0=pickle.load(open(scatteredlightcorr_file,'rb'))

                    pix=np.arange(len(scatteredlightcorr0.data[0]))

                    extract1d_array=[]
                    for k in range(0,len(aperture_array)):
                        print('   extracting aperture ',aperture_array[k].trace_aperture,' of ',len(aperture_array))
                        if apertures_profile_middle.realvirtual[k]:
#                        if aperture_array[k].trace_rms>0.:
                            extract1d_array.append(m2fs.get_extract1d(k,scatteredlightcorr0,apertures_profile_middle,aperture_array,aperture_peak,pix,extract1d_aperture_width))
                
                    pickle.dump(extract1d_array,open(extract1d_array_file,'wb'))

        if id_lines_translate:
            
            with open(directory+'arc_templates') as f:
                data=f.readlines()
            arcfilename=[]
            arcfiltername=[]
            for line in data:
                p=line.split()
                arcfilename.append(p[0])
                arcfiltername.append(p[1])
            arcfilename=np.array(arcfilename)
            arcfiltername=np.array(arcfiltername)

            for j in range(0,len(tharfile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)#use first ThAr from first field in run
                data_file=root0+'_stitched.fits'
                extract1d_array_file=root0+'_extract1d_array.pickle'
                id_lines_array_file=root0+'_id_lines_array.pickle'
                id_lines_array_exist=path.exists(id_lines_array_file)

                if (not(id_lines_array_exist))|(id_lines_array_exist & overwrite):

                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    extract1d_array=pickle.load(open(extract1d_array_file,'rb'))

                    filtername=data.header['FILTER']
                    if filtername=='Mgb_Rev2':
                        filtername='Mgb_HiRes'
                    arc=np.where(arcfiltername==filtername)[0][0]
                    template_root=str(arcfilename[arc])
                    extract1d_array_template_file=template_root+'_extract1d_array.pickle'
                    id_lines_template_file=template_root+'_id_lines_template.pickle'
                    extract1d_array_template=pickle.load(open(extract1d_array_template_file,'rb'))
                    id_lines_template0=pickle.load(open(id_lines_template_file,'rb'))
                    print('translating line IDs and wavelength solution from ',template_root,' to ',root0)
                    if overwrite:
                        print('will overwrite existing version')

                    linelist_file=directory+'m2fs_'+filtername+'_thar_linelist'
                    linelist=m2fs.get_linelist(linelist_file)

                    id_lines_array=[]
                    for j in range(0,len(extract1d_array)):
#                    for j in range(109,111):
#                        plt.plot(extract1d_array[j].spec1d_flux)
#                        plt.show()
#                        plt.close()
                                 
#                        print(extract1d_array[j].spec1d_flux)
#                        print(j)
#                        print(j,np.where(extract1d_array[j].spec1d_mask==False),' bbbbbbbbbbbbbbbbbbbbbbbbbbb')
#                        print(' ')
                        id_lines_array.append(m2fs.get_id_lines_translate(extract1d_array_template[0],id_lines_template0,extract1d_array[j],linelist,id_lines_continuum_rejection_low,id_lines_continuum_rejection_high,id_lines_continuum_rejection_iterations,id_lines_threshold_factor,id_lines_window,id_lines_order,id_lines_continuum_rejection_iterations,id_lines_continuum_rejection_sigma,id_lines_tol_angs,id_lines_tol_pix,resolution_order,resolution_rejection_iterations,id_lines_translate_add_lines_iterations))
#                    np.pause()
                        
                    pickle.dump(id_lines_array,open(id_lines_array_file,'wb'))

        if throughputcorr:

            for j in range(0,len(allfile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(allfile0[i][j]).zfill(4)
                throughput_array_file=root0+'_throughput_array.pickle'
                throughputcorr_array_file=root0+'_throughputcorr_array.pickle'
                throughput_array_exists=path.exists(throughput_array_file)
                throughputcorr_array_exists=path.exists(throughputcorr_array_file)
                extract1d_array_file=root0+'_extract1d_array.pickle'
                id_lines_array_file=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][0]).zfill(4)+'_id_lines_array.pickle'
                print('creating throughput-corrected frame: \n'+throughputcorr_array_file)
                if (not(throughputcorr_array_exists))|(throughputcorr_array_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')
                    extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
                    extract1d_array_flat=pickle.load(open(extract1d_array_flat_file,'rb'))
                    id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                    if len(id_lines_array)!=len(extract1d_array):
                        print('ERROR!!!!! number of apertures in id_lines_array differs from number in extract1d_array')
                        np.pause()
                    throughput_array,throughputcorr_array=m2fs.get_throughputcorr(extract1d_array,extract1d_array_flat,id_lines_array,throughputcorr_continuum_rejection_low,throughputcorr_continuum_rejection_high,throughputcorr_continuum_rejection_iterations,id_lines_minlines)
#                    if len(extract1d_array)>0:
#                        for k in range(0,len(throughputcorr_array)):
#                            wav=id_lines_array[k].func(throughputcorr_array[k].spec1d_pixel)
#                            plt.plot(wav,throughputcorr_array[k].spec1d_flux)
#                    plt.show()
#                    plt.close()

                    pickle.dump(throughput_array,open(throughput_array_file,'wb'))
                    pickle.dump(throughputcorr_array,open(throughputcorr_array_file,'wb'))

#                np.pause()

        if plugmap:

            for j in range(0,len(scifile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(scifile0[i][j]).zfill(4)
                data_file=root0+'_stitched.fits'
                throughputcorr_array_file=root0+'_throughputcorr_array.pickle'
                plugmap_file=root0+'_plugmap.pickle'
                plugmap_exists=path.exists(plugmap_file)
                print('plugmap for: \n'+root0)
                if (not(plugmap_exists))|(plugmap_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')
                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    throughputcorr_array=pickle.load(open(throughputcorr_array_file,'rb'))
                    with open(fibermap_file[i]) as f:
                        fibermap=f.readlines()
                    plugmap0=m2fs.get_plugmap(data.header,throughputcorr_array,fibermap)
                    pickle.dump(plugmap0,open(plugmap_file,'wb'))

#                    skies=np.where(plugmap0['objtype']=='SKY')[0]
#                    targets=np.where(plugmap0['objtype']=='TARGET')[0]
#                    gs=plt.GridSpec(7,7) # define multi-panel plot
#                    gs.update(wspace=0,hspace=0) # specify inter-panel spacing
#                    fig=plt.figure(figsize=(6,6)) # define plot size
#                    ax1=fig.add_subplot(gs[0:7,0:3])
#                    ax2=fig.add_subplot(gs[0:7,4:7])
#                    for k in range(0,len(skies)):
#                        ax1.plot(throughputcorr_array[skies[k]].spec1d_flux[throughputcorr_array[skies[k]].spec1d_mask==False],lw=0.3)
#                    for k in range(0,len(targets)):
#                        ax2.plot(throughputcorr_array[targets[k]].spec1d_flux[throughputcorr_array[targets[k]].spec1d_mask==False],lw=0.3)
#                    ax1.set_ylim([0,500])
#                    ax2.set_ylim([0,500])
#                    plt.show()
#                    plt.close()
#                np.pause()

        if skysubtract:

            for j in range(0,len(scifile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(scifile0[i][j]).zfill(4)
                throughputcorr_array_file=root0+'_throughputcorr_array.pickle'
                id_lines_array_file=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][0]).zfill(4)+'_id_lines_array.pickle'
                plugmap_file=root0+'_plugmap.pickle'
                skysubtract_array_file=root0+'_skysubtract_array.pickle'
                skysubtract_array_exists=path.exists(skysubtract_array_file)

                print('creating sky-subtracted frame: \n'+root0)
                if (not(skysubtract_array_exists))|(skysubtract_array_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')
                    throughputcorr_array=pickle.load(open(throughputcorr_array_file,'rb'))
                    id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                    plugmap0=pickle.load(open(plugmap_file,'rb'))
                    skysubtract_array=m2fs.get_skysubtract(throughputcorr_array,id_lines_array,plugmap0,id_lines_minlines)

                    
                    pickle.dump(skysubtract_array,open(skysubtract_array_file,'wb'))

#                    skies=np.where(plugmap0['objtype']=='SKY')[0]
#                    targets=np.where(plugmap0['objtype']=='TARGET')[0]
#                    gs=plt.GridSpec(7,7) # define multi-panel plot
#                    gs.update(wspace=0,hspace=0) # specify inter-panel spacing
#                    fig=plt.figure(figsize=(6,6)) # define plot size
#                    ax1=fig.add_subplot(gs[0:7,0:3])
#                    ax2=fig.add_subplot(gs[0:7,4:7])
#                    for k in range(0,len(skies)):
#                        ax1.plot(skysubtract_array[skies[k]].spec1d_flux[skysubtract_array[skies[k]].spec1d_mask==False],lw=0.3)
#                    for k in range(0,len(targets)):
#                        ax2.plot(skysubtract_array[targets[k]].spec1d_flux[skysubtract_array[targets[k]].spec1d_mask==False],lw=0.3)
#                    ax1.set_ylim([0,500])
#                    ax2.set_ylim([0,500])
#                    plt.show()
#                    plt.close()

        if pickthar:

            thar=[]
            lines=[]
            plot_file=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][0]).zfill(4)+'_arc.pdf'
            for k in range(0,len(tharfile0[i])):
                data_file=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][k]).zfill(4)+'_stitched.fits'
                id_lines_array_file=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][k]).zfill(4)+'_id_lines_array.pickle'
                id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                thar.append(id_lines_array)
                lines.append([id_lines_array[q].wav.data[id_lines_array[q].wav.mask==False] for q in range(0,len(id_lines_array))])
            if len(thar)>1:
                if len(thar[0])>0:
                    wav=[]
                    pix=[]
                    wav_func=[]
                    vel_func=[]
                    ap=[]
                    wav0=[]
                    ap0=[]
                    for q in range(0,len(lines[0])):
                        keep_ap=True
                        for xxx in range(0,len(lines)):
                            if not thar[0][q].aperture in [thar[xxx][jjj].aperture for jjj in range(0,len(thar[xxx]))]:
                                keep_ap=False
                        if keep_ap:
                            ap0.append(thar[0][q].aperture)
                    for q in range(0,len(lines[0][0])):
                        keep_wav=True
                        for xxx in range(0,len(lines)):
                            for yyy in range(0,len(lines[xxx])):
                                if not lines[0][0][q] in lines[xxx][yyy]:
                                    keep_wav=False
                        if keep_wav:
                            wav0.append(lines[0][0][q])
                    ap0=np.array(ap0)
                    wav0=np.array(wav0)
                    for q in range(0,len(wav0)):
                        for xxx in range(0,len(ap0)):
                            pix0=[]
                            wav_func0=[]
                            vel_func0=[]
                            for yyy in range(0,len(thar)):
                                shite1=np.where(np.array([thar[yyy][zzz].aperture for zzz in range(0,len(thar[yyy]))])==ap0[xxx])[0][0]
                                shite2=np.where(thar[yyy][shite1].wav==wav0[q])[0][0]
                                pix0.append(thar[yyy][shite1].fit_lines.fit[shite2].mean.value)
                                wav_func0.append(thar[yyy][shite1].func(pix0[0]))
                                vel_func0.append((wav_func0[len(wav_func0)-1]-wav_func0[0])/wav_func0[0]*3.e+5)
                            wav.append(wav0[q])
                            ap.append(ap0[xxx])
                            pix.append(pix0)
                            wav_func.append(wav_func0)
                            vel_func.append(vel_func0)
                    wav=np.array(wav)
                    ap=np.array(ap)
                    pix=np.array(pix)
                    wav_func=np.array(wav_func)
                    vel_func=np.array(vel_func)
                    pix_std=[]
                    pix_dev=[]
                    wav_func_std=[]
                    wav_func_dev=[]
                    vel_func_std=[]
                    vel_func_dev=[]
                    for qqq in range(0,len(pix)):
                        pix_std.append(np.std(pix[qqq]))
                        print(qqq,pix[qqq],np.std(pix[qqq]))
                        print(qqq,wav_func[qqq],np.std(wav_func[qqq]))
                        print(qqq,vel_func[qqq],np.std(vel_func[qqq]))
                        pix_dev.append(pix[qqq][len(pix[qqq])-1]-pix[qqq][0])
                        wav_func_std.append(np.std(wav_func[qqq]))
                        wav_func_dev.append(wav_func[qqq][len(wav_func[qqq])-1]-wav_func[qqq][0])
                        vel_func_std.append(np.std(vel_func[qqq]))
                        vel_func_dev.append(vel_func[qqq][len(vel_func[qqq])-1]-vel_func[qqq][0])
                    pix_std=np.array(pix_std)
                    pix_dev=np.array(pix_dev)
                    wav_func_std=np.array(wav_func_std)
                    wav_func_dev=np.array(wav_func_dev)
                    vel_func_std=np.array(vel_func_std)
                    vel_func_dev=np.array(vel_func_dev)

                    gs=plt.GridSpec(7,7) # define multi-panel plot
                    gs.update(wspace=0,hspace=0) # specify inter-panel spacing
                    fig=plt.figure(figsize=(6,6)) # define plot size
                    ax1=fig.add_subplot(gs[0:7,0:3])
                    ax2=fig.add_subplot(gs[0:7,4:7])
#                    ax1.scatter(wav,ap,c=pix_std,s=3,cmap='inferno')
#                    ax2.scatter(wav,ap,c=vel_func_std,s=3,cmap='inferno')
                    cb1=ax1.scatter(wav,ap,c=pix_std,s=3,cmap='inferno',vmin=0,vmax=0.5)
                    cb2=ax2.scatter(wav,ap,c=vel_func_std,s=3,cmap='inferno',vmin=0,vmax=1.0)

                    ax1.set_xlabel('wavelength of line [Angs.]')
                    ax2.set_xlabel('wavelength of line [Angs.]')
                    ax1.set_ylabel('aperture')
                    fig.colorbar(cb1,ax=ax1,orientation='horizontal')
                    fig.colorbar(cb2,ax=ax2,orientation='horizontal')
                    plt.savefig(plot_file,dpi=200)
#                    plt.show()
#                    plt.close()

#                    np.pause()
#                    [thar[q][ for q in range(0,len(thar))]
#                    gs=plt.GridSpec(7,7) # define multi-panel plot
#                    gs.update(wspace=0,hspace=0) # specify inter-panel spacing
#                    fig=plt.figure(figsize=(6,6)) # define plot size
#                    ax1=fig.add_subplot(gs[0:3,0:7])
#                    ax2=fig.add_subplot(gs[4:7,0:7])
#                    col=['r','g','b','c','m']
#                    for k in range(0,len(thar)):
#                        for q in range(0,len(thar[k])):
#                            ax1.scatter(q,len(np.where(thar[k][q].wav>0.)[0]),color=col[k])
#                            ax2.scatter(q,thar[k][q].rms,color=col[k])
                
#                plt.show()
#                plt.close()
#                    id_lines_array=pickle.load(open(id_lines_array_file,'rb'))

