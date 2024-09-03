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
from specutils.spectra import Spectrum1D
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
from astropy import time, coordinates as coord, units as u
from astropy.coordinates import SkyCoord, EarthLocation
from mpl_toolkits.axes_grid1 import make_axes_locatable
#matplotlib.use('pdf')
matplotlib.use('TkAgg')

shite=False
trim=False#define boundaries of useful spectral region
initialize=False
find=False#find apertures in image
trace_all=False#trace all apertures
trace_edit=False#edit traces of individual apertures
make_image=False#generate PDF of data frame with aperture traces overlaid
apmask=False#mask apertures to create 2d data frame containing only extra-aperture light
apflat=False
apflatcorr=False
scatteredlightcorr=False#fit 2d function to extra-aperture light and subtract from data frame
extract1d_flat=False#extract 1d spectra for flat frames
extract1d_thar=False#extract 1d spectra for thar frames
extract1d_sci=False#extract 1d spectra for science frames
id_lines_template=False#identify lines in thar template and fit wavelength solution
id_lines_translate=False#translate line IDs from template to new spectrum, fit new wavelenth solution
id_lines_check=False
tharcheck=False
wavcal=False
stack_twilight=False
throughputcorr=True#perform throughput correction (wavelength-dependent)
plugmap=False
skysubtract=False
stack_frames=False
writefits=False
overwrite=False#overwrite previous results

#linelist_file='/nfs/nas-0-9/mgwalker.proj/m2fs/m2fs_config1b_thar_list'
lco=coord.EarthLocation.of_site('lco')
threshold_factor=25.#multiple of continuum residual rms threshold to impose for aperture detection in find_lines_derivative
n_lines=20#columns to combine when scanning across rows to identify apertures (as 'emission lines')
columnspec_continuum_rejection_low=-5.
columnspec_continuum_rejection_high=1.
columnspec_continuum_rejection_iterations=10#number of iterations of outlier rejection for fitting "continuum"
columnspec_continuum_rejection_order=10
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
id_lines_continuum_rejection_order=10
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
throughputcorr_continuum_rejection_low=-1.
throughputcorr_continuum_rejection_high=3.
throughputcorr_continuum_rejection_iterations=5#number of iterations of outlier rejection for fitting "continuum"
throughputcorr_continuum_rejection_order=4

directory='/nfs/nas-0-9/mgwalker.proj/m2fs/'
m2fsrun='feb14' 
datadir=m2fs.get_datadir(m2fsrun)

edit_header_filename=[]
edit_header_keyword=[]
edit_header_value=[]
with open(directory+'edit_headers') as f:
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    edit_header_filename.append(p[0])
    edit_header_keyword.append(p[1])
    edit_header_value.append(p[2])
edit_header_filename=np.array(edit_header_filename)
edit_header_keyword=np.array(edit_header_keyword)
edit_header_value=np.array(edit_header_value)
for i in range(0,len(edit_header_filename)):
    print('editing header for ',edit_header_filename[i])
    fits.setval(edit_header_filename[i]+'_stitched.fits',edit_header_keyword[i],value=edit_header_value[i])

utdate=[]
file1=[]
file2=[]
flatfile=[]
tharfile=[]
field_name=[]
scifile=[]
fibermap_file=[]
filter_correction=[]

with open(directory+m2fsrun+'_8_science_raw') as f: 
    data=f.readlines()[0:]
for line in data:
    p=line.split()
    if p[0]!='none':
        utdate.append(str(p[0]))
        file1.append(int(p[1]))
        file2.append(int(p[2]))
        flatfile.append(p[3])
        tharfile.append(p[4])
        field_name.append(p[5])
        scifile.append(p[6])
        fibermap_file.append(p[7])
    if 'filter_correction' in line:
        p2=line.split('filter_correction:')
        filter_correction.append(p2[1])
    else:
        filter_correction.append('none')
utdate=np.array(utdate)
file1=np.array(file1)
file2=np.array(file2)
flatfile=np.array(flatfile)
tharfile=np.array(tharfile)
field_name=np.array(field_name)
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
        id_lines_template0=m2fs.get_id_lines_template(extract1d_array[0],linelist,id_lines_continuum_rejection_low,id_lines_continuum_rejection_high,id_lines_continuum_rejection_iterations,id_lines_continuum_rejection_order,id_lines_threshold_factor,id_lines_window,id_lines_order,id_lines_continuum_rejection_iterations,id_lines_continuum_rejection_sigma,id_lines_tol_angs,id_lines_template_fiddle,id_lines_template0,resolution_order,resolution_rejection_iterations)
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
        data_file=root+'_stitched.fits'
        extract1d_array_file=root+'_extract1d_array.pickle'
        id_lines_array_file=root+'_id_lines_array.pickle'

        extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
        print('identify line template for '+root)

        linelist_file=directory+'m2fs_'+arcfiltername[i]+'_thar_linelist'
        linelist=m2fs.get_linelist(linelist_file)
        id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
        id_lines_template_fiddle=True#says that we already have a template, and will let us fiddle with that
        data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
        id_lines_check0=m2fs.get_id_lines_template(extract1d_array[0],linelist,id_lines_continuum_rejection_low,id_lines_continuum_rejection_high,id_lines_continuum_rejection_iterations,id_lines_threshold_factor,id_lines_window,id_lines_order,id_lines_continuum_rejection_iterations,id_lines_continuum_rejection_sigma,id_lines_tol_angs,id_lines_template_fiddle,id_lines_array[0],resolution_order,resolution_rejection_iterations)

for i in range(0,len(utdate)):
    for ccd in ('b'): 
        root=datadir+utdate[i]+'/'+ccd+str(flatfile[i]).zfill(4)
        root2=datadir+utdate[i]+'/'+ccd+'_'+field_name[i]+'_'+m2fsrun
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
        thars_array_file=root2+'_thars_array.pickle'
        thars_plot_file=root2+'_thars.pdf'
        throughput_continuum_file=root+'_throughput_continuum.pickle'
        stack_array_file=root2+'_stack_array.pickle'
        stack_fits_file=root2+'stackskysub.fits'
        twilightstack_array_file=root2+'_twilightstack_array.pickle'
        twilightstack_fits_file=root2+'_twilightstack.fits'
        stack_skysub_file=root2+'_stackskysub.dat'

        image_boundary_exists=path.exists(image_boundary_file)
        columnspec_array_exists=path.exists(columnspec_array_file)
        apertures_profile_middle_exists=path.exists(apertures_profile_middle_file)
        aperture_array_exists=path.exists(aperture_array_file)
        apmask_exists=path.exists(apmask_file)
        apflat_exists=path.exists(apflat_file)
        apflat_residual_exists=path.exists(apflat_residual_file)
        thars_array_exists=path.exists(thars_array_file)
        throughput_continuum_exists=path.exists(throughput_continuum_file)
        stack_array_exists=path.exists(stack_array_file)
        stack_fits_exists=path.exists(stack_fits_file)
        twilightstack_array_exists=path.exists(twilightstack_array_file)
        twilightstack_fits_exists=path.exists(twilightstack_fits_file)
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
            if ((image_boundary_exists)&(overwrite)):
                image_boundary0=pickle.load(open(image_boundary_file,'rb'))
                image_boundary_fiddle=True
                image_boundary=m2fs.get_image_boundary(data,image_boundary_fiddle,image_boundary0)
                pickle.dump(image_boundary,open(image_boundary_file,'wb'))#save pickle to file
            elif (not(image_boundary_exists)):
                image_boundary0=m2fs.image_boundary()
                image_boundary_fiddle=False
                image_boundary=m2fs.get_image_boundary(data,image_boundary_fiddle,image_boundary0)
                pickle.dump(image_boundary,open(image_boundary_file,'wb'))#save pickle to file

        if initialize:#find apertures in each column stack and save to pickle files
            print('initializing '+root)
            if (not(columnspec_array_exists))|(columnspec_array_exists & overwrite):
                if columnspec_array_exists:
                    print('will overwrite existing '+columnspec_array_file)
                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                columnspec_array=m2fs.get_columnspec(data,trace_step,n_lines,columnspec_continuum_rejection_low,columnspec_continuum_rejection_high,columnspec_continuum_rejection_iterations,columnspec_continuum_rejection_order,threshold_factor,window)
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
            
            if((apertures_profile_middle_exists)&(overwrite)):
                print('loading existing '+apertures_profile_middle_file+', will overwrite')
                apertures_profile_middle,middle_column=pickle.load(open(apertures_profile_middle_file,'rb'))
                apertures_profile_middle=m2fs.fiddle_apertures(columnspec_array,middle_column,window,apertures_profile_middle)
                pickle.dump([apertures_profile_middle,middle_column],open(apertures_profile_middle_file,'wb'))#save pickle to file

            elif(not(apertures_profile_middle_exists)):
                middle_column=np.long(len(columnspec_array)/2)
                apertures_profile_middle=columnspec_array[middle_column].apertures_profile
                apertures_profile_middle=m2fs.fiddle_apertures(columnspec_array,middle_column,window,apertures_profile_middle)
                pickle.dump([apertures_profile_middle,middle_column],open(apertures_profile_middle_file,'wb'))#save pickle to file
                
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
                    scattered_model=CCDData(scatteredlightfunc.func(x,y)*u.electron,mask=np.full((len(data.data),len(data.data[0])),False,dtype=bool),uncertainty=StdDevUncertainty(np.full((len(data.data),len(data.data[0])),scatteredlightfunc.rms,dtype='float')))
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
                    print(len(aperture_array))
                    for k in range(0,len(aperture_array)):
#                    for k in range(0,5):
                        print('   extracting aperture ',aperture_array[k].trace_aperture,' of ',len(aperture_array))
                        if apertures_profile_middle.realvirtual[k]:
#                        if aperture_array[k].trace_rms>0.:
                            extract=m2fs.get_extract1d(k,scatteredlightcorr0,apertures_profile_middle,aperture_array,aperture_peak,pix,extract1d_aperture_width)
                            print(len(np.where(extract.spec1d_mask==False)[0]))
                            if len(np.where(extract.spec1d_mask==False)[0])>0:
                                if np.max(extract.spec1d_flux)>0.:
                                    extract1d_array.append(extract)
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
                    extract1d_array_flat=pickle.load(open(extract1d_array_flat_file,'rb'))
                    for k in range(0,len(extract1d_array_flat)):
                        this=np.where(np.array([aperture_array[q].trace_aperture for q in range(0,len(aperture_array))])==extract1d_array_flat[k].aperture)[0][0]
                        print('   extracting aperture ',aperture_array[this].trace_aperture,' of ',len(extract1d_array_flat))
#                        if apertures_profile_middle.realvirtual[k]:
#                        if aperture_array[k].trace_rms>0.:
                        extract=m2fs.get_extract1d(this,scatteredlightcorr0,apertures_profile_middle,aperture_array,aperture_peak,pix,extract1d_aperture_width)
#                            if len(np.where(extract.spec1d_mask==False)[0])>0:
#                                if np.max(extract.spec1d_flux)>0.:
                        extract1d_array.append(extract)
#                            plt.plot(extract1d_array[len(extract1d_array)-1].spec1d_flux)
#                            plt.show()
#                            plt.close()
#                            np.pause()
                    pickle.dump(extract1d_array,open(extract1d_array_file,'wb'))

        if extract1d_sci:
            
            for j in range(0,len(scifile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(scifile0[i][j]).zfill(4)
                data_file=root0+'_stitched.fits'
                apflatcorr_file=root0+'_apflatcorr.pickle'
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
                    apflatcorr0=pickle.load(open(apflatcorr_file,'rb'))
                    extract1d_array_flat=pickle.load(open(extract1d_array_flat_file,'rb'))

                    pix=np.arange(len(scatteredlightcorr0.data[0]))

                    extract1d_array=[]
                    extract1d_array_flat=pickle.load(open(extract1d_array_flat_file,'rb'))
                    for k in range(0,len(extract1d_array_flat)):
                        this=np.where(np.array([aperture_array[q].trace_aperture for q in range(0,len(aperture_array))])==extract1d_array_flat[k].aperture)[0][0]
                        print('   extracting aperture ',aperture_array[this].trace_aperture,' of ',len(extract1d_array_flat))
#                        if apertures_profile_middle.realvirtual[k]:
#                        if aperture_array[k].trace_rms>0.:
                        extract=m2fs.get_extract1d(this,scatteredlightcorr0,apertures_profile_middle,aperture_array,aperture_peak,pix,extract1d_aperture_width)
#                        extract=m2fs.get_extract1d(this,apflatcorr0,apertures_profile_middle,aperture_array,aperture_peak,pix,extract1d_aperture_width)
#                            if len(np.where(extract.spec1d_mask==False)[0])>0:
#                                if np.max(extract.spec1d_flux)>0.:
                        extract1d_array.append(extract)
                
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
#                    for j in range(39,42):
#                        plt.plot(extract1d_array[j].spec1d_flux)
#                        plt.show()
#                        plt.close()
                                 
#                        print(extract1d_array[j].spec1d_flux)
#                        print(j)
#                        print(j,np.where(extract1d_array[j].spec1d_mask==False),' bbbbbbbbbbbbbbbbbbbbbbbbbbb')
#                        print(' ')
#                        extract1d_array_template[0].spec1d_mask[363]=False
                        print('working on aperture ',j)
                        id_lines_array.append(m2fs.get_id_lines_translate(extract1d_array_template[0],id_lines_template0,extract1d_array[j],linelist,id_lines_continuum_rejection_low,id_lines_continuum_rejection_high,id_lines_continuum_rejection_iterations,id_lines_continuum_rejection_order,id_lines_threshold_factor,id_lines_window,id_lines_order,id_lines_continuum_rejection_iterations,id_lines_continuum_rejection_sigma,id_lines_tol_angs,id_lines_tol_pix,resolution_order,resolution_rejection_iterations,id_lines_translate_add_lines_iterations))
#                    np.pause()
                        
                    pickle.dump(id_lines_array,open(id_lines_array_file,'wb'))

        if tharcheck:

            print('creating thars_array: \n'+root2)
            if (not(thars_array_exists))|(thars_array_exists & overwrite):
                if overwrite:
                    print('will overwrite existing version')

                thar=[]
                lines=[]
                temperature=[]
                for j in range(0,len(tharfile0[i])):
                    root0=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)
                    data_file=root0+'_stitched.fits'
                    id_lines_array_file=root0+'_id_lines_array.pickle'
                    id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    if np.float(data.header['exptime'])>=29.:
                        thar.append(id_lines_array)
                        lines.append([id_lines_array[q].wav.data[id_lines_array[q].wav.mask==False] for q in range(0,len(id_lines_array))])
                        temperature.append(data.header['T-DOME'])
                temperature=np.array(temperature)

                thars_array=[]
                if len(thar)>1:
                    if len(thar[0])>0:
                        for q in range(0,len(thar[0])):
                            thars_array.append(m2fs.get_thars(q,thar,temperature))
                pickle.dump(thars_array,open(thars_array_file,'wb'))

                gs=plt.GridSpec(7,7) # define multi-panel plot
                gs.update(wspace=0,hspace=0) # specify inter-panel spacing
                fig=plt.figure(figsize=(6,6)) # define plot size
                ax1=fig.add_subplot(gs[0:5,0:3])
                ax2=fig.add_subplot(gs[0:5,4:7])
                ax3=fig.add_subplot(gs[6:7,0:7])
                ax1.set_xlabel('wavelength of line [Angs.]')
                ax2.set_xlabel('wavelength of line [Angs.]')
                ax1.set_ylabel('aperture')
                for q in range(0,len(thars_array)):
                    cb1=ax1.scatter(thars_array[q].wav,np.full(len(thars_array[q].wav),thars_array[q].aperture),c=thars_array[q].pix_std,s=3,cmap='inferno',vmin=0,vmax=1.0)
                    cb2=ax2.scatter(thars_array[q].wav,np.full(len(thars_array[q].wav),thars_array[q].aperture),c=thars_array[q].vel_func_std,s=3,cmap='inferno',vmin=0,vmax=1.0)
                    for abc in range(0,len(thars_array[q].pix)):
#                            ax3.scatter(pix0[abc]-pix0[abc][0],vel_func0[abc]-vel_func0[abc][0],alpha=0.3)
#                            ax3.scatter(temperature-temperature[0],pix0[abc]-pix0[abc][0])
                        ax3.plot(thars_array[q].temperature-thars_array[q].temperature[0],thars_array[q].pix[abc]-thars_array[q].pix[abc][0],alpha=0.3)

                fig.colorbar(cb1,ax=ax1,orientation='horizontal')
                fig.colorbar(cb2,ax=ax2,orientation='horizontal')
                ax1.text(0,1,ccd+str(tharfile0[i]),transform=ax1.transAxes)
                ax3.set_ylim([-2,2])
                plt.savefig(thars_plot_file,dpi=200)
#                plt.show()
                plt.close()


        if wavcal:

            thar=[]
            thar_mjd=[]
            for j in range(0,len(tharfile0[i])):
                root0=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)
                data_file=root0+'_stitched.fits'
                id_lines_array_file=root0+'_id_lines_array.pickle'
                id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                times=time.Time(time0,location=lco,precision=6)
                if np.float(data.header['exptime'])>=29.:
                    thar_mjd.append(np.mean(times.mjd))
                    thar.append(id_lines_array)
            thar_mjd=np.array(thar_mjd)    

            for j in range(0,len(allfile0[i])):
                if allfile0[i][j] not in tharfile0[i]:#no need to do this for thars
                    root0=datadir+utdate[i]+'/'+ccd+str(allfile0[i][j]).zfill(4)
                    data_file=root0+'_stitched.fits'
                    extract1d_array_file=root0+'_extract1d_array.pickle'
                    wavcal_array_file=root0+'_wavcal_array.pickle'
                    wavcal_array_exists=path.exists(wavcal_array_file)

                    print('creating wavcal_array: \n'+root0)
                    if (not(wavcal_array_exists))|(wavcal_array_exists & overwrite):
                        if overwrite:
                            print('will overwrite existing version')

                        extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
                        data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                        time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                        times=time.Time(time0,location=lco,precision=6)
                        mjd=np.mean(times.mjd)

                        wavcal_array=[]
                        for j in range(0,len(extract1d_array)):
                            print('working on aperture',j+1,' of ',len(extract1d_array))
                            wavcal_array.append(m2fs.get_wav(j,thar,extract1d_array,thar_mjd,mjd,id_lines_minlines))
                        pickle.dump(wavcal_array,open(wavcal_array_file,'wb'))

        if stack_twilight:
            
            if 'twilight' in field_name[i]:
                print('stacking twilight subexposures_array: \n'+root2)

                if (not(twilightstack_array_exists))|(twilightstack_array_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')
                
                    twilightstack_wavcal_array_file=root2+'_twilightstack_wavcal_array.pickle'

                    thar=[]
                    thar_mjd=[]
                    for j in range(0,len(tharfile0[i])):
                        root0=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)
                        data_file=root0+'_stitched.fits'
                        id_lines_array_file=root0+'_id_lines_array.pickle'
                        id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                        data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                        time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                        times=time.Time(time0,location=lco,precision=6)
                        if np.float(data.header['exptime'])>=29.:
                            thar_mjd.append(np.mean(times.mjd))
                            thar.append(id_lines_array)
                    thar_mjd=np.array(thar_mjd)    
                    
                    temperature=[]
                    stack0=[]
                    stack=[]
                    mjd_mid=[]
                    var_med=[]
                    for j in range(0,len(scifile0[i])):
                        root0=datadir+utdate[i]+'/'+ccd+str(scifile0[i][j]).zfill(4)
                        data_file=root0+'_stitched.fits'
                        extract1d_array_file=root0+'_extract1d_array.pickle'
                        wavcal_array_file=root0+'_wavcal_array.pickle'
                        wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
                        extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
#                    wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
#                    id_lines_array_file=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)+'_id_lines_array.pickle'
#                    id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                        data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                        stack0.append(extract1d_array)
                        temperature.append(data.header['T-DOME'])

                        time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                        times=time.Time(time0,location=lco,precision=6)
                        mjd_mid.append(np.mean(times.mjd))
                        var_med0=[]
                        for k in range(0,len(extract1d_array)):
                            var_med0.append(np.median(extract1d_array[k].spec1d_uncertainty.quantity.value[extract1d_array[k].spec1d_mask==False]**2))
                        var_med0=np.array(var_med0)
                        var_med.append(np.median(var_med0[var_med0==var_med0]))
                    mjd_mid=np.array(mjd_mid)
                    var_med=np.array(var_med)
                    var_med[var_med!=var_med]=1.e+30
                    mjd_weightedmean=np.sum(mjd_mid/var_med)/np.sum(1./var_med)
                    print('stacking ',data_file,' dome temperature=',temperature[len(temperature)-1],' deg C')
                    temperature=np.array(temperature)
       
                    twilightstack_wavcal_array=[]
                    for j in range(0,len(extract1d_array)):
#                        print('twilight stackwavcal working on aperture',j+1,' of ',len(extract1d_array))
                        twilightstack_wavcal_array.append(m2fs.get_wav(j,thar,extract1d_array,thar_mjd,mjd_weightedmean,id_lines_minlines))
#                        print(twilightstack_wavcal_array[len(twilightstack_wavcal_array)-1].wav)
                        shite00=len(twilightstack_wavcal_array[len(twilightstack_wavcal_array)-1].wav)
#                        print(shite00)
#                        if shite==0:
#                            print('asdf')
                    pickle.dump(twilightstack_wavcal_array,open(twilightstack_wavcal_array_file,'wb'))
                    print(twilightstack_wavcal_array_file)
                    
                    twilightstack_array=[]
                    for j in range(0,len(stack0[0])):
                        twilightstack_array.append(m2fs.get_stack(stack0,j))
                    pickle.dump(twilightstack_array,open(twilightstack_array_file,'wb'))#save pickle to file

        if throughputcorr:

            twi_name=[]
            twi_wavcal_name=[]
            twi_mjd=[]
            twi_filtername=[]
            for qqq in range(0,len(utdate)):
                if 'twilight' in field_name[qqq]:
                    root4=datadir+utdate[qqq]+'/'+ccd+'_'+field_name[qqq]+'_'+m2fsrun
                    wavcal0=pickle.load(open(root4+'_twilightstack_wavcal_array.pickle','rb'))
                    data_file=datadir+utdate[qqq]+'/'+ccd+str(scifile0[qqq][0]).zfill(4)+'_stitched.fits'
                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    filtername=data.header['FILTER']
                    if filtername=='Mgb_Rev2':
                        filtername='Mgb_HiRes'
                    if len(wavcal0)>0:
                        twi_wavcal_name.append(root4+'_twilightstack_wavcal_array.pickle')
                        twi_name.append(root4+'_twilightstack_array.pickle')
                        twi_mjd.append(wavcal0[0].mjd)
                        twi_filtername.append(filtername)
            twi_name=np.array(twi_name)
            twi_wavcal_name=np.array(twi_wavcal_name)
            twi_mjd=np.array(twi_mjd)
            twi_filtername=np.array(twi_filtername)

            for j in range(0,len(allfile0[i])):
                if ((allfile0[i][j] not in tharfile0[i])&(allfile0[i][j] not in flatfile0[i])):#no need to do this for thars or flats
                    root0=datadir+utdate[i]+'/'+ccd+str(allfile0[i][j]).zfill(4)
                    data_file=root0+'_stitched.fits'
                    data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                    filtername=data.header['FILTER']
                    if filtername=='Mgb_Rev2':
                        filtername='Mgb_HiRes'
                    time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                    times=time.Time(time0,location=lco,precision=6)
                    mjd=np.mean(times.mjd)
                    throughput_array_file=root0+'_throughput_array.pickle'
                    throughputcorr_array_file=root0+'_throughputcorr_array.pickle'
                    wavcal_array_file=root0+'_wavcal_array.pickle'
                    throughput_array_exists=path.exists(throughput_array_file)
                    throughputcorr_array_exists=path.exists(throughputcorr_array_file)
                    extract1d_array_file=root0+'_extract1d_array.pickle'

                    keep=np.where(twi_filtername==filtername)[0]
                    if len(keep)>0:
                        diff=mjd-twi_mjd[keep]
                        best=np.where(np.abs(diff)==np.min(np.abs(diff)))[0][0]
                        twilightstack_wavcal_array_file=twi_wavcal_name[keep][best]
                        twilightstack_array_file=twi_name[keep][best]
                        twilightstack_wavcal_array=pickle.load(open(twilightstack_wavcal_array_file,'rb'))
                        twilightstack_array=pickle.load(open(twilightstack_array_file,'rb'))
                        print('creating throughputcorr_array: \n'+throughputcorr_array_file)
                        if (not(throughputcorr_array_exists))|(throughputcorr_array_exists & overwrite):
                            if overwrite:
                                print('will overwrite existing version')

                            extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
                            wavcal_array=pickle.load(open(wavcal_array_file,'rb'))

                            twilightstack_continuum_array,throughput_continuum=m2fs.get_throughput_continuum(twilightstack_array,twilightstack_wavcal_array,throughputcorr_continuum_rejection_low,throughputcorr_continuum_rejection_high,throughputcorr_continuum_rejection_iterations,throughputcorr_continuum_rejection_order)

                            throughput_array=[]
                            throughputcorr_array=[]
                            print(len(extract1d_array))
                            for k in range(0,len(extract1d_array)):
                                print('working on ',k+1,' of ',len(extract1d_array))

                                throughput,throughputcorr=m2fs.get_throughputcorr(k,throughput_continuum,extract1d_array,twilightstack_array,twilightstack_continuum_array,wavcal_array,twilightstack_wavcal_array)

                                throughput_array.append(throughput)
                                throughputcorr_array.append(throughputcorr)

                            pickle.dump(throughput_array,open(throughput_array_file,'wb'))
                            pickle.dump(throughputcorr_array,open(throughputcorr_array_file,'wb'))

                    else:
                        print('WARNING: no suitable twilight for throughput correction for \n'+root0)
                        print('so not performing throughput correction')
                        extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
                        throughputcorr_array=extract1d_array
                        throughput_array=[]

#                        gs=plt.GridSpec(10,10) # define multi-panel plot
#                        gs.update(wspace=0,hspace=0) # specify inter-panel spacing
#                        fig=plt.figure(1)
#                        ax1=fig.add_subplot(gs[0:10,0:5])
#                        ax2=fig.add_subplot(gs[0:10,5:10])
#                        ax1.set_ylim([0,30000])
#                        ax2.set_ylim([0,30000])
#                        for k in range(0,len(throughputcorr_array)): 
#                            ax1.plot(throughputcorr_array[k].spec1d_flux[throughputcorr_array[k].spec1d_mask==False],color='k',lw=0.1,alpha=0.1)
#                            ax2.plot(extract1d_array[k].spec1d_flux[extract1d_array[k].spec1d_mask==False],color='r',lw=0.1,alpha=0.1)
#                        ax1.set_ylim([0,30000])
#                        ax2.set_ylim([0,30000])
#                        plt.show()
#                        plt.close()
#                        np.pause()

                        pickle.dump(throughput_array,open(throughput_array_file,'wb'))
                        pickle.dump(throughputcorr_array,open(throughputcorr_array_file,'wb'))

        if plugmap:
            if (('twilight' not in field_name[i])&('standard' not in field_name[i])):
                for j in range(0,len(scifile0[i])):
                    root0=datadir+utdate[i]+'/'+ccd+str(scifile0[i][j]).zfill(4)
                    data_file=root0+'_stitched.fits'
                    extract1d_array_file=root0+'_extract1d_array.pickle'
                    throughput_array_file=root0+'_throughput_array.pickle'
                    throughputcorr_array_file=root0+'_throughputcorr_array.pickle'
                    plugmap_file=root0+'_plugmap.pickle'
                    plugmap_exists=path.exists(plugmap_file)
                    id_lines_array_file=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][0]).zfill(4)+'_id_lines_array.pickle'
                    print('plugmap for: \n'+root0)
                    if (not(plugmap_exists))|(plugmap_exists & overwrite):
                        if overwrite:
                            print('will overwrite existing version')
                        data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                        throughputcorr_array=pickle.load(open(throughputcorr_array_file,'rb'))
                        throughput_array=pickle.load(open(throughput_array_file,'rb'))
                        extract1d_array=pickle.load(open(extract1d_array_file,'rb'))
                        with open(fibermap_file[i]) as f:
                            fibermap=f.readlines()
                        plugmap0=m2fs.get_plugmap(data.header,throughputcorr_array,fibermap,ccd)
                        pickle.dump(plugmap0,open(plugmap_file,'wb'))
                        id_lines_array=pickle.load(open(id_lines_array_file,'rb'))

                        skies=np.where(((plugmap0['objtype']=='SKY')|(plugmap0['objtype']=='unused')))[0]
                        targets=np.where(plugmap0['objtype']=='TARGET')[0]

#                        for k in range(0,len(targets)):
#                            this=np.where(np.array([id_lines_array[q].aperture for q in range(0,len(id_lines_array))])==throughputcorr_array[targets[k]].aperture)[0][0]
#                            print(k,len(np.where(throughputcorr_array[targets[k]].spec1d_mask==False)[0]))
#                            print(id_lines_array[this].npoints,np.median(throughput_array[targets[k]][extract1d_array[targets[k]].spec1d_mask==False]))

#                        gs=plt.GridSpec(7,7) # define multi-panel plot
#                        gs.update(wspace=0,hspace=0) # specify inter-panel spacing
#                        fig=plt.figure(figsize=(6,6)) # define plot size
#                        ax1=fig.add_subplot(gs[0:7,0:3])
#                        ax2=fig.add_subplot(gs[0:7,4:7])
#                    for k in range(0,len(throughputcorr_array)):
#                        this=np.where(np.array([id_lines_array[q].aperture for q in range(0,len(id_lines_array))])==throughputcorr_array[k].aperture)[0][0]
#                        if this>=0:
#                            print(k,len(np.where(throughputcorr_array[k].spec1d_mask==False)[0]))
#                            ax2.plot(id_lines_array[this].func(throughputcorr_array[k].spec1d_pixel[throughputcorr_array[k].spec1d_mask==False]),throughput_array[k][throughputcorr_array[k].spec1d_mask==False],lw=0.3)
                        
                        for k in range(0,len(skies)):
                            this=np.where(np.array([id_lines_array[q].aperture for q in range(0,len(id_lines_array))])==throughputcorr_array[skies[k]].aperture)[0][0]
#                            if this>=0:
#                                ax1.plot(id_lines_array[this].func(extract1d_array[skies[k]].spec1d_pixel[extract1d_array[skies[k]].spec1d_mask==False]),extract1d_array[skies[k]].spec1d_flux[extract1d_array[skies[k]].spec1d_mask==False],lw=0.3,color='r',alpha=0.3)
#                                ax2.plot(id_lines_array[this].func(throughputcorr_array[skies[k]].spec1d_pixel[throughputcorr_array[skies[k]].spec1d_mask==False]),throughputcorr_array[skies[k]].spec1d_flux[throughputcorr_array[skies[k]].spec1d_mask==False],lw=0.3,color='k',alpha=0.3)
#                            ax2.plot(id_lines_array[this].func(throughputcorr_array[skies[k]].spec1d_pixel[throughputcorr_array[skies[k]].spec1d_mask==False]),throughput_array[skies[k]][throughputcorr_array[skies[k]].spec1d_mask==False],lw=0.3)
#                        for k in range(0,len(targets)):
#                            this=np.where(np.array([id_lines_array[q].aperture for q in range(0,len(id_lines_array))])==throughputcorr_array[targets[k]].aperture)[0][0]
#                            ax2.plot(id_lines_array[this].func(throughputcorr_array[targets[k]].spec1d_pixel[throughputcorr_array[targets[k]].spec1d_mask==False]),throughput_array[targets[k]][throughputcorr_array[targets[k]].spec1d_mask==False],lw=0.3)
#                            if this>=0.:
#                                ax2.plot(id_lines_array[this].func(throughputcorr_array[targets[k]].spec1d_pixel[throughputcorr_array[targets[k]].spec1d_mask==False]),throughputcorr_array[targets[k]].spec1d_flux[throughputcorr_array[targets[k]].spec1d_mask==False],lw=0.3,alpha=0.3)
#                        ax1.set_ylim([0,500])
#                        ax2.set_ylim([0,500])
#                        ax1.set_xlim([5120,5195])
#                        ax2.set_xlim([5120,5195])
#                    plt.show()
#                    plt.close()

        if skysubtract:

            if (('twilight' not in field_name[i])&('standard' not in field_name[i])):
                for j in range(0,len(scifile0[i])):
                    root0=datadir+utdate[i]+'/'+ccd+str(scifile0[i][j]).zfill(4)

                    data_file=root0+'_stitched.fits'
                    throughputcorr_array_file=root0+'_throughputcorr_array.pickle'
#                id_lines_array_file=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][0]).zfill(4)+'_id_lines_array.pickle'
                    plugmap_file=root0+'_plugmap.pickle'
                    wavcal_array_file=root0+'_wavcal_array.pickle'
                    sky_array_file=root0+'_sky_array.pickle'
                    skysubtract_array_file=root0+'_skysubtract_array.pickle'
                    skysubtract_array_exists=path.exists(skysubtract_array_file)

                    print('creating sky-subtracted frame: \n'+root0)
                    if (not(skysubtract_array_exists))|(skysubtract_array_exists & overwrite):
                        if overwrite:
                            print('will overwrite existing version')

                        data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                        time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                        times=time.Time(time0,location=lco,precision=6)
                        mjd=np.mean(times.mjd)

                        thar=[]
                        thar_mjd=[]
                        for j in range(0,len(tharfile0[i])):
                            root0=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)
                            data_file=root0+'_stitched.fits'
                            id_lines_array_file=root0+'_id_lines_array.pickle'
                            id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                            data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                            time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                            times=time.Time(time0,location=lco,precision=6)
                            if np.float(data.header['exptime'])>=29.:
                                thar_mjd.append(np.mean(times.mjd))
                                thar.append(id_lines_array)
                        thar_mjd=np.array(thar_mjd)    
                
                        throughputcorr_array=pickle.load(open(throughputcorr_array_file,'rb'))
                        id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                        plugmap0=pickle.load(open(plugmap_file,'rb'))
                        wavcal_array=pickle.load(open(wavcal_array_file,'rb'))

                        meansky=m2fs.get_meansky(throughputcorr_array,wavcal_array,plugmap0)#thar,id_lines_minlines,thar_mjd,mjd,plugmap0)
                    
                        sky_array=[]
                        skysubtract_array=[]
                        for k in range(0,len(throughputcorr_array)):
                            sky,skysubtract=m2fs.get_skysubtract(meansky,k,throughputcorr_array,wavcal_array)
#                            sky,skysubtract=m2fs.get_skysubtract(meansky,k,extract1d_array,wavcal_array)
                            sky_array.append(sky)
                            skysubtract_array.append(skysubtract)
                    
                        pickle.dump(sky_array,open(sky_array_file,'wb'))
                        pickle.dump(skysubtract_array,open(skysubtract_array_file,'wb'))

                        id_nlines=np.array([len(np.where(np.array(id_lines_array[q].wav)>0.)[0]) for q in range(0,len(id_lines_array))],dtype='int')
                        skies=np.where(((plugmap0['objtype']=='SKY')|(plugmap0['objtype']=='unused'))&(id_nlines>=id_lines_minlines))[0]
                        targets=np.where(plugmap0['objtype']=='TARGET')[0]

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


        if stack_frames:

            if (('twilight' not in field_name[i])&('standard' not in field_name[i])):
                print('stacking subexposures_array: \n'+root2)
                if (not(stack_array_exists))|(stack_array_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')
                
                    stack_wavcal_array_file=root2+'_stack_wavcal_array.pickle'

                    thar=[]
                    thar_mjd=[]
                    for j in range(0,len(tharfile0[i])):
                        root0=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)
                        data_file=root0+'_stitched.fits'
                        id_lines_array_file=root0+'_id_lines_array.pickle'
                        id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                        data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                        time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                        times=time.Time(time0,location=lco,precision=6)
                        if np.float(data.header['exptime'])>=29.:
                            thar_mjd.append(np.mean(times.mjd))
                            thar.append(id_lines_array)
                    thar_mjd=np.array(thar_mjd)    
    
                    temperature=[]
                    stack0=[]
                    stack=[]
                    mjd_mid=[]
                    var_med=[]
                    for j in range(0,len(scifile0[i])):
                        root0=datadir+utdate[i]+'/'+ccd+str(scifile0[i][j]).zfill(4)
                        data_file=root0+'_stitched.fits'
                        skysubtract_array_file=root0+'_skysubtract_array.pickle'
                        throughputcorr_array_file=root0+'_throughputcorr_array.pickle'
                        wavcal_array_file=root0+'_wavcal_array.pickle'
                        wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
                        skysubtract_array=pickle.load(open(skysubtract_array_file,'rb'))
                        throughputcorr_array=pickle.load(open(throughputcorr_array_file,'rb'))
#                    wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
#                    id_lines_array_file=datadir+utdate[i]+'/'+ccd+str(tharfile0[i][j]).zfill(4)+'_id_lines_array.pickle'
#                    id_lines_array=pickle.load(open(id_lines_array_file,'rb'))
                        data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                        stack0.append(skysubtract_array)
                        temperature.append(data.header['T-DOME'])

                        time0=[data.header['DATE-OBS']+'T'+data.header['UT-TIME'],data.header['DATE-OBS']+'T'+data.header['UT-END']]
                        times=time.Time(time0,location=lco,precision=6)
                        mjd_mid.append(np.mean(times.mjd))
                        var_med0=[]
                        for k in range(0,len(skysubtract_array)):
                            var_med0.append(np.median(skysubtract_array[k].spec1d_uncertainty.quantity.value[skysubtract_array[k].spec1d_mask==False]**2))
                        var_med0=np.array(var_med0)
                        var_med.append(np.median(var_med0[var_med0==var_med0]))
                    mjd_mid=np.array(mjd_mid)
                    var_med=np.array(var_med)
                    var_med[var_med!=var_med]=1.e+30
                    mjd_weightedmean=np.sum(mjd_mid/var_med)/np.sum(1./var_med)
                    print('stacking ',data_file,' dome temperature=',temperature[len(temperature)-1],' deg C')
                    temperature=np.array(temperature)
       
                    stack_wavcal_array=[]
                    for j in range(0,len(skysubtract_array)):
                        print('stackwavcal working on aperture',j+1,' of ',len(skysubtract_array))
                        stack_wavcal_array.append(m2fs.get_wav(j,thar,skysubtract_array,thar_mjd,mjd_weightedmean,id_lines_minlines))
                    pickle.dump(stack_wavcal_array,open(stack_wavcal_array_file,'wb'))

                    stack_array=[]
                    for j in range(0,len(stack0[0])):
                        stack_array.append(m2fs.get_stack(stack0,j))
                    pickle.dump(stack_array,open(stack_array_file,'wb'))#save pickle to file

        if writefits:

            if (('twilight' not in field_name[i])&('standard' not in field_name[i])):
                if (not(stack_fits_exists))|(stack_fits_exists & overwrite):
                    if overwrite:
                        print('will overwrite existing version')                

                    stack_array_file=root2+'_stack_array.pickle'
                    stack_wavcal_array_file=root2+'_stack_wavcal_array.pickle'
                    stack_array=pickle.load(open(stack_array_file,'rb'))
                    stack_wavcal_array=pickle.load(open(stack_wavcal_array_file,'rb'))

                    for j in range(0,len(scifile0[i])):
                        root0=datadir+utdate[i]+'/'+ccd+str(scifile0[i][j]).zfill(4)
                        data_file=root0+'_stitched.fits'
                        data=astropy.nddata.CCDData.read(data_file)#format is such that data.data[:,0] has column-0 value in all rows
                        wavcal_array_file=root0+'_wavcal_array.pickle'
                        skysubtract_array_file=root0+'_skysubtract_array.pickle'
                        sky_array_file=root0+'_sky_array.pickle'
                        fits_file=root0+'_skysubtract.fits'
                        plugmap_file=root0+'_plugmap.pickle'

                        filtername=data.header['FILTER']
                        if filtername=='Mgb_Rev2':
                            filtername='Mgb_HiRes'

                        wavcal_array=pickle.load(open(wavcal_array_file,'rb'))
                        skysubtract_array=pickle.load(open(skysubtract_array_file,'rb'))
                        sky_array=pickle.load(open(sky_array_file,'rb'))
                        plugmap0=pickle.load(open(plugmap_file,'rb'))

                        new_hdul=m2fs.get_hdul(data,skysubtract_array,sky_array,wavcal_array,plugmap0)
                        new_hdul[0].header['filtername']=filtername
                        new_hdul[0].header['m2fsrun']=m2fsrun
                        new_hdul[0].header['field_name']=field_name[i]

                        if len(skysubtract_array)>0:
                            new_hdul[0].header['weighted_mjd']=str(wavcal_array[0].mjd)
                        print('writing individual frame .fits for: \n'+root0)
                        new_hdul.writeto(fits_file,overwrite=True)

#                stack_mjd=stack_wavcal_array[0].mjd
                    new_hdul=m2fs.get_hdul(data,stack_array,sky_array,stack_wavcal_array,plugmap0)
                    new_hdul[0].header['filtername']=filtername
                    new_hdul[0].header['m2fsrun']=m2fsrun
                    new_hdul[0].header['field_name']=field_name[i]
                    if len(stack_array)>0:
                        new_hdul[0].header['weighted_mjd']=str(stack_wavcal_array[0].mjd)
                    new_hdul[0].header['continuum_lamp_frame']=flatfile[i]
                    tharframe0=tharfile[i].split('-')
                    for q in range(0,len(tharframe0)):
                        keyword='comparison_arc_frame_'+str(q)
                        new_hdul[0].header[keyword]=tharframe[q]
#                    new_hdul[0].header['comparison_arc_frame']=tharfile[i]
                    sciframe0=scifile[i].split('-')
                    for q in range(0,len(sciframe0)):
                        keyword='sub_frame_'+str(q)
                        new_hdul[0].header[keyword]=sciframe[q]
#                    new_hdul[0].header['sub_frames']=scifile[i]
                    print('writing stack fits for: \n'+root2)
                    print(len(new_hdul[0].data),len(new_hdul[5].data))
                    if len(new_hdul[0].data)!=len(new_hdul[5].data):
                        np.pause()
                    new_hdul.writeto(stack_fits_file,overwrite=True)
#                for j in range(0,len(stack_array)):
#                    this=np.where(plugmap0['aperture']==stack_array[j].aperture)[0][0]
#                    if plugmap0['objtype'][this]=='TARGET':
#                        coords=coord.SkyCoord(plugmap0['ra'][this],plugmap0['dec'][this],unit=(u.deg,u.deg))
#                        lco=coord.EarthLocation.of_site('lco')
#                        times=time.Time(stack_wavcal_array[j].mjd,format='mjd',location=lco)
#                        ltt_helio=times.light_travel_time(coords,'heliocentric')
#                        hjd=times.jd+ltt_helio.value
#                        out=datadir+utdate[i]+'/m2fs_'+filtername+'_'+field_name[i]+'_'+str(i+1)+'_'+m2fsrun+'_ra'+str('{0:6f}'.format(round(plugmap0['ra'][this],6)))+'_dec'+str('{0:6f}'.format(round(plugmap0['dec'][this],6)))+'_hjd'+str(round(hjd,3))+'_ap'+ccd+str(plugmap0['aperture'][this]).zfill(3)+'_stackskysub.dat'
#                        print(out)
#                        g1=open(out,'w')
#                        for k in range(0,len(stack_array[j].spec1d_pixel)):
#                            if stack_array[j].spec1d_mask[k]==False:
#                                string=str(round(stack_wavcal_array[j].wav[k],5))+' '+str(round(stack_array[j].spec1d_flux.value[k],3))+' '+str(round(stack_array[j].spec1d_uncertainty.quantity.value[k]**2,3))+' \n'
#                                print(i,j,k,string)
#                                g1.write(string)
#                        g1.close()
