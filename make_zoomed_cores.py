import aplpy
import matplotlib.pyplot as plt
import numpy
import pyregion
import astropy
from astropy.io import fits
import matplotlib.patheffects as Patheffects

import good_cores_getsources

#NO,    XCO_P,   YCO_P,  WCS_ACOOR,  WCS_DCOOR,  SIG_GLOB,  FG,  GOOD, SIG_MONO01, FM01, FXP_BEST01, FXP_ERRO01, FXT_BEST01, FXT_ERRO01, AFWH01, BFWH01, THEP01, SIG_MONO02, FM02, FXP_BEST02, FXP_ERRO02, FXT_BEST02, FXT_ERRO02, AFWH02, BFWH02, THEP02

def auto_zoomed_cores(region='Aquila', distance=260.):

	print region

	getsources_core_catalog = '/mnt/scratch-lustre/jkeown/GS_Extractions/GS-Extractions/' + region + '/' + region + '.sw.final.reliable.ok.cat'
	core_figure_directory = '/mnt/scratch-lustre/jkeown/JCMT_GBS_Jybeam/JCMT_pipeline/Figures/' + region + '/core_figures/'
	JCMT_450um_image='/mnt/scratch-lustre/jkeown/GS_Extractions/GS-Extractions/'+ region + '/' + region + '_450_Jybeam.m.fits'
	JCMT_850um_image='/mnt/scratch-lustre/jkeown/GS_Extractions/GS-Extractions/'+ region + '/' + region + '_850_Jybeam.m.fits'

	# Import getsources "good core" data table
	cores_array1 = numpy.loadtxt(getsources_core_catalog,comments='!')
	good_core_indices = good_cores_getsources.get_good_cores(region=region)
	cores_array = cores_array1[numpy.array(good_core_indices)]

	if region=='PipeE1':
		cores_array = numpy.array(cores_array1)
		z = numpy.zeros(len(cores_array))
		cores_array = numpy.stack((cores_array, z))

	core_center_RA = cores_array[:,3]
	core_center_Dec = cores_array[:,4]
	
	core_index = 1
	for line in cores_array:

		x_coor = str(line[3])
		y_coor = str(line[4])
		AFWHM_array = [line[14], line[23]] # 450, 850
		BFWHM_array = [line[15], line[24]]
		Theta_array = [line[16], line[25]]
		SIG_array = [line[8], line[17]]
		images_array = [JCMT_450um_image, JCMT_850um_image]
		wavelengths = ['450', '850']
		maximums = numpy.zeros(len(AFWHM_array))
		minimums = numpy.zeros(len(AFWHM_array))
		counter = 0
		for i in wavelengths:
			### Create a DS9 region string for the core's getsources ellipse, 
			### from which a mask will be created. 
			region1 = ('fk5;ellipse(' + x_coor + ', ' + y_coor + ', ' + str(AFWHM_array[counter]/3600.)+ ', ' + str(BFWHM_array[counter]/3600.) + ', ' + str(Theta_array[counter]+90.)+')')
			r = pyregion.parse(region1)

			f = fits.open(images_array[counter])
			header_primary = fits.getheader(images_array[counter])
			data = fits.getdata(images_array[counter])

			mymask = r.get_mask(hdu=f[0])
			newmask=numpy.where(mymask!=0)
			maximums[counter] = max(data[newmask])
			minimums[counter] = min(data[newmask])
				
			region2 = ('fk5;box(' + x_coor + ', ' + y_coor + ', ' + str(0.04) + ', ' + 
			str(0.04) + ', ' + str(0.0)+')')
			r = pyregion.parse(region2)

			mymask = r.get_mask(hdu=f[0])
			newmask=numpy.where(mymask!=0)
			f.close()
			newmask2=numpy.where(mymask==0, 0, data)
			fits.writeto(region + '_' +i + '_contour_mask.fits', newmask2, header_primary, clobber=True)
			counter+=1
	
		microns = ['mu_450', 'mu_850']
		if maximums[0]<0.:
			maximums[0]=10.0
		if minimums[0]<0.:
			minimums[0]=-5.0
		v_max = maximums
		v_min = minimums
		
		contour_levels = [(maximums[0]*0.3,maximums[0]*0.5,maximums[0]*0.7,maximums[0]*0.9),
		(maximums[1]*0.3,maximums[1]*0.5,maximums[1]*0.7,maximums[1]*0.9)]

		fig = plt.figure() #figsize=(8,5.5)

		f=0
		for i in wavelengths:
			microns[f]=aplpy.FITSFigure(region + '_' + i + '_contour_mask.fits', figure=fig, subplot=(2,1,f+1))
			microns[f].recenter(line[3],line[4],0.01)
			microns[f].show_contour(region + '_' + i + '_contour_mask.fits', colors=('white', 'white', 'white','grey'), levels=contour_levels[f], overlap=True)
			microns[f].show_colorscale(cmap='gist_stern', vmin=v_min[f], vmax=v_max[f])
			#microns[f].show_regions('/mnt/scratch-lustre/jkeown/DS9_regions/HGBS_pipeline/L1157/L1157_core_SED/255_all_cores_good.reg')
			microns[f].add_colorbar()
			microns[f].colorbar.set_axis_label_text('Jy/beam')
				
			scale = str('{:.2f}'.format(2*(distance*numpy.tan(numpy.radians(0.5*(1.0/60.))))))
			if f==0:
				microns[f].add_scalebar((1./60.), path_effects=[Patheffects.withStroke(linewidth=3, foreground='white')])
				microns[f].scalebar.show(1./60.)  # length in degrees (one arcminute)
				microns[f].scalebar.set_corner('top left')
				microns[f].scalebar.set_label('1' + "'" + ' = ' + scale + ' pc')
			if SIG_array[f]>5.:
				line_type = 'solid'
			else:
				line_type = 'dashed'	
			microns[f].show_ellipses(line[3], line[4], AFWHM_array[f]/3600., BFWHM_array[f]/3600., Theta_array[f]+90., color='#00FF00', zorder=10, linewidth=1.0, linestyle=line_type)
			#microns[f].show_markers(line[3], line[4], c='#00FF00', marker='x', zorder=10, linewidths=1.5,s=20)
			#microns[f].show_markers(line[3], line[4], c='black', marker='x', zorder=9, linewidths=2.0,s=30)
			microns[f].tick_labels.hide() 
			microns[f].axis_labels.hide()
			#microns[f].set_title('L' + name)
			microns[f].add_label(0.3,0.1,i + ' $\mu$' + 'm',relative=True, path_effects=[Patheffects.withStroke(linewidth=3, foreground='white')])
			f=f+1
		fig.subplots_adjust(hspace=0.05, wspace=0.35)
		fig.canvas.draw()
		fig.suptitle('Core Number ' + str(core_index) + ' - RA: ' + str(line[3]) + ' Dec: ' + str(line[4]))
		fig.savefig(core_figure_directory + 'core' + str(core_index) + '.pdf')
		print 'Core Number = ' + str(core_index)
		for i in range(2):
			microns[i].close()
		core_index = core_index + 1
		#plt.show()

#auto_zoomed_cores(region = 'Aquila', distance=436.)
#auto_zoomed_cores(region = 'Auriga', distance=450.)
#auto_zoomed_cores(region = 'CepheusL1228', distance=200.)
#auto_zoomed_cores(region = 'CepheusL1251', distance=300.)
#auto_zoomed_cores(region = 'CepheusSouth', distance=288.)
#auto_zoomed_cores(region = 'CrA', distance=130.)
#auto_zoomed_cores(region = 'IC5146', distance=950.)
#auto_zoomed_cores(region = 'Lupus', distance=140.)
#auto_zoomed_cores(region = 'OphL1688', distance=139.)
#auto_zoomed_cores(region = 'OphL1689_1709_12', distance=139.)
#auto_zoomed_cores(region = 'OphScoN2', distance=139.)
#auto_zoomed_cores(region = 'OphScoN3', distance=139.) ## No sources pass selection criteria 
#auto_zoomed_cores(region = 'OrionA', distance=450.)
#auto_zoomed_cores(region = 'OrionB_L1622', distance=415.)
#auto_zoomed_cores(region = 'OrionB_N2023', distance=415.)
#auto_zoomed_cores(region = 'OrionB_N2068', distance=415.)
#auto_zoomed_cores(region = 'PerseusIC348', distance=250.)
auto_zoomed_cores(region = 'PerseusWest', distance=250.)
#auto_zoomed_cores(region = 'PipeB59', distance=145.)
#auto_zoomed_cores(region = 'PipeE1', distance=145.)
#auto_zoomed_cores(region = 'SerpensE', distance=436.)
#auto_zoomed_cores(region = 'SerpensMain', distance=436.)
#auto_zoomed_cores(region = 'SerpensMWC297', distance=250.)
#auto_zoomed_cores(region = 'SerpensN', distance=436.)
#auto_zoomed_cores(region = 'TaurusB18_E', distance=140.)
#auto_zoomed_cores(region = 'TaurusB18_W', distance=140.)
#auto_zoomed_cores(region = 'TaurusL1495', distance=140.)
#auto_zoomed_cores(region = 'TaurusTMC', distance=140.)
