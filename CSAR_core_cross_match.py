import aplpy
import matplotlib.pyplot as plt
import pyfits
import numpy
import pyregion
import astropy
from astropy.io import fits
from astropy import wcs
from astropy import coordinates
import matplotlib.patheffects as Patheffects
from astropy import units as u

import good_cores_getsources

def cross_match_CSAR(getsources_core_catalog = '/mnt/scratch-lustre/jkeown/Getsources/Extract/cep1157/120115_flat/combo/+catalogs/L1157.sw.final.reliable.ok.cat', CSAR_catalog = '/mnt/scratch-lustre/jkeown/DS9_regions/L1157/CSAR/CEPl1157_CSAR.dat', high_res_coldens_image = '/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1157/080615/cep1157_255_mu.image.resamp.fits', CSAR_core_indices='L1157_matched_CSAR_cores.dat'):
	# Import getsources "good core" data table
	cores_array1 = numpy.loadtxt(getsources_core_catalog,comments='!')
	
	good_core_indices = good_cores_getsources.get_good_cores(getsources_core_catalog)
	cores_array = cores_array1[numpy.array(good_core_indices)]

	### Import the CSAR catalog 
	### ***MAKE SURE FIRST TWO COLUMNS OF "CSAR_catalog" FILE ARE X_POSITION AND Y_POSITION OF SOURCE IN DECIMAL DEGREES***
	CSAR_array = numpy.loadtxt(CSAR_catalog,comments='#')
	#print CSAR_array
	CSAR_positions = numpy.column_stack((CSAR_array[:,0], CSAR_array[:,1])) 
	#print CSAR_positions
	w = wcs.WCS(high_res_coldens_image)
	pos_pix = w.wcs_world2pix(CSAR_positions, 1)

	### Loop through the potential matched cores identified in the step above.
	counter = 0
	matched_cores = []
	for line in cores_array:	

		x_coor = str(line[3])
		y_coor = str(line[4])
	
		### Create a DS9 region string for the core's getsources ellipse, 
		### from which a mask will be created. 
		region = ('fk5;ellipse(' + x_coor + ', ' + y_coor + ', ' + str((line[50]/2.0)/3600.) + ', ' + 
		str((line[51]/2.0)/3600.) + ', ' + str(line[52]+90.0)+')')
	
		r = pyregion.parse(region)
		f=fits.open(high_res_coldens_image)
		mymask = r.get_mask(hdu=f[0])
		f.close()
		newmask=mymask
		### Set all values outside the core's ellipse to zero, 
		### all values inside the ellipse are set to one.
		newmask=numpy.where(newmask==0,0,1)
		mask_shape = numpy.shape(newmask)
		### Loop through the CSAR catalog
		### If any CSAR cores fall within a getsources core's ellipse, 
		### store the getsources core's index
		match_counter=0
		for i in pos_pix:
			ypos = int(round(i[1],0))-1
			xpos = int(round(i[0],0))-1
			if ypos<=mask_shape[0] and xpos<=mask_shape[1]:
				if newmask[ypos][xpos]==1 and match_counter==0:
					matched_cores.append(counter)
					# match_counter prevents counting indices twice 
					# if two CSAR cores fall within getsources ellipse
					match_counter+=1	
		#print len(matched_cores)
		counter += 1

	print 'CSAR_matched:total ratio =' + str(round(float(len(matched_cores)) / float(len(cores_array[:,0])),3))
	### Save the CSAR matched core indices to a file
	#numpy.savetxt(CSAR_core_indices, matched_cores, fmt='%i')
	return numpy.array(matched_cores)

#cross_match_CSAR()

#cross_match_CSAR(getsources_good_cores_catalog = '/mnt/scratch-lustre/jkeown/DS9_regions/L1172/core_SED/L1172_good_sources.dat', CSAR_catalog = '/mnt/scratch-lustre/jkeown/DS9_regions/L1172/CSAR/CEPl1172_CSAR.dat', high_res_coldens_image = '/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1172/082315/cep1172_255_mu.image.resamp.fits', CSAR_core_indices='L1172_matched_CSAR_cores.dat')

#cross_match_CSAR(getsources_good_cores_catalog = '/mnt/scratch-lustre/jkeown/DS9_regions/L1228/core_SED/L1228_good_sources.dat', CSAR_catalog = '/mnt/scratch-lustre/jkeown/DS9_regions/L1228/CSAR/CEPl1228_CSAR.dat', high_res_coldens_image = '/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1228/082315/cep1228_255_mu.image.resamp.fits', CSAR_core_indices='L1228_matched_CSAR_cores.dat')

#cross_match_CSAR(getsources_core_catalog = '/mnt/scratch-lustre/jkeown/Getsources/Extract/cep1241/120115_flat/combo/+catalogs/L1241.sw.final.reliable.ok.cat', CSAR_catalog = '/mnt/scratch-lustre/jkeown/DS9_regions/L1241/CSAR/CEPl1241_CSAR.dat', high_res_coldens_image = '/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1241/071415/cep1241_255_mu.image.resamp.fits', CSAR_core_indices='L1241_matched_CSAR_cores.dat')

#cross_match_CSAR(getsources_good_cores_catalog = '/mnt/scratch-lustre/jkeown/DS9_regions/L1251/core_SED/L1251_good_sources.dat', CSAR_catalog = '/mnt/scratch-lustre/jkeown/DS9_regions/L1251/CSAR/CEPl1251_CSAR.dat', high_res_coldens_image = '/mnt/scratch-lustre/jkeown/Getsources/Prepare/Images/cep1251/082315/cep1251_255_mu.image.resamp.fits', CSAR_core_indices='L1251_matched_CSAR_cores.dat')
