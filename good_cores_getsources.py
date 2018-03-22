import numpy

def get_good_cores(region='Auriga', visual_checks = 'l1157.gs.vischeck.jk.txt'):

	getsources_core_catalog = '/mnt/scratch-lustre/jkeown/GS_Extractions/GS-Extractions/' + region + '/' + region + '.sw.final.reliable.ok.cat'

	NO,    XCO_P,   YCO_P,  WCS_ACOOR,  WCS_DCOOR,  SIG_GLOB,  FG,  GOOD, SIG_MONO01, FM01, FXP_BEST01, FXP_ERRO01, FXT_BEST01, FXT_ERRO01, AFWH01, BFWH01, THEP01, SIG_MONO02, FM02, FXP_BEST02, FXP_ERRO02, FXT_BEST02, FXT_ERRO02, AFWH02, BFWH02, THEP02 = numpy.loadtxt(getsources_core_catalog,comments='!',unpack=True)

	#num,vis_checks = numpy.loadtxt(visual_checks, skiprows=1, usecols=(0,1), dtype=[('bg','S5'),('type','S10')], unpack=True)

	good_source_indices = []
	index = 0
	#Remove bad sources using the Herschel selection criteria
	if region=='PipeE1':

		if GOOD>=1.0 and SIG_GLOB>10 and SIG_MONO02 > 5.0 and (FXP_BEST02/FXP_ERRO02) > 1.0:
			good_source_indices.append(index)
	else:
		for i in range(len(NO)):
			if GOOD[i]>=1.0 and SIG_GLOB[i]>10 and SIG_MONO02[i] > 5.0 and (FXP_BEST02[i]/FXP_ERRO02[i]) > 1.0:
				good_source_indices.append(index)
			index += 1

	print len(good_source_indices)
	#print good_source_indices

	### Save the "good core" indices to a file
	#numpy.savetxt(good_source_indices_file, good_source_indices, fmt='%i')
	#no = numpy.where(vis_checks=="no")
	#no_2 = numpy.where(vis_checks=="no?")
	#no_tot = numpy.append(no, no_2)
	#updated_good_source_indices = numpy.delete(good_source_indices, no_tot) 
	#print len(updated_good_source_indices)
	#print updated_good_source_indices
	return numpy.array(good_source_indices)

#get_good_cores()
