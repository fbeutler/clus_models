import os, sys
import numpy as np
import scipy.integrate as integrate
from scipy.special import legendre


def kaiser(para_dict, pk_model, ells=[]):
	'''
	Calculates the linear power spectrum multipoles including RSD

	Input:
		para_dict: Dictionary including model parameters
			para_dict['b1']: Linear bias parameter
			para_dict['f']: Linear growth rate
			para_dict['k']: Array of k-bins
		pk_model: Function returning the linear power spectrum
		ells: Multipoles to be calculated

	Output: 
		model_dict: Dictionary containing the power spectrum multipoles
			model_dict['k']: Input k-bins
			model_dict['pkX']: Power spectrum multipole X (specified in ells)
	'''
	if not ells:
		print("WARNING: You have to set the multipoles to be returned ells=[0,2,4]")
		return None

	if (not 'k' in para_dict or
		not 'b1' in para_dict or
		not 'f' in para_dict):
		print("WARNING: Required parameter is missing")
		# Print required parameters
		return None

	model = {}
	model['k'] = para_dict['k']
	for ell in ells:
		model['pk%d' % int(ell)] = []
		P = legendre(ell)
		for k in para_dict['k']:
			int_func = lambda mu: ((2.*int(ell) + 1.)/2.)*P(mu)*(para_dict['b1'] + para_dict['f']*mu**2)**2*pk_model(k)
			dummy = integrate.quad(int_func, -1, 1.)
			model['pk%d' % int(ell)].append(dummy[0])
		model['pk%d' % int(ell)] = np.array(model['pk%d' % int(ell)])
	return model
