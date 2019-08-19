import os, sys
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.special import legendre

import subprocess
import time


class BaseClass:
	'''
	This class contains functions and parameters which are 
	common to all models 
	'''
	def f():
		return 


class EFTofLSSClass(BaseClass):
	'''
	'''
	def __init__(self):
		# Default parameters
		self.km = 1.
		self.knl = 1.
		self.nbar = 3.e-4
		self.z_pk = 0.55
		self.omega_b = 0.0171688
		self.omega_cdm = 0.0926812
		self.h = 0.6777
		self.ns = 0.96
		self.lnAS = 2.6666 # ln10^{10}A_s
		self.path_to_linear_pk = None
		self.b1 = 2.23
		self.b2 = np.sqrt(2)*1.2
		self.b3 = -0.6
		self.b4 = np.sqrt(2)*1.2
		self.b5 = 0.2 
		self.b6 = -6.9 
		self.b7 = 0. 
		self.b8 = 2 
		self.b9 = 0.
		self.b10 = -2.3 
		self.b11 = 0
		self.e1 = 0 
		self.e2 = 0 
		self.e3 = 0
		# if executabel does not exist we need to run the Makefile
		if not os.path.exists('cbird_v0p4_EH/cbird'):
			print("Compile EFT code to get executable")
			subprocess.check_output(['make'], cwd="cbird_v0p4_EH")
			#subprocess.check_output(['cd cbird_v0p4_EH && make'])

	def citation(self):
		'''
		Print out the citation information for this model
		'''
		return 

	def parameters(self):
		'''
		Print out the parameters which this model has
		'''
		return 

	def _write_ini_file(self):
		'''
		'''
		with open('cbird_v0p4_EH/cbird_dynamic.ini', 'w') as f:
			f.write('PathToOutput = cbird_v0p4_EH/output/\n')
			f.write('PathToLinearPowerSpectrum = %s\n' % self.path_to_linear_pk)
			f.write('PathToTriangles = ./testbird/TrianglesConfiguration.dat\n')
			f.write('z_pk = %f\n' % self.z_pk)
			f.write('omega_b = %f\n' % self.omega_b)
			f.write('omega_cdm = %f\n' % self.omega_cdm)
			f.write('h = %f\n' % self.h)
			f.write('ns = %f\n' % self.ns)
			f.write('ln10^{10}A_s = %f\n' % self.lnAS)
			f.write('ComputePowerSpectrum = yes\n')
			f.write('ResumPowerSpectrum = yes\n')
			f.write('ComputeBispectrum = no\n')
			f.write('km = %f\n' % self.km)
			f.write('knl = %f\n' % self.knl)
			f.write('nbar = %f\n' % self.nbar)
		return 

	def get_pk(self, k_array):
		'''
		'''
		self._write_ini_file()
		# Run EFT code and write to files in cbird_v0p4_EH/output
		start = time.time()
		subprocess.check_output(['cbird_v0p4_EH/cbird','cbird_v0p4_EH/cbird_dynamic.ini'])
		print("Power spectrum calculated after %0.3fs" % (time.time()-start))
		return self._assemble_pk(k_array)

	def _assemble_pk(self, k_array):
		'''
		'''
		# Read the new power spectrum files
		Plin = np.loadtxt('cbird_v0p4_EH/output/PowerSpectraLinear.dat', unpack=True)
		Ploop = np.loadtxt('cbird_v0p4_EH/output/PowerSpectra1loop.dat', unpack=True)[:19]
		vlin, vloop = self._get_bias()
		kk = Plin[0, :len(Plin[0])//3]
		Plin = Plin.reshape(Plin.shape[0], 3, len(kk))[1:]
		Ploop = Ploop.reshape(Ploop.shape[0], 3, len(kk))[1:]
		toadd = np.array([self.b8/self.nbar*np.ones_like(kk),
		                  self.b10*kk**2/self.km**2/self.nbar,
		                  np.zeros_like(kk)])
		PS = np.einsum('c,c...->...', vlin, Plin) + np.einsum('c,c...->...', vloop, Ploop) + toadd

		iPS = interp1d(kk, PS[0], kind='cubic', fill_value="extrapolate", bounds_error=False)
		return iPS(k_array)

	def _get_bias(self):
		km = 1.
		knl = 1.
		vlin = np.array([1, self.b1, self.b1**2])
		vloop = np.array([1, self.b1, self.b2, self.b3, self.b4,
		              self.b1**2, self.b1 * self.b2, self.b1 * self.b3, self.b1 * self.b4,
		              self.b2**2, self.b2 * self.b4, self.b4**2,
		              self.b1*self.b5/self.knl**2, self.b1 * self.b6/self.km**2,
		              self.b1*self.b7/self.km**2, self.b5/self.knl**2,
		              self.b6/self.km**2, self.b7/self.km**2])
		return vlin, vloop


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
