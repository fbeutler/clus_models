import matplotlib.pyplot as plt

import clus_models


def main():
	Nk = 400
	dkp = 0.001
	kp = [i*dkp + dkp/2 for i in range(0,Nk)]

	model = clus_models.EFTofLSSClass()
	model.path_to_linear_pk = '/Users/xflorian/GR_sims/halo_catalog_boxlen2625_n4096_lcdmw7v2_00000_fullsky_fofb02000m_v00002/pk_lcdmw7v2.dat'
	pk = model.get_pk(kp)

	plt.plot(kp, pk)
	plt.show()
	return 


# to call the main() function to begin the program.
if __name__ == '__main__':
    main()