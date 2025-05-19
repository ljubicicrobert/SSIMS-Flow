"""
This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this package. If not, you can get eh GNU GPL from
https://www.gnu.org/licenses/gpl-3.0.en.html.

Created by Robert Ljubicic.
"""


try:
	from __init__ import *
	from class_console_printer import tag_print
	from utilities import exit_message, present_exception_and_exit, cfg_get
	from scipy.optimize import minimize

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


def main(cfg, ys, mags):
	def mse(params, array_fixed):
		array_changing = generate_profile(*params, thr_data_ys)
		return np.nansum((array_fixed - array_changing) ** 2)

	def generate_profile(pBeta, pUmax, pBratio, _ys):
		arr = np.ndarray([_ys.size])

		B1 = B * pBratio
		B2 = B - B1

		for i, y in enumerate(_ys):
			try:
				arr[i] = pUmax * (1 - ((y - B1)/B1)**2)**pBeta if y < B1 else pUmax * (1 - ((y - B1)/B2)**2)**pBeta 
			except RuntimeWarning:
				arr[i] = 0.0

		return arr
	
	try:
		section = 'Optical flow'

		Ymin_perc = cfg_get(cfg, section, 'RiverbankStart', int, default=0)
		Ymax_perc = 100 - cfg_get(cfg, section, 'RiverbankEnd', int, default=0)
		
		chain_max = np.max(ys)

		Ymin = Ymin_perc / 100 * chain_max
		Ymax = Ymax_perc / 100 * chain_max

		if Ymin >= 100 - Ymax:
			tag_print('error', 'Riverbank start is greater than or equal to riverbank end!')
			tag_print('error', 'Check riverbank settings!')
			exit_message()

		B = Ymax - Ymin

		Beta = 0.25
		Umax = 1.00
		Bratio = 0.50

		fit_thr = cfg_get(cfg, section, 'FitThreshold', float, default=0.67)
		beta_min = cfg_get(cfg, section, 'BetaMin', float, default=0.15)
		beta_max = cfg_get(cfg, section, 'BetaMax', float, default=0.50)
		Umax_min = cfg_get(cfg, section, 'UMaxMin', float, default=0.0)
		Umax_max = cfg_get(cfg, section, 'UMaxMax', float, default=10.0)
		Bratio_min = cfg_get(cfg, section, 'BratioMin', float, default=0.1)
		Bratio_max = cfg_get(cfg, section, 'BratioMax', float, default=0.9)

		bounds = [[beta_min, beta_max], [Umax_min, Umax_max], [Bratio_min, Bratio_max]]

		data_ys = ys - Ymin
		data_mag = mags

		count_below_0 = np.sum(data_ys < 0)
		count_above_B = np.sum(data_ys > B)
		
		mask = data_ys < 0
		data_ys = data_ys[~mask]
		data_mag = data_mag[~mask]

		mask = data_ys > B
		data_ys = data_ys[~mask]
		data_mag = data_mag[~mask]

		bound_data_ys = data_ys.copy()

		thr_data_ys = data_ys.copy()
		thr_data_mag = data_mag.copy()

		mask = data_mag > fit_thr*np.max(data_mag)
		thr_data_ys[mask == False] = np.nan
		thr_data_mag[mask == False] = np.nan

		result = minimize(mse, (Beta, Umax, Bratio), args=(thr_data_mag,), bounds=bounds, method='L-BFGS-B')
		Beta, Umax, Bratio = result.x

		optimized_mags = generate_profile(Beta, Umax, Bratio, bound_data_ys)

		print()
		tag_print('info', 'Optimized power-law fit parameters:')
		tag_print('info', f'    Beta   = {Beta:.3f}')
		tag_print('info', f'    Umax   = {Umax:.3f}')
		tag_print('info', f'    Bratio = {Bratio:.3f}')
		print()

		mags_fitted = np.hstack([np.zeros([count_below_0]), optimized_mags, np.zeros([count_above_B])])
		mags_fitted = np.nan_to_num(mags_fitted, nan=0.0, posinf=0.0, neginf=0.0)

		return mags_fitted, Beta, Umax, Bratio
	
	except Exception as ex:
		present_exception_and_exit()
