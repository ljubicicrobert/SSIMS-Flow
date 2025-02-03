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
	from class_console_printer import tag_print, unix_path
	from utilities import cfg_get, exit_message, present_exception_and_exit
	from scipy.io import loadmat
	from SDI_estimate import custom_medfilt
	from scipy.ndimage import label

	import matplotlib.pyplot as plt

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--cfg', type=str, help='Path to config file')
		args = parser.parse_args()

		cfg = configparser.ConfigParser()
		cfg.optionxform = str

		try:
			cfg.read(args.cfg, encoding='utf-8-sig')
		except Exception:
			tag_print('error', 'There was a problem reading the configuration file!')
			tag_print('error', 'Check if project has valid configuration.')
			exit_message()

		section = 'Optical flow'

		project_folder = unix_path(cfg_get(cfg, 'Project settings', 'Folder', str))
		results_folder = unix_path(f'{project_folder}/SDI')

		data = loadmat(f'{results_folder}/SDI.mat')
		SDI = data['SDI'].squeeze()
		mean_SDI = data['MeanSDI'].squeeze()
		optimal_start_frame = data['OptimalStartFrame'].squeeze()
		optimal_end_frame = data['OptimalEndFrame'].squeeze()

		filtered_SDI = custom_medfilt(SDI, 10)
		binary_filtered_SDI = filtered_SDI < mean_SDI
		labeled_binary_filtered_SDI, num_regions = label(binary_filtered_SDI)

		fig, ax = plt.subplots(nrows=2, sharex=True)

		ax[0].plot(SDI, label='SDI')
		ax[0].plot(filtered_SDI, '-r', label='Filtered SDI')
		ax[0].axhline(mean_SDI, linestyle='--', color='k', linewidth=1, label='SDI threshold')
		ax[0].fill_between(range(optimal_start_frame, optimal_end_frame + 1), filtered_SDI[optimal_start_frame: optimal_end_frame + 1], mean_SDI,
							where=filtered_SDI[optimal_start_frame: optimal_end_frame + 1] < mean_SDI, color=[0.9290, 0.6940, 0.1250], label='Optimal window')
		ax[0].set_ylabel('SDI')
		ax[0].legend()
		ax[0].set_xlim(0, len(SDI))

		ax[1].plot(binary_filtered_SDI, '-r')
		ax[1].set_xlabel('Frame Number')
		ax[1].set_ylabel('Candidate frame')
		ax[1].set_xlim(0, len(SDI))
		ax[1].fill_between(range(optimal_start_frame, optimal_end_frame + 1), 1, color=[0.9290, 0.6940, 0.1250])

		try:
			mng = plt.get_current_fig_manager()
			mng.window.state('zoomed')
			mng.set_window_title('Inspect frames')
		except Exception:
			pass

		plt.tight_layout()
		plt.show()

	except Exception as ex:
		present_exception_and_exit()
