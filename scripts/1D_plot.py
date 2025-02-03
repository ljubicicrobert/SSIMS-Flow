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

	import matplotlib.pyplot as plt

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


__types__ = [
	'Coverage',
	'Mean magnitude',
	'Median magnitude',
	'Maximal magnitude',
	'Mean + Median + Max. magnitude',
]

__units__ = [
	'[%]',
	'[px/frame]',
	'[px/frame]',
	'[px/frame]',
	'[px/frame]',
]

__files__ = [
	'coverage.txt',
	'disp_mean.txt',
	'disp_median.txt',
	'disp_max.txt',
]


def try_load_file(fname):
	try:
		return np.loadtxt(fname)
	except Exception:
		return None


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--cfg', type=str, help='Path to project configuration file')
		parser.add_argument('--data', type=int, help='Which data to plot, see __types__ for more details')
		args = parser.parse_args()

		cfg = configparser.ConfigParser()
		cfg.optionxform = str

		try:
			cfg.read(args.cfg, encoding='utf-8-sig')
		except Exception:
			tag_print('error', 'There was a problem reading the configuration file!')
			tag_print('error', 'Check if project has valid configuration.')
			exit_message()

		project_folder = unix_path(cfg_get(cfg, 'Project settings', 'Folder', str))
		diagnostics_folder = f'{project_folder}/optical_flow/diagnostics'

		data_type = __types__[args.data]
		units = __units__[args.data]

		fig, ax = plt.subplots()
		ax.set_xlabel('Frame # [-]')
		ax.set_ylabel(f'{data_type} {units}')

		if args.data in range(4):
			diagnostics_data_path = f'{diagnostics_folder}/{__files__[args.data]}'
			diagnostics_data = try_load_file(diagnostics_data_path)

			ax.plot(range(diagnostics_data.size), diagnostics_data)

		elif args.data == 4:
			for i in range(1, 4):
				diagnostics_data_path = f'{diagnostics_folder}/{__files__[i]}'
				diagnostics_data = try_load_file(diagnostics_data_path)
				ax.plot(range(diagnostics_data.size), diagnostics_data)

			plt.legend([x for x in __types__[1:4]])

		try:
			mng = plt.get_current_fig_manager()
			mng.window.state('zoomed')
			mng.set_window_title('Inspect frames')
		except Exception:
			pass

		ax.set_title(data_type)
		plt.show()

	except Exception as ex:
		present_exception_and_exit()
