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
	from sys import exit
	from class_console_printer import tag_print, unix_path
	from scipy.ndimage import map_coordinates
	from scipy.signal import medfilt
	from math import atan2
	from utilities import cfg_get

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


def values_along_line(img: np.ndarray, xs: list, ys: list, count: int, order=3):
	length = ((xs[0] - xs[1])**2 + (ys[0] - ys[1])**2)**0.5
	dl = length / count
	coordinates = np.arange(0, length, dl)
	x_range, y_range = np.linspace(xs[0], xs[1], count), np.linspace(ys[0], ys[1], count)

	return coordinates, map_coordinates(img, np.vstack((y_range, x_range)), order=order)


def main(cfg_path=None):
	try:
		try:
			if cfg_path is None:
				parser = ArgumentParser()
				parser.add_argument('--cfg', type=str, help='Path to project configuration file')
				args = parser.parse_args()

				cfg = configparser.ConfigParser()
				cfg.optionxform = str
				cfg.read(args.cfg, encoding='utf-8-sig')
			else:
				cfg.read(cfg_path, encoding='utf-8-sig')
		except Exception:
			tag_print('error', 'There was a problem reading the configuration file!')
			tag_print('error', 'Check if project has valid configuration.')
			print('\n{}'.format(format_exc()))
			input('\nPress ENTER/RETURN key to exit...')
			exit()

		project_folder = cfg['Project settings']['Folder']
		tag_print('start', 'Getting profile data for project in [{}]'.format(project_folder))

		section = 'Optical flow'

		input_folder = unix_path(project_folder) + '/optical_flow'

		sources = [
			'{}/mag_mean.txt'.format(input_folder),
			'{}/diagnostics/T0.txt'.format(input_folder),
			'{}/diagnostics/T1.txt'.format(input_folder),
			'{}/diagnostics/T2.txt'.format(input_folder),
			'{}/mag_max.txt'.format(input_folder),
	    ]

		source_index = int(cfg[section]['ProfileSource'])
		field_raw_mag = np.loadtxt(sources[source_index])
		field_raw_angle = np.loadtxt('{}/angle_mean.txt'.format(input_folder))

		frames_step = cfg_get(cfg, 'Frames', 'Step', float, 1.0)
		optical_flow_step = cfg_get(cfg, section, 'Step', float)
		scale = cfg_get(cfg, section, 'Scale', float)
		fps = cfg_get(cfg, section, 'Framerate', float)		# frames/sec
		gsd = cfg_get(cfg, section, 'GSD', float)           # px/m
		pooling = cfg_get(cfg, section, 'Pooling', float)  	# px
		gsd_pooled = gsd / pooling  				# blocks/m, 1/m

		v_ratio = fps / gsd / (frames_step * optical_flow_step) / scale         	# (frame*m) / (s*px)
		field_raw_mag *= v_ratio					# px/frame * ((frame*m) / (s*px)) = m/s

		field_median_mag = medfilt(field_raw_mag, [3, 3])
		field_median_angle = medfilt(field_raw_angle, [3, 3])

		field_raw_us, field_raw_vs = cv2.polarToCart(field_raw_mag, field_raw_angle, angleInDegrees=True)
		field_median_us, field_median_vs = cv2.polarToCart(field_median_mag, field_median_angle, angleInDegrees=True)

		x_start, y_start = [float(x)/pooling*scale for x in cfg[section]['ChainStart'].replace(' ', '').split(',')[:2]]
		x_end, y_end = [float(x)/pooling*scale for x in cfg[section]['ChainEnd'].replace(' ', '').split(',')[:2]]
		count = int(cfg[section]['ChainCount'])
		order = 3

		dx = x_end - x_start
		dy = y_end - y_start

		chainage_direction = atan2(dy, dx) * 180 / np.pi
		if chainage_direction < 0:
			chainage_direction += 360
		
		length = (dx**2 + dy**2)**0.5 / scale
		dl = length / (count-1)
		chainage = np.arange(0, length+0.001, dl)
		chainage_pooled = [c / gsd_pooled for c in chainage]

		x_range = np.linspace(x_start, x_end, count)
		y_range = np.linspace(y_start, y_end, count)

		field_angle_diff = field_raw_angle - chainage_direction
		field_angle_corr = np.abs(np.sin(field_angle_diff / 180 * np.pi))

		line_raw_mag = np.abs(map_coordinates(field_raw_mag, np.vstack((y_range, x_range)), order=order))
		line_raw_us = map_coordinates(field_raw_us, np.vstack((y_range, x_range)), order=order)
		line_raw_vs = map_coordinates(field_raw_vs, np.vstack((y_range, x_range)), order=order)
		line_raw_angle = cv2.cartToPolar(line_raw_us, line_raw_vs, angleInDegrees=True)[1].ravel()

		line_median_mag = np.abs(map_coordinates(field_median_mag, np.vstack((y_range, x_range)), order=order))
		line_median_us = map_coordinates(field_median_us, np.vstack((y_range, x_range)), order=order)
		line_median_vs = map_coordinates(field_median_vs, np.vstack((y_range, x_range)), order=order)
		line_median_angle = cv2.cartToPolar(line_median_us, line_median_vs, angleInDegrees=True)[1].ravel()

		line_corr = map_coordinates(field_angle_corr, np.vstack((y_range, x_range)), order=1)

		line_raw_mag_corr = line_corr * line_raw_mag
		line_median_mag_corr = line_corr * line_median_mag

		fmt_vel = '%.4f'
		fmt_angle = '%.1f'

		table_data = np.vstack([
			chainage_pooled,
			line_corr,

			line_raw_mag,
			line_raw_mag_corr,
			line_raw_angle,
			line_raw_us,
			line_raw_vs,

			line_median_mag,
			line_median_mag_corr,
			line_median_angle,
			line_median_us,
			line_median_vs,
			]).T

		table_data_header = [
			'Chainage [m]',
			'Flow direction corr. factor [-]',

			'Magnitude total (raw) [m/s]',
			'Magnitude normal (raw) [m/s]',
			'Flow direction (raw) [deg]',
			'U comp. (raw) [m/s]',
			'V comp. (raw) [m/s]',

			'Magnitude total (filt.) [m/s]',
			'Magnitude normal (filt.) [m/s]',
			'Flow direction (filt.) [deg]',
			'U comp. (filt.) [m/s]',
			'V comp. (filt.) [m/s]',
		]

		table_data_fmt = [
			fmt_vel,
			fmt_vel,

			fmt_vel,
			fmt_vel,
			fmt_angle,
			fmt_vel,
			fmt_vel,

			fmt_vel,
			fmt_vel,
			fmt_angle,
			fmt_vel,
			fmt_vel,
		]

		table_data_header_str = str(',').join(table_data_header)
		table_data_fmt_str = str(',').join(table_data_fmt)

		np.savetxt('{}/profile_data.txt'.format(input_folder), table_data, fmt=table_data_fmt_str, header=table_data_header_str, delimiter=',', comments='')

		tag_print('end', 'Chainage data saved to [{}/profile_data.txt]'.format(input_folder))
		input('\nPress ENTER/RETURN key to exit...')

	except Exception as ex:
		print()
		tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
		print('\n{}'.format(format_exc()))
		input('\nPress ENTER/RETURN key to exit...')


if __name__ == '__main__':
	main()
