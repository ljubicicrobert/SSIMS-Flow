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
	from scipy.ndimage import map_coordinates
	from scipy.signal import medfilt
	from math import atan2
	from utilities import cfg_get, exit_message, present_exception_and_exit
	from glob import glob
	from optical_flow import nan_locate
	from profile_fit import main as profile_fit_main

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


fmt_vel = '%.4f'
fmt_angle = '%.1f'

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
	'Magnitude normal (fit) [m/s]',
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
	fmt_vel,
]

table_data_header_str = str(',').join(table_data_header)
table_data_fmt_str = str(',').join(table_data_fmt)


def main(cfg_path=None, quiet=0):
	try:
		try:
			parser = ArgumentParser()
			parser.add_argument('--cfg', type=str, help='Path to project configuration file')
			parser.add_argument('--quiet', type=int, help='Quiet mode for batch processing, no RETURN confirmation on success', default=0)
			args = parser.parse_args()

			cfg = configparser.ConfigParser()
			cfg.optionxform = str

			if cfg_path is None:
				cfg.read(args.cfg, encoding='utf-8-sig')
			else:
				cfg.read(cfg_path, encoding='utf-8-sig')

		except Exception:
			tag_print('error', 'There was a problem reading the configuration file!')
			tag_print('error', 'Check if project has valid configuration.')
			print()
			print(format_exc())
			exit_message()

		is_quiet = 1 if args.quiet == 1 else quiet

		project_folder = unix_path(cfg_get(cfg, 'Project settings', 'Folder', str))
		input_folder = unix_path(project_folder + '/optical_flow')
		tag_print('start', f'Getting profile data for project in [{project_folder}]')

		section = 'Optical flow'

		frames_folder = unix_path(cfg_get(cfg, section, 'Folder', str))
		frames_ext = cfg_get(cfg, section, 'Extension', str, 'jpg')
		frames_list = glob(f'{frames_folder}/*.{frames_ext}')

		img = cv2.imread(frames_list[0], 0)
		h, w = img.shape

		sources = [
			f'{input_folder}/mag_mean.txt',
			f'{input_folder}/diagnostics/T1.txt',
			f'{input_folder}/diagnostics/T2.txt',
			f'{input_folder}/diagnostics/T3.txt',
			f'{input_folder}/mag_max.txt',
	    ]

		version_created = int(cfg_get(cfg, 'Project settings', 'VersionCreated', str, '0').replace('v', '').replace('.', ''))
		if 0 < version_created < 500:
			sources[1] = f'{input_folder}/diagnostics/T0.txt'
			sources[2] = f'{input_folder}/diagnostics/T1.txt'
			sources[3] = f'{input_folder}/diagnostics/T2.txt'

		source_index = cfg_get(cfg, section, 'ProfileSource', int)
		field_raw_mag = np.loadtxt(sources[source_index])
		field_raw_angle = np.loadtxt(f'{input_folder}/angle_mean.txt')

		angle_main = cfg_get(cfg, section, 'AngleMain', float)
		nans, x = nan_locate(field_raw_angle)
		try:
			if 315 <= angle_main <= 360 or \
				 0 <= angle_main <= 45 or \
			   135 <= angle_main <= 225:
				field_raw_angle[nans] = np.interp(x(nans), x(~nans), field_raw_angle[~nans], period=360)
			else:
				field_raw_angle[nans] = np.interp(x(nans.T), x(~nans.T), field_raw_angle[~nans].T, period=360).T
		except ValueError:
			pass

		nans, x = nan_locate(field_raw_mag)
		field_raw_mag[nans] = np.interp(x(nans), x(~nans), field_raw_mag[~nans])

		frames_step = cfg_get(cfg, 'Frames', 'Step', float, 1.0)
		optical_flow_step = cfg_get(cfg, section, 'Step', float)
		scale = cfg_get(cfg, section, 'Scale', float)
		fps = cfg_get(cfg, section, 'Framerate', float)		# frames/sec
		gsd = cfg_get(cfg, section, 'GSD', float)           # px/m
		gsd_units = cfg_get(cfg, section, 'GSDUnits', str, 'px/m')           # px/m
		magnitudes_fit_from = cfg_get(cfg, section, 'FitFrom', int, default=1)

		if gsd_units != 'px/m':
			gsd = 1/gsd
		pooling = cfg_get(cfg, section, 'Pooling', float)
		gsd_pooled = gsd / pooling  				# blocks/m, 1/m

		padd_x = w % pooling // 2
		padd_y = h % pooling // 2

		v_ratio = fps / gsd / (frames_step * optical_flow_step) / scale         	# (frame*m) / (s*px)
		field_raw_mag *= v_ratio					# px/frame * ((frame*m) / (s*px)) = m/s

		field_median_mag = medfilt(field_raw_mag, [3, 3])
		field_median_angle = medfilt(field_raw_angle, [3, 3])

		field_raw_us, field_raw_vs = cv2.polarToCart(field_raw_mag, field_raw_angle, angleInDegrees=True)
		field_median_us, field_median_vs = cv2.polarToCart(field_median_mag, field_median_angle, angleInDegrees=True)

		chain_start = cfg_get(cfg, section, 'ChainStart', str)
		chain_end = cfg_get(cfg, section, 'ChainEnd', str)

		x_start, y_start = [float(x) for x in chain_start.replace(' ', '').split(',')[:2]]
		x_end, y_end = [float(x) for x in chain_end.replace(' ', '').split(',')[:2]]

		x_start = (x_start - padd_x) / pooling * scale
		x_end = (x_end - padd_x) / pooling * scale
		y_start = (y_start - padd_y) / pooling * scale
		y_end = (y_end - padd_y) / pooling * scale

		count = cfg_get(cfg, section, 'ChainCount', int)
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

		line_fitted_mag, beta, Umax, Bratio = profile_fit_main(cfg, chainage_pooled, line_median_mag if magnitudes_fit_from == 1 else line_raw_mag)

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
			line_fitted_mag,
			]).T

		np.savetxt(f'{input_folder}/profile_data.txt', table_data, fmt=table_data_fmt_str, header=table_data_header_str, delimiter=',', comments='')

		cfg[section]['Beta'] = f'{beta:.3f}'
		cfg[section]['Umax'] = f'{Umax:.3f}'
		cfg[section]['Bratio'] = f'{Bratio:.3f}'

		with open(args.cfg, 'w', encoding='utf-8-sig') as configfile:
			cfg.write(configfile)

		tag_print('end', f'Chainage data saved to [{input_folder}/profile_data.txt]')

		if is_quiet == 0:
			exit_message()

	except Exception as ex:
		present_exception_and_exit()


if __name__ == '__main__':
	main()
