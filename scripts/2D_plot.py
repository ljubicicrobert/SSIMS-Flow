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
	from matplotlib.widgets import Slider
	from sys import exit
	from glob import glob
	from class_console_printer import tag_print, unix_path
	from vel_ratio import L1
	from utilities import cfg_get, exit_message, present_exception_and_exit

	import matplotlib.pyplot as plt

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


__types__ = [
	'U(x,y)',
	'V(x,y)',
	'Magnitudes',
	'Flow direction',
	'Threshold ratio',
]

__units__ = [
	'[m/s]',
	'[m/s]',
	'[m/s]',
	'[deg]',
	'[-]',
]


def update_vmax(val):
	img_shown.set_clim(vmax=sl_ax_vmax.val)
	plt.draw()


def update_slider(val):
	mags = try_load_file(mag_list[val]) * v_ratio
	dirs = try_load_file(dir_list[val])
	us, vs = cv2.polarToCart(mags, dirs, angleInDegrees=True)

	data_new = [us, vs, mags, dirs]
	img_new = data_new[plot_type]

	if frames_available:
		back_new = cv2.imread(frames_list[val], cv2.COLOR_BGR2RGB)[::-1]
		back_shown.set_data(back_new)

	img_shown.set_data(img_new)
	img_shown.set_clim(vmin=np.nanmin(img_new[cbar_cutoff_h: -cbar_cutoff_h, cbar_cutoff_w: -cbar_cutoff_w]),
					   vmax=np.nanmax(img_new[cbar_cutoff_h: -cbar_cutoff_h, cbar_cutoff_w: -cbar_cutoff_w]))
	ax.set_title(f'{data_type}, frame #{sl_ax_vmax.val}/{num_frames - 1}')
	plt.draw()

	return


def keypress(event):
	if event.key == 'escape':
		exit()

	elif event.key == 'down':
		if sl_ax_vmax.val == 0:
			sl_ax_vmax.set_val(num_frames - 1)
		else:
			sl_ax_vmax.set_val(sl_ax_vmax.val - 1)

	elif event.key == 'up':
		if sl_ax_vmax.val == num_frames - 1:
			sl_ax_vmax.set_val(0)
		else:
			sl_ax_vmax.set_val(sl_ax_vmax.val + 1)

	elif event.key == 'pageup':
		if sl_ax_vmax.val >= num_frames - 10:
			sl_ax_vmax.set_val(0)
		else:
			sl_ax_vmax.set_val(sl_ax_vmax.val + 10)

	elif event.key == 'pagedown':
		if sl_ax_vmax.val <= 9:
			sl_ax_vmax.set_val(num_frames - 1)
		else:
			sl_ax_vmax.set_val(sl_ax_vmax.val - 10)

	update_slider(sl_ax_vmax.val)


def try_load_file(fname):
	try:
		return np.loadtxt(fname)
	except Exception:
		return None


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--cfg', type=str, help='Path to project configuration file')
		parser.add_argument('--mode', type=int, help='0 = time averaged, 1 = maximal, 2 = instantaneous', default=0)
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

		plot_mode = args.mode
		plot_type = args.data
		data_type = __types__[plot_type]
		units = __units__[plot_type]
		project_folder = unix_path(cfg_get(cfg, 'Project settings', 'Folder', str))

		section = 'Optical flow'

		frames_step = cfg_get(cfg, 'Frames', 'Step', float)
		optical_flow_step = cfg_get(cfg, section, 'Step', float)
		scale = cfg_get(cfg, section, 'Scale', float)
		fps = cfg_get(cfg, section, 'Framerate', float)		# frames/sec

		try:
			gsd = cfg_get(cfg, section, 'GSD', float)       # px/m
		except Exception as ex:
			gsd = cfg_get(cfg, 'Transformation', 'GSD', float)        # px/m

		gsd_units = cfg_get(cfg, section, 'GSDUnits', str, 'px/m')           # px/m
		
		if gsd_units != 'px/m':
			gsd = 1/gsd
			
		pooling = cfg_get(cfg, section, 'Pooling', float)   	# px
		gsd_pooled = gsd / pooling  				# blocks/m, 1/m

		v_ratio = fps / gsd / (frames_step * optical_flow_step) / scale         	# (frame*m) / (s*px)

		average_only = cfg_get(cfg, section, 'AverageOnly', int)    # px
		frames_folder = cfg_get(cfg, section, 'Folder', str)
		frames_ext = cfg_get(cfg, 'section', 'Extension', str, 'jpg')
		frames_list = glob(f'{frames_folder}/*.{frames_ext}')
		frames_available = len(frames_list) > 0

		alpha = 1.0 if not frames_available else 0.5

		if average_only == 0:
			mag_list = glob(f'{project_folder}/optical_flow/magnitudes/*.txt')
			dir_list = glob(f'{project_folder}/optical_flow/directions/*.txt')
			num_frames = len(mag_list)

			if num_frames == 0:
				print()
				tag_print('error', f'No optical flow data found in [{project_folder}/optical_flow/]')
				exit_message()

		fig, ax = plt.subplots()
		plt.subplots_adjust(bottom=0.13)
		plt.axis('off')

		legend = 'Use slider to select frame,\n' \
				 'use UP and DOWN keys to move by +/- 1 frame\n' \
				 'or PageUP and PageDOWN keys to move by +/- 10 frames\n' \
				 'Press ESC or Q to exit'

		legend_toggle = plt.text(0.02, 0.97, legend,
								 horizontalalignment='left',
								 verticalalignment='top',
								 transform=ax.transAxes,
								 bbox=dict(facecolor='white', alpha=0.5),
								 fontsize=9,
								 )

		if plot_mode == 0:       # Time averaged
			legend_toggle.set_visible(False)

			mags = try_load_file(f'{project_folder}/optical_flow/mag_mean.txt') * v_ratio	# px/frame
			dirs = try_load_file(f'{project_folder}/optical_flow/angle_mean.txt')
			thrs = try_load_file(f'{project_folder}/optical_flow/threshold_ratios.txt')
			h, w = mags.shape

			us, vs = cv2.polarToCart(mags, dirs, angleInDegrees=True)
			data_list = [us, vs, mags, dirs, thrs]
			data = data_list[plot_type]

		elif plot_mode == 1:     # Maximal
			legend_toggle.set_visible(False)

			mags = try_load_file(f'{project_folder}/optical_flow/mag_max.txt') * v_ratio	# px/frame
			dirs = try_load_file(f'{project_folder}/optical_flow/angle_mean.txt')
			h, w = mags.shape

			us, vs = cv2.polarToCart(mags, dirs, angleInDegrees=True)
			data_list = [us, vs, mags]
			data = data_list[plot_type]

		elif plot_mode == 2:     # Instantaneous      
			mags = try_load_file(mag_list[0]) * v_ratio
			dirs = try_load_file(dir_list[0])
			h, w = mags.shape

			us, vs = cv2.polarToCart(mags, dirs, angleInDegrees=True)
			data_list = [us, vs, mags, dirs]
			data = data_list[plot_type]
			
			axcolor = 'lightgoldenrodyellow'
			valfmt = "%d"

			fig.canvas.mpl_connect('key_press_event', keypress)
			ax_vmax = plt.axes([0.2, 0.05, 0.63, 0.03], facecolor=axcolor)
			sl_ax_vmax = Slider(ax_vmax, 'Frame #', 0, num_frames-1, valinit=0, valstep=1, valfmt=valfmt)
			sl_ax_vmax.on_changed(update_slider)

		cbar_cutoff_h = h//5
		cbar_cutoff_w = w//5


		if plot_mode in [0, 1]:
			axcolor = 'lightgoldenrodyellow'
			valfmt = "%.3f"

			ax_vmax = plt.axes([0.2, 0.05, 0.63, 0.03], facecolor=axcolor)

			real_max = np.nanmax(data)
			cut_max = np.nanmax(data[cbar_cutoff_h: -cbar_cutoff_h, cbar_cutoff_w: -cbar_cutoff_w])

			sl_ax_vmax = Slider(ax_vmax, 'Max. value [m/s]', np.nanmin(data), np.nanmax(data), valinit=cut_max, valstep=real_max/100, valfmt=valfmt)
			sl_ax_vmax.on_changed(update_vmax)

		if frames_available:
			back = cv2.imread(frames_list[0], cv2.COLOR_BGR2RGB)[::-1]
			padd_x = (back.shape[1] % pooling) / 2 / pooling / scale
			padd_y = (back.shape[0] % pooling) / 2 / pooling / scale
			
			extent = (-padd_x, w + padd_x, -padd_y, h + padd_y)
			back_shown = ax.imshow(back, extent=extent)

		img_shown = ax.imshow(data, cmap='jet', interpolation='hanning', alpha=alpha)

		if plot_type == 4:
			img_shown.set_clim(vmin=0, vmax=L1)
		else:
			img_shown.set_clim(vmin=np.nanmin(data[cbar_cutoff_h: -cbar_cutoff_h, cbar_cutoff_w: -cbar_cutoff_w]),
							   vmax=np.nanmax(data[cbar_cutoff_h: -cbar_cutoff_h, cbar_cutoff_w: -cbar_cutoff_w]))
		
		cbar = plt.colorbar(img_shown, ax=ax)
		cbar.set_label(f'{data_type} {units}')

		try:
			mng = plt.get_current_fig_manager()
			mng.window.state('zoomed')
			mng.set_window_title('Inspect frames')
		except Exception:
			pass

		ax.set_title(f'{data_type}, frame #0/{num_frames - 1}'
					 if plot_mode == 2
					 else f'Time averaged values: {data_type}')
		
		plt.show()

	except Exception as ex:
		present_exception_and_exit()
