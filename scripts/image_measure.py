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
	from os import path
	from glob import glob
	from class_console_printer import tag_print, unix_path
	from utilities import exit_message, present_exception_and_exit, cfg_get
	from matplotlib.widgets import Slider

	import matplotlib.pyplot as plt

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


def xy2str(points_list: list, distance: float) -> str:
	"""
	Formats a display of distance between two points.
	"""

	s = ''
	i = 1
	for x, y in points_list:
		s += f'Point {i}: x={x}, y={y}\n'
		i += 1

	if distance > 0:
		s += f'Distance = {distance:.1f} px'
	if get_profile:
		s += '\nPress ENTER/RETURN to accept profile'

	return s


def update_riverbank_start(val):
	global riverbank_start
	global riverbank_start_ax

	riverbank_start[0] = val * dx + xs[0]
	riverbank_start[1] = val * dy + ys[0]

	riverbank_start_ax.set_offsets(riverbank_start)
	return


def update_riverbank_end(val):
	global riverbank_end
	global riverbank_end_ax

	riverbank_end[0] = val * dx + xs[0]
	riverbank_end[1] = val * dy + ys[0]

	sl_ax_riverbank_end.ax.texts[-1].set_text(100 - val)
	riverbank_end_ax.set_offsets(riverbank_end)
	return


def get_measurement(event):
	global d
	global dx
	global dy
	global xs
	global ys
	global ax
	global riverbank_start_ax
	global riverbank_end_ax
	global points
	global plt_image
	global riverbank_start
	global riverbank_end
	global riverbank_start_ax
	global riverbank_end_ax
	
	if event.button == 3:
		d = 0
		p = [event.xdata, event.ydata]
		points.append(np.round(p, 1))

		if len(points) == 1:
			try:
				ax.lines[-1].set_visible(False)
			except IndexError:
				pass

			ax.plot(points[0][0], points[0][1],
			        points[0][0], points[0][1], 'ro')

			d = 0

		elif len(points) == 2:
			ax.lines[-1].set_visible(False)
			xs = [row[0] for row in points]
			ys = [row[1] for row in points]
			ax.plot(xs, ys, 'ro-')

			d = ((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)**0.5

			dx = (xs[1] - xs[0]) / 100
			dy = (ys[1] - ys[0]) / 100

			if get_profile:
				riverbank_start = [xs[0], ys[0]]
				riverbank_end = [xs[1], ys[1]]

				riverbank_start_ax.set_offsets(riverbank_start)
				riverbank_end_ax.set_offsets(riverbank_end)

			distance_box.set_text(xy2str(points, d))
			distance_box.set_x((xs[0]+xs[1])/2)
			distance_box.set_y((ys[0]+ys[1])/2)

		elif len(points) > 2:
			points = []
			d = 0

			if get_profile:
				riverbank_start = [0, 0]
				riverbank_end = [0, 0]

				sl_ax_riverbank_start.set_val(0)
				sl_ax_riverbank_end.set_val(100)

				try:
					riverbank_start_ax.set_offsets([-1, -1])
					riverbank_end_ax.set_offsets([-1, -1])
				except IndexError:
					pass

			try:
				ax.lines[-1].set_visible(False)
				ax.lines[-2].set_visible(False)
				ax.lines[-3].set_visible(False)
			except IndexError:
				pass

			distance_box.set_text('')

	plt.draw()


def select_profile(event):
	global cfg

	if event.key == 'enter':
		if len(points) == 2:
			section = 'Optical flow'

			cfg[section]['ChainStart'] = f'{points[0][0]}, {points[0][1]}'
			cfg[section]['ChainEnd'] = f'{points[1][0]}, {points[1][1]}'

			if get_profile:
				cfg[section]['RiverbankStart'] = f'{min(sl_ax_riverbank_start.val, 100 - sl_ax_riverbank_end.val)}'
				cfg[section]['RiverbankEnd'] = f'{max(sl_ax_riverbank_start.val, 100 - sl_ax_riverbank_end.val)}'

			with open(args.cfg, 'w', encoding='utf-8-sig') as configfile:
				cfg.write(configfile)

			plt.close()

	return


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--cfg', type=str, help='Path to config file')
		parser.add_argument('--img_path', type=str, help='Path to image file or folder with images')
		parser.add_argument('--ext', type=str, help='Image extension', default='jpg')
		parser.add_argument('--profile', type=int, help='Whether to save profile', default=0)
		args = parser.parse_args()

		points = []
		d = 0

		cfg = configparser.ConfigParser()
		cfg.optionxform = str

		try:
			cfg.read(args.cfg, encoding='utf-8-sig')
		except Exception:
			tag_print('error', 'There was a problem reading the configuration file!')
			tag_print('error', 'Check if project has valid configuration.')
			exit_message()

		img_path = unix_path(args.img_path)
		ext = args.ext
		get_profile = args.profile

		section = 'Optical flow'

		x_start, y_start = cfg_get(cfg, section, 'ChainStart', str, default='0, 0').split(', ')
		x_end, y_end = cfg_get(cfg, section, 'ChainEnd', str, default='0, 0').split(', ')

		riverbank_start_val = cfg_get(cfg, section, 'RiverbankStart', int, default=0)
		riverbank_end_val = cfg_get(cfg, section, 'RiverbankEnd', int, default=100)

		x_start = float(x_start)
		y_start = float(y_start)
		x_end = float(x_end)
		y_end = float(y_end)

		initial_profile = x_start + x_end + y_start + y_end > 0

		try:
			path.exists(img_path)
		except Exception:
			raise ValueError('The path argument [--img_path] does not seem to correspond to an image or folder with images, or could be missing!')

		if path.isfile(img_path):
			img = cv2.imread(img_path)
		elif path.isdir(img_path):
			images = glob(f'{img_path}/*.{ext}')
			img = cv2.imread(images[0])
		else:
			raise ValueError('The path argument [--img_path] does not seem to correspond to an image or folder with images, or could be missing!')

		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		fig, ax = plt.subplots()
		plt_image = ax.imshow(img_rgb)
		fig.canvas.mpl_connect('button_press_event', get_measurement)

		if get_profile:
			fig.canvas.mpl_connect('key_press_event', select_profile)

		legend = 'Right click to select starting and end point.\n' \
		         'Right click again to start over.\n' \
		         'ENTER/RETURN = select profile' if get_profile else '' \
		         'O = zoom to window\n' \
				 'P = pan image'

		distance_box = plt.text(0, 0, '',
		                       horizontalalignment='right',
		                       verticalalignment='bottom',
		                       transform=ax.transData,
		                       bbox=dict(facecolor='white', alpha=0.5),
		                       fontsize=9,
		                       )

		plt.text(0.01, 0.98, legend,
				 horizontalalignment='left',
				 verticalalignment='top',
				 transform=ax.transAxes,
				 bbox=dict(facecolor='white', alpha=0.5),
				 fontsize=9,
				 )

		try:
			mng = plt.get_current_fig_manager()
			mng.window.state('zoomed')
			mng.set_window_title('Inspect frames')
		except Exception:
			pass

		if get_profile:
			if initial_profile:
				points = [np.array([x_start, y_start]), np.array([x_end, y_end])]

				try:
					ax.lines[-1].set_visible(False)
				except:
					pass

				xs = [row[0] for row in points]
				ys = [row[1] for row in points]
				
				ax.plot(xs, ys, 'ro-')
				d = ((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)**0.5

				distance_box.set_text(xy2str(points, d))
				distance_box.set_x((xs[0]+xs[1])/2)
				distance_box.set_y((ys[0]+ys[1])/2)

			axcolor = 'lightgoldenrodyellow'
			valfmt = "%d"

			ax_riverbank_start_slider = plt.axes([0.2, 0.05, 0.63, 0.03], facecolor=axcolor)
			ax_riverbank_end_slider = plt.axes([0.2, 0.025, 0.63, 0.03], facecolor=axcolor)

			sl_ax_riverbank_start = Slider(ax_riverbank_start_slider, 'Start', 0, 100, valinit=riverbank_start_val, valstep=1, valfmt=valfmt)
			sl_ax_riverbank_end = Slider(ax_riverbank_end_slider, 'End', 0, 100, valinit=100 - riverbank_end_val, valstep=1, valfmt=valfmt)
			
			sl_ax_riverbank_start.on_changed(update_riverbank_start)
			sl_ax_riverbank_end.on_changed(update_riverbank_end)

			if initial_profile:
				dx = (xs[1] - xs[0]) / 100
				dy = (ys[1] - ys[0]) / 100

				riverbank_start = [xs[0], ys[0]]
				riverbank_end = [xs[1], ys[1]]

				riverbank_start_ax = ax.scatter(xs[0], ys[0], s=200, marker='x', c='b')
				riverbank_end_ax = ax.scatter(xs[1], ys[1], s=200, marker='x', c='b')

				sl_ax_riverbank_start.set_val(riverbank_start_val)
				sl_ax_riverbank_end.set_val(100 - riverbank_end_val)

		plt.show()

	except Exception as ex:
		present_exception_and_exit('Import failed! See traceback below:')
