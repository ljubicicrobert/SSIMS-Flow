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
	from matplotlib.widgets import Slider
	import matplotlib.backend_bases as backend_bases
	from glob import glob

	import matplotlib.pyplot as plt
	import matplotlib.patches as patches

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


def xy2str(points_list: list) -> str:
	"""
	Formats a display of distance between two points.
	"""

	s = ''
	i = 1
	for x, y in points_list:
		s += f'Point {i}: x={x}, y={y}\n'
		i += 1

	s += f'Resulting W = {int(abs(points_list[0][0] - points_list[1][0]))} px\n'
	s += f'Resulting H = {int(abs(points_list[0][1] - points_list[1][1]))} px'
	s += '\nPress ENTER/RETURN to accept profile'

	return s


def select_frame(val):
	global img_gray
	global img_thr
	global img_rgb
	global img_shown

	img = cv2.imread(img_list[val])
	
	h, w = img.shape[:2]

	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_thr = cv2.threshold(img_gray, int(current_threshold*255), 255, cv2.THRESH_BINARY)[1]
	img_thr = cv2.merge([img_thr, img_thr, img_thr])

	img_shown = img_rgb.copy()

	if len(points) > 0:
		img_shown[y_start: y_end, x_start: x_end, :] = img_thr[y_start: y_end, x_start: x_end, :]

	plt_image.set_data(img_shown)
	plt.draw()

	return


def change_threshold(val):
	global img_shown
	global img_thr
	global current_threshold

	current_threshold = val

	if len(points) > 0:
		img_thr = cv2.threshold(img_gray, int(current_threshold*255), 255, cv2.THRESH_BINARY)[1]
		img_thr = cv2.merge([img_thr, img_thr, img_thr])
		img_shown[y_start: y_end, x_start: x_end, :] = img_thr[y_start: y_end, x_start: x_end, :]
	
	plt_image.set_data(img_shown)
	plt.draw()

	return


def highlight_roi(event):
	global d
	global x_start
	global y_start
	global x_end
	global y_end
	global ax
	global points
	global img_shown
	global plt_image

	if event.button == 3:
		p = [event.xdata, event.ydata]
		points.append(np.round(p, 0).astype(int))

		print(points)

		if len(points) == 1:
			try:
				ax.lines[-1].set_visible(False)
			except IndexError:
				pass

			ax.plot(points[0][0], points[0][1],
					points[0][0], points[0][1], 'ro')

		elif len(points) == 2:
			xs = [row[0] for row in points]
			ys = [row[1] for row in points]

			x_start = min(xs)
			y_start = min(ys)
			x_end = max(xs)
			y_end = max(ys)

			img_shown = img_rgb.copy()
			img_shown[y_start: y_end, x_start: x_end, :] = img_thr[y_start: y_end, x_start: x_end, :]
			plt_image.set_data(img_shown)

			ax.plot(points[1][0], points[1][1],
					points[1][0], points[1][1], 'ro')
			roi = patches.Rectangle((xs[0], ys[0]-1), xs[1] - xs[0], ys[1] - ys[0], linewidth=2, edgecolor='r', facecolor='none')
			ax.add_patch(roi)

			roi_box.set_text(xy2str(points))

		elif len(points) > 2:
			points = []
			img_shown = img_rgb.copy()
			plt_image.set_data(img_shown)

			try:
				ax.patches[-1].set_visible(False)
				ax.lines[-1].set_visible(False)
				ax.lines[-2].set_visible(False)
				ax.lines[-3].set_visible(False)
			except IndexError:
				pass

			roi_box.set_text('')

	plt.draw()


def select_roi(event):
	global cfg

	if event.key == 'enter':
		if len(points) == 2:
			section = 'SDI'

			try:
				cfg[section]['XStart'] = f'{points[0][0]}'
			except KeyError:
				cfg.add_section(section)
				cfg[section]['XStart'] = f'{points[0][0]}'

			cfg[section]['YStart'] = f'{points[0][1]}'
			cfg[section]['XEnd'] = f'{points[1][0]}'
			cfg[section]['YEnd'] = f'{points[1][1]}'
			cfg[section]['BinarizationThreshold'] = f'{current_threshold:.2f}'

			with open(args.cfg, 'w', encoding='utf-8-sig') as configfile:
				cfg.write(configfile)

			plt.close()

	return


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--cfg', type=str, help='Path to config file')
		args = parser.parse_args()

		points = []

		cfg = configparser.ConfigParser()
		cfg.optionxform = str

		try:
			cfg.read(args.cfg, encoding='utf-8-sig')
		except Exception:
			tag_print('error', 'There was a problem reading the configuration file!')
			tag_print('error', 'Check if project has valid configuration.')
			exit_message()

		section = 'SDI'
			
		project_path = unix_path(cfg_get(cfg, 'Project settings', 'Folder', str))
		frames_path = unix_path(cfg_get(cfg, section, 'Folder', str, default=f'{project_path}/frames'))
		ext = cfg_get(cfg, section, 'Extension', str, default='jpg')

		points = []
		x_start = cfg_get(cfg, section, 'XStart', int, default=0)
		y_start = cfg_get(cfg, section, 'YStart', int, default=0)
		x_end = cfg_get(cfg, section, 'XEnd', int, default=0)
		y_end = cfg_get(cfg, section, 'YEnd', int, default=0)

		if x_start > x_end:
			x_start, x_end = x_end, x_start
		if y_start > y_end:
			y_start, y_end = y_end, y_start

		initial_roi = x_start - x_end < 0 and y_start - y_end < 0

		img_list = glob(f'{frames_path}/*.{ext}')
		num_frames = len(img_list)

		initial_threshold = cfg_get(cfg, 'SDI', 'BinarizationThreshold', float, default=0.8)
		current_threshold = initial_threshold

		img = cv2.imread(img_list[0])
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_thr = cv2.threshold(img_gray, int(initial_threshold*255), 255, cv2.THRESH_BINARY)[1]
		img_thr = cv2.merge([img_thr, img_thr, img_thr])

		fig, ax = plt.subplots()
	
		plt_image = ax.imshow(img_rgb)
		fig.canvas.mpl_connect('button_press_event', highlight_roi)
		fig.canvas.mpl_connect('key_press_event', select_roi)

		axcolor = 'lightgoldenrodyellow'

		ax_frame_num = plt.axes([0.2, 0.02, 0.63, 0.03], facecolor=axcolor)
		ax_threshold = plt.axes([0.2, 0.050, 0.63, 0.03], facecolor=axcolor)

		sl_ax_frame_num = Slider(ax_frame_num, 'Frame #', 0, num_frames-1, valinit=0, valstep=1, valfmt="%d")
		sl_ax_threshold = Slider(ax_threshold, 'Threshold', 0, 1.00, valinit=initial_threshold, valstep=0.01, valfmt="%.2f")
		
		sl_ax_frame_num.on_changed(select_frame)
		sl_ax_threshold.on_changed(change_threshold)

		legend = 'Right click to select starting and end point.\n' \
				 'Right click again to start over.\n' \
				 'ENTER/RETURN = select ROI\n' \
				 'O = zoom to window\n' \
				 'P = pan image'

		roi_box = plt.text(0.01, 0.02, '',
						   horizontalalignment='left',
						   verticalalignment='bottom',
						   transform=ax.transAxes,
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
			mng.set_window_title('Select ROI for cropping')
		except Exception:
			pass

		if initial_roi:
			points = [np.array([x_start, y_start]), np.array([x_end, y_end])]

			xs = [row[0] for row in points]
			ys = [row[1] for row in points]

			img_shown = img_rgb.copy()
			img_shown[y_start: y_end, x_start: x_end, :] = img_thr[y_start: y_end, x_start: x_end, :]
			plt_image.set_data(img_shown)

			ax.plot(points[1][0], points[1][1],
					points[1][0], points[1][1], 'ro')
			roi = patches.Rectangle((xs[0], ys[0]-1), xs[1] - xs[0], ys[1] - ys[0], linewidth=2, edgecolor='r', facecolor='none')
			ax.add_patch(roi)

			roi_box.set_text(xy2str(points))

		plt.show()

	except Exception as ex:
		present_exception_and_exit()
