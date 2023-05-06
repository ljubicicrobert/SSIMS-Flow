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
	from os import path
	from glob import glob
	from class_console_printer import tag_print, unix_path
	from utilities import cfg_get

	import matplotlib.pyplot as plt
	import matplotlib.patches as patches

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


def xy2str(points_list: list) -> str:
	"""
	Formats a display of distance between two points.
	"""

	s = ''
	i = 1
	for x, y in points_list:
		s += 'Point {}: x={}, y={}\n'.format(i, x, y)
		i += 1

	s += 'Resulting W = {} px\n'.format(int(abs(points_list[0][0] - points_list[1][0])))
	s += 'Resulting H = {} px'.format(int(abs(points_list[0][1] - points_list[1][1])))
	s += '\nPress ENTER/RETURN to accept profile'

	return s


def get_measurement(event):
	global d
	global ax
	global points
	global plt_image

	if event.button == 3:
		p = [event.xdata, event.ydata]
		points.append(np.round(p, 0).astype(int))

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

			img_shown = img_gray3
			img_shown[min(ys): max(ys), min(xs): max(xs), :] = img_rgb[min(ys): max(ys), min(xs): max(xs), :]
			plt_image.set_data(img_shown)

			ax.plot(points[1][0], points[1][1],
					points[1][0], points[1][1], 'ro')
			roi = patches.Rectangle((xs[0], ys[0]), xs[1] - xs[0], ys[1] - ys[0], linewidth=2, edgecolor='r', facecolor='none')
			ax.add_patch(roi)

			roi_box.set_text(xy2str(points))

		elif len(points) > 2:
			points = []
			plt_image.set_data(img_rgb)

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
			section = 'Frames'

			cfg[section]['Crop'] = '{}, {}, {}, {}'.format(int(points[0][0]), int(points[1][0]), int(points[0][1]), int(points[1][1]))

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
			input('\nPress ENTER/RETURN key to exit...')
			exit()

		section = 'Frames'
			
		video_path = unix_path(cfg[section]['VideoPath'])
		unpack_start = cfg_get(cfg, section, 'Start', int, 0)

		vidcap = cv2.VideoCapture(video_path)
		vidcap.set(cv2.CAP_PROP_POS_FRAMES, unpack_start)
		success, img = vidcap.read()
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_gray3 = cv2.merge([img_gray, img_gray, img_gray])
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		fig, ax = plt.subplots()
		plt_image = ax.imshow(img_rgb)
		fig.canvas.mpl_connect('button_press_event', get_measurement)
		fig.canvas.mpl_connect('key_press_event', select_roi)

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

		plt.show()

	except Exception as ex:
		print()
		tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
		print('\n{}'.format(format_exc()))
		input('\nPress ENTER/RETURN key to exit...')
