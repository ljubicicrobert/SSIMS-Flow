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

	import matplotlib.pyplot as plt

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


def xy2str(points_list: list, distance: float) -> str:
	"""
	Formats a display of distance between two points.
	"""

	s = ''
	i = 1
	for x, y in points_list:
		s += 'Point {}: x={}, y={}\n'.format(i, x, y)
		i += 1

	if distance > 0:
		s += 'Distance = {:.1f} px'.format(distance)
	if get_profile:
		s += '\nPress ENTER/RETURN to accept profile'

	return s


def get_measurement(event):
	global d
	global ax
	global points
	global plt_image

	if event.button == 3:
		d = 0
		p = [event.xdata, event.ydata]
		points.append(np.round(p, 1))

		if len(points) == 1:
			try:
				ax.lines.pop(0)
			except IndexError:
				pass

			ax.plot(points[0][0], points[0][1],
			        points[0][0], points[0][1], 'ro')

			d = 0

		elif len(points) == 2:
			ax.lines.pop(0)
			xs = [row[0] for row in points]
			ys = [row[1] for row in points]
			ax.plot(xs, ys, 'ro-')

			d = ((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)**0.5

			distance_box.set_text(xy2str(points, d))
			distance_box.set_x((xs[0]+xs[1])/2)
			distance_box.set_y((ys[0]+ys[1])/2)

		elif len(points) > 2:
			points = []
			d = 0

			try:
				ax.lines.pop(0)
				ax.lines.pop(0)
			except IndexError:
				pass

			distance_box.set_text('')

	plt.draw()


def select_profile(event):
	global cfg

	if event.key == 'enter':
		if len(points) == 2:
			section = 'Optical flow'

			cfg[section]['ChainStart'] = '{}, {}'.format(points[0][0], points[0][1])
			cfg[section]['ChainEnd'] = '{}, {}'.format(points[1][0], points[1][1])

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
			input('\nPress ENTER/RETURN key to exit...')
			exit()

		img_path = unix_path(args.img_path)
		ext = args.ext
		get_profile = args.profile

		try:
			path.exists(img_path)
		except Exception:
			raise ValueError('The path argument [--img_path] does not seem to correspond to an image or folder with images, or could be missing!')

		if path.isfile(img_path):
			img = cv2.imread(img_path)
		elif path.isdir(img_path):
			images = glob('{}/*.{}'.format(img_path, ext))
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

		plt.show()

	except Exception as ex:
		print()
		tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
		print('\n{}'.format(format_exc()))
		input('\nPress ENTER/RETURN key to exit...')
