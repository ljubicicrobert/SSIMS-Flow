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
	from class_console_printer import tag_print, unix_path

	import matplotlib.pyplot as plt

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


def keypress(event):
	if event.key == 'escape':
		plt.close()


try:
	parser = ArgumentParser()
	parser.add_argument('--folder', type=str, help='Path to stabilization folder')
	args = parser.parse_args()

	folder = unix_path(args.folder)
	ssim_path = '{}/ssim_scores.txt'.format(folder)

	if not path.exists(ssim_path):
		tag_print('error', 'SSIM scores file not found in folder [{}]'.format(folder))
		exit()

	data = np.loadtxt(ssim_path, dtype='float')
	num_markers = data.shape[1]
	marker_indices = list(range(1, num_markers + 1))

	fig, ax = plt.subplots()
	fig.canvas.mpl_connect('key_press_event', keypress)

	plt.boxplot(data)
	xlim = plt.xlim()
	plt.plot(xlim, [1, 1], 'k--')
	plt.xlim(xlim)
	plt.xticks(marker_indices)
	plt.xlabel('Feature # [-]')
	plt.ylabel('SSIM score [-]')
	plt.title('Higher score is better, perfect score = 1')

	try:
		mng = plt.get_current_fig_manager()
		mng.window.state('zoomed')
		mng.set_window_title('SSIM score comparison by GCP')
	except Exception:
		pass

	plt.show()

	fig, ax = plt.subplots()
	fig.canvas.mpl_connect('key_press_event', keypress)

	for i in range(num_markers):
		plt.plot(data[:, i])

	plt.legend(["Marker #" + str(x) for x in list(range(num_markers))])
	plt.xlabel('Frame # [-]')
	plt.ylabel('SSIM score [-]')
	plt.title('Higher score is better, perfect score = 1')

	try:
		mng = plt.get_current_fig_manager()
		mng.window.state('zoomed')
		mng.set_window_title('SSIM score variation over time')
	except Exception:
		pass

	plt.show()

except Exception as ex:
	print()
	tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
