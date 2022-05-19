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

	import matplotlib.pyplot as plt

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


def update_frame(val):
	img_new_path = frames_list[sl_ax_frame_num.val]
	img_new = cv2.cvtColor(cv2.imread(img_new_path), cv2.COLOR_BGR2RGB)
	h, w = img_new.shape[:2]

	img_shown.set_data(img_new)
	ax.set_title('Frame #{}/{}, {}x{}px'.format(sl_ax_frame_num.val, num_frames - 1, w, h))
	plt.draw()

	return


def keypress(event):
	if event.key == 'escape':
		exit()

	elif event.key == 'down':
		if sl_ax_frame_num.val == 0:
			sl_ax_frame_num.set_val(num_frames - 1)
		else:
			sl_ax_frame_num.set_val(sl_ax_frame_num.val - 1)

	elif event.key == 'up':
		if sl_ax_frame_num.val == num_frames - 1:
			sl_ax_frame_num.set_val(0)
		else:
			sl_ax_frame_num.set_val(sl_ax_frame_num.val + 1)

	elif event.key == 'pageup':
		if sl_ax_frame_num.val >= num_frames - 10:
			sl_ax_frame_num.set_val(0)
		else:
			sl_ax_frame_num.set_val(sl_ax_frame_num.val + 10)

	elif event.key == 'pagedown':
		if sl_ax_frame_num.val <= 9:
			sl_ax_frame_num.set_val(num_frames - 1)
		else:
			sl_ax_frame_num.set_val(sl_ax_frame_num.val - 10)

	update_frame(sl_ax_frame_num.val)


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--folder', type=str, help='Path to frames folder')
		parser.add_argument('--ext', type=str, help='Frames'' extension')
		args = parser.parse_args()

		frames_folder = unix_path(args.folder)
		ext = str(args.ext).replace('.', '')

		if frames_folder is None:
			print()
			tag_print('error', 'Frames folder not provided!')
			input('\nPress ENTER/RETURN to exit...')
			exit()
		elif ext is None:
			print()
			tag_print('error', 'No frame extension provided!')
			input('\nPress ENTER/RETURN to exit...')
			exit()

		frames_list = glob('{}/*.{}'.format(frames_folder, ext))
		num_frames = len(frames_list)

		if num_frames == 0:
			print()
			tag_print('error', 'No frames were found in folder [{}] with extension [{}]'.format(frames_folder, ext))
			input('\nPress ENTER/RETURN to exit...')
			exit()

		fig, ax = plt.subplots()
		plt.subplots_adjust(bottom=0.13)
		plt.axis('off')

		axcolor = 'lightgoldenrodyellow'
		valfmt = "%d"

		ax_frame_num = plt.axes([0.2, 0.05, 0.63, 0.03], facecolor=axcolor)
		sl_ax_frame_num = Slider(ax_frame_num, 'Frame #', 0, num_frames-1, valinit=0, valstep=1, valfmt=valfmt)
		sl_ax_frame_num.on_changed(update_frame)

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

		img = cv2.cvtColor(cv2.imread(frames_list[0]), cv2.COLOR_BGR2RGB)
		h, w = img.shape[:2]
		fig.canvas.mpl_connect('key_press_event', keypress)
		img_shown = ax.imshow(img)

		try:
			mng = plt.get_current_fig_manager()
			mng.window.state('zoomed')
			mng.set_window_title('Inspect frames')
		except Exception:
			pass

		ax.set_title('Frame #0/{}, {}x{}px'.format(num_frames - 1, w, h))
		plt.show()

	except Exception as ex:
		print()
		tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
		print('\n{}'.format(format_exc()))
		input('\nPress ENTER/RETURN key to exit...')
