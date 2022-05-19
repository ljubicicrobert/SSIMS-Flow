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
	from class_console_printer import tag_print

	import matplotlib.pyplot as plt

except Exception:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


def snr(x: np.ndarray) -> float:
	"""
	Calculate signal-to-noise ratio of :x:.
	"""
	
	m = np.mean(x)
	sd = np.std(x)
	return m/sd


def update_frame(val):
	global current_frame

	current_frame = frames_list[sl_ax_frame_num.val]
	get_colospaces(current_frame, xlim, ylim)

	plt.draw()


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


def get_colospaces(path: str, xlim: list, ylim: list):
	"""
	Update plot with new colorspace channels.
	"""

	img_bgr = cv2.imread(path)
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
	img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

	cs_list = [
		img_rgb,
		img_rgb[:, :, 0],
		img_rgb[:, :, 1],
		img_rgb[:, :, 2],
		img_gray,
		img_hsv[:, :, 0],
		img_hsv[:, :, 1],
		img_hsv[:, :, 2],
		None,
		img_lab[:, :, 0],
		img_lab[:, :, 1],
		img_lab[:, :, 2],
	]

	for i in range(len(cs_list)):
		if cs_list[i] is not None:
			img = cs_list[i]
			imshow_list[i].set_data(img)
			ax_list[i].set_xlim(xlim)
			ax_list[i].set_ylim(ylim)
			ax_list[i].set_title(cs_names[i])

			if i == 0:
				ax_list[i].set_title(cs_names[i])
			else:
				ax_snr = snr(img[ylim[1]: ylim[0], xlim[0]: xlim[1]])
				ax_list[i].set_title('{}, SNR={:.2f}'.format(cs_names[i], ax_snr))


def on_lims_change(event_ax):
	"""
	Matplotlib event to connect all axes to zoom and pan simultaneously.
	"""
	
	global xlim
	global ylim

	cid_list = list(event_ax.callbacks.callbacks['ylim_changed'].keys())
	for cid in cid_list:
		event_ax.callbacks.disconnect(cid)

	xlim = [int(x) for x in event_ax.get_xlim()]
	ylim = [int(y) for y in event_ax.get_ylim()]

	event_ax.set_xlim(xlim)
	event_ax.set_ylim(ylim)

	for i in range(1, len(cs_names)):
		if cs_names[i] != '':
			a = ax_list[i]
			ax_title = a.get_title().split(', ')[0]
			ax_img = np.array(a.get_images()[0]._A)
			ax_img_crop = ax_img[ylim[1]: ylim[0], xlim[0]: xlim[1]]
			ax_snr = snr(ax_img_crop)

			a.set_title('{}, SNR={:.2f}'.format(ax_title, ax_snr))

	event_ax.callbacks.connect('ylim_changed', on_lims_change)


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--folder', type=str, help='Path to image file or folder with images')
		parser.add_argument('--ext', type=str, help='Path to image file')
		args = parser.parse_args()

		frames_list = glob('{}/*.{}'.format(args.folder, args.ext))
		num_frames = len(frames_list)
		first_frame = cv2.imread(frames_list[0], 0)

		h, w = first_frame.shape
		xlim = [0, w]
		ylim = [h, 0]

		nrows, ncols = 3, 4
		fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
		plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.06, wspace=0.02, hspace=0.1)
		fig.canvas.mpl_connect('key_press_event', keypress)

		legend = 'Use O to zoom and P to pan images,\n' \
		         'Use slider to select frame,\n' \
		         'use UP and DOWN keys to move by +/- 1 frame\n' \
		         'or PageUP and PageDOWN keys to move by +/- 10 frames\n' \
		         'Press ESC or Q to exit'

		legend_toggle = plt.text(0.5, 0.5, legend,
				                 horizontalalignment='center',
				                 verticalalignment='center',
		                         transform=ax[2][0].transAxes,
				                 bbox=dict(facecolor='white', alpha=0.5),
				                 fontsize=9,
				                 )

		axcolor = 'lightgoldenrodyellow'
		valfmt = "%d"

		ax_frame_num = plt.axes([0.2, 0.02, 0.63, 0.03], facecolor=axcolor)
		sl_ax_frame_num = Slider(ax_frame_num, 'Frame #\n({} total)'.format(num_frames), 0, num_frames - 1, valinit=0, valstep=1, valfmt=valfmt)
		sl_ax_frame_num.on_changed(update_frame)

		cs_names = [
			'Original RGB',
			'[R]GB',
			'R[G]B',
			'RG[B]',
			'Grayscale',
			'[H]SV',
			'H[S]V',
			'HS[V]',
			'',
			'[L*]a*b*',
			'L*[a*]b*',
			'L*a*[b*]',
		]

		ax_list = ax.reshape(-1)
		imshow_list = [None] * 12

		current_frame = frames_list[0]

		img_bgr = cv2.imread(current_frame)
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
		img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
		img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

		cs_list = [
			img_rgb,
			img_rgb[:, :, 0],
			img_rgb[:, :, 1],
			img_rgb[:, :, 2],
			img_gray,
			img_hsv[:, :, 0],
			img_hsv[:, :, 1],
			img_hsv[:, :, 2],
			None,
			img_lab[:, :, 0],
			img_lab[:, :, 1],
			img_lab[:, :, 2],
		]

		for i in range(len(cs_list)):
			if cs_list[i] is not None:
				img = cs_list[i]
				imshow_list[i] = ax_list[i].imshow(img)
				ax_list[i].set_xlim(xlim)
				ax_list[i].set_ylim(ylim)
				ax_list[i].set_title(cs_names[i])

				if i == 0:
					ax_list[i].set_title(cs_names[i])
				else:
					ax_snr = snr(img[ylim[1]: ylim[0], xlim[0]: xlim[1]])
					ax_list[i].set_title('{}, SNR={:.2f}'.format(cs_names[i], ax_snr))

		[a.set_axis_off() for a in ax_list]
		[a.callbacks.connect('ylim_changed', on_lims_change) for a in ax_list]

		try:
			mng = plt.get_current_fig_manager()
			mng.window.state('zoomed')
			mng.set_window_title('Inspect frames')
		except Exception:
			pass

		plt.show()

	except Exception:
		print()
		tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
		print('\n{}'.format(format_exc()))
		input('\nPress ENTER/RETURN key to exit...')
