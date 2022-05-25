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
	from math import log
	from itertools import product
	from os import makedirs, remove, path, name
	from sys import exit
	from matplotlib.widgets import Slider
	from class_logger import Logger
	from class_console_printer import Console_printer, tag_string, tag_print, unix_path
	from class_progress_bar import Progress_bar
	from class_timing import Timer, time_hms
	from glob import glob
	from CPP.dll_import import DLL_Loader

	import matplotlib.pyplot as plt

	dll_path = path.split(path.realpath(__file__))[0]
	dll_name = 'CPP/ssim.dll' if name == 'nt' else 'CPP/libssim.so'
	dll_loader = DLL_Loader(dll_path, dll_name)
	# float SSIM_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int win_size, int maxVal);
	fast_ssim = dll_loader.get_function('float', 'SSIM_Byte', ['byte*', 'byte*', 'int', 'int', 'int', 'int', 'int'])

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


show_legend = True
legend_toggle = False


def to_odd(x, side=0) -> int:
	"""
	Returns the closest odd ingeter of :x:, if :x: not already odd.

	:param side:	If 0 then returns next larger odd integer, else next smaller.
	:return:		Odd integer.
	"""
	x = int(x)
	if x % 2 == 1:
		return x

	if side == 0:
		return x + 1
	else:
		return x - 1


def to_int(x: float) -> int:
	return int(round(x))


def get_gcps_from_image(image_orig: np.ndarray, verbose=False, ia=11, sa=21, hide_sliders=False) -> list:
	"""
	Extracts [x, y] pixel coordinates from an image using right mouse click.
	Use middle mouse button or BACKSPACE key to remove the last selected point from the list.
	Pres ENTER to accept the list of points, or ESC to cancel the operation.

	:param image_orig:		Image as numpy array to be rectified.
	:param verbose:			Whether to use verbose output. Default is False.
	:param ia:				Interrogation area size. Default is 11.
	:param sa:				Search area size. Default is 21.
	:param hide_sliders:	Whether to hide the IA/SA size sliders.
	:return:				A list of initial pixel positions to use with coordTransform().
	"""

	global show_legend
	global legend_toggle

	image = image_orig.copy()

	iw = ia // 2
	sw = sa // 2
	points = []
	org = []
	axcolor = 'lightgoldenrodyellow'
	valfmt = "%d"

	if not hide_sliders:
		ia_c, sa_c = 0.8, 0.9
	else:
		ia_c, sa_c = 1.0, 1.0

	fig, ax = plt.subplots()
	plt.subplots_adjust(bottom=0.2)

	def xy2str(l):
		s = ''
		i = 1
		for x, y in l:
			s += '#{}: X={}, Y={}\n'.format(i, x, y)
			i += 1

		return s[:-1]

	def getPixelValue(event):
		global p_list

		if event.button == 2 and len(points) > 0:
			p = points.pop()
			o = org.pop()

			image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1] = o

			points_list.set_text(xy2str(points))
			img_ref.set_data(image)
			update_ia(sl_ax_ia_size.val)
			update_sa(sl_ax_sa_size.val)
			plt.draw()

		if event.button == 3:
			p = [to_int(event.xdata), to_int(event.ydata)]
			points.append(p)

			sec_s = image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1]
			sec_i = image[p[1] - iw: p[1] + iw + 1, p[0] - iw: p[0] + iw + 1]

			org.append(image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1].copy())

			sec_i = sec_i ** ia_c
			image[p[1] - iw: p[1] + iw + 1, p[0] - iw: p[0] + iw + 1] = sec_i

			sec_s = sec_s ** sa_c
			image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1] = sec_s

			image[p[1], p[0]] = 0

			points_list.set_text(xy2str(points))
			img_ref.set_data(image)
			update_ia(sl_ax_ia_size.val)
			update_sa(sl_ax_sa_size.val)
			plt.draw()

	def keypress(event):
		global p_list
		global show_legend

		if event.key == 'd' and len(points) > 0:
			p = points.pop()
			o = org.pop()

			image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1] = o

			points_list.set_text(xy2str(points))
			img_ref.set_data(image)
			update_ia(sl_ax_ia_size.val)
			update_sa(sl_ax_sa_size.val)
			plt.draw()

		elif event.key == 'enter':
			plt.close()

		elif event.key in ['escape', 'q']:
			plt.close()
			input(tag_string('abort', 'EXECUTION STOPPED: Operation aborted by user! Press ENTER/RETURN key to exit...'))
			exit()

		elif event.key == 'f1':
			show_legend = not show_legend
			legend_toggle.set_visible(show_legend)
			points_list.set_visible(show_legend)
			event.canvas.draw()

	if verbose:
		print()
		tag_print('info', 'Click MOUSE RIGHT to add a point to the list')
		tag_print('info', 'Press ENTER key to accept the list of points')
		tag_print('info', 'Press D or click MOUSE MIDDLE to remove the last point in the list')
		tag_print('info', 'Press ESC or Q to cancel the operation\n')

	fig.canvas.mpl_connect('button_press_event', getPixelValue)
	fig.canvas.mpl_connect('key_press_event', keypress)
	img_ref = plt.imshow(image)

	def update_sa(val):
		global search_size
		global k_size
		global k_span

		if val <= sl_ax_ia_size.val - 2:
			sl_ax_ia_size.set_val(val - 2)

		sa = int(val)
		sw = sa // 2

		ia = int(sl_ax_ia_size.val)
		iw = ia // 2

		search_size = sa
		k_size = ia
		k_span = iw

		image = image_orig.copy()
		org = []

		for p in points:
			sec_s = image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1]
			sec_i = image[p[1] - iw: p[1] + iw + 1, p[0] - iw: p[0] + iw + 1]

			org.append(image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1].copy())

			sec_i = sec_i ** ia_c
			image[p[1] - iw: p[1] + iw + 1, p[0] - iw: p[0] + iw + 1] = sec_i

			sec_s = sec_s ** sa_c
			image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1] = sec_s

			image[p[1], p[0]] = 0

		img_ref.set_data(image)
		plt.draw()

	def update_ia(val):
		global search_size
		global k_size
		global k_span

		if val >= sl_ax_sa_size.val + 2:
			sl_ax_sa_size.set_val(val + 2)

		sa = int(sl_ax_sa_size.val)
		sw = sa // 2

		ia = int(val)
		iw = ia // 2

		search_size = sa
		k_size = ia
		k_span = iw

		image = image_orig.copy()
		org = []

		for p in points:
			sec_s = image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1]
			sec_i = image[p[1] - iw: p[1] + iw + 1, p[0] - iw: p[0] + iw + 1]

			org.append(image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1].copy())

			sec_i = sec_i ** ia_c
			image[p[1] - iw: p[1] + iw + 1, p[0] - iw: p[0] + iw + 1] = sec_i

			sec_s = sec_s ** sa_c
			image[p[1] - sw: p[1] + sw + 1, p[0] - sw: p[0] + sw + 1] = sec_s

			image[p[1], p[0]] = 0

		img_ref.set_data(image)
		plt.draw()

	ax_ia_size = plt.axes([0.3, 0.1, 0.40, 0.03], facecolor=axcolor)
	sl_ax_ia_size = Slider(ax_ia_size, 'IA size', 7, 51, valinit=ia, valstep=2, valfmt=valfmt)
	sl_ax_ia_size.on_changed(update_ia)

	ax_sa_size = plt.axes([0.3, 0.05, 0.40, 0.03], facecolor=axcolor)
	sl_ax_sa_size = Slider(ax_sa_size, 'SA size', 13, 101, valinit=sa, valstep=2, valfmt=valfmt)
	sl_ax_sa_size.on_changed(update_sa)

	legend = 'O = zoom to window\n' \
			 'P = pan image\n' \
			 'Mouse RIGHT = select feature\n' \
			 'Mouse MIDDLE or D = remove last feature\n' \
			 'ENTER/RETURN = accept selected features\n' \
			 'ESC or Q = abort operation\n' \
			 'Use sliders to change IA/SA window size'

	if hide_sliders:
		ax_ia_size.set_visible(False)
		ax_sa_size.set_visible(False)

		ax.set_title('Choose GCPs in the same order as in the orthorectification GCP list')

		legend = 'O = zoom to window\n' \
		         'P = pan image\n' \
		         'Mouse RIGHT = select GCP\n' \
		         'Mouse MIDDLE or D = remove last GCP\n' \
		         'ENTER/RETURN = accept selected GCPs\n' \
		         'ESC or Q = abort operation'

	legend_toggle = ax.text(0.02, 0.97, legend,
							 horizontalalignment='left',
							 verticalalignment='top',
							 transform=ax.transAxes,
							 bbox=dict(facecolor='white', alpha=0.5),
							 fontsize=9,
							 )

	hint = ax.text(0.02, 1.02, 'F1: toggle legend',
					 horizontalalignment='left',
					 verticalalignment='bottom',
					 transform=ax.transAxes,
					 fontsize=9,
					 )

	points_list = ax.text(0.99, 0.02, '',
				  horizontalalignment='right',
				  verticalalignment='bottom',
				  transform=ax.transAxes,
				  bbox=dict(facecolor='white', alpha=0.5),
				  fontsize=9,
				  )

	try:
		mng = plt.get_current_fig_manager()
		mng.window.state('zoomed')
		mng.set_window_title('Feature selection')
	except Exception:
		pass

	plt.show()

	if verbose:
		print(points)

	return points


def find_gcp(search_area: np.ndarray, kernel: np.ndarray) -> tuple:
	"""
	Detects a GCP center in the using SSIM.

	:param search_area: 	Input image to search for GCP in.
	:param kernel:			Kernel to use for SSIM comparison.
	:param show: 			Whether to show the SSIM map. Default is False.
	:param padding:			Padding inside search_area.
	:return: 				Position (x,y) of the GCP center in the input image.
	"""
	
	padding = kernel.shape[0] // 2
	ssim_map = np.zeros(search_area.shape)
	search_range = range(padding, search_area.shape[0] - padding)

	for i, j in product(search_range, search_range):
		subarea = cv2.getRectSubPix(search_area, (k_size, k_size), (i, j))
		ssim_map[i, j] = fast_ssim(subarea.ravel(), kernel.ravel(), kernel.shape[0], kernel.shape[0], kernel.shape[0], 7, 255)

	try:
		score_max = np.max(ssim_map)
	except ValueError:
		score_max = 0
		x_pix, y_pix = 0, 0
		
		return (x_pix, y_pix), score_max

	x_pix, y_pix = np.unravel_index(np.argmax(ssim_map), ssim_map.shape)

	# Gaussian 2x3 fit
	dx = (log(ssim_map[x_pix-1, y_pix]) - log(ssim_map[x_pix+1, y_pix])) / (2*(log(ssim_map[x_pix-1, y_pix]) + log(ssim_map[x_pix+1, y_pix]) - 2*log(ssim_map[x_pix, y_pix])))
	dy = (log(ssim_map[x_pix, y_pix-1]) - log(ssim_map[x_pix, y_pix+1])) / (2*(log(ssim_map[x_pix, y_pix-1]) + log(ssim_map[x_pix, y_pix+1]) - 2*log(ssim_map[x_pix, y_pix])))

	x_sub = x_pix + dx
	y_sub = y_pix + dy

	return (x_sub, y_sub), score_max


def fresh_folder(folder_path, ext='*', exclude=list()):
	if not path.exists(folder_path):
		makedirs(folder_path)
	else:
		files = glob('{}/*.{}'.format(folder_path, ext))
		for f in files:
			if f not in exclude:
				remove(f)


def print_markers(marker_list):
	s = ''
	for i, marker in enumerate(marker_list):
		x, y = marker
		s += '{}=({:.2f}, {:.2f}), '.format(i, x, y)
	
	return s[:-2]



def print_and_log(string, printer_obj: Console_printer, logger_obj: Logger):
	logger_obj.log(printer_obj.add_line(string))


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--cfg', type=str, help='Path to configuration file')
		args = parser.parse_args()

		cfg = configparser.ConfigParser()
		cfg.optionxform = str

		try:
			cfg.read(args.cfg, encoding='utf-8-sig')
		except Exception:
			input(tag_string('error', 'There was a problem reading the configuration file!\nCheck if project has valid configuration.'))
			exit()

		# Project root
		project_folder = unix_path(cfg.get('Project settings', 'Folder'))

		# Folder with raw frames
		frames_folder = '{}/frames'.format(project_folder)

		# Folder for result output
		results_folder = '{}/transformation'.format(project_folder)

		# Extension for image files
		ext = cfg.get('Frames', 'Extension', fallback='jpg')

		section = 'Feature tracking'

		# Search area size (high cost)
		search_size = int(cfg.get(section, 'SearchAreaSize', fallback='21'))

		# Interrogation area size (low cost)
		k_size = int(cfg.get(section, 'InterrogationAreaSize', fallback='11'))
		k_span = k_size // 2

		# Expand the search area width/height if SSIM score is below :expand_ssim_thr:
		expand_ssim_search = int(cfg.get(section, 'ExpandSA', fallback='0'))

		# Search area expansion factor (high cost)
		expand_coef = float(cfg.get(section, 'ExpandSACoef', fallback='2.0'))

		# SSIM score threshold for expanded search
		expand_ssim_thr = float(cfg.get(section, 'ExpandSAThreshold', fallback='0.5'))

		# If significant image rotation is expected
		update_kernels = int(cfg.get(section, 'UpdateKernels', fallback='0'))

		# Do not change from this point on ------------------------------------------------------------
		assert search_size > 13 and search_size % 2 == 1 and type(search_size) == int, \
			tag_string('error', 'Search area size must be an integer >= 13!')
		assert 7 < k_size < search_size and k_size % 2 == 1 and type(k_size) == int, \
			tag_string('error', 'Interrogation area size (kernel size) must be an integer >= 7, and smaller than search area size!')
		assert expand_coef > 1, \
			tag_string('error', 'Search area expansion coefficient must be higher than 1!')
		assert 0 < expand_ssim_thr < 1, \
			tag_string('error', 'Search area expansion threshold must be in range (0, 1)!')

		raw_frames_list = glob('{}/*.{}'.format(frames_folder, ext))
		num_frames = len(raw_frames_list)
		numbering_len = int(log(num_frames, 10)) + 1

		is_expanded_search = False
		init_search_size = search_size
		exp_search_size = to_odd(search_size*expand_coef)

		img_path = raw_frames_list[0]
		img = cv2.imread(img_path)
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		markers = get_gcps_from_image(img_rgb, ia=k_size, sa=search_size, verbose=False)

		if len(markers) == 1:
			tag_print('error', 'Number of GCPs must be at least 2!')
			input('\nPress ENTER/RETURN key to exit...')
			exit()

		# Override initial configuration
		cfg[section]['SearchAreaSize'] = str(search_size)
		cfg[section]['InterrogationAreaSize'] = str(k_size)
		cfg['Transformation']['FeatureMask'] = '1'*len(markers)
		
		with open(args.cfg, 'w') as configfile:
			cfg.write(configfile)

		markers_mask = [1] * len(markers)

		folders_to_check = ['{}/gcps_csv'.format(results_folder),
							'{}/kernels'.format(results_folder)]

		for f in folders_to_check:
			fresh_folder(f)

		try:
			log_path = '{}/log_gcps.txt'.format(results_folder)
			logger = Logger(log_path)
			logger.log(tag_string('start', 'Feature tracking for frames in [{}]'.format(frames_folder)), to_print=True)
			print()
			logger.log(tag_string('info', 'Output to [{}]'.format(results_folder)), to_print=True)
			logger.log(tag_string('info', 'Total frames = {}'.format(num_frames)), to_print=True)
			logger.log(tag_string('info', 'Number of markers = {}'.format(len(markers))), to_print=True)
			logger.log(tag_string('info', 'Log file {}/\n'.format(log_path)), to_print=True)

			printer = Console_printer()
			progress_bar = Progress_bar(total=num_frames, prefix=tag_string('info', 'Frame '))

			kernels = []
			img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
			img_ch, *_ = cv2.split(img)

			for m in markers:
				k = cv2.getRectSubPix(img_ch, (k_size, k_size), (m[0], m[1]))
				cv2.imwrite('{}/kernels/{}.{}'.format(results_folder, len(kernels), ext), k)
				kernels.append(k)

			timer = Timer(total_iter=num_frames)

			ssim_scores = np.zeros([num_frames, len(markers)])

			for n in range(num_frames):
				try:
					print_and_log(progress_bar.get(n), printer, logger)

					img_path = raw_frames_list[n]
					img = cv2.imread(img_path)

					img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
					img_ch, *_ = cv2.split(img)

					for j in range(len(markers)):
						xx, yy = markers[j]

						if xx != 0 and yy != 0:
							search_space = cv2.getRectSubPix(img_ch, (search_size, search_size), (xx, yy))

							rel_center, ssim_max = find_gcp(search_space,
															kernels[j],
															)

							if expand_ssim_search and ssim_max < expand_ssim_thr:
								print_and_log(
									tag_string('warning', 'Expanding the search area, SSIM={:.3f} < {:.3f}'.format(ssim_max, expand_ssim_thr)), printer, logger
								)

								is_expanded_search = True
								search_size = exp_search_size
								search_space = cv2.getRectSubPix(img_ch, (search_size, search_size), (xx, yy))

								rel_center, ssim_max = find_gcp(search_space,
															kernels[j],
															)

							real_x = rel_center[0] + xx - (search_size - 1) / 2
							real_y = rel_center[1] + yy - (search_size - 1) / 2

							markers[j] = [real_x, real_y]
							ssim_scores[n, j] = ssim_max

							if is_expanded_search:
								is_expanded_search = False
								search_size = init_search_size

							if update_kernels:
								kernels[j] = cv2.getRectSubPix(img_ch, (k_size, k_size), (real_x, real_y))

							try:
								cv2.getRectSubPix(img_ch, (search_size, search_size), (real_x, real_y))
							except SystemError:
								print_and_log(
									tag_string('warning', 'Marker {} lost! Setting coordinates to (0, 0).'.format(j)), printer, logger
								)

								markers[j] = [0, 0]
								markers_mask[j] = 0

						else:
							print_and_log(
								tag_string('warning', 'Marker {} lost! Setting coordinates to (0, 0).'.format(j)), printer, logger
							)

							markers[j] = [0, 0]
							markers_mask[j] = 0

				except (AttributeError, IOError, IndexError):
					break

				timer.update()

				np.savetxt('{}/gcps_csv/{}.txt'.format(results_folder, str(n).rjust(numbering_len, '0')), markers, fmt='%.3f', delimiter=' ')
				np.savetxt('{}/ssim_scores.txt'.format(results_folder), ssim_scores, fmt='%.3f', delimiter=' ')

				# print_and_log(
				# 	tag_string('info', 'Markers: {}'.format(print_markers(markers))), printer, logger
				# )

				print_and_log(
					tag_string('info', 'Frame processing time = {:.3f} sec'.format(timer.interval())), printer, logger
				)

				print_and_log(
					tag_string('info', 'Elapsed time = {} hr {} min {} sec'.format(*time_hms(timer.elapsed()))), printer, logger
				)

				print_and_log(
					tag_string('info', 'Remaining time ~ {} hr {} min {} sec'.format(*time_hms(timer.remaining()))), printer, logger
				)

				printer.overwrite()

		except IOError:
			logger.close()

		else:
			logger.close()

		np.savetxt('{}/markers_mask.txt'.format(results_folder), markers_mask, fmt='%d', delimiter=' ')

		print()
		tag_print('end', 'Feature tracking complete!')
		print('\a')
		input('\nPress ENTER/RETURN key to exit...')

	except Exception as ex:
		print()
		tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
		print('\n{}'.format(format_exc()))
		input('\nPress ENTER/RETURN key to exit...')
