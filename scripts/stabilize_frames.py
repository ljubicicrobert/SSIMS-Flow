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
	from shutil import copy, SameFileError
	from time import time
	from os import path, makedirs, remove
	from feature_tracking import get_gcps_from_image
	from math import log
	from glob import glob
	from class_console_printer import Console_printer, tag_string, tag_print, unix_path
	from class_progress_bar import Progress_bar
	from class_logger import time_hms
	from class_timing import Timer, time_hms
	from utilities import fresh_folder, cfg_get, exit_message, present_exception_and_exit

	import ctypes

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


def coordTransform(image: np.ndarray,
				   points_old: np.ndarray, points_new: np.ndarray,
				   width: int, height: int,
				   method=cv2.findHomography,
				   M_ortho=None,
				   use_ransac=False, ransac_thr=None,
				   confidence=0.995, LM_iters=10) -> tuple:
	"""
	Performs coordinate transform an image or set of images using point handles. Original handle positions and new handle
	positions must be specifies, along with the height and width of the new image(s). See parameters for more details.

	:param image:			Image as numpy.ndarray to be transformed.
	:param points_old:		Handle points for transformation. A list of handles (min. 2) must be specified as [x, y].
	:param points_new:		New position for handles. A list of four handles must be specified as [x, y].
	:param width:			Width of the transformed image. Must be specified.
	:param height:			Height of the transformed image. Must be specified.
	:param method:			Method to use for the transformation: 	cv2.estimateAffinePartial2D,
																	cv2.getAffineTransform,
																	cv2.estimateAffine2D,
																	cv2.getPerspectiveTransform, or
																	cv2.findHomography (default).
	:param M_ortho:			Orthorectification matrix.
	:param use_ransac:		Whether to use RANSAC filtering for outlier detection. Default is False.
	:param ransac_thr:		Threshold for RANSAC outlier detection. Default is 1.
	:param confidence:		Required confidence for the transformation matrix. Default is 0.995.
	:param LM_iters:		Number of Levenberg-Marquardt iterations for refining. Default is 10.
	:return:				New transformed image as numpy.ndarray.
	"""

	assert len(points_old) >= 2 and len(points_new) >= 2, \
		tag_string('error', 'Minimal number of origin and destination points is 2!')
	assert len(points_old) == len(points_new), \
		tag_string('error', 'Number of origin points must be equal to the number of destination points!')

	if method is None:
		M_stable = np.identity(3, dtype='float64')
		status = []

	elif method in [cv2.estimateAffine2D, cv2.estimateAffinePartial2D]:
		if use_ransac:
			M_stable, status = method(points_old, points_new, method=cv2.RANSAC, ransacReprojThreshold=ransac_thr, confidence=confidence, refineIters=LM_iters)
		else:
			M_stable, status = method(points_old, points_new, confidence=confidence, refineIters=LM_iters)

	elif method == cv2.getAffineTransform:
		if len(points_old) != 3:
			tag_print('warning', 'Origin array not of size 3 for strict affine transformation!')
			tag_print('warning', 'Using only the first 3 features in from the file to estimate the transformation matrix.')
		M_stable = method(points_old[:3], points_new[:3])
		status = []

	elif method == cv2.getPerspectiveTransform:
		if len(points_old) != 4:
			tag_print('warning', 'Origin array not of size 4 for simple projective transformation!')
			tag_print('warning', 'Using only the first 4 features in from the file to estimate the transformation matrix.')
		M_stable = method(points_old[:4], points_new[:4])
		status = []

	elif method == cv2.findHomography:
		if use_ransac:
			M_stable, status = method(points_old, points_new, method=cv2.RANSAC, ransacReprojThreshold=ransac_thr, confidence=confidence)
		else:
			M_stable, *_ = method(points_old, points_new, 0)
			status = []

	else:
		tag_print('error', 'Unknown transformation method for stabilization point set!')
		exit_message()

	if M_ortho is None:
		M_ortho = np.identity(3)

	M_stable = extend_matrix_to_3x3(M_stable)
	M_ortho = extend_matrix_to_3x3(M_ortho)
	M_final = np.matmul(M_ortho, M_stable)

	stab_ortho = cv2.warpPerspective(image, M_final, (width, height))[::-1]

	# if method is not None:
	# 	if method in [cv2.getPerspectiveTransform, cv2.findHomography]:
	# 		stab_ortho = cv2.warpPerspective(image, M_final, (width, height))[::-1]
	# 	elif method in [cv2.estimateAffine2D, cv2.estimateAffinePartial2D, cv2.getAffineTransform]:
	# 		stab_ortho = cv2.warpAffine(image, M_final[:2, :], (width, height))[::-1]
	# else:
	# 	stab_ortho = cv2.warpPerspective(image, M_final, (width, height))[::-1]

	return stab_ortho, M_stable, status


def extend_matrix_to_3x3(m):
	if m.shape[0] == 2:
		m = np.vstack([m, [0, 0, 1]])
	
	return m


def imcrop(img: np.ndarray, bbox: list) -> np.ndarray:
	"""
	Crop image to boundary box.

	:param img: 	Image.
	:param bbox: 	Boundary box as [Xstart, Xend, Ystart, Yend].
	:return: 		Cropped image.
	"""

	x1, x2, y1, y2 = bbox

	if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
		img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)

	return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img: np.ndarray, x1: int, x2: int, y1: int, y2:int) -> tuple:
	"""
	Pad image with zeros for cropping.

	:param img: Image.
	:param x1:  X from.
	:param x2: 	X to.
	:param y1: 	Y from.
	:param y2: 	Y to.
	:return: 	Padded image.
	"""

	img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
			   (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0, 0)), mode="constant")

	y1 += np.abs(np.minimum(0, y1))
	y2 += np.abs(np.minimum(0, y1))
	x1 += np.abs(np.minimum(0, x1))
	x2 += np.abs(np.minimum(0, x1))

	return img, x1, x2, y1, y2


def compress(data: np.ndarray, selectors: list) -> list:
	"""
	Removal of unselected GCP markers from stabilization.

	:param data:		Previously tracked GCP markers loaded from file.
	:param selectors:	List of 0s and 1s to mark used GCP markers.
	:return:			List of selected GCP markers.
	"""
	return list(d for d, s in zip(data, selectors) if s)


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
			tag_print('error', 'There was a problem reading the configuration file!')
			tag_print('error', 'Check if project has valid configuration.')
			exit_message()

		project_folder = unix_path(cfg_get(cfg, 'Project settings', 'Folder', str))
		frames_folder = f'{project_folder}/frames'
		results_folder = f'{project_folder}/transformation'

		try:
			copy(args.cfg, results_folder)
		except SameFileError:
			pass

		ext_in = cfg_get(cfg, 'Frames', 'Extension', str, 'jpg')

		section = 'Transformation'

		moving_camera = cfg_get(cfg, "Feature tracking", 'MovingCamera', int, 1)
		ext_out = cfg_get(cfg, section, 'Extension', str, 'jpg')
		qual = cfg_get(cfg, section, 'Quality', int, 95)
		stabilization_method = cfg_get(cfg, section, 'Method', int)
		use_ransac_filtering = cfg_get(cfg, section, 'UseRANSAC', int, 0)
		ransac_filtering_thr = cfg_get(cfg, section, 'RANSACThreshold', float, 2.0)
		orthorectify = cfg_get(cfg, section, 'Orthorectify', int, 0)
		gsd = cfg_get(cfg, section, 'GSD', float)
		gcps_mask = cfg_get(cfg, section, 'FeatureMask', str)
		pdx = cfg_get(cfg, section, 'PaddX', str)
		pdy = cfg_get(cfg, section, 'PaddX', str)

		padd_x = [int(float(x) * gsd) for x in pdx.split('-')]
		padd_y = [int(float(y) * gsd) for y in pdy.split('-')]

		# Do not change from this point on ---------------------------------------------------------------------------------
		methods = {0: cv2.estimateAffinePartial2D,
				   1: cv2.getAffineTransform,
				   2: cv2.estimateAffine2D,
				   3: cv2.getPerspectiveTransform,
				   4: cv2.findHomography}

		methods_alias = ['similarity',
						 'affine_2D_strict',
						 'affine_2D_optimal',
						 'projective_strict',
						 'projective_optimal']

		gcp_folder = f'{results_folder}/gcps_csv'
		stabilized_folder = f'{results_folder}/frames_{"orthorectified" if orthorectify else "stabilized"}'
		transform_folder = f'{results_folder}/transform_{"orthorectified" if orthorectify else "stabilized"}'
		end_file = f'{results_folder}/end'

		fresh_folder(stabilized_folder)
		fresh_folder(transform_folder)

		if path.exists(end_file):
			remove(end_file)

		tag_print('start', f'Starting image transformation using data in [{results_folder}]')
		print()

		folders_to_check = [stabilized_folder,
							transform_folder]

		for f in folders_to_check:
			if not path.exists(f):
				makedirs(f)

		raw_frames_list = glob(f'{frames_folder}/*.{ext_in}')
		num_frames = len(raw_frames_list)
		num_len = int(log(num_frames, 10)) + 1
		
		img = cv2.imread(raw_frames_list[0], cv2.COLOR_BGR2RGB)
		img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		
		h, w = img.shape[:2]

		if moving_camera:
			features_coord = glob(f'{gcp_folder}/*.txt')
			anchors = np.loadtxt(features_coord[0], dtype='float32', delimiter=' ')

			if gcps_mask == '1':
				num_features = gcps_mask.count(True)
				gcps_mask = [1] * num_features
			else:
				gcps_mask = [int(x) for x in gcps_mask]
				num_features = gcps_mask.count(True)
				anchors = np.asarray(compress(anchors, gcps_mask))
		else:
			anchors = np.array([[0, 0], [h, 0], [h, w], [0, w]], dtype='float32')
			features_coord = np.dstack(anchors * (num_frames - 1))
		
		num_avail_gcps = anchors.shape[0]

		assert num_avail_gcps >= 2,\
			tag_string('error', 'Number of available GCPs is not >= 2 in all frames!\n') + \
			'        Consider repeating the feature tracking with features which are available in all frames.\n' \
			'        If this cannot be achieved, consider splitting the video into several segments and then' \
			'        stabilizing each one individually.'
		if stabilization_method > 3: assert num_avail_gcps > 3,\
			tag_string('error', 'Number of available GCPs is not >= 4 in all frames for projective transform!\n') + \
			'        Consider switching to one of the available affine transformation methods or\n' \
			'        repeat the feature tracking with features which are available in all frames.'

		if orthorectify:
			overwrite_GCPs = True

			if path.exists(f'{results_folder}/gcps_image.txt'):
				MessageBox = ctypes.windll.user32.MessageBoxW

				response = MessageBox(None, f'Ground control point positions have already been set.\n' +
						  				     'Do you wish to overwrite them?\n' + 
											 'YES = select new GCP positions\n' +
											 'NO = use previously selected GCP positions', 'Overwrite GCPs', 68)

				if response != 6:
					overwrite_GCPs = False
					gcps_image = np.loadtxt(f'{results_folder}/gcps_image.txt', dtype='float32', delimiter=' ')

			gcps_real = np.multiply(np.loadtxt(f'{results_folder}/gcps_real.txt', dtype='float32', delimiter=' '), gsd)

			if overwrite_GCPs:
				initial_gcps = []

				dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
				parameters =  cv2.aruco.DetectorParameters()
				detector = cv2.aruco.ArucoDetector(dictionary, parameters)
				corners, ids, rejectedImgPoints = detector.detectMarkers(img_gray)

				try:
					if len(ids) > 0:
						ids_sorted = ids[:, 0].argsort()
						corners = [corners[x] for x in ids_sorted]
						MessageBox = ctypes.windll.user32.MessageBoxW
						response = MessageBox(None, f'A total of {ids.shape[0]} ArUco markers have been detected in the first frame.\nDo you wish to add them to the list of tracked GCPs?', 'ArUco markers detected', 68)

						if response == 6:
							for i in range(len(ids)):
								c = corners[i][0]
								initial_gcps.append([c[:, 0].mean(), c[:, 1].mean()])

				except Exception as ex:
					pass

				gcps_image = np.asarray(get_gcps_from_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), initial=initial_gcps, verbose=False, hide_sliders=True))
				np.savetxt(f'{results_folder}/gcps_image.txt', gcps_image, fmt='%.3f', delimiter=' ')

			assert gcps_real.shape == gcps_image.shape,\
				tag_string('error', f'Number of GCPs [{gcps_real.shape[0]}] not equal to number of selected features [{gcps_image.shape[0]}]')

			min_x = np.min(gcps_real[:, 0])
			min_y = np.min(gcps_real[:, 1])

			for p in gcps_real:
				p[0] = p[0] - min_x + padd_x[0]
				p[1] = p[1] - min_y + padd_y[0]

			max_x = int(np.max(gcps_real[:, 0]) + padd_x[1])
			max_y = int(np.max(gcps_real[:, 1]) + padd_y[1])

			h, w = max_y, max_x

			if gcps_real.shape[0] >= 4:
				if use_ransac_filtering:
					M_ortho, *_ = cv2.findHomography(gcps_image, gcps_real, cv2.RANSAC, ransac_filtering_thr)
				else:
					M_ortho, *_ = cv2.findHomography(gcps_image, gcps_real, 0)

			elif gcps_real.shape[0] == 3:
				if use_ransac_filtering:
					M_ortho, *_ = cv2.estimateAffine2D(gcps_image, gcps_real, method=cv2.RANSAC, ransacReprojThreshold=ransac_filtering_thr, confidence=0.995, refineIters=10)
				else:
					M_ortho, *_ = cv2.estimateAffine2D(gcps_image, gcps_real, confidence=0.995, refineIters=10)

		else:
			M_ortho = None

		i = 0

		console_printer = Console_printer()
		progress_bar = Progress_bar(total=num_frames, prefix=tag_string('info', 'Stabilized frame '))
		timer = Timer(total_iter=num_frames)

		while True:
			try:
				start_time = time()

				if moving_camera:
					features = np.asarray(compress(np.loadtxt(features_coord[i], dtype='float32', delimiter=' '), gcps_mask))
				else:
					features = anchors

				img_path = raw_frames_list[i]
				img = cv2.imread(img_path)

				stabilized, M, status = coordTransform(img, features, anchors, width=w, height=h,
														method=methods[stabilization_method] if moving_camera else None,
														M_ortho=M_ortho,
														use_ransac=use_ransac_filtering,
														ransac_thr=ransac_filtering_thr)

				if not orthorectify:
					stabilized = stabilized[::-1]

				n = str(i).rjust(num_len, '0')
				np.savetxt(f'{transform_folder}/{n}.txt', M, delimiter=' ')
				save_str_img = f'{stabilized_folder}/{n}.{ext_out}'

				if ext_out.lower() in ['jpg', 'jpeg']:
					cv2.imwrite(save_str_img,
								stabilized,
								[int(cv2.IMWRITE_JPEG_QUALITY), qual])
				elif ext_out.lower() == 'png':
					cv2.imwrite(save_str_img,
								stabilized,
								[int(cv2.IMWRITE_PNG_COMPRESSION), int(9 - 0.09 * qual)])
				elif ext_out.lower() == 'webp':
					cv2.imwrite(save_str_img,
								stabilized,
								[int(cv2.IMWRITE_WEBP_QUALITY), qual + 1])
				else:
					cv2.imwrite(save_str_img, stabilized[::-1])

				note = ''
				if [0] in status:
					note = '{}'.format([i[0] for i in status])			

				if use_ransac_filtering:
					console_printer.add_line(tag_string('info', f'Outliers: {note}'))

				timer.update()
				
				console_printer.add_line(progress_bar.get(i))
				console_printer.add_line(tag_string('info', f'Frame processing time = {timer.interval():.3f} sec'))
				he, me, se = time_hms(timer.elapsed())
				console_printer.add_line(tag_string('info', f'Elapsed time = {he} hr {me} min {se} sec'))
				hr, mr, sr = time_hms(timer.remaining())
				console_printer.add_line(tag_string('info', f'Remaining time = {hr} hr {mr} min {sr} sec'))

				console_printer.overwrite()

				i += 1

			except (IOError, IndexError):
				break

		# Touch
		open(end_file, 'w').close()

		print()
		tag_print('end', 'Image stabilization complete!')
		print('\a')
		exit_message()

	except Exception as ex:
		present_exception_and_exit()
