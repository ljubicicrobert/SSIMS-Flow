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
	from os import path, mkdir
	from glob import glob
	from class_console_printer import tag_print, unix_path
	from utilities import exit_message, present_exception_and_exit

	import matplotlib.pyplot as plt

except Exception:
	present_exception_and_exit('Import failed! See traceback below:')


if __name__ == '__main__':
	try:
		parser = ArgumentParser(add_help=False)
		parser.add_argument('--model', type=str, help='Camera model name', default='')
		parser.add_argument('--folder', type=str, help='Path to images folder')
		parser.add_argument('--ext', type=str, help='Frames'' extension', default='jpg')
		parser.add_argument('-w', type=int, help='Number of squares in horizontal direction')
		parser.add_argument('-h', type=int, help='Number of squares in vertical direction')
		parser.add_argument('--use_k3', type=int, help='Whether to use three radial distortion coefficients', default=0)
		parser.add_argument('--output', type=int, help='Whether to output undistorted images', default=0)
		args = parser.parse_args()

		frames_folder = unix_path(args.folder)
		output_folder = f'{unix_path(frames_folder)}/undistorted'
		extension = args.ext
		camera_model = 'camera_parameters' if args.model == '' else args.model

		if not path.exists(output_folder):
			mkdir(output_folder)

		cheq_w = int(args.w) - 1
		cheq_h = int(args.h) - 1

		board_size = (cheq_w, cheq_h)
		fixed_k3 = 1 if args.use_k3 == 0 else 0
		calibration_flags = cv2.CALIB_FIX_K3 if fixed_k3 else 0

		objpoints = []
		imgpoints = []

		objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float32)
		objp[0, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

		fig, ax = plt.subplots()

		images = glob(f'{frames_folder}/*.{extension}')
		image_names = [path.basename(x) for x in images]
		good_image_names = []
		num_images = len(images)
		ret_list = [None] * num_images

		image_0 = cv2.imread(images[0], 0)
		h, w = image_0.shape
		h, w = min(h, w), max(h, w)
		rotations = [0] * num_images

		try:
			mng = plt.get_current_fig_manager()
			mng.set_window_title('Chequerboard corners detection')
		except Exception:
			pass

		tag_print('start', f'Starting camera calibration using images in folder [{frames_folder}]\n')
		tag_print('info', f'Camera model: {camera_model}\n')

		for i, fname in enumerate(images):
			img_gray = cv2.imread(fname, 0)

			if img_gray.shape != (h, w):
				if img_gray.shape == (w, h):
					tag_print('info', f'Rotating image {i+1}/{num_images}')
					img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)
					rotations[i] = 1
				else:
					tag_print('error', f'All images in the folder [{frames_folder}] must be of the same size!')
					exit_message()

			ret, corners = cv2.findChessboardCorners(img_gray, board_size)
			plt.cla()

			if ret:
				tag_print('success', f'Detected corners in image {i+1}/{num_images}')
				ret_list[i] = ret
				objpoints.append(objp)
				corners_subpixel = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
				imgpoints.append(corners_subpixel)
				good_image_names.append(image_names[i])
			else:
				tag_print('failed', f'Corner detection failed in image {i+1}/{num_images}')

			if ret:
				xs = corners_subpixel[:, 0, 0].tolist()
				ys = corners_subpixel[:, 0, 1].tolist()
				plt.scatter(xs, ys, facecolors='none', edgecolors='r')

			plt.imshow(img_gray, cmap='gray')
			plt.title(f'Image {i+1}/{num_images}')
			plt.axis('off')
			plt.draw()
			plt.pause(0.01)

		plt.pause(1.0)
		plt.close()

		print()
		tag_print('info', 'Calculating camera intrinsics and distortion coefficients... ')
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None, None, None, flags=calibration_flags)
		tag_print('info', 'DONE!\n')

		mean_img_errors = []

		j = 0
		for ret in ret_list:
			if ret is not None:
				reprojected_points, _ = cv2.projectPoints(objpoints[j], rvecs[j], tvecs[j], mtx, dist)
				s = cv2.norm(imgpoints[j], reprojected_points, cv2.NORM_L2) / np.sqrt(len(reprojected_points))
				mean_img_errors.append(s)
				j += 1
			else:
				mean_img_errors.append(np.nan)

		mean_error = np.nanmean(mean_img_errors)
		stdev_error = np.nanstd(mean_img_errors)
		good_images = np.count_nonzero(~np.isnan(mean_img_errors))

		np.savetxt(f'{frames_folder}/ret_list.txt', mean_img_errors, fmt='%.4f')

		tag_print('info',  f'Mean reprojection error = {mean_error:.3f} px')

		mtx_scaled = mtx / w

		print('\nCamera matrix (f=F/W, c=C/W):')
		print(f'fx = {mtx_scaled[0, 0]:.8f}')
		print(f'fy = {mtx_scaled[1, 1]:.8f}')
		print(f'cx = {mtx_scaled[0, 2]:.8f}')
		print(f'cy = {mtx_scaled[1, 2]:.8f}')

		print(f'\nk1 = {dist[0, 0]:.8f}')
		print(f'k2 = {dist[0, 1]:.8f}')
		print(f'k3 = {dist[0, 4]:.8f}\n')
		print(f'p1 = {dist[0, 2]:.8f}')
		print(f'p2 = {dist[0, 3]:.8f}')

		plt.bar(list(range(num_images)), mean_img_errors)
		plt.xticks(list(range(num_images)), [s.split('.')[0] for s in image_names], rotation=90)
		plt.ylabel('Reprojection error [px]')
		plt.title(f'Mean reprojection error = {mean_error:.3f} px')
		plt.show()

		if args.output == 1:
			print()
			tag_print('info', f'Writing undistorted images to [{output_folder}]\n')

			for i, iname in enumerate(image_names):
				img_bgr = cv2.imread(images[i])
				if rotations[i] == 1:
					img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)

				dst = cv2.undistort(img_bgr, mtx, dist)
				cv2.imwrite(f'{output_folder}/{iname}', dst)
				tag_print('info', f'Undistorting image {i+1}/{num_images}')

		cfg = configparser.ConfigParser()
		cfg.optionxform = str

		cfg['Camera'] = {'Model': args.model}
		cfg['Calibration'] = {
			'Total images': f'{num_images:d}',
			'Detected patterns': f'{good_images:d}',
			'Failed detections': f'{num_images - good_images:d}',
			'Cheq. W': args.w,
			'Cheq. H': args.h,
			'Mean repr. error [px]': f'{mean_error:.4f}',
			'Stdev repr. error [px]': f'{stdev_error:.4f}',
		}
		cfg['Intrinsics'] = {
			'fx': f'{mtx_scaled[0, 0]:.8f}',
			'fy': f'{mtx_scaled[1, 1]:.8f}',
			'cx': f'{mtx_scaled[0, 2]:.8f}',
			'cy': f'{mtx_scaled[1, 2]:.8f}',
		}
		cfg['Radial'] = {
			'k1': f'{dist[0, 0]:.8f}',
			'k2': f'{dist[0, 1]:.8f}',
			'k3': f'{dist[0, 4]:.8f}',
		}
		cfg['Tangential'] = {
			'p1': f'{dist[0, 2]:.8f}',
			'p2': f'{dist[0, 3]:.8f}',
		}

		with open(f'{frames_folder}/{camera_model}.cpf', 'w', encoding='utf-8-sig') as configfile:
			cfg.write(configfile)

		print()
		tag_print('end', 'Camera calibration complete!')
		exit_message()

	except Exception:
		present_exception_and_exit()