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
	from math import log
	from class_console_printer import Console_printer, tag_string, tag_print, unix_path
	from class_progress_bar import Progress_bar
	from feature_tracking import fresh_folder

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


MAX_FRAMES_DEFAULT = 60**3


def get_camera_parameters(path: str) -> tuple:
	"""
	Retreives camera parameters from .cpf file (INI format).

	:param path:    Path as string.
	:return:        Camera matrix and distortion coeffs vector.
	"""

	cp = configparser.ConfigParser()
	cp.optionxform = str
	cp.read(path, encoding='utf-8-sig')

	sec = 'Intrinsics'
	fx = float(cp.get(sec, 'fx'))
	fy = float(cp.get(sec, 'fy'))
	cx = float(cp.get(sec, 'cx'))
	cy = float(cp.get(sec, 'cy'))

	sec = 'Radial'
	k1 = float(cp.get(sec, 'k1'))
	k2 = float(cp.get(sec, 'k2'))
	k3 = float(cp.get(sec, 'k3', fallback='0'))

	sec = 'Tangential'
	p1 = float(cp.get(sec, 'p1'))
	p2 = float(cp.get(sec, 'p2'))

	camera_matrix = np.array([[fx, 0,  cx],
							  [0,  fy, cy],
							  [0,  0,  1]])

	distortion = np.array([k1, k2, p1, p2, k3])

	return camera_matrix, distortion


def videoToFrames(video: str, folder='.', frame_prefix='', ext='jpg',
				  start=0, start_num=0, end=MAX_FRAMES_DEFAULT, qual=95, scale=None, step=1, interp=cv2.INTER_CUBIC,
				  camera_matrix=None, dist=None, cp=None, pb=None, crop=None, verbose=False,) -> bool:
	"""
	Extracts all num_frames from a video to separate images. Optionally writes to a specified folder,
	creates one if it does not exist. If no folder is specified, it writes to the parent folder.
	Option to choose an image file prefix and extension. Returns True (if success) or False (if error).

	:param video: 			Path to the video. Should be a string.
	:param folder: 			Folder name to put the image files in. Default is '.', so it writes all the num_frames to the
							parent folder. Creates a folder if it does not already exist.
	:param frame_prefix:	Default prefix for the image files. Default is 'frame'.
	:param ext: 			Extension for the image files. Should be a string, without the leading dot. Default is 'jpg'.
	:param start:			Starting frame. Default is 0, i.e. the first frame.
	:param start_num:		Frame numbering sequence start. Default is 0.
	:param end:				End frame MAX_FRAMES_DEFAULT global.
	:param qual:			Output image quality in range (1-100). Default is 95.
	:param scale:			Scale parameter for the output images. Default is None, which preserves the original size.
	:param step:			Step to export every Nth frame. Default is 1, i.e., every frame.
	:param interp:			Interpolation algorithm for image resizing from cv2 package. Default is cv2.INTER_CUBIC.
	:param camera_matrix:	Camera matrix. If None, no camera rectification will be performed.
							Note that parameters [fx, fy, cx, cy] are divided by image size so they are dimensionless here.
	:param dist:			Camera distortion parameters. If None, no camera rectification will be performed.
	:param pb:				Progress bar writer object.
	:param cp:				Console printer writer object.
	:param verbose: 		Whether to use a verbose output. Default is False.
	:return: 				True (if success) or False (if error).
	"""

	if not path.exists(video):
		tag_print('error', 'Video file not found at {}'.format(video))
		input('\nPress ENTER/RETURN to exit...')
		exit()

	vidcap = cv2.VideoCapture(video)
	num_frames_total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
	success, image = vidcap.read()

	height, width = image.shape[:2]

	if verbose:
		tag_print('start', 'Starting frame extraction')
		print()
		tag_print('info', 'Extraction of frames from [{}] starting from frame {}/{}'.format(video, start, num_frames_total))
		tag_print('info', 'Extension: {}'.format(ext))
		tag_print('info', 'Quality: {}'.format(qual))
		tag_print('info', 'Scale: {:.2f}'.format(scale))
		tag_print('info', 'Step: {}'.format(step))
		if crop:
			tag_print('info', 'Crop: X={}..{}, Y={}..{}'.format(*crop))
		else:
			tag_print('info', 'Crop: ')
		print()

	i = start
	j = start_num
	extracted_frames = 0
	success = True
	extracted_size = 0

	if not end:
		end = MAX_FRAMES_DEFAULT

	frame_range = end - start - 1

	if cp and pb:
		pb.set_total(frame_range + 1)

	num_len = int(log(end-start, 10)) + 1
	fresh_folder(folder, ext=ext)

	while success and i < end:  # If new frame exists

		if folder is None:
			n = str(j).zfill(num_len)
			save_str = '{}{}.{}'.format(frame_prefix, n, ext)
		else:
			n = str(j).zfill(num_len)
			save_str = '{}/{}{}.{}'.format(folder, frame_prefix, n, ext)

		if camera_matrix and dist:
			camera_matrix[0, 0] = camera_matrix[0, 0] * width			# fx
			camera_matrix[1, 1] = camera_matrix[1, 1] * width			# fy
			camera_matrix[0, 2] = camera_matrix[0, 2] * width			# cx
			camera_matrix[1, 2] = camera_matrix[1, 2] * width		    # cy
			image = cv2.undistort(image, camera_matrix, dist)

		if crop:
			xs, xe, ys, ye = crop
			image = image[ys: ye, xs: xe]

		if scale and scale != 1.0:
			image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp)

		if verbose:
			if cp and pb:
				cp.single_line(pb.get(int(i - start)))
			else:
				tag_print('info', 'Extracting frame: {}'.format(int(i)))

		if ext.lower() in ['jpg', 'jpeg']:
			cv2.imwrite(save_str, image,
						[int(cv2.IMWRITE_JPEG_QUALITY), qual])
		elif ext.lower() == 'png':
			cv2.imwrite(save_str, image,
						[int(cv2.IMWRITE_PNG_COMPRESSION), 9 - int(0.09 * qual)])
		elif ext.lower() == 'webp':
			cv2.imwrite(save_str, image,
						[int(cv2.IMWRITE_WEBP_QUALITY), qual + 1])
		else:
			cv2.imwrite(save_str, image)

		extracted_size += path.getsize(save_str) / (1024 * 1024)

		if step != 1:
			vidcap.set(cv2.CAP_PROP_POS_FRAMES, i + step)

		success, image = vidcap.read()

		i = int(i + step)
		j = int(j + step)
		extracted_frames += 1

	if verbose:
		print()
		tag_print('end', 'Images written to folder [{}]'.format(folder))
		tag_print('end', 'Total number of extracted images is {}'.format(extracted_frames))
		tag_print('end', 'Total size of extracted images is {:.2f} MB'.format(extracted_size))

	vidcap.release()  # Clear video from memory

	if i == num_frames_total or i == end + 1:
		return True
	else:
		return False


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
			tag_print('error', 'There was a problem reading the configuration file!\nCheck if project has valid configuration.')
			exit()

		project_folder = unix_path(cfg.get('Project settings', 'Folder'))
		frames_folder = '{}/frames'.format(project_folder)

		section = 'Frames'

		video_path = unix_path(cfg.get(section, 'VideoPath'))
		remove_distortion = int(cfg.get(section, 'Undistort'))
		frame_ext = cfg.get(section, 'Extension', fallback='jpg')
		frame_qual = int(cfg.get(section, 'Quality', fallback='95'))
		frame_scale = float(cfg.get(section, 'Scale', fallback='1.0'))
		frame_step = int(cfg.get(section, 'Step', fallback='1'))
		unpack_start = int(cfg.get(section, 'Start', fallback='0'))

		vidcap = cv2.VideoCapture(video_path)
		w  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
		h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		num_frames_total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
		vidcap.release()

		unpack_end = int(cfg.get(section, 'End', fallback=str(num_frames_total)))
		unpack_end = min(unpack_end, num_frames_total)

		try:
			crop_labels = ['X_start', 'X_end', 'Y_start', 'Y_end']
			crop_str = cfg.get(section, 'Crop', fallback='')
			crop_limits = [int(c) for c in crop_str.replace(' ', '').split(',')]
			skip_crop_prompt = False

			for i, c in enumerate(crop_limits):
				if c < 0:
					tag_print('error', 'Crop limit {} (={}) is lower than 0!'.format(crop_labels[i], crop_limits[i]))
					skip_crop_prompt = True
			if crop_limits[0] >= crop_limits[1]:
				tag_print('error', 'Crop limit X_start (={}) is larger than or equal to crop limit X_end (={})!'.format(crop_limits[0], crop_limits[1]))
				skip_crop_prompt = True
			if crop_limits[2] >= crop_limits[3]:
				tag_print('error', 'Crop limit Y_start (={}) is larger than or equal to crop limit Y_end (={})!'.format(crop_limits[2], crop_limits[3]))
				skip_crop_prompt = True
			if max(crop_limits[0], crop_limits[1]) > w:
				tag_print('error', 'Crop limits X ([{}, {}]) out of image bounds: width = {}!'.format(crop_limits[0], crop_limits[1], w))
				skip_crop_prompt = True
			if max(crop_limits[2], crop_limits[3]) > h:
				tag_print('error', 'Crop limits Y ([{}, {}]) out of image bounds: height = {}!'.format(crop_limits[2], crop_limits[3], h))
				skip_crop_prompt = True

			if skip_crop_prompt:
				input = input('Do you wish to continue unpacking video without cropping? [y/n]').lower()
				if input == 'y':
					crop_limits = ''
				else:
					print()
					tag_print('end', 'Terminated by user!')
					input('\nPress ENTER/RETURN key to exit...')
					exit()
		except Exception as ex:
			pass

		camera_matrix, distortion = get_camera_parameters('{}/camera_parameters.cpf'.format(project_folder))\
										if remove_distortion else None, None

		console_printer = Console_printer()
		progress_bar = Progress_bar(total=1, prefix=tag_string('info', 'Extracting frame '))

		videoToFrames(video=		 video_path,
					  folder=		 frames_folder,
					  ext=			 frame_ext,
					  qual=			 frame_qual,
					  scale=		 frame_scale,
					  step=			 frame_step,
					  start=		 unpack_start,
					  end=			 unpack_end,
					  camera_matrix= camera_matrix,
					  dist=			 distortion,
					  pb=			 progress_bar,
					  cp=			 console_printer,
					  crop=			 crop_limits,
					  verbose=		 True,
					  )

		print('\a')
		input('\nPress ENTER/RETURN to exit...')

	except Exception as ex:
		print()
		tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
		print('\n{}'.format(format_exc()))
		input('\nPress ENTER/RETURN key to exit...')
