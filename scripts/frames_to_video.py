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
	from class_console_printer import Console_printer, tag_print, tag_string, unix_path
	from class_progress_bar import Progress_bar
	from os import path, listdir
	from datetime import timedelta
	from glob import glob

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


MAX_FRAMES_DEFAULT = 60**3  # 60 minutes at 60fps


def framesToVideo(output, folder='.', ext='jpg', codec='MJPG', fps=30.00, scale=1.0,
				  max_frames=MAX_FRAMES_DEFAULT, interp=cv2.INTER_LINEAR, size_adj=False,
				  pb=None, cp=None, verbose=True) -> bool:
	"""
	Convert individual images to a video. Images are specified using prefix and extension and the code iterates through
	corresponding images in a specified folder. Several video codecs are available, scaling is provided, FPS
	and a maximum number of num_frames can be set.

	:param output:		Video output filename as a string. Enter without the extension as only .avi is available.
	:param folder:		Folder containing the images to be written to a video. The video will be created in the same
						folder. Default is the parent folder, i.e. '.'.
	:param prefix:		String prefix of files to be written to a video. Default is 'frame'.
	:param ext:			Extension of the image files. Should be a string, without the leading dot. Default is 'jpg'.
	:param codec:		Codec for the output video. Default is MJPG which has the best quality but a large file size.
						Other available are DIVX, XVID, WMV1 and WMV2, which are all smaller in size but provide worse quality video.
	:param fps:			FPS count for the output video. Default is 30.
	:param scale:		Scale factor for the video. Linear interpolation is used if scale != 1.0. Default is 1.0.
	:param max_frames:	Maximum number of num_frames to write to video. Default is defined by MAX_FRAMES_DEFAULT global variable.
						Be careful not to exceed the PC RAM limit since the video is stored in RAM until the final frame is encoded.
	:param interp:		Interpolation algorithm for image resizing from cv2 package. Default is cv2.INTER_LINEAR.
	:param size_adj:	Whether to adjust the size of all frames to the size of the first frame. Default is False.
	:param pb:			Progress bar object.
	:param cp:			Console printer object.
	:param verbose:		Whether to use a verbose output. Default is False.
	:return:			True is success, raises an error if something is wrong.
	"""

	# Check for allowed codec
	# codec = 'H264' if '264' in codec else ('HEVC' if '265' in codec else codec)

	assert codec in ('MJPG', 'DIVX', 'XVID', 'WMV1', 'WMV2')

	# Get the first image shape
	for filename in listdir(folder):
		if filename.endswith('.' + ext):
			image = cv2.imread(folder + '/' + filename)
			height, width, *_ = image.shape
			break

	# Verify shape available
	try:
		height, width
	except [NameError, ValueError]:
		tag_print('error', 'Could not obtain image shape. Check frames folder path or file type!')

	# TODO: Should I include other extensions?
	saveStr = '{}/{}.avi'.format(folder, output)

	out = cv2.VideoWriter(saveStr, cv2.VideoWriter_fourcc(*codec), fps,
						  (int(scale * width), int(scale * height)))

	if verbose:
		tag_print('start', 'Creating video from frames')
		print()
		tag_print('info', 'Encoding frames to video from folder [{}]'.format(folder))
		tag_print('info', 'Writing results to [{}]'.format(saveStr))
		tag_print('info', 'Codec: {}'.format(codec))
		tag_print('info', 'Framerate: {:.2f}'.format(fps))
		tag_print('info', 'Scale: {:.2f}'.format(scale))
		print()

	i = 0

	if max_frames is None:
		max_frames = MAX_FRAMES_DEFAULT  # 60 minutes at 60fps

	# Go through all frames
	for filename in listdir(folder):
		if filename.endswith('.' + ext) and i < max_frames:
			try:
				image = cv2.imread(folder + '/' + filename)
				h, w = image.shape[:2]

				# Happens sometimes with oddly packed videos
				if h != height or w != width:
					if not size_adj:
						tag_print('error', 'Frame {} does not have the same size as the first frame!'.format(i))
						tag_print('error', 'OpenCV Video writer requires all frames to be the same size!')
						input('\nPress ENTER/RETURN to exit...')
						exit()
					else:
						tag_print('warning', 'Adjusting the size of frame {} to {}x{} px'.format(i, width, height))
						cv2.resize(image, [height, width], interpolation=interp)

				if scale != 1.0:
					image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp)

				out.write(image)

			except Exception:
				# Release output video from memory
				tag_print('error', 'Something went wrong, releasing video file from memory...')
				out.release()

			if verbose:
				if cp and pb:
					cp.single_line(pb.get(i))
				else:
					tag_print('info', 'Writting frame {}'.format(i))

			i += 1

	# Release output video from memory
	out.release()

	if verbose:
		print()
		tag_print('end', 'Video written to {} using {} codec'.format(saveStr, codec))
		tag_print('end', 'Total number of frames written is {}'.format(i))
		tag_print('end', 'Total duration of the video is {}'.format(timedelta(seconds=(i / fps))))
		size = path.getsize(saveStr) / (1024 * 1024)
		tag_print('end', 'Total size of the file is {:.2f} MB'.format(size))

	return True


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--cfg', type=str, help='Path to configuration file')
		args = parser.parse_args()

		cfg = configparser.ConfigParser()
		cfg.optionxform = str
		section = 'Create video'

		try:
			cfg.read(args.cfg, encoding='utf-8-sig')
		except Exception:
			tag_print('error', 'There was a problem reading the configuration file!')
			tag_print('error', 'Check if project has valid configuration.')
			exit()

		interp_methods = {0: cv2.INTER_LINEAR,
						  1: cv2.INTER_CUBIC,
						  2: cv2.INTER_LANCZOS4}

		video_name = cfg.get(section, 'VideoName')
		frames_folder = unix_path(cfg.get(section, 'Folder'))
		frames_ext = cfg.get(section, 'Extension', fallback='jpg')
		video_fps = float(cfg.get(section, 'Framerate'))
		video_codec = cfg.get(section, 'Codec', fallback='MJPG')
		video_scale = float(cfg.get(section, 'Scale', fallback='1.0'))
		scale_interp = interp_methods[int(cfg.get(section, 'Interpolation', fallback='0'))]

		frames_list = glob('{}/*.{}'.format(frames_folder, frames_ext))
		num_frames = min(len(frames_list), MAX_FRAMES_DEFAULT)

		progress_bar = Progress_bar(total=num_frames, prefix=tag_string('info', 'Writing frame '))
		console_printer = Console_printer()

		framesToVideo(video_name,
					  folder=	frames_folder,
					  ext=		frames_ext,
					  codec=	video_codec,
					  fps=		video_fps,
					  scale=	video_scale,
					  interp=	scale_interp,
					  pb=		progress_bar,
					  cp=		console_printer
					  )

		print('\a')
		input('\nPress ENTER/RETURN to exit...')

	except Exception as ex:
		print()
		tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
		print('\n{}'.format(format_exc()))
		input('\nPress ENTER/RETURN key to exit...')