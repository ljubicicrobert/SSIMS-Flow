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
	from os import makedirs, path
	from class_console_printer import Console_printer, tag_string, unix_path, tag_print
	from class_progress_bar import Progress_bar
	from class_timing import Timer, time_hms
	from glob import glob
	from inspect import getfullargspec
	from feature_tracking import fresh_folder
	from filters import *

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


# Has to remain here
def remove_background(img, num_frames_background=10):
	num_frames_background = int(num_frames_background)
	h, w = img.shape[:2]

	if len(img_list) < num_frames_background:
		num_frames_background = len(img_list)

	step = len(img_list) // num_frames_background
	img_back_path = '{}/../median_{}.{}'.format(path.dirname(img_list[0]), num_frames_background, ext)

	if path.exists(img_back_path):
		back = cv2.imread(img_back_path)
	else:
		stack = np.ndarray([h, w, 3, num_frames_background], dtype='uint8')

		for i in range(num_frames_background):
			stack[:, :, :, i] = cv2.imread(img_list[i*step])

		back = np.median(stack, axis=3)
		cv2.imwrite(img_back_path, back)

	return cv2.subtract(back.astype('uint8'), img.astype('uint8'))


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
			input('\nPress ENTER/RETURN key to exit...')
			exit()

		section = 'Enhancement'

		frames_folder = unix_path(cfg[section]['Folder'])
		results_folder = unix_path('{}/enhancement'.format(cfg['Project settings']['Folder']))
		ext = cfg[section]['Extension']
		
		img_list = glob('{}/*.{}'.format(frames_folder, ext))
		num_frames = len(img_list)
		filters_data = np.loadtxt(results_folder + '/filters.txt', dtype='str', delimiter='/', ndmin=2)

		progress_bar = Progress_bar(total=num_frames, prefix=tag_string('info', 'Frame '))
		console_printer = Console_printer()

		legend = 'Filters:'

		tag_print('start', 'Frame filtering\n')
		tag_print('info', 'Filtering frames from folder [{}]'.format(frames_folder))
		tag_print('info', 'Filters to apply:')
		for f in filters_data:
			func_args_names = globals()[f[0]]
			func_args = getfullargspec(func_args_names)[0][1:] if f[1] != '' else []
			arg_values = ['{}={}'.format(p, v) for p, v in zip(func_args, f[1].split(','))]
			filter_text = '{}: {}'.format(f[0], ', '.join(arg_values if f[1] != '' else ''))
			legend += '\n    ' + filter_text
			print(' ' * 11, filter_text)
		print()

		fresh_folder(results_folder, ext='avi')
		fresh_folder(results_folder, ext=ext)
		timer = Timer(total_iter=num_frames)

		if not path.exists(results_folder):
			makedirs(results_folder)

		for j in range(len(img_list)):
			colorspace = 'rgb'
			img_path = img_list[j]
			img = cv2.imread(img_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			img = apply_filters(img, filters_data)
			img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

			cv2.imwrite('{}/{}'.format(results_folder, path.basename(img_path)), img_bgr)

			timer.update()
			console_printer.add_line(progress_bar.get(j))
			console_printer.add_line(
				tag_string('info', 'Frame processing time = {:.3f} sec'
					.format(timer.interval())
				)
			)
			console_printer.add_line(
				tag_string('info', 'Elapsed time = {} hr {} min {} sec'
					.format(*time_hms(timer.elapsed()))
				)
			)
			console_printer.add_line(
				tag_string('info', 'Remaining time = {} hr {} min {} sec'
					.format(*time_hms(timer.remaining()))
				)
			)
			console_printer.overwrite()

		print()
		tag_print('end', 'Filtering complete!')
		tag_print('end', 'Results available in folder [{}]'.format(results_folder))
		print('\a')
		input('\nPress ENTER/RETURN to exit...')
	
	except Exception as ex:
		print()
		tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
		print('\n{}'.format(format_exc()))
		input('\nPress ENTER/RETURN key to exit...')
