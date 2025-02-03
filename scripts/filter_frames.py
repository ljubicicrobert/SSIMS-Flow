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
	from filters import *
	from utilities import fresh_folder, cfg_get, exit_message, present_exception_and_exit

	import ctypes

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--cfg', type=str, help='Path to configuration file')
		parser.add_argument('--quiet', type=int, help='Quiet mode for batch processing, no RETURN confirmation on success', default=0)
		args = parser.parse_args()

		cfg = configparser.ConfigParser()
		cfg.optionxform = str

		try:
			cfg.read(args.cfg, encoding='utf-8-sig')
		except Exception:
			tag_print('error', 'There was a problem reading the configuration file!')
			tag_print('error', 'Check if project has valid configuration.')
			exit_message()

		section = 'Enhancement'

		project_folder = unix_path(cfg_get(cfg, 'Project settings', 'Folder', str))
		frames_folder = unix_path(cfg_get(cfg, section, 'Folder', str))
		frames_folder = frames_folder if frames_folder != '' else project_folder + '/frames'
		results_folder = unix_path(f'{project_folder}/enhancement')
		ext = cfg_get(cfg, section, 'Extension', str)
		useOnlySDIFrames = cfg_get(cfg, section, 'UseSDIFramesOnly', int, 0)
		
		img_list = glob(f'{frames_folder}/*.{ext}')

		if useOnlySDIFrames:
			try:
				optimal_window = np.loadtxt(f'{project_folder}/SDI/optimal_frame_window.txt', dtype=int)
				img_list = img_list[optimal_window[0]: optimal_window[1] + 1]
			except Exception:
				MessageBox = ctypes.windll.user32.MessageBoxW

				response = MessageBox(None, f'There was a problem reading the optimal frame window. Was the SDI analysis performed?\n' +
						  				     'Would you like to proceed with filtering all frames?',
											 'Optimal frame window read error', 68)

				if response != 6:
					tag_print('end', 'Filtering aborted!')
					exit_message()

		num_frames = len(img_list)
		filters_data = np.loadtxt(results_folder + '/filters.txt', dtype='str', delimiter='/', ndmin=2)

		progress_bar = Progress_bar(total=num_frames, prefix=tag_string('info', 'Frame '))
		console_printer = Console_printer()

		tag_print('start', 'Frame filtering\n')
		tag_print('info', f'Filtering frames from folder [{frames_folder}]')
		tag_print('info', f'Results folder [{results_folder}]')
		tag_print('info', 'Filters and parameters:')
		for f in filters_data:
			func_args_names = globals()[f[0]]
			func_args = getfullargspec(func_args_names)[0][1:] if f[1] != '' else []
			arg_values = ['{}={}'.format(p, v) for p, v in zip(func_args, f[1].split(','))]
			filter_text = '{}: {}'.format(f[0], ', '.join(arg_values if f[1] != '' else ''))
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

			img = apply_filters(img, filters_data, img_list, ext)
			img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

			cv2.imwrite(f'{results_folder}/{path.basename(img_path)}', img_bgr)

			timer.update()

			console_printer.add_line(progress_bar.get(j))
			console_printer.add_line(tag_string('info', f'Frame processing time = {timer.interval():.3f} sec'))
			he, me, se = time_hms(timer.elapsed())
			console_printer.add_line(tag_string('info', f'Elapsed time = {he} hr {me} min {se} sec'))
			hr, mr, sr = time_hms(timer.remaining())
			console_printer.add_line(tag_string('info', f'Remaining time = {hr} hr {mr} min {sr} sec'))

			console_printer.overwrite()

		print()
		tag_print('end', 'Filtering complete!')
		print('\a')

		if args.quiet == 0:
			exit_message()
	
	except Exception as ex:
		present_exception_and_exit()
