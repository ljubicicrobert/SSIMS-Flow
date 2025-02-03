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
	from os import path, remove
	from class_console_printer import unix_path, tag_print
	from glob import glob
	from utilities import cfg_get, exit_message, present_exception_and_exit
	from filters import *
	
	import shutil

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--cfg', type=str, help='Path to configuration file')
		parser.add_argument('-i', type=int, help='Image number', default=0)
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
		frames_folder = unix_path(cfg_get(cfg, section, 'Folder', str, f'{project_folder}/frames'))
		results_folder = f'{project_folder}/enhancement'
		ext = cfg_get(cfg, section, 'Extension', str, 'jpg')
		
		save_path_original = f'{project_folder}/original.{ext}'
		save_path_filtered = f'{project_folder}/preview.{ext}'
		if path.exists(save_path_filtered):
			remove(save_path_filtered)
		
		img_list = glob(f'{frames_folder}/*.{ext}')
		num_frames = len(img_list)
		filters_data = np.loadtxt(results_folder + '/filters_preview.txt', dtype='str', delimiter='/', ndmin=2)

		img_path = img_list[args.i]
		img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

		img = apply_filters(img, filters_data, img_list, ext)
		img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		
		shutil.copyfile(img_path, save_path_original)
		cv2.imwrite(save_path_filtered, img_bgr)

	except Exception as ex:
		print('[EXCEPTION] An exception has occurred! See traceback below: \n')
		print(format_exc())
		print('[END_TRACEBACK]')
		exit_message()
