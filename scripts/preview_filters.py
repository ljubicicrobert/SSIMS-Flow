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
    from filter_frames import *     # Chain importing filters.py
    from os import remove
    import shutil

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()


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
            input('\nPress ENTER/RETURN key to exit...')
            exit()

        section = 'Enhancement'

        frames_folder = unix_path(cfg[section]['Folder'])
        results_folder = unix_path('{}/enhancement'.format(cfg['Project settings']['Folder']))
        ext = cfg[section]['Extension']
        
        save_path_original = unix_path(r'{}/original.{}'.format(cfg['Project settings']['Folder'], ext))
        save_path_filtered = unix_path(r'{}/preview.{}'.format(cfg['Project settings']['Folder'], ext))
        if path.exists(save_path_filtered):
            remove(save_path_filtered)
        
        img_list = glob('{}/*.{}'.format(frames_folder, ext))
        num_frames = len(img_list)
        filters_data = np.loadtxt(results_folder + '/filters_preview.txt', dtype='str', delimiter='/', ndmin=2)

        img_path = img_list[args.i]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        img = apply_filters(img, filters_data)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        shutil.copyfile(img_path, save_path_original)
        cv2.imwrite(save_path_filtered, img_bgr)

    except Exception as ex:
        print()
        tag_print('exception', 'Import failed! \n')
        print('\n{}'.format(format_exc()))
        input('\nPress ENTER/RETURN key to exit...')
        exit()
