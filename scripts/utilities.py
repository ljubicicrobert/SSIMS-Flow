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

from os import path, makedirs, remove
from glob import glob
from traceback import format_exc
from class_console_printer import tag_print


def cfg_get(cfg, section, name, type, default=None):
	try:
		s = cfg[section][name]
		if s == '':
			return default
		if type == str:
			return s
		else:
			return type(s)
	except KeyError:
		if default is not None:
			return default
		else:
			raise ValueError(f'Value [{name}] not found for key [{section}] and default value not provided!')


def exit_message():
	input('\nPress ENTER/RETURN key to exit...')
	exit()


def present_exception_and_exit(message='An exception has occurred! See traceback below:'):
	print()
	tag_print('exception', message)
	print()
	print(format_exc())
	exit_message()


def fresh_folder(folder_path, ext='*', exclude=list()):
	if not path.exists(folder_path):
		makedirs(folder_path)
	else:
		files = glob(f'{folder_path}/*.{ext}')
		for f in files:
			if path.basename(f) not in exclude:
				remove(f)
