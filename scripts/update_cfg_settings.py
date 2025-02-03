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

# Use this script to batch update certain project parameters for multiple projects.

import configparser
from class_console_printer import unix_path


def update_parameter(folder_path, section, parameter, new_value):
	project_file_path = unix_path(fr'{folder_path}/project.ssims')
	config = configparser.RawConfigParser()
	config.optionxform = str
	config.read(project_file_path, encoding='utf-8-sig')
	
	if not section in config:
		print(fr'Section [{section}] not found in project file [{project_file_path}]')
	elif not parameter in config[section]:
		print(fr'Parameter [{parameter}] not found in section [{section}] of the project file [{project_file_path}]')
	else:
		print(fr'Updating parameter [{parameter}] in section [{section}] to value [{new_value}] for project {unix_path(folder_path)}')

		config[section][parameter] = new_value

		with open(project_file_path, 'w', encoding='utf-8-sig') as configfile:
			config.write(configfile)


folders = [
	fr'path-to-project-folder-1',
	fr'path-to-project-folder-2',
	fr'path-to-project-folder-3',
]

num_folders = len(folders)

# Change the variables in the update_parameter() function to your liking.
# Copy the function to change multiple parameters at once.
# Make sure that section, parameter, and value are all passed as strings
for f in folders:
	update_parameter(f, 'section_name1', 'parameter_name1', 'new_value1')
	update_parameter(f, 'section_name2', 'parameter_name2', 'new_value2')
