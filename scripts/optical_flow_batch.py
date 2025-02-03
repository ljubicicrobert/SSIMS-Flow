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

# Use this script to batch process velocity estimation using optical flow for multiple projects.
# This can be useful when reprocessing existing projects, to avoid doing it manually from the GUI.

import subprocess

# Add the project folder paths in this list, remove the placeholder values
folders = [
    fr'path-to-project-folder-1',
    fr'path-to-project-folder-2',
    fr'path-to-project-folder-3',
]

# Set to True if you want to also produce the profile data, make sure that the ProfileSource is set correctly in project.ssims
generate_profile = True

# No need to change anything from this point onward, unless you know what you're doing
num_folders = len(folders)

for i, f in enumerate(folders):
    project_file_path = fr'{f}/project.ssims'
    print(fr'Velocity estimation for project {f} [{i+1}/{num_folders}]')
    subprocess.call(['python', 'optical_flow.py', '--cfg', project_file_path, '--quiet', '1'])

    if generate_profile:
        print(fr'Generating profile data for project {f} [{i+1}/{num_folders}]')
        subprocess.call(['python', 'profile_data.py', '--cfg', project_file_path, '--quiet', '1'])
