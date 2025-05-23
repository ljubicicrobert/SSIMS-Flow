# Common imports for all scripts
try:
	import cv2
	import numpy as np
	import configparser
	from traceback import format_exc
	from argparse import ArgumentParser
except Exception as ex:
	from utilities import present_exception_and_exit
	present_exception_and_exit('Import failed! See traceback below:')


__package_name__ = 'SSIMS-Flow: Image velocimetry workbench'
__description__ = 'Workbench for obtaining open-channel flow rate from videos.'
__version__ = '0.7.0.0'
__status__ = 'beta'
__date_deployed__ = '2025-05-20'

__author__ = 'Robert Ljubicic @ Faculty of Civil Engineering, University of Belgrade'
__author_email__ = 'rljubicic@grf.bg.ac.rs, ljubicicrobert@gmail.com'
__author_webpage__ = 'https://www.grf.bg.ac.rs/fakultet/pro/e?nid=216'
__project_url__ = 'https://github.com/ljubicicrobert/SSIMS-Flow'
__license__ = 'GPL 3.0'

__header__ = '''
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
			 '''
