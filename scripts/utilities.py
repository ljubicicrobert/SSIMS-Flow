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

def cfg_get(cfg, section, name, type, default=None):
    try:
        s = cfg[section][name]
        if type == str:
            return s
        else:
            return type(s)
    except KeyError:
        if default is not None:
            return default
        else:
            raise ValueError('Value [{}] not found for key [{}] and default value not provided!'.format(name, section))
