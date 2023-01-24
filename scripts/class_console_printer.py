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

from sys import stdout
from os import name as os_name

# Enable ANSI escape sequences for Windows CMD
if os_name == 'nt':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

ansi_color_codes = {
    'red':     '\033[31m',
    'green':   '\033[32m',
    'yellow':  '\033[33m',
    'blue':    '\033[34m',
    'magenta': '\033[35m',
    'cyan':    '\033[36m',
}


def color_text(string: str, color: str) -> str:
    return ansi_color_codes[color] + string + '\033[0m'


tags = {
    'info':      color_text('[INFO]',      'cyan'),
    'log':       color_text('[LOG]',       'cyan'),
    'success':   color_text('[SUCCESS]',   'green'),
    'start':     color_text('[START]',     'green'),
    'end':       color_text('[END]',       'green'),
    'warning':   color_text('[WARNING]',   'yellow'),
    'error':     color_text('[ERROR]',     'red'),
    'exception': color_text('[EXCEPTION]', 'red'),
    'failed':    color_text('[FAILED]',    'red'),
    'abort':     color_text('[ABORT]',     'red'),
}


def tag_string(tag: str, string: str) -> str:
    return '{} {}'.format(tags[tag], string)


def tag_print(tag: str, string: str, *kwargs):
    print(tag_string(tag, string), *kwargs)


def unix_path(path: str) -> str:
    if path == '':
        return path
    else:
        path = path.replace('\\', '/')
        if path[-1] == '/':
            path = path[:-1]
        return path


# Class for handling console printing
# with overwritting
class Console_printer:
    """
    Handling console printing with line overwritting.
    Use .add_line(line) to keep adding lines to stack, end with
      .console_overwrite() to clear all previous lines and write new ones.
    Use .single_line(line) if you need to refresh a single console line.
    """
    
    first_iter = True

    def __init__(self):
        self.lines = []

    def add_line(self, line: str) -> str:
        self.lines.append(str(line))
        return line

    def reset(self):
        self.first_iter = True
        self.lines = []

    def count(self) -> int:
        return len(self.lines)

    def clear_lines(self, num_lines: int):
        # Move cursor up and overwrite
        # with empty line
        for _ in range(num_lines):
            stdout.write('\033[F')
            stdout.write('\033[K')

    def clear_all(self):
        # If no lines added, prevent
        # falsely changing :first_iter:
        if not self.lines:
            return

        # To avoid clearing console
        # on the first iteration
        if self.first_iter:
            self.first_iter = False
        else:
            self.clear_lines(len(self.lines))

    def write(self):
        for line in self.lines:
            stdout.write(line + '\n')
        
        stdout.flush()
        self.lines = []

    def overwrite(self):
        self.clear_all()
        self.write()


if __name__ == '__main__':
    import time

    cp = Console_printer()

    for i in range(101):
        time.sleep(0.01)

        cp.add_line(i)
        cp.add_line(i*2)
        cp.add_line(i*3)

        cp.overwrite()
