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

from math import log10, floor


class Progress_bar:
    """
    Simple progress bar for console.
    Requires number of iterations to be known at creation time.
    Use .get(iter) to update the bar and return it as string.
    Use .show(iter) to update the bar and print it to console.
    """
    
    def __init__(self, total: int, prefix='Progress ', length=50, char_done='#', char_remain=' ', suffix_fmt='{:.1f}%'):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.char_done = char_done
        self.char_remain = char_remain
        self.suffix_fmt = suffix_fmt
        self.bar = '[{}]'.format(char_remain * length)
        self.percent = 0
        self.num_digits = floor(log10(total)) + 1

    def set_total(self, total):
        self.total = total
        self.num_digits = floor(log10(total)) + 1

    def update_bar(self, iteration: int):
        self.percent = (iteration + 1)/self.total * 100
        if self.percent > 100:
            self.percent = 100
        len_done = int(self.percent / 100 * self.length)
        len_remain = self.length - len_done
        self.bar = '{}/{} [\033[32m{}\033[0m\033[31m{}\033[0m]'.format(str(iteration+1).rjust(self.num_digits, ' '),
                                                                       self.total,
                                                                       self.char_done * len_done,
                                                                       self.char_remain * len_remain)

    def show(self, iteration: int):
        self.update_bar(iteration)
        print('{}{} '.format(self.prefix, self.bar) + self.suffix_fmt.format(self.percent))

    def get(self, iteration: int) -> str:
        self.update_bar(iteration)
        return '{}{} '.format(self.prefix, self.bar) + self.suffix_fmt.format(self.percent)


if __name__ == '__main__':
    import time

    pb = Progress_bar(total=100)

    for i in range(100):
        time.sleep(0.01)
        pb.show(i)
