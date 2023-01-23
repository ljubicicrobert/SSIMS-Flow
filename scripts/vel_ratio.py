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


LIM_1 = 1.00
LIM_2 = 0.67
LIM_3 = 0.33


def slope(x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1)


def intercept(x1, y1, x2, y2):
    return y1 - x1*slope(x1, y1, x2, y2)


def ab(x1, y1, x2, y2):
    return slope(x1, y1, x2, y2), intercept(x1, y1, x2, y2)


def C1(ratio: float) -> float:
    if ratio <= LIM_2:
        return 0.0
    elif ratio >= LIM_1:
        return 1.0
    else:
        a, b = ab(LIM_2, 0, LIM_1, 1)
        return a*ratio + b


def C2(ratio: float) -> float:
    if ratio <= LIM_3 or ratio >= LIM_1:
        return 0.0
    elif ratio <= LIM_2:
        a, b = ab(LIM_3, 0, LIM_2, 1)
        return a*ratio + b
    else:
        a, b = ab(LIM_2, 1, LIM_1, 0)
        return a*ratio + b


def C3(ratio: float) -> float:
    if ratio <= LIM_3:
        return 1.0
    elif ratio >= LIM_2:
        return 0.0
    else:
        a, b = ab(LIM_3, 1, LIM_2, 0)
        return a*ratio + b


def vel_ratio(v1, v2, v3):
    if v2 != v1:
        ratio = (v3 - v2) / (v2 - v1)
    else:
        ratio = LIM_1

    c1 = C1(ratio)
    c2 = C2(ratio)
    c3 = C3(ratio)

    return c1 * v1 + c2 * v2 + c3 * v3
