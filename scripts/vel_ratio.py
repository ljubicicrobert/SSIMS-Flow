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


L0 = 1.00
L1 = 0.67
L2 = 0.33


def slope(x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1)


def intercept(x1, y1, x2, y2):
    return y1 - x1*slope(x1, y1, x2, y2)


def ab(x1, y1, x2, y2):
    return slope(x1, y1, x2, y2), intercept(x1, y1, x2, y2)


def C0(ratio: float) -> float:
    if ratio <= L1:
        return 0.0
    elif ratio >= L0:
        return 1.0
    else:
        a, b = ab(L1, 0, L0, 1)
        return a*ratio + b


def C1(ratio: float) -> float:
    if ratio <= L2 or ratio >= L0:
        return 0.0
    elif ratio <= L1:
        a, b = ab(L2, 0, L1, 1)
        return a*ratio + b
    else:
        a, b = ab(L1, 1, L0, 0)
        return a*ratio + b


def C2(ratio: float) -> float:
    if ratio <= L2:
        return 1.0
    elif ratio >= L1:
        return 0.0
    else:
        a, b = ab(L2, 1, L1, 0)
        return a*ratio + b


def vel_ratio(t0, t1, t2):
    if t1 != t0:
        ratio = (t2 - t1) / (t1 - t0)
    else:
        ratio = L0

    c0 = C0(ratio)
    c1 = C1(ratio)
    c2 = C2(ratio)

    return c0 * t0 + c1 * t1 + c2 * t2


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    xs = [0, L2, L1, L0, L0+L2]
    y0 = [C0(x) for x in xs]
    y1 = [C1(x) for x in xs]
    y2 = [C2(x) for x in xs]

    plt.plot(xs, y0, 'r')
    plt.plot(xs, y1, 'g')
    plt.plot(xs, y2, 'b')
    plt.legend(['C0', 'C1', 'C2'])
    plt.xlabel('R [-]')
    plt.ylabel('C [-]')
    plt.xticks([L2, L1, L0], ['L0=0.33', 'L1=0.67', 'L2=1.00'])
    plt.show()
