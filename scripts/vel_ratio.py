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


L1 = 1.00
L2 = 0.67
L3 = 0.33


def slope(x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1)


def intercept(x1, y1, x2, y2):
    return y1 - x1*slope(x1, y1, x2, y2)


def ab(x1, y1, x2, y2):
    return slope(x1, y1, x2, y2), intercept(x1, y1, x2, y2)


def C1(ratio: float) -> float:
    if ratio <= L2:
        return 0.0
    elif ratio >= L1:
        return 1.0
    else:
        a, b = ab(L2, 0, L1, 1)
        return a*ratio + b


def C2(ratio: float) -> float:
    if ratio <= L3 or ratio >= L1:
        return 0.0
    elif ratio <= L2:
        a, b = ab(L3, 0, L2, 1)
        return a*ratio + b
    else:
        a, b = ab(L2, 1, L1, 0)
        return a*ratio + b


def C3(ratio: float) -> float:
    if ratio <= L3:
        return 1.0
    elif ratio >= L2:
        return 0.0
    else:
        a, b = ab(L3, 1, L2, 0)
        return a*ratio + b


def vel_ratio(t1, t2, t3):
    if t2 != t1:
        ratio = (t3 - t2) / (t2 - t1)
    else:
        ratio = L1

    c1 = C1(ratio)
    c2 = C2(ratio)
    c3 = C3(ratio)

    return c1 * t1 + c2 * t2 + c3 * t3


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    xs = [0, L3, L2, L1, L1+L3]
    y1 = [C1(x) for x in xs]
    y2 = [C2(x) for x in xs]
    y3 = [C3(x) for x in xs]

    plt.plot(xs, y1, 'r')
    plt.plot(xs, y2, 'g')
    plt.plot(xs, y3, 'b')
    plt.legend(['C0', 'C1', 'C2'])
    plt.xlabel('R [-]')
    plt.ylabel('C [-]')
    plt.xticks([L3, L2, L1], ['L1=0.33', 'L2=0.67', 'L3=1.00'])
    plt.show()
