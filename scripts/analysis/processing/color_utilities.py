from numba import njit


@njit
def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    s = (maxc - minc) / maxc if maxc != 0 else 0

    if maxc == minc:
        h = 0
    elif maxc == r:
        h = (g - b) / (maxc - minc)
    elif maxc == g:
        h = 2.0 + (b - r) / (maxc - minc)
    else:
        h = 4.0 + (r - g) / (maxc - minc)

    h = (h * 60) % 360
    if h < 0:
        h += 360

    return h, s, v
