# Place your custom filter function definitions here.
# You can also use the available colorspace conversions and filters from the filters.py.
# Just remember a few things:
#   1. To avoid the problem of circular imports, when trying to use some functions that exist in the filters.py,
#      do not import them at the header of this file. Instead, import them into individual functions after the
#      function header, like shown below.
#   2. All the filter functions MUST both TAKE as input and RETURN a three-channel numpy array!
#   3. If the filter changes the colorspace, make sure to either do it using convert_to_* functions
#      or using global varible colorspace from filters.py. This enables correct conversions when
#      potentially using convert_to_* functions further down the filtering stack.


def custom_negative(img):
    from filters import negative

    return negative(img)
