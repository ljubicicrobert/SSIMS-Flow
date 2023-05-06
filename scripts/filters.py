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

try:
	from __init__ import *
	from os import path
	from class_console_printer import tag_print
	from CPP.dll_import import DLL_Loader
	from ctypes import c_size_t, c_double

	import scipy.stats as stats

	dll_path = path.split(path.realpath(__file__))[0]
	dll_name = 'CPP/filtering.dll'
	dll_loader = DLL_Loader(dll_path, dll_name)
	cpp_intensity_capping = dll_loader.get_function('void', 'intensity_capping', ['byte*', 'size_t', 'double'])

except Exception as ex:
	print()
	tag_print('exception', 'Import failed! \n')
	print('\n{}'.format(format_exc()))
	input('\nPress ENTER/RETURN key to exit...')
	exit()

colorspaces_list = ['rgb', 'hsv', 'lab', 'grayscale', 'ycrcb']

# V from / > to
color_conv_codes = (
	[[], 	[67], 		[45], 		[7], 		[37]],
	[[71], 	[], 		[71, 45], 	[71, 7], 	[71, 37]],
	[[57], 	[57, 67], 	[], 		[57, 7], 	[57, 37]],
	[[8], 	[8, 67], 	[8, 45], 	[], 		[8, 37]],
	[[39], 	[39, 67], 	[39, 45], 	[39, 7], 	[]]
)

colorspace = 'rgb'


def single_channel(img):
	try:
		a = img[:, :, 0]
		return img[:, :, 0]
	except IndexError:
		return img


def three_channel(img):
	try:
		a = img[:, :, 0]
		return img[:, :, :3]
	except IndexError:
		return cv2.merge([img, img, img])


def convert_img(img: str, from_cs: str, to_cs: str) -> np.ndarray:
	global colorspace
	"""
	Convert image from one colorspace to another. Extends cv2.cvtColor() to
	encompass all transformation possibilities between RGB, HSV, LAB, and grayscale.
	"""

	colorspace = to_cs

	from_cs_index = colorspaces_list.index(from_cs)
	to_cs_index = colorspaces_list.index(to_cs)
	conv_codes = color_conv_codes[from_cs_index][to_cs_index]

	if to_cs == from_cs:
		return img

	if len(conv_codes) == 0:
		return img

	for code in conv_codes:
		try:
			img = cv2.cvtColor(three_channel(img), code)
		except Exception as ex:
			img = cv2.cvtColor(single_channel(img), code)

	return three_channel(img)


def is_grayscale(img: np.ndarray) -> bool:
	"""
	Checks whether all three image channels are identical,
	i.e., if image is grayscale.
	"""
	try:
		img[:, :, 0]
	except IndexError:
		return True
	
	if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 0] == img[:, :, 2]).all():
		return True
	else:
		return False


def func(name, image, params):
	"""
	Template function for all filtering functions.
	"""
	return name(image, *params)


def crop(img, xs, xe, ys, ye):
	return img[xs: xe, ys: ye, :]

	
def negative(img):
	return ~img


def to_grayscale(img):
	img_gray = convert_img(img, colorspace, 'grayscale')[:, :, 0]
	return three_channel(img_gray)


def to_rgb(img):
	return convert_img(img, colorspace, 'rgb')
	
	
def to_hsv(img):
	return convert_img(img, colorspace, 'hsv')
	

def to_lab(img):
	return convert_img(img, colorspace, 'lab')


def to_ycrcb(img):
	return convert_img(img, colorspace, 'ycrcb')


def rearrange_channels(img, c1=1, c2=2, c3=3):
	return cv2.merge([img[:, :, int(c1 - 1)], img[:, :, int(c2 - 1)], img[:, :, int(c3 - 1)]])
	

def select_channel(img, channel=1):
	global colorspace

	try:
		img_single = img[:, :, int(channel)-1]
	except IndexError:
		return img

	colorspace = 'grayscale'
	return three_channel(img_single)
	
	
def highpass(img, sigma=3.0):
	if is_grayscale(img):
		img = img[:, :, 0]
	
	blur = cv2.GaussianBlur(img, (0, 0), sigma)
	img_highpass = ~cv2.subtract(cv2.add(blur, 127), img)

	return three_channel(img_highpass)
	
	
def normalize_image(img, lower=None, upper=None):
	img_gray = convert_img(img, colorspace, 'grayscale')[:, :, 0]
	
	if lower is None:
		lower = 0
	if upper is None:
		upper = 255

	img_norm = ((img_gray - lower) / (upper - lower) * 255).astype('uint8')

	return three_channel(img_norm)


def intensity_capping(img, n_std=0.0, mode=1):
	img_gray = convert_img(img, colorspace, 'grayscale')[:, :, 0]
	img_gray = ~img_gray if mode == 1 else img_gray

	img_ravel = img_gray.ravel()
	cpp_intensity_capping(img_ravel, c_size_t(img_ravel.size), c_double(n_std))
	img_cap = ~img_ravel.reshape(img_gray.shape) if mode == 1 else img_ravel.reshape(img_gray.shape)

	return three_channel(img_cap)
	
	
def brightness_contrast(img, alpha=1.0, beta=0.0):
	return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def adjust_channels(img, shift_c1=0, shift_c2=0, shift_c3=0):
	c1, c2, c3 = cv2.split(img)

	c1 = cv2.add(c1, shift_c1)
	c2 = cv2.add(c2, shift_c2)
	c3 = cv2.add(c3, shift_c3)

	return cv2.merge([c1, c2, c3])


def gamma(img, gamma=1.0):
	invGamma = 1.0 / gamma
	
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
		
	return cv2.LUT(img, table)


def gaussian_lookup(img, sigma=51, mean=127):
	x = np.arange(0, 256)
	pdf = stats.norm.pdf(x, mean, sigma)

	cdf = np.cumsum(pdf)
	cdf_norm = np.array([(x - np.min(cdf))/(np.max(cdf) - np.min(cdf)) * 255 for x in cdf]).astype('uint8')

	return cv2.LUT(img, cdf_norm)
	
	
def thresholding(img, c1l=0, c1u=255, c2l=0, c2u=255, c3l=0, c3u=255):
	global colorspace
	
	mask = cv2.inRange(img, (int(c1l), int(c2l), int(c3l)), (int(c1u), int(c2u), int(c3u)))
	
	colorspace = 'grayscale'
	return three_channel(mask)


def denoise(img, ksize=3):
	return cv2.medianBlur(img, int(ksize))


def channel_ratios(img, c1=1, c2=2, limit=2.0):
	divisor = img[:, :, int(c2 - 1)]
	divisor[divisor == 0] = 255

	ratio = img[:, :, int(c1 - 1)].astype('float') / divisor.astype('float')
	ratio[ratio > limit] = limit
	ratio = ((ratio - np.min(ratio)) / (np.max(ratio) - np.min(ratio)) * 255).astype('uint8')

	return three_channel(ratio)


def histeq(img):
	img_gray = convert_img(img, colorspace, 'grayscale')[:, :, 0]
	eq = cv2.equalizeHist(img_gray)
	eq = convert_img(eq, 'grayscale', colorspace)
	
	return three_channel(eq)


def clahe(img, clip=2.0, tile=8):
	img_gray = convert_img(img, colorspace, 'grayscale')[:, :, 0]
	clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(int(tile), int(tile)))
	img_clahe = clahe.apply(img_gray)
	img_clahe = convert_img(img_clahe, 'grayscale', colorspace)
	
	return three_channel(img_clahe)


def remove_background(img, num_frames_background, gray=1, img_list = list(), ext = 'jpg'):
	num_frames_background = int(num_frames_background)
	h, w = img.shape[:2]

	if len(img_list) < num_frames_background:
		num_frames_background = len(img_list)

	step = len(img_list) // num_frames_background
	img_back_path = '{}/../median_{}.{}'.format(path.dirname(img_list[0]), num_frames_background, ext)

	if path.exists(img_back_path):
		back = cv2.imread(img_back_path)
		if gray:
			back = three_channel(cv2.cvtColor(back, cv2.COLOR_RGB2GRAY))
	else:
		stack = np.ndarray([h, w, 3, num_frames_background], dtype='uint8')

		for i in range(num_frames_background):
			back = cv2.imread(img_list[i*step])
			if gray:
				back = three_channel(cv2.cvtColor(back, cv2.COLOR_RGB2GRAY))
			stack[:, :, :, i] = back

		back = np.median(stack, axis=3)
		cv2.imwrite(img_back_path, back)

	if gray:
		img = convert_img(img, colorspace, 'grayscale')

	return cv2.subtract(img.astype('uint8'), back.astype('uint8'))


def params_to_list(params: str) -> list:
	"""
	Splits filter parameters into a list.
	"""
	
	if params == '':
		return []
	else:
		return [float(x) for x in params.split(',')]


def apply_filters(img: np.ndarray, filters_data: np.ndarray, img_list: list, ext: str) -> np.ndarray:
	"""
	Applies multiple filters consecutively using a template function.
	"""

	global colorspace
	
	for i in range(filters_data.shape[0]):
		func_name = filters_data[i][0]
		func_pointer = globals()[func_name]
		params = params_to_list(filters_data[i][1])

		if func_name == 'remove_background':
			params = params + [img_list, ext]
		
		img = func(func_pointer, img, params)

		if func_name.startswith('to_'):
			colorspace = func_name.split('_')[1]

	colorspace = 'rgb'
	return img
