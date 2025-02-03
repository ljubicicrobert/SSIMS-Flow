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
	from class_console_printer import Console_printer, tag_string, tag_print, unix_path
	from class_progress_bar import Progress_bar
	from class_logger import time_hms
	from class_timing import Timer, time_hms
	from utilities import fresh_folder, cfg_get, exit_message, present_exception_and_exit
	from skimage import measure
	from scipy.io import savemat
	from scipy.ndimage import label
	from glob import glob
	from warnings import filterwarnings

	import matplotlib.pyplot as plt
	import ctypes

	# To remove runtime warnings about division
	filterwarnings("ignore")

except Exception as ex:
	present_exception_and_exit('Import failed! See traceback below:')


def custom_medfilt(signal, window_size):
    signal_length = len(signal)
    filtered_signal = np.zeros(signal_length)
    pad_width = window_size // 2
    padded_signal = np.pad(signal, (pad_width, pad_width), mode='constant')

    for i in range(signal_length):
        filtered_signal[i] = np.median(padded_signal[i: window_size + i])

    return filtered_signal


def aggregate(img, block_size, min_tracer_area, max_tracer_area, mean_area_filtered):
	img_norm = img/255
	h, w = img_norm.shape
		
	# Dividing the ROI in blocks of size: blockSizeR x blockSizeC
	block_rows = h // block_size
	block_vector_rows = [h % block_size // 2] + [block_size] * block_rows

	block_cols = w // block_size
	block_vector_cols = [w % block_size // 2] + [block_size] * block_cols

	num_rows = len(block_vector_rows) - 1
	num_cols = len(block_vector_cols) - 1

	row_indices = np.cumsum(block_vector_rows)
	col_indices = np.cumsum(block_vector_cols)

	# Analysis for Grayscale or full RGB images
	# ca = [np.split(row, row_indices, axis=1) for row in np.split(img_norm, col_indices, axis=0)]
	# ca = [[subarray for subarray in row if subarray.size > 0] for row in ca if len(row) > 0]
	# ca = [lst for lst in ca if lst]

	# sum_area = []
	num_particles = []
	# particles_count = 0

	for r in range(num_rows):
		for c in range(num_cols):
			row_start = row_indices[r]
			row_end = row_indices[r+1]
			col_start = col_indices[c]
			col_end = col_indices[c+1]

			block = img_norm[row_start: row_end, col_start: col_end]
			block_rows, block_cols = block.shape

			regions = measure.regionprops(measure.label(block, connectivity=1))
			filtered_regions = [region for region in regions if min_tracer_area <= region.area <= max_tracer_area]
			area = [region.area for region in filtered_regions]

			array_area = np.array(area, dtype=float)
			array_area[array_area > (block_size/2)**2] = np.nan

			# sum_area.append(np.nansum(area_filtered))
			s_area_filtered = np.ceil(array_area / mean_area_filtered)
			s_area_filtered = s_area_filtered[~np.isnan(s_area_filtered)]
			num_particles.append(np.nansum(s_area_filtered))

			# particles_count += 1

	var = np.nanvar(num_particles)
	mean = np.nanmean(num_particles)
	nu = var / mean if any(num_particles) > 0 else 0

	return nu


def seeding_metrics(img_path_list, ROI, threshold, block_size, min_tracer_area, max_tracer_area):
	num_frames = len(img_path_list)
	
	array_density = np.ndarray(num_frames)
	array_nu = np.ndarray(num_frames)
	array_mean_area_filtered = np.ndarray(num_frames)
	# array_num_particles = np.ndarray(num_frames)

	roi_area = abs((ROI[0, 0] - ROI[1, 0])) * abs((ROI[0, 1]) - ROI[1, 1])
	ys = min(ROI[0, 1], ROI[1, 1]) - 1
	ye = max(ROI[0, 1], ROI[1, 1])
	xs = min(ROI[0, 0], ROI[1, 0]) - 1
	xe = max(ROI[0, 0], ROI[1, 0])

	console_printer = Console_printer()
	progress_bar = Progress_bar(total=num_frames, prefix=tag_string('info', 'SDI estimation for frame '))
	timer = Timer(total_iter=num_frames)
	
	for i, img_path in enumerate(img_path_list):
		img = cv2.imread(img_path, 0)
		img_crop = img[ys: ye, xs: xe]
		img_binary = cv2.threshold(img_crop, int(threshold*255), 255, cv2.THRESH_BINARY)[1]

		regions = measure.regionprops(measure.label(img_binary, connectivity=1))
		filtered_regions = [region for region in regions if min_tracer_area <= region.area <= max_tracer_area]

		array_area = np.array([region.area for region in filtered_regions], dtype=float)
		array_area[array_area > (block_size/2)**2] = np.nan

		mean_area_filtered = np.nanmean(array_area) if array_area.size >0 else np.nan
		array_mean_area_filtered[i] = mean_area_filtered

		s_area_filtered = np.ceil(array_area / mean_area_filtered)
		s_area_filtered = s_area_filtered[~np.isnan(s_area_filtered)]
		# array_num_particles[i] = np.nansum(s_area_filtered)

		array_density[i] = np.nansum(s_area_filtered) / roi_area
		array_nu[i] = aggregate(img_binary, block_size, min_tracer_area, max_tracer_area, mean_area_filtered)

		timer.update()
		
		console_printer.add_line(progress_bar.get(i))
		console_printer.add_line(tag_string('info', f'Frame processing time = {timer.interval():.3f} sec'))
		he, me, se = time_hms(timer.elapsed())
		console_printer.add_line(tag_string('info', f'Elapsed time = {he} hr {me} min {se} sec'))
		hr, mr, sr = time_hms(timer.remaining())
		console_printer.add_line(tag_string('info', f'Remaining time = {hr} hr {mr} min {sr} sec'))

		console_printer.overwrite()

		i += 1
	
	mean_density = np.nanmean(array_density)
	mean_area_tracers = np.nanmean(array_mean_area_filtered)
	mean_nu = np.nanmean(array_nu)

	SDI = array_nu ** 0.1 / (array_density / 1.52E-03)

	return mean_density, mean_area_tracers, mean_nu, SDI


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument('--cfg', type=str, help='Path to config file')
		args = parser.parse_args()

		cfg = configparser.ConfigParser()
		cfg.optionxform = str

		try:
			cfg.read(args.cfg, encoding='utf-8-sig')
		except Exception:
			tag_print('error', 'There was a problem reading the configuration file!')
			tag_print('error', 'Check if project has valid configuration.')
			exit_message()

		tag_print('start', 'Starting estimation of best frame sequence using SDI!')
		print()

		section = 'Optical flow'

		project_folder = unix_path(cfg_get(cfg, 'Project settings', 'Folder', str))
		frames_folder = unix_path(cfg_get(cfg, 'SDI', 'Folder', str))
		results_folder = unix_path(f'{project_folder}/SDI')
		ext = cfg_get(cfg, 'SDI', 'Extension', str, default='jpg')
		Xstart = cfg_get(cfg, 'SDI', 'XStart', int)
		Ystart = cfg_get(cfg, 'SDI', 'YStart', int)
		Xend = cfg_get(cfg, 'SDI', 'XEnd', int)
		Yend = cfg_get(cfg, 'SDI', 'YEnd', int)
		binarization_threshold = cfg_get(cfg, 'SDI', 'BinarizationThreshold', float, default=0.8)
		frame_window_threshold = cfg_get(cfg, 'SDI', 'FrameWindowThreshold', int, default=10)
		block_size = cfg_get(cfg, 'SDI', 'BlockSize', int, default=30)
		sequence_min_length = cfg_get(cfg, 'SDI', 'SequenceMinLength', int, default=20)
		max_tracer_area = cfg_get(cfg, 'SDI', 'MaxArea', int, default=1)
		min_tracer_area = cfg_get(cfg, 'SDI', 'MinArea', int, default=14000)

		if abs(Xstart - Xend + 1) < block_size or abs(Ystart - Yend + 1) < block_size:
			MessageBox = ctypes.windll.user32.MessageBoxW

			response = MessageBox(None, f'Block size is larger than the ROI in one of the directions!\n' +
										f'ROI size in X direction = {Xstart - Xend + 1} px\n' +
										f'ROI size in Y direction = {Ystart - Yend + 1} px\n' +
										f'Block size = {block_size} px',
										 'Block size error', 16)

		tag_print('info', f'Using frames from folder {frames_folder}')
		tag_print('info', f'Results folder {results_folder}')
		tag_print('info', f'ROI = [[{Xstart}, {Ystart}], [{Xend}, {Yend}]]')
		tag_print('info', f'Binarization threshold = {binarization_threshold:.2f}')
		tag_print('info', f'Frame window threshold = {frame_window_threshold}')
		tag_print('info', f'Block size = {block_size}')
		tag_print('info', f'Minimal sequence length = {sequence_min_length}')
		tag_print('info', f'Min. tracer area = {min_tracer_area}')
		tag_print('info', f'Max. tracer area = {max_tracer_area}')
		print()

		img_list = glob(f'{frames_folder}/*.{ext}')
		roi = np.array([[Xstart, Ystart], [Xend, Yend]])

		fresh_folder(results_folder)

		mean_density, mean_area_tracers, mean_nu, SDI = seeding_metrics(img_list, roi, binarization_threshold, block_size, min_tracer_area, max_tracer_area)
		mean_SDI = np.nanmean(SDI)

		with open(f'{results_folder}/mean_values.txt', 'w') as file:
			file.write(f'{mean_density:.5f}\n')
			file.write(f'{mean_area_tracers:.3f}\n')
			file.write(f'{mean_nu:.3f}\n')
			file.write(f'{mean_SDI:.3f}')

		with open(f'{results_folder}/SDI_list.txt', 'w') as file:
			for s in SDI:
				file.write(f'{s:.3f}\n')

		filtered_SDI = custom_medfilt(SDI, 10)
		binary_filtered_SDI = filtered_SDI < mean_SDI
		labeled_binary_filtered_SDI, num_regions = label(binary_filtered_SDI)

		SDI_analysis = []

		for label in np.unique(labeled_binary_filtered_SDI):
			if label == 0: 
				continue

			mask = labeled_binary_filtered_SDI == label
			area = np.nansum(mask)
			pixel_values = binary_filtered_SDI[mask]
			SDI_analysis.append({'Label': label, 'Area': area, 'PixelValues': pixel_values})

		number_of_frames_in_window = []

		for region in SDI_analysis:
			num_frames = region['Area']
			number_of_frames_in_window.append(num_frames)

		candidate_by_min_length = [i for i, nf in enumerate(number_of_frames_in_window) if nf >= sequence_min_length]
		max_candidate_by_min_length = np.max(number_of_frames_in_window)
		if not candidate_by_min_length:
			candidate_by_min_length = [i for i, nf in enumerate(number_of_frames_in_window) if nf >= max_candidate_by_min_length - frame_window_threshold]
			print()
			tag_print('warning', f'No candidate window satisfies condition of min. length = {sequence_min_length}!')
			tag_print('warning', f'Maximal candidate window length = {max_candidate_by_min_length}')

		num_candidates_by_min_length = len(candidate_by_min_length)		

		if num_candidates_by_min_length > 1:
			longest_candidate_window = np.argmax(number_of_frames_in_window)
			candidates_by_window_threshold = [i for i in candidate_by_min_length if number_of_frames_in_window[i] > (number_of_frames_in_window[longest_candidate_window] - frame_window_threshold)]
			num_candidates_by_window_threshold = len(candidates_by_window_threshold)

			print()
			tag_print('info', f'Number of candidate frame windows = {num_candidates_by_window_threshold}:')

			start_frame = []
			end_frame = []
			frame_window_mean_SDI = []
			
			for i in range(num_candidates_by_window_threshold):
				sf = np.min(np.where(labeled_binary_filtered_SDI == candidates_by_window_threshold[i]+1))
				ef = np.max(np.where(labeled_binary_filtered_SDI == candidates_by_window_threshold[i]+1))
				mSDI = np.nanmean(SDI[sf: ef + 1])

				start_frame.append(sf)
				end_frame.append(ef)
				frame_window_mean_SDI.append(mSDI)

				print(f'       Candidate frame window {i+1}: start = {sf}, end = {ef}, length = {ef - sf + 1}, mean SDI = {mSDI:.3f}')
			
			position_optimal_frame_window = candidates_by_window_threshold[np.argmin(frame_window_mean_SDI)]
		
		elif len(candidate_by_min_length) == 1:
			position_optimal_frame_window = candidate_by_min_length[0]
		else:
			position_optimal_frame_window = 0

		optimal_start_frame = np.min(np.where(labeled_binary_filtered_SDI == position_optimal_frame_window + 1))
		optimal_end_frame = np.max(np.where(labeled_binary_filtered_SDI == position_optimal_frame_window + 1))
		mean_SDI_in_optimal_window = np.nanmean(SDI[optimal_start_frame: optimal_end_frame + 1])

		export_data = {
			'MeanDensity': mean_density,
			'MeanAreaTracers': mean_area_tracers,
			'MeanNu': mean_nu,
			'MeanSDI': mean_SDI,
			'OptimalStartFrame': optimal_start_frame,
			'OptimalEndFrame': optimal_end_frame,
			'SDI': SDI,
		}

		savemat(f'{results_folder}/SDI.mat', export_data)

		with open(f'{results_folder}/optimal_frame_window.txt', 'w') as file:
			file.write(f'{optimal_start_frame}\n')
			file.write(f'{optimal_end_frame}')

		fig, ax = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)

		ax[0].plot(SDI, label='SDI')
		ax[0].plot(filtered_SDI, '-r', label='Filtered SDI')
		ax[0].axhline(mean_SDI, linestyle='--', color='k', linewidth=1, label='SDI threshold')
		ax[0].fill_between(range(optimal_start_frame, optimal_end_frame + 1), filtered_SDI[optimal_start_frame: optimal_end_frame + 1], mean_SDI,
							where=filtered_SDI[optimal_start_frame: optimal_end_frame + 1] < mean_SDI, color=[0.9290, 0.6940, 0.1250], label='Optimal window')
		ax[0].set_ylabel('SDI')
		ax[0].legend()
		ax[0].set_xlim(0, len(SDI))

		ax[1].plot(binary_filtered_SDI, '-r')
		ax[1].set_xlabel('Frame Number')
		ax[1].set_ylabel('Candidate frame')
		ax[1].set_xlim(0, len(SDI))
		ax[1].fill_between(range(optimal_start_frame, optimal_end_frame + 1), 1, color=[0.9290, 0.6940, 0.1250])

		try:
			mng = plt.get_current_fig_manager()
			mng.window.state('zoomed')
			mng.set_window_title('Inspect frames')
		except Exception:
			pass

		plt.tight_layout()
		plt.savefig(f'{results_folder}/SDI_results.png')
		plt.show()

		print()
		tag_print('info', f'Mean density     = {mean_density:.3e}')
		tag_print('info', f'Mean tracer area = {mean_area_tracers:.3f}')
		tag_print('info', f'Mean nu          = {mean_nu:.3f}')
		tag_print('info', f'Mean SDI         = {mean_SDI:.3f}')

		print()
		tag_print('info', f'Optimal first frame index    = {optimal_start_frame}')
		tag_print('info', f'Optimal last frame index     = {optimal_end_frame}')
		tag_print('info', f'Length of the optimal window = {optimal_end_frame - optimal_start_frame + 1}')
		tag_print('info', f'Mean SDI in optimal window   = {mean_SDI_in_optimal_window:.3f}')

		print()
		tag_print('end', 'Best frame sequence estimation complete!')
		print('\a')
		exit_message()

	except Exception as ex:
		present_exception_and_exit()
