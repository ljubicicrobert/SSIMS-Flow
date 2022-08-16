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
    from feature_tracking import fresh_folder
    from math import log10, floor
    from glob import glob
    from class_console_printer import Console_printer, tag_string, tag_print, unix_path
    from class_progress_bar import Progress_bar
    from class_timing import Timer, time_hms
    from CPP.dll_import import DLL_Loader
    from skimage.measure import block_reduce
    from ctypes import c_uint, c_double
    from warnings import catch_warnings, simplefilter

    import matplotlib.pyplot as plt

    dll_path = path.split(path.realpath(__file__))[0]
    dll_name = 'CPP/pooling.dll'
    dll_loader = DLL_Loader(dll_path, dll_name)
    # float mag_pool(float* array, int size, float k)
    mag_pool = dll_loader.get_function('double', 'mag_pool', ['float*', 'uint', 'double'])
    
except Exception:
    print()
    tag_print('exception', 'Import failed! \n')
    print('\n{}'.format(format_exc()))
    input('\nPress ENTER/RETURN key to exit...')
    exit()


def get_angle_range(angle_main, angle_range):
    angle_lower = angle_main - angle_range
    angle_upper = angle_main + angle_range

    underflow = True if angle_lower < 0 else False
    overflow = True if angle_upper >= 360 else False

    angle_lower = angle_lower + 360 if underflow else angle_lower
    angle_upper = angle_upper - 360 if overflow else angle_upper

    func = np.bitwise_or if overflow or underflow else np.bitwise_and

    return func, angle_lower, angle_upper


def pooling_mask(array, axis: int, k: float = 1.0):
    res = np.ndarray(array.shape[:2])

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            subarray = array[i, j, :, :]
            res[i, j] = mag_pool(subarray.ravel(), c_uint(subarray.size), c_double(k))

    return res


def nan_locate(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
    

if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        parser.add_argument('--cfg', type=str, help='Path to configuration file')
        args = parser.parse_args()

        cfg = configparser.ConfigParser()
        cfg.optionxform = str

        try:
            cfg.read(args.cfg, encoding='utf-8-sig')
        except Exception:
            tag_print('error', 'There was a problem reading the configuration file!')
            tag_print('error', 'Check if project has valid configuration.')
            input('\nPress ENTER/RETURN key to exit...')
            exit()
        
        section = 'Optical flow'

        project_folder = unix_path(cfg['Project settings']['Folder'])
        frames_folder = unix_path(cfg[section]['Folder'])
        results_folder = unix_path('{}/optical_flow'.format(cfg['Project settings']['Folder']))
        ext = cfg[section]['Extension']
        step = int(cfg[section]['Step'])
        scale = float(cfg[section]['Scale'])
        pooling = int(cfg[section]['Pooling'])
        angle_main = float(cfg[section]['AngleMain'])
        angle_range = float(cfg[section]['AngleRange'])

        fresh_folder(results_folder, exclude=['depth_profile.txt'])
        fresh_folder(results_folder + '/magnitudes')
        fresh_folder(results_folder + '/directions')
        
        frames_list = glob('{}/*.{}'.format(frames_folder, ext))
        num_frames = len(frames_list)
        num_digits = floor(log10(num_frames)) + 1
        angle_func, angle_lower, angle_upper = get_angle_range(angle_main, angle_range)

        prev_frame = cv2.imread(frames_list[0], 0)
        h, w = prev_frame.shape

        try:
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            mng.set_window_title('Filtering')
        except Exception:
            pass

        if scale != 1.0:
            prev_frame = cv2.resize(prev_frame, (int(w * scale), int(h * scale)))
            h, w = prev_frame.shape

        flow_hsv = np.zeros([h, w, 2], dtype='uint8')
        mag_max = None
        angle_stack = None
        padd_h = h//pooling//10
        padd_w = w//pooling//10

        flow_shown = plt.imshow(flow_hsv[:, :, 0], cmap='jet', vmax=1, vmin=0)
        cbar = plt.colorbar(flow_shown)
        cbar.set_label('Velocity magnitude [px/frame]')
        plt.title('Frame: 1/{}'.format(num_frames))
        plt.axis('off')
        plt.tight_layout()

        console_printer = Console_printer()
        progress_bar = Progress_bar(total=(num_frames - num_frames % step), prefix=tag_string('info', 'Frame '))
        timer = Timer(total_iter=num_frames//step)

        tag_print('start', 'Optical flow estimation using Farneback algorithm\n')
        tag_print('info', 'Using frames from folder [{}]'.format(frames_folder))
        tag_print('info', 'Frame extension = {}'.format(ext))
        tag_print('info', 'Number of frames = {}'.format(num_frames))
        tag_print('info', 'Frame step = {}'.format(step))
        tag_print('info', 'Frame scaling = {:.1f}'.format(scale))
        tag_print('info', 'Pooling = {} px'.format(pooling))
        tag_print('info', 'Main flow direction = {:.0f} deg'.format(angle_main))
        tag_print('info', 'Direction range = {:.0f} deg\n'.format(angle_range))

        j = 0

        for i in range(step, num_frames, step):
            next_frame = cv2.imread(frames_list[i], 0)

            if scale != 1.0:
                next_frame_bgr = cv2.resize(next_frame_bgr, (w, h))
                next_frame = cv2.resize(next_frame, (w, h))

            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, *[0.5, 3, 15, 1, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN])
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

            # Filter by vector angle
            angle = np.where(angle_func((angle >= angle_lower), (angle <= angle_upper)), angle, np.NaN)
            mask_magnitude = np.where(angle >= 0, 1, 0)
            magnitude *= mask_magnitude

            if angle_func.__name__ == "bitwise_or":
                angle -= angle_upper
                angle = np.where(angle <= 0, angle + 360, angle)

            if pooling > 1:
                with catch_warnings():
                    simplefilter("ignore", category=RuntimeWarning)
                    magnitude = block_reduce(magnitude, (pooling, pooling), pooling_mask)
                    angle = block_reduce(angle, (pooling, pooling), np.nanmean)

            if mag_max is None:
                mag_max = np.zeros(magnitude.shape)
                angle_stack = np.ndarray([mag_max.shape[0], mag_max.shape[1], len(range(1, num_frames, step))])
            mag_max = np.maximum(mag_max, magnitude)

            nans, x = nan_locate(angle)
            try:
                angle[nans] = np.interp(x(nans), x(~nans), angle[~nans], period=360)
            except ValueError:
                pass

            angle_stack[:, :, j] = angle

            n = str(i).rjust(num_digits, '0')
            np.savetxt('{}/magnitudes/{}.txt'.format(results_folder, n), magnitude, fmt='%.2f')
            np.savetxt('{}/directions/{}.txt'.format(results_folder, n), angle, fmt='%.1f')

            flow_shown.set_data(mag_max)
            flow_shown.set_clim(vmax=np.max(mag_max[padd_h: -padd_h, padd_w: -padd_w]))
            plt.title('Frame: {}/{}'.format(i+1, num_frames))
            plt.pause(0.001)
            plt.draw()

            prev_frame = next_frame

            timer.update()

            console_printer.add_line(progress_bar.get(i))
            console_printer.add_line(
                tag_string('info', 'Frame processing time = {:.3f}'
                    .format(timer.interval())
                )
            )
            console_printer.add_line(
                tag_string('info', 'Elapsed time = {} hr {} min {} sec'
                    .format(*time_hms(timer.elapsed()))
                )
            )
            console_printer.add_line(
                tag_string('info', 'Remaining time = {} hr {} min {} sec'
                    .format(*time_hms(timer.remaining()))
                )
            )
            console_printer.overwrite()
            
            j += 1

        angle_mean = np.nanmean(angle_stack, axis=2)
        if angle_func.__name__ == "bitwise_or":
            angle_mean += angle_upper
            angle_mean = np.where(angle_mean >= 360, angle_mean - 360, angle_mean)

        np.savetxt('{}/mag_max.txt'.format(results_folder), mag_max, fmt='%.3f')
        np.savetxt('{}/angle_mean.txt'.format(results_folder), angle_mean, fmt='%.3f')
        
        print()
        tag_print('end', 'Optical flow estimation complete!')
        tag_print('end', 'Results available in [{}]!'.format(results_folder))
        print('\a')
        input('\nPress ENTER/RETURN key to exit...')

    except Exception as ex:
        print()
        tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
        print('\n{}'.format(format_exc()))
        input('\nPress ENTER/RETURN key to exit...')
