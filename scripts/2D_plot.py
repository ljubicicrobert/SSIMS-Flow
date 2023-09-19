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
    from matplotlib.widgets import Slider
    from sys import exit
    from glob import glob
    from class_console_printer import tag_print, unix_path
    from vel_ratio import L1
    from utilities import cfg_get

    import matplotlib.pyplot as plt

except Exception as ex:
    print()
    tag_print('exception', 'Import failed! \n')
    print('\n{}'.format(format_exc()))
    input('\nPress ENTER/RETURN key to exit...')
    exit()


__types__ = [
    'U(x,y)',
    'V(x,y)',
    'Magnitudes',
    'Flow direction',
    'Threshold ratio',
]

__units__ = [
    '[m/s]',
    '[m/s]',
    '[m/s]',
    '[deg]',
    '[-]',
]


def update_frame(val):
    mags = try_load_file(mag_list[val]) * v_ratio
    dirs = try_load_file(dir_list[val])
    us, vs = cv2.polarToCart(mags, dirs, angleInDegrees=True)

    data_new = [us, vs, mags, dirs]
    img_new = data_new[args.data]

    if frames_available:
        back_new = cv2.imread(frames_list[val], cv2.COLOR_BGR2RGB)[::-1]
        back_shown.set_data(back_new)

    img_shown.set_data(img_new)
    img_shown.set_clim(vmin=np.nanmin(img_new[cbar_cutoff_h: -cbar_cutoff_h, cbar_cutoff_w: -cbar_cutoff_w]),
                       vmax=np.nanmax(img_new[cbar_cutoff_h: -cbar_cutoff_h, cbar_cutoff_w: -cbar_cutoff_w]))
    ax.set_title('{}, frame #{}/{}'.format(data_type, sl_ax_frame_num.val, num_frames - 1))
    plt.draw()

    return


def keypress(event):
    if event.key == 'escape':
        exit()

    elif event.key == 'down':
        if sl_ax_frame_num.val == 0:
            sl_ax_frame_num.set_val(num_frames - 1)
        else:
            sl_ax_frame_num.set_val(sl_ax_frame_num.val - 1)

    elif event.key == 'up':
        if sl_ax_frame_num.val == num_frames - 1:
            sl_ax_frame_num.set_val(0)
        else:
            sl_ax_frame_num.set_val(sl_ax_frame_num.val + 1)

    elif event.key == 'pageup':
        if sl_ax_frame_num.val >= num_frames - 10:
            sl_ax_frame_num.set_val(0)
        else:
            sl_ax_frame_num.set_val(sl_ax_frame_num.val + 10)

    elif event.key == 'pagedown':
        if sl_ax_frame_num.val <= 9:
            sl_ax_frame_num.set_val(num_frames - 1)
        else:
            sl_ax_frame_num.set_val(sl_ax_frame_num.val - 10)

    update_frame(sl_ax_frame_num.val)


def try_load_file(fname):
    try:
        return np.loadtxt(fname)
    except Exception:
        return None


if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        parser.add_argument('--cfg', type=str, help='Path to project configuration file')
        parser.add_argument('--mode', type=int, help='0 = time averaged, 1 = maximal, 2 = instantaneous', default=0)
        parser.add_argument('--data', type=int, help='Which data to plot, see __types__ for more details')
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

        mode = args.mode
        data_type = __types__[args.data]
        units = __units__[args.data]

        project_folder = cfg['Project settings']['Folder']

        section = 'Optical flow'

        frames_step = cfg_get(cfg, 'Frames', 'Step', float)
        optical_flow_step = cfg_get(cfg, section, 'Step', float)
        scale = cfg_get(cfg, section, 'Scale', float)
        fps = cfg_get(cfg, section, 'Framerate', float)		# frames/sec
        try:
            gsd = cfg_get(cfg, section, 'GSD', float)       # px/m
        except Exception as ex:
            gsd = cfg_get(cfg, 'Transformation', 'GSD', float)        # px/m

        gsd_units = cfg_get(cfg, section, 'GSDUnits', str, 'px/m')           # px/m
        if gsd_units != 'px/m':
            gsd = 1/gsd
            
        pooling = cfg_get(cfg, section, 'Pooling', float)   	# px
        gsd_pooled = gsd / pooling  				# blocks/m, 1/m

        v_ratio = fps / gsd / (frames_step * optical_flow_step) / scale         	# (frame*m) / (s*px)

        average_only = int(cfg[section]['AverageOnly'])   	# px

        frames_folder = cfg_get(cfg, section, 'Folder', str)
        frames_ext = cfg_get(cfg, 'section', 'Extension', str, 'jpg')
        frames_list = glob('{}/*.{}'.format(frames_folder, frames_ext))
        frames_available = len(frames_list) > 0

        alpha = 1.0 if not frames_available else 0.5

        if average_only == 0:
            mag_list = glob('{}/optical_flow/magnitudes/*.txt'.format(project_folder))
            dir_list = glob('{}/optical_flow/directions/*.txt'.format(project_folder))
            num_frames = len(mag_list)

            if num_frames == 0:
                print()
                tag_print('error', 'No optical flow data found in [{}/optical_flow/]'.format(project_folder))
                input('\nPress ENTER/RETURN to exit...')
                exit()

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.13)
        plt.axis('off')

        legend = 'Use slider to select frame,\n' \
                 'use UP and DOWN keys to move by +/- 1 frame\n' \
                 'or PageUP and PageDOWN keys to move by +/- 10 frames\n' \
                 'Press ESC or Q to exit'

        legend_toggle = plt.text(0.02, 0.97, legend,
                                 horizontalalignment='left',
                                 verticalalignment='top',
                                 transform=ax.transAxes,
                                 bbox=dict(facecolor='white', alpha=0.5),
                                 fontsize=9,
                                 )

        if mode == 0:       # Time averaged
            legend_toggle.set_visible(False)

            mags = try_load_file('{}/optical_flow/mag_mean.txt'.format(project_folder)) * v_ratio	# px/frame
            dirs = try_load_file('{}/optical_flow/angle_mean.txt'.format(project_folder))
            thrs = try_load_file('{}/optical_flow/threshold_ratios.txt'.format(project_folder))
            h, w = mags.shape

            us, vs = cv2.polarToCart(mags, dirs, angleInDegrees=True)
            data_list = [us, vs, mags, dirs, thrs]
            data = data_list[args.data]

        elif mode == 1:     # Maximal
            legend_toggle.set_visible(False)

            mags = try_load_file('{}/optical_flow/mag_max.txt'.format(project_folder)) * v_ratio	# px/frame
            dirs = try_load_file('{}/optical_flow/angle_mean.txt'.format(project_folder))
            h, w = mags.shape

            us, vs = cv2.polarToCart(mags, dirs, angleInDegrees=True)
            data_list = [us, vs, mags]
            data = data_list[args.data]

        elif mode == 2:     # Instantaneous      
            mags = try_load_file(mag_list[0]) * v_ratio
            dirs = try_load_file(dir_list[0])
            h, w = mags.shape

            us, vs = cv2.polarToCart(mags, dirs, angleInDegrees=True)
            data_list = [us, vs, mags, dirs]
            data = data_list[args.data]
            
            axcolor = 'lightgoldenrodyellow'
            valfmt = "%d"

            fig.canvas.mpl_connect('key_press_event', keypress)
            ax_frame_num = plt.axes([0.2, 0.05, 0.63, 0.03], facecolor=axcolor)
            sl_ax_frame_num = Slider(ax_frame_num, 'Frame #', 0, num_frames-1, valinit=0, valstep=1, valfmt=valfmt)
            sl_ax_frame_num.on_changed(update_frame)

        cbar_cutoff_h = h//10
        cbar_cutoff_w = w//10

        if frames_available:
            back = cv2.imread(frames_list[0], cv2.COLOR_BGR2RGB)[::-1]
            padd_x = back.shape[1] % pooling // 2
            padd_y = back.shape[0] % pooling // 2
            back_shown = ax.imshow(back, extent=(padd_x / pooling, (back.shape[1] - padd_x) / pooling, padd_y / pooling, (back.shape[0] - padd_y) / pooling))

        img_shown = ax.imshow(data, cmap='jet', interpolation='hanning', alpha=alpha)

        if args.data == 4:
            img_shown.set_clim(vmin=0, vmax=L1)
        else:
            img_shown.set_clim(vmin=np.nanmin(data[cbar_cutoff_h: -cbar_cutoff_h, cbar_cutoff_w: -cbar_cutoff_w]),
                               vmax=np.nanmax(data[cbar_cutoff_h: -cbar_cutoff_h, cbar_cutoff_w: -cbar_cutoff_w]))
        
        cbar = plt.colorbar(img_shown, ax=ax)
        cbar.set_label('{} {}'.format(data_type, units))

        try:
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            mng.set_window_title('Inspect frames')
        except Exception:
            pass

        ax.set_title('{}, frame #0/{}'.format(data_type, num_frames - 1)
                     if mode == 2
                     else 'Time averaged values: {}'.format(data_type))
        plt.show()

    except Exception as ex:
        print()
        tag_print('exception', 'An exception has occurred! See traceback bellow: \n')
        print('\n{}'.format(format_exc()))
        input('\nPress ENTER/RETURN key to exit...')
