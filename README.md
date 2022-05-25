# SSIMS-Flow: UAV image velocimetry workbench

This tool represents a feature-complete solution for the estimation of open-channel flow (discharge) from UAV videos using Farneback dense optical flow algorithm. The integrated workflow enables:

1. Unpacking of video data into frames,
2. Camera calibration and removal of camera distortions,
3. Digital image stabilization,
4. Image orthorectification,
5. Image filtering/enhancement,
6. Application of optical flow to estimate surface velocity field,
7. Estimation of open channel flow/discharge using surface velocity field and bathymetric data,
8. Creation of videos from individual frames.

The tool is based on an older tool SSIMS, which was only aimed at image preprocessing for UAV velocimetry, which it effectively replaces.


## Requirements and installation

The package consists of a backend (written in Python 3 and C++) and frontend GUI (written in C# with .NET Framework 4.5.1). **Unlike its predecessor ([SSIMS](https://github.com/ljubicicrobert/SSIMS)), this tool is written entirely (backend and GUI) for Windows OS and only compiled for x64 architecture only**. At request, I can compile for x86 version, but will likely not bother otherwise.

The GUI requires .NET Framework 4.5.1+, which can be downloaded from the [official site](https://dotnet.microsoft.com/download/dotnet-framework).

The backend requires that Python 3+ exists in the **%PATH% environmental variable** in Windows. Please make sure that this is the case <ins>before</ins> running SSIMS-Flow! The tool will recognize multiple instances of Python in %PATH% and allow you to choose the correct interpreter in the GUI.

So far, the package was tested using **Python** versions **3.6.8, 3.7.2, 3.7.6, 3.8.5** and **3.9.10**. You can download latest Python from the [official site](https://www.python.org/downloads/)

Python library requirements (other than the standard library):
```python
numpy >= 1.19                       # pip install numpy
opencv-python >= 4.0                # pip install opencv-python
matplotlib >= 3.0                   # pip install matplotlib
mplcursors >= 0.3                   # pip install mplcursors
scipy >= 1.0                        # pip install scipy
skimage (scikit-image) >= 0.16.1    # pip install scikit-image
```

>**Note #1**: If you are using distributions of Python such as Anaconda or WinPython, you will likely have all the necessary libraries with the possible exception of `opencv-python`.

>**Note #2**: The tool might also work fine with `opencv-python` version 3.6, but this is yet to be tested.

>**Note #3**: It has been noted that newer versions of `matplotlib` (apparently those which have been installed using PIP) are failing to be properly imported due to missing `ft2font` DLL. In this case you can try installing `matplotlib` version 3.2.1 which worked well during testing:
>```bash
>pip install matplotlib==3.2.1
>```


## New versions (checking for updates)

The tool will automatically check for latest release of the tool on program start. If new release was found in the [official repository](https://github.com/ljubicicrobert/SSIMS-Flow), a form will be displayed from where you can read the new release information, choose to view/download the new release on/from GitHub, or pause automatic checking for new versions for some period of time.

> **Note #4**: You can also manually check for new releases through the _About_ tab.


## Usage

**Unlike its predecessor, SSIMS-Flow is meant to be used ONLY through the graphical user interface (GUI)**. Usage through the terminal is possible but discouraged. Those interested in such approach can check the `ArgumentParser` objects in source files for more details.


### Project settings

<img align="right" width="500" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/main.png">

Main information about the project will be shown in the **Project settings** panel once the GUI has started. These include project name, location on disk, description, data of creation, and interpreter used for backend. Interpreted choice will be memorized for each project.

You can choose to either:
1. **Create** a new project, or 
2. **Open** an existing one.

Creating a new project involves a selection of a folder which will host all of the resulting files created by this tool. **Selected folder should be empty, or will it be emptied for the user after prompt**. Project folder **does not** have to contain the video itself - the video can be hosted anywhere on disk and will not be moved/copied/deleted.

All of the project metadata will be contained in the file `%PROJECT_FOLDER%\project.ssims` in the selected project folder, which is automatically created.

Loading an existing project requires that you select an appropriate `project.ssims` file, which will load all the project information to the GUI.

> **Note #5**: Please note that some sections of the GUI will be unavailable until you create a new project or load an existing one.


### Project status

In the lower section of the **Project settings** panel a summary of the project status will be shown. This information is meant to briefly summarize the currently available results of the project. The status of the project is also indicated by the two-letter icons in the lower right of the footer, for each of the project stages:

- **FR** = Frames available,
- **TR** = Feature tracking,
- **ST** = Stabilization,
- **OR** = Orthorectification,
- **EN** = Image enhancement,
- **OF** = Optical flow.

Completed project stages, which have available results, will be indicated by the green icons in the footer, while uncompleted stages will be indicated by the gray icons.


### Video unpacking

<img align="right" width="500" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/unpack_video.png">

The workflow of the tool requires that the video is unpacked into individual frames. The source video can be selected in the **Unpack video** panel, after which the video metadata will be shown below. You can specify the resulting frames' file extension, image quality parameter (0-100), and a scaling factor. Furthermore, you can choose to only unpack a section of the original video, defined by the starting and end time in the video.

The resulting frames will be unpacked to folder `%PROJECT_FOLDER%\frames`.


### Camera calibration

<img align="right" width="530" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/camera_calibration.png">

During the video unpacking, it is also possible to remove camera distortion using intrinsic camera parameters - focal length, principal point, radial and tangential distortion coefficients. These parameters can be provided by:

1. Manual input in the corresponding boxes,
2. Selecting an available camera model from the dropdown list, or
3. By performing a new camera calibration.

Last option requires that you provide a set of calibrating images using a chequerboard marker (between 20 and 30 images is usually enough). Please follow the instructions from the **Camera calibration** form, or refer to the webpage link from the Step 2 of said form.

When option (3) is selected, the form will expand to reveal additional options and explanations on how to perform a calibration of a new camera. The calibration process involves detection of chequerboard patterns in a series of images, and minimizing the reprojection errors between the image- and object-space by modifying the camera parameters. At the end of the procedure, a report will show the RMS reprojection errors obtained using the new camera parameters.

> **Note #6**: If certain images have a relatively high reprojection error value, it can be beneficial to remove them from the calibration folder and repeat the calibration procedure, as this might improve the calibration accuracy. For best results try to keep between 15 and 30 images with chequerboard (target) pattern. Also, generation and manual inspection of undistorted chequerboard images is highly advised.

Camera parameters will be saved to `%PROJECT_FOLDER%\camera_parameters.cpf`, and can be viewed and modified using any text editor. Copying this file to the `%INSTALATION_FOLDER%\camera_profiles` folder of the SSIMS-Flow tool will make this camera profile available in the dropdown list of the Camera parameters form for all projects.


### Stabilization/orthorectification

<img align="right" width="500" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/stab_ortho.png">

#### Feature tracking

Despite the modern UAVs having sophisticated in-built camera/video stabilization, in most cases it is necessary to perform additional stabilization to ensure that the coordinate system of the video remains constant throughout the entire sequence of frames. SSIMS-Flow uses the same stabilization strategy as its precursor (SSIMS tool), which consists of the following steps:

1. Selecting features for tracking, to estimate the camera motion,
2. Selecting features for image transformation,
3. Image transformation.

Clicking the **Track features** button in the bottom-left corner of the **Stabilize/Orthorectify** panel will open a new window to allow you to manually select static features which will be tracked in order to estimate the direction and magnitude of camera motion. This information will later be used to annul such motion and to provide a constant coordinate system. Selected static features should be motionless and (optimally) present throughout the entire video.

Use the RIGHT mouse button to select the static feature. Once a feature is selected, a regions representing interrogation area (IA) and search area (SA) will be shown around it.

Use the sliders at the bottom to adjust the IA/SA sizes.

If you wish to delete the previously selected feature, use the MIDDLE mouse button or the key D on keyboard. This will also clear the IA/SA regions from the image.

Toggle visibility of legend and point list using F1 key.
Use keys O and P to zoom and pan the image. Use keys LEFT and RIGHT to undo and redo the zoom or pan commands. Use key ENTER/RETURN to accept the selected features and start the tracking in all frames.

The results of the feature tracking will be stored in the `%PROJECT_FOLDER\transformation%` folder.

> **Note #7**: Feature tracking will not immediately produce stabilized images. This will be done after the two following steps have been completed.


#### Feature selection

<img align="right" width="830" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/select_features.png">

Not all of the tracked features have to be used for the transformation (stabilization) of frames. You can select features that will be used for the transformation using the **Select features for transformation** button. This will open a new form which display the positions and coordinates of tracked features. From the given list, you can choose which ones will be used to stabilize the original images.

> **Note #8**: An average SSIM tracking score (higher is better, 1 is ideal) will be shown next to each feature coordinates on the right side of the form.

To further help you choose the best features, an additional analysis is available by clicking the **Plot SSIM** scores button in the top-left corner of the Select features for transformation window. This will run the ssim_scores.py script and will show a bar graph of SSIM tracking scores for all frames. In the bar graph, better features will have a higher SSIM score and lower variance, which can help you decide which ones to keep and which ones to remove from the transformation.

<img align="right" width="500" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/ssim_scores.png">


#### Image transformation

The tool offers several options for the final stage, i.e. image transformation: (1) choosing the output images' extension, (2) choosing image quality, (3) image transformation method, (4) whether to use RANSAC filtering/outlier detection, and (5) orthorectification. The latter is explained in the **Orthorectification** section below.

The most important parameter is the **transformation method** which can significantly impact the stabilization accuracy. Five methods are available:

1. **Similarity**, based on [cv2.estimateAffinePartial2D](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad767faff73e9cbd8b9d92b955b50062d), which requires at least 2 features (4 degrees of freedom),
2. **Affine 2D (strict)**, based on [cv2.getAffineTransform](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga8f6d378f9f8eebb5cb55cd3ae295a999), which requires exactly 3 features,
3. **Affine 2D (optimal)**, based on [cv2.estimateAffine2D](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga27865b1d26bac9ce91efaee83e94d4dd), which requires at least 3 features,
4. **Projective (strict)**, based on [cv2.getPerspectiveTransform](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga8c1ae0e3589a9d77fffc962c49b22043), which requires exactly 4 features, and
5. **Projective (optimal)**, based on [cv2.findHomography](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780), which requires at least 4 features. This is the default option and is usually the best starting point.

> **Note #9:** RANSAC filtering option is only available for methods labeled as **(optimal)**.


#### Orthorectification

<img align="right" width="210" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/orthorectification.png">

The tool also offers a simple orthorectification to be performed by estimating the transformation matrix between the in-image positions of tracking features and their corresponding real-world coordinates.

**Orthorectification form** can be shown by clicking **Orthorectification** button on the righthand side of the **Stabilize/Orthorectify** panel. Here you can specify the known real-world coordinates (in meters) of a number of ground control points (at least 3).

By clicking **Apply** the GCP list will be saved and number of selected GCPs will be shown in the button label. IF orthorectification is configured, after clicking the **Image transformation** button you will be prompted to select in-image positions of these GCPs, which **CAN BE DIFFERENT** from those features tracked for the stabilization purposes.

> **Note #10**: The number of GCPs defined in the table and those selected in the image after clicking **Image transformation** button MUST BE THE SAME!

Clicking **Cancel** will turn the orthorectification step off for the project workflow, and only image stabilization will then be performed.

You should also set a ground sampling distance (GSD, in px/m) to rescale the image and plays an important role during postprocessing - use this feature carefully as it will always introduce additional errors/noise in the transformed images. **It's best to keep this ratio as close as possible to the original GSD!**. For these reasons you can click **Measure** button in the **Orthorectification** form to quickly measure in-image distances and to compare them with real-world data in order to help determine appropriate GSD value.


### Image enhancement

<img align="right" width="830" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/filtering.png">

Image filtering/enhancement is often a crucial part of the image velocimetry workflow. This tools offers a form in which such filtering can easily be performed. For those users unfamiliar with the image filtering/enhancement process, it is advisable to visit the [Image enhancement for UAV velocimetry](https://github.com/ljubicicrobert/Image-enhancement-for-UAV-velocimetry) repository. In that repository, a detailed overview of different aspects of image enhancement is provided through a series of Jupyter notebooks.

Before filtering, it is highly advisable to inspect the different colorspace models of the frames. This can be done by clicking on the **Explore colorspaces** button at the bottom-left of the panel.

Users can perform filtering of images in a folder using the **Enhance images** panel. Available filters are listed in the dropdown menu through which they can be added to the **filtering stack**. Filters will be added to the end of the stack by default, but can be reordered using the buttons on the righthand side. Filter parameters will also be shown in the brackets for each filter in the stack. By clicking on the filter, its paramters will be shown on the righthand side of the panel, which can be modified using either the trackbar or the corresponding boxes. Changes made through these will instantly be reflected in the filtering stack.

> **IMPORTANT**: Filters will be applied in the top-down order in the filtering stack.

Some of the available filter are just colorspace model conversions (titled _Convert to..._). These will transform the image from the previous colorspace to the chosen one. **Default colorspace model**, which is active when the image is loaded for filtering, is **RGB**.

> SSIMS-Flow will keep track of the colorspace conversions during filtering. For example, if _Convert to L\*a\*b\*_ is in the stack, followed by the _Single image channel_, by choosing the channel in the **Filter parameters** form, the user will effectively be selecting a channel of the image from its L\*a\*b\* colorspace model. Some filters will implicitly convert the image to a single-channel grayscale colorspace.

You can preview the results of the current stack (with the currently set parameters) by clicking the **Preview results** button. Once you are happy with the results, you can start creating filtered/enhanced images by clicking the **Filter frames** button in the lower right corner. This will apply the selected filters to all frames in the selected folder, and the resulting images will be stored in the `%PROJECT_FOLDER\enhancement%` folder.


### Optical flow (OF)

<img align="right" width="500" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/optical_flow.png">

Estimation of surface velocity field is performed using the Gunnar Farnebäck method (Farnebäck 2003). This method is a dense optical flow method, which provides estimation of motion (in X and Y directions) for all pixels in the image.

However, keeping information about per-pixel motion in high-resolution images requires significant amount of storage space (easily tens of MB per image). Additionally, if tracer particle seeding is sparse and/or periodic, only a smaller percentage of the image will experience motion between consecutive frames. Hence, an aggregation/pooling technique is applied to the raw results:

1. Results are first converted from U and V components (velicities in X and Y direction) to **magnitude and flow direction** (angle) representation.
2. The user should specify the **main direction** of the flow, either by entering the value in the box, or by pointing and clicking at the desired angle in the circle. Then, a flow direction tolerance (**angle range**) should be provided. Pixels whose flow angles do not fall into the range _main flow direction_ +/- _angle range_ will be masked out and their magnitudes will be replaced by 0.
3. The idea further is to reduce the size of the resulting matrix by aggregating the results from PxP sized blocks, where P is the **pooling block size** in pixels. We can assume that the tracer particle seeding is sparse (which is often true) which means that only a handful of pixels in each block will be likely to represent actual tracer motion, while the rest of the pixels will have magnitudes close to zero. A fair strategy for isolating valid pixels and their magnitudes is to calculate the mean magnitude of the block, select pixels with magnitudes greater than said mean, and then adopt the mean of those selected pixels as the representative magnitude of that pooling block. The same procedure is performed for all blocks in the image.
4. The resulting matrix will have P*P times fewer pixels, whilst still adequately representing the flow field.
5. Angles are pooled in a similar manner, but just using the mean of the block values.
6. While steps 3-5 cover the issue of spatial aggregation, temporal aggregation is performed by selecting the maximal pooled magnitude of the pixels obtained using the provided frame sequence. In order to reduce the chances of accidental overestimation of time-aggregated magnitudes, the parameters of the Farnebäck optical flow method have been chosen in such way to sacrifice a bit of computational speed for additional robustness.


### Post-OF analyses

<img align="right" width="830" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/optical_flow_analyses.png">

Once the optical flow analyses have been completed and the surface velocity field has been estimated, an estimation of the flow/discharge can be obtained by clicking on the **Analyze results** button in the Optical flow panel. This will open a new window, where you can define a profile (cross-section) of the channel, either from pixel coordinate values, or by choosing two points from an appropriate image.

> KEEP IN MIND that if you choose the profile from an image, you should choose it from a frame of the same folder which was used for optical flow analysis. In certain cases the image enhancement step will degrade visual information which can be used to identify the channel cross-section. If this is the case, you should use the frames from folder which was used to obtain enhanced images (e.g., orthorectified or stabilized frames).

Once the profile (cross-section) has been selected, click the **Generate profile data** to create an analysis of the time averaged surface velocities in that profile. You can control the interpolation of values from 2D optical flow data onto 1D profile data by defining a number of interpolation points and the interpolation order used for spline interpolation.

Graphical representation of the data can be obtained by selecting columns using checkboxes just below the column headers. Velocity data will be shown on the primary Y axis and direction/angle data on the secondary Y axis.

Finally, in order to obtain the flow estimation, you need to enter the depth profile/bathymetry data in the table in the lower left of the window. Please note that the water surface is assumed to be zero, and that depths should all be defined using positive numbers. Estimated flow will be shown on the righthand size of the window.


### Video creation

<img align="right" width="440" src="https://github.com/ljubicicrobert/SSIMS-Flow/blob/main/screenshots/create_video.png">

Videos can be created using this tool regardless of whether the project has been set up or loaded. In the **Create video** panel, frames folder can be either selected manually, or, if the project results are available, from stabilized, orthorectified, or enhanced images.


## Future features

&#9744; Adding options for easier and more complete post-OF analyses

&#9744; Creation of videos using optical flow data

&#9744; Compiling bottlenecks into C++ to reduce the overall computational complexity


## Acknowledgements

I wish to express my gratitude to the following people (in no particular order):

[Mrs. Sophie Pierce](https://www.worcester.ac.uk/about/profiles/sophie-pearce) - for motivating me to start the work in the first place;

[Dr Budo Zindović](https://www.grf.bg.ac.rs/fakultet/pro/e?nid=153) - for helping me with many implementational details and extensive testing of the tool;

[Mrs. Dariia Strelnikova](https://www.fh-kaernten.at/en/en/faculty-and-staff-details?personId=4298793872) for supporting and reviewing the work related to the image enhancement;

[Mrs. Dariia Strelnikova](https://www.fh-kaernten.at/en/en/faculty-and-staff-details?personId=4298793872) (again) and [Dr Anette Eltner](https://tu-dresden.de/bu/umwelt/geo/ipf/photogrammetrie/die-professur/beschaeftigte/Anette_Eltner?set_language=en) - for testing the software and allowing me to learn from their own work;

[Dr Alonso Pizarro](https://www.researchgate.net/profile/Alonso_Pizarro) and [Dr Salvador Peña‐Haro](https://www.researchgate.net/profile/Salvador_Pena-Haro) - for providing me with valuable insights into their own work;

[Dr Salvatore Manfreda](https://www.salvatoremanfreda.it) and [Dr Silvano Fortunato Dal Sasso](https://www.researchgate.net/profile/Silvano_Fortunato_Dal_Sasso) - for hosting me at the Università Basilicata in Potenza where I have learned and improved my algorithm;

[Dr Matthew T. Perks](https://www.ncl.ac.uk/gps/staff/profile/matthewperks.html) - for providing me with helpful comments, as well as providing most of the camera parameters.


## References

Wang, Z., Bovik, A. C., Sheikh, H. R. and Simoncelli, E. P. (2004) *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE Trans. Image Process., 13(4), 600–612, [https://doi.org/10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861)

Fast-SSIM by Chen Yu ([https://github.com/chinue/Fast-SSIM](https://github.com/chinue/Fast-SSIM))

Farnebäck, G. (2003) *Two-frame motion estimation based on polynomial expansion*, Scandinavian conference on Image analysis. Springer, Berlin, Heidelberg.


## How to cite
Ljubicic, R. (2022) *SSIMS-Flow: Preprocessing tool for UAV image velocimetry*, [https://github.com/ljubicicrobert/SSIMS-Flow](https://github.com/ljubicicrobert/SSIMS-Flow)

&nbsp;&nbsp;&nbsp;&nbsp;or (for image stabilization/orthorectification only)

Ljubičić R., Strelnikova D., Perks M.T., Eltner A., Pena-Haro S., Pizarro A., Dal Sasso S.F., Scherling U., Vuono P. and Manfreda S (2021) _A comparison of tools and techniques for stabilising unmanned aerial system (UAS) imagery for surface flow observations_. Hydrology and Earth System Sciences. 25 (9), pp.5105--5132, [https://doi.org/10.5194/hess-25-5105-2021](https://doi.org/10.5194/hess-25-5105-2021)

Performance evaluation of image stabilization accuracy and comparison of results to similar tools available at [https://doi.org/10.5281/zenodo.4557921](https://doi.org/10.5281/zenodo.4557921).


## License and disclaimer

This tool is published under the [General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) and can be used and distributed freely and without charge. The author does not bear the responsibility regarding any possible (mis)use of the software/code, as well as for any damages (physical, hardware, and/or software) that may arise from the use of this tool. The package was scanned using Avast Antivirus prior to the upload, but a rescan after download is always recommended.
