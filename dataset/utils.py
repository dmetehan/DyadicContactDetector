class Aug:
    hflip = 'hflip'
    crop = 'crop'
    rotate = 'rotate'
    swap = 'swap'

class Options:
    debug = 0
    gaussian = 1  # gaussian heatmaps around detected keypoints
    jointmaps = 2  # detected heatmaps mapped onto cropped image around interacting people
    gaussian_rgb = 3
    jointmaps_rgb = 4
    gaussian_rgb_bodyparts = 5
    jointmaps_rgb_bodyparts = 6
    all = {"debug": debug, "gaussian": gaussian, "jointmaps": jointmaps, "gaussian_rgb": gaussian_rgb,
           "jointmaps_rgb": jointmaps_rgb, "gaussian_rgb_bodyparts": gaussian_rgb_bodyparts,
           "jointmaps_rgb_bodyparts": jointmaps_rgb_bodyparts}