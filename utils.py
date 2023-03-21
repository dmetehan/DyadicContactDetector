import yaml

# Aug.swap: Swapping the order of pose detections between people (50% chance)
#           NOT WORKING WELL because I believe the order is always top left to bottom right and switching it randomly doesn't help
# Aug.hflip: Horizontally flipping the rgb image as well as flipping left/right joints
# Aug.crop: Cropping
class Aug:
    hflip = 'hflip'
    crop = 'crop'
    rotate = 'rotate'
    swap = 'swap'
    all = {'hflip': hflip, 'crop': crop, 'rotate': rotate, 'swap': swap}

class Options:
    debug = "debug"
    gaussian = "gaussian"  # gaussian heatmaps around detected keypoints
    jointmaps = "jointmaps"  # detected heatmaps mapped onto cropped image around interacting people
    gaussian_rgb = "gaussian_rgb"
    jointmaps_rgb = "jointmaps_rgb"
    gaussian_rgb_bodyparts = "gaussian_rgb_bodyparts"
    jointmaps_rgb_bodyparts = "jointmaps_rgb_bodyparts"
    all = {"debug": debug, "gaussian": gaussian, "jointmaps": jointmaps, "gaussian_rgb": gaussian_rgb,
           "jointmaps_rgb": jointmaps_rgb, "gaussian_rgb_bodyparts": gaussian_rgb_bodyparts,
           "jointmaps_rgb_bodyparts": jointmaps_rgb_bodyparts}


def check_config(cfg):
    assert cfg.OPTION in Options.all
    for aug in cfg.AUGMENTATIONS:
        assert aug in Aug.all

def parse_config(config_file):
    class Config:
        pass
    with open(config_file, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    cfg_obj = Config()
    for section in cfg:
        print(section, cfg[section])
        setattr(cfg_obj, section, cfg[section])
    setattr(cfg_obj, "ID", config_file.split('/')[-1].split('_')[0])
    check_config(cfg_obj)
    return cfg_obj