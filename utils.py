import os
import glob
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Aug.swap: Swapping the order of pose detections between people (50% chance)
#           NOT WORKING WELL because I believe the order is always top left to bottom right and switching it randomly doesn't help
# Aug.hflip: Horizontally flipping the rgb image as well as flipping left/right joints
# Aug.crop: Cropping
class Aug:
    hflip = 'hflip'
    crop = 'crop'
    rotate = 'rotate'
    swap = 'swap'
    color = 'color'
    all = {'hflip': hflip, 'crop': crop, 'rotate': rotate, 'swap': swap, 'color': color}

class Options:
    debug = "debug"
    rgb = "rgb"
    bodyparts = "bodyparts"
    gaussian = "gaussian"  # gaussian heatmaps around detected keypoints
    jointmaps = "jointmaps"  # detected heatmaps mapped onto cropped image around interacting people
    gaussian_rgb = "gaussian_rgb"
    jointmaps_rgb = "jointmaps_rgb"
    rgb_bodyparts = "rgb_bodyparts"
    jointmaps_bodyparts = "jointmaps_bodyparts"
    gaussian_rgb_bodyparts = "gaussian_rgb_bodyparts"
    jointmaps_rgb_bodyparts = "jointmaps_rgb_bodyparts"
    all = {"debug": debug, "rgb": rgb, "bodyparts": bodyparts, "gaussian": gaussian, "jointmaps": jointmaps, "gaussian_rgb": gaussian_rgb,
           "jointmaps_rgb": jointmaps_rgb, "rgb_bodyparts": rgb_bodyparts, "jointmaps_bodyparts": jointmaps_bodyparts,
           "gaussian_rgb_bodyparts": gaussian_rgb_bodyparts, "jointmaps_rgb_bodyparts": jointmaps_rgb_bodyparts}


def check_config(cfg):
    print(cfg.AUGMENTATIONS)
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


def get_experiment_name(cfg):
    experiment_name = f'{cfg.ID}' \
                      f'_{cfg.OPTION}' \
                      f'{"_pretr" if cfg.PRETRAINED else ""}{"Copy" if cfg.PRETRAINED and cfg.COPY_RGB_WEIGHTS else ""}' \
                      f'_{cfg.TARGET_SIZE[0]}' \
                      f'{"_Aug-" if len(cfg.AUGMENTATIONS) > 0 else ""}{"-".join(cfg.AUGMENTATIONS)}' \
                      f'{"_strat" if cfg.STRATIFIED else ""}' \
                      f'_lr{cfg.LR}' \
                      f'_b{cfg.BATCH_SIZE}'
    print("Experiment name:")
    print(experiment_name)
    return experiment_name


def find_last_values_tensorboard(log_dir, tag):
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    assert len(event_files) > 0, "No event files found in log directory."
    event_files.sort(key=os.path.getmtime)

    event_file = event_files[-1]  # Get the latest event file.
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    scalar_events = event_acc.Scalars(tag)
    assert len(scalar_events) > 0, f"No events found for tag '{tag}' in {log_dir}."
    return scalar_events[-1].value
