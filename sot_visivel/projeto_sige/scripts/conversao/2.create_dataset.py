import yaml
config = {
    'path': '/Users/ighorribeiro/Downloads/Anti_uav_3uav_visible',
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': 1,
    'names': ['drone']
}
with open('/Users/ighorribeiro/Downloads/Anti_uav_3uav_visible/dataset.yaml', 'w') as f:
    yaml.dump(config, f)