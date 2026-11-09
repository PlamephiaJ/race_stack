from setuptools import setup
import os
from glob import glob

package_name = 'stack_master'
source_root = os.path.dirname(os.path.realpath(__file__))


#handle the maps subfolders
map_subfolders = []
for map_dir in os.listdir('maps'):
    map_subfolders.append((os.path.join('share', package_name, 'maps', map_dir), glob('maps/{}/*'.format(map_dir), recursive=True)))

#handle the config folder, which nests arbitrarily deep (e.g. config/NUC5/pacejka/<floor>/archive/*.yaml)
config_subfolders = []
# colcon symlink builds retain links to removed files in the build directory.
# Discover the current source files even when setup.py runs from that directory.
for root, _, files in os.walk(os.path.join(source_root, 'config')):
    relative_root = os.path.relpath(root, source_root)
    sources = [os.path.join(relative_root, f) for f in files if os.path.isfile(os.path.join(root, f))]
    if sources:
        config_subfolders.append((os.path.join('share', package_name, relative_root), sources))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'launch', 'subsystems'), glob(os.path.join('launch', 'subsystems', '*launch.[pxy][yma]*'))),
        *config_subfolders,
        *map_subfolders,
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ForzaETH',
    maintainer_email='nicolas.baumann@pbl.ee.ethz.ch',
    description='The package to rule them all: Launch files and Configs',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_parameter_node = stack_master.global_parameter_node:main',
            'static_tf_publisher = stack_master.static_tf_publisher:main',
        ],
    },
)
