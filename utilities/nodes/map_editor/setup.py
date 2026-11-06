from setuptools import find_packages, setup

setup(
    name='map_editor',
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/map_editor']),
        ('share/map_editor', ['package.xml', 'README.md', 'LICENSE']),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML', 'PySide6>=6.4,<6.12'],
    zip_safe=False,
    maintainer='ForzaETH',
    maintainer_email='nicolas.baumann@pbl.ee.ethz.ch',
    description='Independent Qt occupancy map editor with offline Global Planner preview.',
    license='MIT',
    entry_points={'console_scripts': ['map_editor = map_editor.app:main']},
)
