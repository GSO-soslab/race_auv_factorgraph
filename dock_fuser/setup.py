from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dock_fuser'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        (os.path.join('share', package_name), ["package.xml"]),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@example.com',
    description='gtsam factor graph for docking station pose estimation.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'dock_fuser_node = dock_fuser.ros.node:main',
        ],
    },
)
