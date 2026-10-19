from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'race_auv_factorgraph'

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
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
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
            'dock_fuser_node = race_auv_factorgraph.ros.node:main',
            'replay_bag = race_auv_factorgraph.tools.replay_bag:main',
            'evaluate = race_auv_factorgraph.tools.evaluate:main',
            'plot_bev = race_auv_factorgraph.tools.plot_bev:main',
            'plot_graph = race_auv_factorgraph.tools.plot_graph:main',
            'batch_lm = race_auv_factorgraph.tools.batch_lm:main',
            'compare_backends = race_auv_factorgraph.tools.compare_backends:main',
            'synthetic_demo = race_auv_factorgraph.tools.synthetic:main',
        ],
    },
)
