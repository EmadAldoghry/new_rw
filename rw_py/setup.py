from setuptools import find_packages, setup

package_name = 'rw_py'

setup(
    name=package_name,
    version='1.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Emad Aldoghry',
    maintainer_email='Aldoghry@isac.rwth-aachen.de',
    description='Refactored Python nodes for modular navigation with visualization',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orchestrator = rw_py.orchestrator_node:main',
            'waypoint_publisher = rw_py.waypoint_publisher_node:main',
            'visual_fuser = rw_py.visual_fusion_node:main', # Replaces two old nodes
            'centroid_calculator = rw_py.centroid_calculator_node:main',
            'navigation_logic = rw_py.navigation_logic_node:main',
        ],
    },
)