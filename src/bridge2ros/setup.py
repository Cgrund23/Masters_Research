from setuptools import find_packages, setup

package_name = 'bridge2ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    
    data_files=[
    	('share/' + package_name +"/launch", ['launch/bridge_launch.launch.py']),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='parallels',
    maintainer_email='parallels@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "shout = bridge2ros.shout:main",
            "test = bridge2ros.test:main",
            "cbf = bridge2ros.cbf_Node:main",
            "visual = bridge2ros.visual_Node:main",
            
        ],
    },
)
