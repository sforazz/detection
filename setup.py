from setuptools import setup, find_packages

setup(name='Detection',
      version='1.0',
      description='Python repository with some utilities to object detection and tracking',
      url='https://github.com/sforazz/detection.git',
      python_requires='>=3.5',
      author='Francesco Sforazzini',
      author_email='f.sforazzini@dkfz.de',
      license='Apache 2.0',
      zip_safe=False,
      include_package_data=True,
      install_requires=[
      'imutils==0.5.3',
      'matplotlib==3.3.0',
      'numpy==1.19.1',
      'opencv-contrib-python==4.3.0.36',
      'scipy==1.5.2'],
      entry_points={
          'console_scripts': ['plot_object_position = scripts.plot_object_position:main',
			      'histogram_average = scripts.histogram_averaging:main']},
      packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )
