from setuptools import setup

setup(name='km3pipe',
      version='0.1.3',
      url='http://github.com/tamasgal/km3pipe/',
      description='An analysis framework for KM3NeT',
      author='Tamas Gal',
      author_email='tgal@km3net.de',
      packages=['km3pipe', 'km3pipe.testing', 'km3pipe.pumps', 'pipeinspector'],
      include_package_data=True,
      platforms='any',
      install_requires=[
          'numpy',
          'urwid',
          'docopt',
      ],
      entry_points={
          'console_scripts': [
              'pipeinspector=pipeinspector.app:main',
          ],
      },
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
      ],
)

__author__ = 'Tamas Gal'
