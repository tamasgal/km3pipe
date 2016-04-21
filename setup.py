from setuptools import setup

from km3pipe import version

setup(name='km3pipe',
      version=version,
      url='http://github.com/tamasgal/km3pipe/',
      description='An analysis framework for KM3NeT',
      author='Tamas Gal',
      author_email='tgal@km3net.de',
      packages=['km3pipe', 'km3pipe.testing', 'km3pipe.pumps',
                'km3modules', 'pipeinspector'],
      include_package_data=True,
      platforms='any',
      install_requires=[
          'cython',
          'numpy',
          'controlhost',
          'urwid',
          'docopt',
          'pandas',
          'seaborn',
          'pytz',
          'six',
      ],
      entry_points={
          'console_scripts': [
              'km3pipe=km3pipe.cmd:main',
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
