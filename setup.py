from setuptools import setup

setup(name='km3pipe',
      version='0.0.1',
      url='http://github.com/tamasgal/km3pipe/',
      description='An analysis framework for KM3NeT',
      author='Tamas Gal',
      author_email='tgal@km3net.de',
      packages=['km3pipe'],
      include_package_data=True,
      platforms='any',
      install_requires=[
      ],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
      ],
)

__author__ = 'Tamas Gal'
