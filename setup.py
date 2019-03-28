#!/usr/bin/env python3
from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


def get_version(short=False):
    with open('README.rst') as f:
        for line in f:
            if ':Version:' in line:
                ver = line.split(':')[2].strip()
                if short:
                    subver = ver.split('.')
                    return '%s.%s' % tuple(subver[:2])
                else:
                    return ver


setup(name='anesthetic',
      version=get_version(),
      description='anesthetic: nested sampling post-processing',
      long_description=readme(),
      author='Will Handley',
      author_email='wh260@cam.ac.uk',
      url='https://github.com/williamjameshandley/anesthetic',
      packages=find_packages(),
      install_requires=['numpy', 'scipy'],
      setup_requires=['pytest-runner'],
      extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme', 'numpydoc'],
          },
      tests_require=['pytest'],
      include_package_data=True,
      license='MIT',
      classifiers=[
                   'Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Natural Language :: English',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Topic :: Scientific/Engineering :: Visualization',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: Scientific/Engineering :: Mathematics',
      ],
      )
