#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
        'numpy >= 1.13.3',
        'scipy >= 1.1.0',
        'matplotlib >= 2.2.2',
        'mplstereonet == 0.5']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Exneyder A. Montoya-Araque & Ludger O. Suarez-Burgoa",
    author_email='eamontoyaa@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="Application software for applying the Second-order tensor statistical proposal of Jelínek (1978).",
    install_requires=requirements,
    license="BSD 2-Clause License",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords=['Anisotropy of Magnetic Susceptibility', 'Jelínek', 'tensor', 'statistics', 'Python', 'application software'],
    name='jelinekstat',
    packages=find_packages(include=['jelinekstat']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/eamontoyaa/jelinekstat',
    version='0.1.0',
    zip_safe=False,
)
