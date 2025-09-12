from setuptools import setup
import pkg_resources

with open('requirements.txt') as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='MIMo',
    version='2.0.0',
    url='',
    license='',
    author='Francisco M. López, Dominik Mattern, Miles Lenz, Pierre Schumacher',
    packages=['mimoEnv', 'mimoVision', 'mimoVestibular', 'mimoProprioception', 'mimoTouch', 'mimoActuation', 'mimoGrowth'],
    install_requires=install_requires,
    author_email='lopez@fias.uni-frankfurt.de',
    description='MIMo-v2 library'
)
