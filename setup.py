from setuptools import setup, find_packages
import pkg_resources

with open('requirements.txt') as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='MIMo',
    version='1.1.0',
    url='',
    license='',
    author='Dominik Mattern, Francisco M. López, Pierre Schumacher',
    packages=find_packages(),
    install_requires=install_requires,
    author_email='domattee@yahoo.de',
    description='MIMo library'
)
