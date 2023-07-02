from setuptools import setup, find_packages

HYPHEN_E_DOT = "-e ."

def get_requirements_list():
    package_items = []
    with open("requirements.txt") as requirement_file:
        requirement_list = requirement_file.readlines()
        for package in requirement_list:
            package_items.append(package.replace("\n", ""))
            if HYPHEN_E_DOT in package_items:
                package_items.remove(HYPHEN_E_DOT)
                return package_items

setup(
    name='ml_sensor_fault_detection',
    version='0.0.1',
    author='Msn',
    description='SENSOR_FAULT_DETECTION',
    packages=find_packages(),
    install_requires=get_requirements_list()
)