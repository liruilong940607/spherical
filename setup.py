
from setuptools import find_packages, setup

setup(
    name="spherical",
    install_requires=[
        "ninja",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)
