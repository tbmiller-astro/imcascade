from setuptools import setup, Extension, find_packages

with open('requirements.txt') as infd:
    INSTALL_REQUIRES = [x.strip('\n') for x in infd.readlines()]
    print(INSTALL_REQUIRES)

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name="imcascade",
    version="1.1",
    author="Tim Miller",
    author_email="tim.miller@yale.edu",
    description="imcascade: Fitting astronomical images using a 'cascade' of Gaussians",
    long_description_content_type="text/x-rst",
    long_description=readme(),
    url="https://github.com/tbmiller-astro/imcascade",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"],
    license='MIT',

)
