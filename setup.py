from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_libraries = f.read().splitlines()
# import ipdb;ipdb.set_trace()
setup(
    name="enformer_dna_diff",
    version="1.1",
    packages=find_packages(),
    install_requires=required_libraries,
    scripts=['get_data.py'],
    author="Tin M. Tunjic",
    author_email="tunjictin@gmail.com",
    description="This is a package for that combines DeepMind Enformer model with DNA Diffusion project",
    url="https://github.com/ttunja/Enformer_DNA_Diffusion.git"
)
