from setuptools import setup, find_packages

setup(
    name="enformer_dna_diff",
    version="1.0",
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    author="Tin M. Tunjic",
    author_email="tunjictin@gmail.com",
    description="This is a package for that combines DeepMind Enformer \
                model with DNA Diffusion project",
)