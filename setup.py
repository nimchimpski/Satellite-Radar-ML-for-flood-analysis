from setuptools import setup, find_packages

setup(
    name="UNOSAT_FloodAI_v2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g.,
        "wandb",
        "torch",
    ],
)
