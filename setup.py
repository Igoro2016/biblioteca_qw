from setuptools import setup, find_packages

setup(
    name='biblioteca',
    version='0.1.0',
    description='Simulação de caminhadas quânticas em grafos toroidais',
    author='Seu Nome',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytest',
        'requests',
    ],
)