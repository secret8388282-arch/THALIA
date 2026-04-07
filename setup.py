# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="thalia-ai",
    version="0.9.0",  
    description="Thalia: Cognitive Architecture with Dynamic Psyche, Hebb Memory and Self-Reflection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Secret",  
    author_email="thalia@cognitive.arch",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "transformers>=4.39",
        "peft>=0.10.0",
        "accelerate>=0.20.0",
        "numpy",
        "tqdm",
        "optuna",
        "pyyaml",
        "datasets",
        "gradio",  
    ],
    extras_require={
        "dev": [
            "jupyter",
            "matplotlib",
            "seaborn",
            "streamlit",
            "pytest",  
            "black",   
        ],
        "philosophy": [  
            "markdown",
            "pypandoc",
        ]
    },
    entry_points={
        'console_scripts': [
            'thalia-create=thalia.create_model:main',
            'thalia-gui=thalia.interface:launch_gui',
            'thalia-think=thalia.philosophy:reflect', 
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Philosophy",  
    ],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False
)