from setuptools import setup, find_packages

setup(
    name="stud_poker_rl",
    version="0.1.0",
    description="Deep Reinforcement Learning for Stud Poker Variants",
    author="WaterQualityInspector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "open_spiel>=1.2.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.12.0",
        "tqdm>=4.65.0",
    ],
)
