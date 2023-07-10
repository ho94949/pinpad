from setuptools import setup

setup(
    name='gym_pinpad',
    version='2023.7.10',
    keywords='environment, agent, rl, openaigym, openai-gym, gym, 2d',
    packages=['gym_pinpad'],
    install_requires=[
        'gym==0.23.1',
        'numpy>=1.10.0',
    ],
    extras_require={'gui': ['pygame', 'Pillow']},
    entry_points={'console_scripts': ['gym_pinpad=gym_pinpad.run_gui:main']},
    # Include textures and meshes in the package
    include_package_data=True
)
