# Kami: A hair simulator

This is a hair simulator which can compare different models e.g. Augmented Mass Spring or Stable Cosserat Rods.

<video src="img/demo.mp4" controls="true"></video>


https://github.com/user-attachments/assets/99574c63-40f9-47e6-b0ce-2d236b428f0f


## How to build

Before proceeding to the installation steps, make sure you have CUDA Toolkit 11.6 (or later) installed on your machine. Please follow the [installation guide](https://docs.nvidia.com/cuda/) provided by NVIDIA for Microsoft Windows or Linux.

You may have to type the below command to install Python3.

```shell
sudo apt-get update
sudo apt update
sudo apt install python3
```

Please resolve the other dependencies with the command below.

```shell
sudo apt-get update
sudo apt update
sudo apt install ffmpeg libjsoncpp-dev libassimp-dev
pip install mitsuba
```

Please open this directory in a terminal and type the commands below.

```shell
mkdir build
cd build
cmake ..
make
```

## Data preparation

[Please read here](/resources/README.md).

## Running Sample Schene

Please open bin directory, open the terminal at the directory, and run

```shell
cd bin
./kami
```

or

```shell
cd bin
./kami ../scripts/json/KamiSetting.json
```

## for more details

See [Table of Contents for KamiSimulator Guides](doc/toc.md).

## citations

- Digital-Salon
  
```text
	@article{digitalsalon,
		title={Digital Salon: An AI and Physics-Driven Tool for 3D Hair Grooming and Simulation},
		author={He, Chengan and Herrera, Jorge Alejandro Amador and Zhou, Yi and Shu, Zhixin and Sun, Xin and Feng, Yao and Pirk, S{\"o}ren and Michels, Dominik L and Zhang, Meng and Wang, Tuanfeng Y and Rushmeier, Holly},
		url={https://digital-salon.github.io/},
		year={2024}
	}
```

- Mitsuba3

```text
	@software{jakob2022mitsuba3,
    title = {Mitsuba 3 renderer},
    author = {Wenzel Jakob and Sébastien Speierer and Nicolas Roussel and Merlin Nimier-David and Delio Vicini and Tizian Zeltner and Baptiste Nicolet and Miguel Crespo and Vincent Leroy and Ziyi Zhang},
    note = {https://mitsuba-renderer.org},
    version = {3.0.1},
    year = 2022,
	}
```

- [USC-HairSalon]( https://huliwenkidkid.github.io/liwenhu.github.io/ )
