## Prerequisites

* Cuda 11.7
* Conda
* A C++14 capable compiler
  * __Windows:__ Visual Studio 2019 or 2022
  * __Linux:__ GCC/G++ 8 or higher

## Setup
First make sure all the Prerequisites are installed in your operating system. Then, invoke

```bash
conda create --name anim-gaussian python=3.8
conda activate anim-gaussian
bash ./install.sh
```

## Prepare Data
We use PeopleSnapshot and GalaBasketball datasets and correspoding template body model. Please [download](https://drive.google.com/drive/folders/1xyLF7UwIrUaU5KU0IsEjYrz9hdTeZuza?usp=sharing) and organize as follows
```bash
|---data
|   |---Gala
|   |---PeopleSnapshot
|   |---smpl
```

## Train
To train a scene, run

```bash
python train.py --config-name <main config file name> dataset=<path to dataset config file>
```

For the PeopleSnapshot dataset, run
```bash
python train.py --config-name peoplesnapshot.yaml dataset=peoplesnapshot/male-3-casual
```

For the GalaBasketball dataset, run
```bash
python train.py --config-name gala.yaml dataset=gala/idle
```

## Test

To test the trained model performance on the test set, run
```bash
python test.py --config-name <main config file name> dataset=<path to dataset config file>
```

To animate the trained model at the novel poses, run
```bash
python test.py --config-name animate.yaml output_path=<path to train output> dataset=<path to animation dataset config file>
```

For example, to animate the model trained on the idle dataset using poses on the dribble dataset, run
```bash
python test.py --config-name animate.yaml output_path=idle dataset=gala/dribble
```

