
0. Setup

```bash
module load mamba
source activate medmnist
```

Install conda env from YAML

```bash
 wget -i /home/ljwoods2/workspace/MedMNIST-experiments/files.txt /scratch/ljwoods2/medmnist/weights
```

```bash
cd /scratch/ljwoods2/medmnist/data
python -m medmnist download --size 224
```

train_and_eval_pytorch.py modified to use "root" kwarg with `DataClass` and `medmnist.Evaluator` with path to data in "scratch"

1. train a Resnet-50 on ChestMNIST 5 times. take mean and stdev and report

```bash
python MedMNIST2D/train_and_eval_pytorch.py --output_root ./output/chestmnist --num_epochs 100 --size 28 --gpu_ids 0 --model_flag resnet50 --run 1 --data_flag chestmnist
```

--model_path /scratch/ljwoods2/medmnist/weights/weights_chestmnist/resnet50_28_3.pth weights_bloodmnist.zip

2. train a Resnet-50 on all 2D and 3D datasets 5 times. take mean and stdev and report each




3. how to train all together?