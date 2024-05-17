# CAR-DQN Reference Implementation: Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error

This repository contains a reference implementation for Consistent Adversarial Robust Deep
Q Networks (CAR-DQN). See our paper ["Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error"](https://arxiv.org/abs/2402.02165) for more details. This paper has been accepted by **ICML 2024**.

Our PGD version code is based on the [SA-DQN](https://github.com/chenhongge/SA_DQN) codebase and the IBP version code is based on the [RADIAL-DQN](https://github.com/tuomaso/radial_rl_v2) codebase.

## Requirements
To run our code you need to have Python 3 (>=3.7) and pip installed on your systems. Additionally we require PyTorch>=1.4, which should be installed using instructions from https://pytorch.org/get-started/locally/.

To install requirements:

```setup
pip install -r requirements.txt
```

## Training (PGD)
Enter the 'PGD' directory and run the following command to train a CAR-DQN model on Pong:
```shell
cd PGD
python train.py --config config/Pong_pgd.json training_config:load_model_path=models/Pong-natural_sadqn.pt
```

## Training (IBP)
Enter the 'IBP' directory and run the following command to train a CAR-DQN model on Pong:
```shell
cd IBP
python main.py --env PongNoFrameskip-v4 --load-path "models/Pong-natural_sadqn.pt"
```

## Evaluation
To evaluate a trained model on Pong, run the following command:
```shell
cd IBP
python evaluate.py --env PongNoFrameskip-v4 --load-path <model-path> --fgsm --pgd --nominal --acr
```