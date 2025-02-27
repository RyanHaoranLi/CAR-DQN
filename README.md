# CAR-DQN Reference Implementation: Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error

This repository contains a reference implementation for Consistent Adversarial Robust Deep
Q Networks (CAR-DQN). See our paper ["Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error"](https://arxiv.org/abs/2402.02165) for more details. This paper has been accepted by [**ICML 2024**](https://proceedings.mlr.press/v235/li24cl.html) as an [**oral**](https://icml.cc/virtual/2024/oral/35463) presentation.

The reference implementation for Consistent Adversarial Robust Proximal Policy Gradient (CAR-PPO) can be found at [RyanHaoranLi/CAR-RL](https://github.com/RyanHaoranLi/CAR-RL). See our paper ["Towards Optimal Adversarial Robust Reinforcement Learning with Infinity Measurement Error"](https://arxiv.org/abs/2502.16734) for more details.

Our PGD version code is based on the [SA-DQN](https://github.com/chenhongge/SA_DQN) codebase and the IBP version code is based on the [RADIAL-DQN](https://github.com/tuomaso/radial_rl_v2) codebase.

## Requirements
To run our code you need to have Python 3 (>=3.7) and pip installed on your systems. Additionally, we require PyTorch>=1.4, which should be installed using instructions from https://pytorch.org/get-started/locally/.

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

## Citation
```shell
@InProceedings{li2024towards,
  title = 	 {Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error},
  author =       {Li, Haoran and Zhang, Zicheng and Luo, Wang and Han, Congying and Hu, Yudong and Guo, Tiande and Liao, Shichen},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {29324--29372},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/li24cl/li24cl.pdf},
  url = 	 {https://proceedings.mlr.press/v235/li24cl.html}
}
```
