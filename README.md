## Training

Our method relies on a pretrained Histogram-assisted NeRF model and a DFNet model as we stated in the paper. We have provide example config files in our repo. The followings are examples to train the models.

- NeRF model

```sh
python run_nerf.py --config config_nerfh.txt
```

- DFNet model

```sh
python run_feature.py --config config_dfnet.txt
```

- Direct Feature Matching using DSO (DFNet<sub>dm</sub>)

```sh
python train_dfnet_nerf_dso.py --config config_dfnetdm.txt --real --pose_avg_stats_mode 100 --no_grad_update False --lrate 1e-9 --learning_rate 0.0001
```

## Evaluation
We provide methods to evaluate our models.
- To evaluate the NeRF model in PSNR, simply add `--render_test` argument.

```sh
python run_nerf.py --config config_nerfh.txt --render_test
```

- To evaluate APR performance of the DFNet model, you can just add `--eval --testskip=1 --pretrain_model_path=../logs/PATH_TO_CHECKPOINT`. For example:

```sh
python run_feature.py --config config_dfnet.txt --eval --testskip=1 --pretrain_model_path=../logs/heads/dfnet/checkpoint.pt
```

- Same to evaluate APR performance for the DFNet using DSO<sub>dm</sub> model

```sh
python multi_eval.py --opt heads --pose_avg_stats_mode 1 --exp dfnetdm --pose_avg_stats_mode_pred_eval -1
```