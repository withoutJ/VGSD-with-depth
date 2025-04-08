# Video Glass Surface Detection Using Self-Supervised Depth Estimation 

Official code release of my final year project "Multi-view Dynamic Reflection Prior for Video Glass Surface Detection"


### Dataest

The training and testing dataset is available at [Google drive](https://drive.google.com/drive/folders/1QsdYI5Gwi-rKKwGgdE7GFTjhRO4-wIiI?usp=sharing). 


### Evaluation
Download the predicted results from the [link](https://drive.google.com/file/d/1qxpBJvLWVOep1BDAuQSa80QoiNwRSTxF/view?usp=sharing) and run the following command to evaluate the results.

```bash
python eval.py -pred ../results/pred  -gt ../VGSD_dataset/test
```

### Inference
Download the trained model from the [VGSD.pth](https://drive.google.com/file/d/1PAcYNS9LsUd0E9QdUGOGQZcfebJVj8B0/view?usp=sharing) and run the following command to generate the predicted results.

```bash
python infer.py -pred ./results/ -exp ./checkpoints/VGSD_with_depth.pth 
```

### Training
Download backbone weights for encoder from the [resnext_101_32x4d.pth](https://github.com/fawnliu/VGSD/releases/download/1.0/resnext_101_32x4d.pth) and run `train.py` to train the model.

### Acknowledgement

Source code from the following repositories was used: [VGSD](https://github.com/fawnliu/VGSD), [GSD](https://jiaying.link/cvpr2021-gsd/code.zip), [VMD](https://jiaying.link/cvpr2023-vmd/), [BiFormer](https://github.com/rayleizhu/BiFormer), [SIRR](https://github.com/zdlarr/Location-aware-SIRR). Thanks to the authors for their excellent work.

