# R-MSFM: Recurrent Multi-Scale Feature Modulation for Monocular Depth Estimating(ICCV-2021)
This is the official implementation for testing depth estimation using the model proposed in 
>R-MSFM: Recurrent Multi-Scale Feature Modulation for Monocular Depth Estimating
>Zhongkai Zhou, Xinnan Fan, Pengfei Shi, Yuanxue Xin

R-MSFM can estimate a depth map from a single image.

Paper is now available at [ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_R-MSFM_Recurrent_Multi-Scale_Feature_Modulation_for_Monocular_Depth_Estimating_ICCV_2021_paper.pdf)

## Training code
You can train R-MSFM3 with:
```shell
python train.py --iters=3
```
or R-MSFM6 with
```shell
python train.py --iters=6
```
or R-MSFM3-GC with
```shell
python train.py --iters=3  --gc
```
or R-MSFM6-GC with
```shell
python train.py --iters=6  --gc
```

## Improved Version(T-PAMI2024)
Paper is now available at [T-PAMI2024](https://ieeexplore.ieee.org/abstract/document/10574331).
In this paper, we improve our R-MSFM from two aspects and achieve SOTA results.
1.  We propose another lightweight convolutional network R-MSFMX that evolved from our R-MSFM to better address the problem of depth estimation. Our R-MSFMX takes the first three blocks from ResNet50 instead of ResNet18 in our R-MSFM and  further improves the depth accuracy.
2.  We promote the geometry consistent depth learning for both our R-MSFM and R-MSFMX, which prevents the depth artifacts at object borders and thus generates more consistent depth. We denote the models that perform geometry consistent depth estimation by the postfix (GC).

We show the superiority of our R-MSFMX-GC as follows:
![4](https://user-images.githubusercontent.com/32475718/160613575-a924c751-7352-4429-87ff-c6f6bcc19c44.jpg)


The rows (from up to bottom) are RGB images, and the results by [Monodepth2](https://github.com/nianticlabs/monodepth2), R-MSFM6, and the improved version R-MSFMX6-GC.


## R-MSFM Results
![image](https://user-images.githubusercontent.com/32475718/160614132-3e7d25cc-e3d2-4d63-a2de-4fcaf10ef04e.png)
## R-MSFMX Results
![image](https://user-images.githubusercontent.com/32475718/160617371-50e304c0-1266-4ccc-afb7-524231c43bcf.png)




## Precomputed Results
We have updated all the results as follows:
[results](https://drive.google.com/drive/folders/1xLglsHFVxxTlvj5UBEyK5MQ_D0dLIjbS?usp=sharing)

## Pretrained Models
We have updated all the results as follows:
[models](https://drive.google.com/drive/folders/1IhUsEEY-oKfgcsTX2uHuENMe7u-1Pzik?usp=sharing)

## KITTI Evaluation
You can predict scaled disparity for a single image used R-MSFM3 with:
```shell
python test_simple.py --image_path='path_to_image' --model_path='path_to_model' --update=3
```
or R-MSFMX3 with
```shell
python test_simple.py --image_path='path_to_image' --model_path='path_to_model' --update=3 --x
```
or R-MSFM6 with:
```shell
python test_simple.py --image_path='path_to_image' --model_path='path_to_model' --update=6
```
or R-MSFM6X with:
```shell
python test_simple.py --image_path='path_to_image' --model_path='path_to_model' --update=6 --x
```
## License & Acknowledgement
The codes are based on [RAFT](https://github.com/princeton-vl/RAFT), [Monodepth2](https://github.com/nianticlabs/monodepth2). Please also follow their licenses. Thanks for their great works.
