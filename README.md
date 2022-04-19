# R-MSFM: Recurrent Multi-Scale Feature Modulation for Monocular Depth Estimating(ICCV-2021)
This is the official implementation for testing depth estimation using the model proposed in 
>R-MSFM: Recurrent Multi-Scale Feature Modulation for Monocular Depth Estimating

>Zhongkai Zhou, Xinnan Fan, Pengfei Shi, Yuanxue Xin


R-MSFM can estimate a depth map from a single image.

Paper is now available at [ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_R-MSFM_Recurrent_Multi-Scale_Feature_Modulation_for_Monocular_Depth_Estimating_ICCV_2021_paper.pdf)

We will update the training code in the future, if there is any problem before then, please contact us.







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

or R-MSFM6 with:
```shell
python test_simple.py --image_path='path_to_image' --model_path='path_to_model' --update=6
```

## License & Acknowledgement
The codes are based on [RAFT](https://github.com/princeton-vl/RAFT), [Monodepth2](https://github.com/nianticlabs/monodepth2). Please also follow their licenses. Thanks for their great works.
