<div align=center>
<h1> Unsupervised Domain Adaptation for Anatomical Structure Detection in Ultrasound Images.</h1>
</div>
<div align=center>

<!-- <a src="https://img.shields.io/badge/%F0%9F%93%96-ICCV_2023-8A2BE2.svg?style=flat-square" href="https://arxiv.org/abs/2309.11145">
<img src="https://img.shields.io/badge/%F0%9F%93%96-ICCV_2023-8A2BE2.svg?style=flat-square">
</a> -->
   
<a src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square" href="https://xmengli.github.io/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square">
</a>

<!-- <a src="https://img.shields.io/badge/%F0%9F%9A%80-XiaoweiXu's Github-blue.svg?style=flat-square" href="https://github.com/XiaoweiXu/CardiacUDA-dataset">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-Xiaowei Xu's Github-blue.svg?style=flat-square">
</a> -->

</div>


## :hammer: PostScript
&ensp; :smile: This project is the pytorch implemention of ToMo-UDA;

&ensp; :laughing: Our experimental platform is configured with <u>One *RTX3090 (cuda>=11.0)*</u>; 

&ensp; :blush: Currently, this code is avaliable for proposed dataset $FUSH^2$, public dataset <a href="https://github.com/xmed-lab/GraphEcho">CardiacUDA</a> and <a href="https://zmiclab.github.io/zxh/0/mmwhs/">MMWHS</a>;  

<!-- &ensp; :smiley: For codes and accessment that related to dataset ***CardiacUDA***; -->


## :computer: Installation


1. You need to build the relevant environment first, please refer to : [**requirements.yaml**](requirements.yaml)

2. Install Environment:
    ```
    conda env create -f requirements.yaml
    ```

+ We recommend you to use Anaconda to establish an independent virtual environment, and python > = 3.8.3; 


## :blue_book: Data Preparation

### *1. FUSH^2 dataset*
 * This project provides the use case of UDA Ultrasound Anatomical Structure Detection task;

 * The hyper parameters setting of the dataset can be found in the **utils/config.py**, where you could do the parameters modification;

 * For different tasks, the composition of data sets have significant different, so there is no repetition in this file;


   <!-- #### *1.1. Download The **$FUSH^2$**.* -->
   <!-- :speech_balloon: The detail of CAMUS, please refer to: https://www.creatis.insa-lyon.fr/Challenge/camus/index.html/. -->

   1. Download & Unzip the dataset.

      The ***FUSH^2 dataset*** is composed as: /Heart & /Head.

   2. The source code of loading the $FUSH^2$ dataset exist in path :

      ```python
      ..\data\fetus_dataset_coco.py
      and modify the dataset path in
      ..\utils/config.py
      ```
      

   3. In **utils/config.py**, you can set the ```part``` to select the anatomical slice, and choose the source and target domains using ```selected_source_hospital``` and ```selected_target_hospital```, respectively.
   
   <!-- #### *1.2. Download The **CardiacUDA**.*

   :speech_balloon: The detail of CardiacUDA, please refer to: https://echonet.github.io/dynamic/.

   1. Download & Unzip the dataset.

      - The ***CardiacUDA*** dataset is consist of: /Video, FileList.csv & VolumeTracings.csv.

   2. The source code of loading the Echonet dataset exist in path :

      ```python
      ..\datasets\echo.py
      and modify the dataset path in
      ..\train_camus_echo.py
      ``` -->
### *2. FUSH^2 dataset access*
  * Dataset access can be obtained by contacting hospital staff (doc.liangbc@gmail.com) and asking for a license.
    
## :feet: Training

1. In this framework, after the parameters are configured in the file **utils/config.py** and **train.py** , you only need to use the command:

    ```shell
    python train.py
    ```

2. You are also able to start distributed training. 

   - **Note:** Please set the number of graphics cards you need and their id in parameter **"enable_GPUs_id"**.
   ```shell
   python -m torch.distributed.launch --nproc_per_node=4 train.py
   ```

#

## :feet: Testing
1. Download the ```TEST_CHECKPOINT``` <a href="https://drive.google.com/drive/folders/1XvrZR4DOWA58aSsVK6FYTWqtXIid2VPd?usp=sharing">here</a>.

2. you only need to use the command:

   ```shell
    python test.py --path TEST_CHECKPOINT
    ```
#


## :feet: citation

```
@inproceedings{puunsupervised,
  title={Unsupervised Domain Adaptation for Anatomical Structure Detection in Ultrasound Images},
  author={Pu, Bin and Lv, Xingguo and Yang, Jiewen and Guannan, He and Dong, Xingbo and Lin, Yiqun and Shengli, Li and Ying, Tan and Fei, Liu and Chen, Ming and others},
  booktitle={Forty-first International Conference on Machine Learning}
}
```


## :rocket: Code Reference 
  - https://github.com/CityU-AIM-Group/SIGMA

<!-- ###### :rocket: Updates Ver 1.0（PyTorch）
###### :rocket: Project Created by Jiewen Yang : jyangcu@connect.ust.hk -->
