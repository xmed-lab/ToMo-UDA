<div align=center>
<h1>  $M^3$-UDA: A New Benchmark for Unsupervised Domain Adaptive Fetal Cardiac Structure Detection.</h1>
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
&ensp; :smile: This project is the pytorch implemention of $M^3$-UDA;

&ensp; :laughing: Our experimental platform is configured with <u>One *RTX3090 (cuda>=11.0)*</u>; 

&ensp; :blush: Currently, this code is avaliable for proposed dataset FCS and public dataset <u>CardiacUDA</u>;

<!-- &ensp; :smiley: For codes and accessment that related to dataset ***CardiacUDA***; -->

&ensp; &ensp; &ensp;    **:eyes:** The code is now available at:
&ensp; &ensp; &ensp;       ```
                            ..\data\detus_dataset.py
                           ```



## :computer: Installation


1. You need to build the relevant environment first, please refer to : [**requirements.yaml**](requirements.yaml)

2. Install Environment:
    ```
    conda env create -f requirements.yaml
    ```

+ We recommend you to use Anaconda to establish an independent virtual environment, and python > = 3.8.3; 


## :blue_book: Data Preparation

### *1. FCS dataset*
 * This project provides the use case of Unsupervised Domain Adaptive Fetal Cardiac Structure Detection task;

 * The hyper parameters setting of the dataset can be found in the **utils/config.py**, where you could do the parameters modification;

 * For different tasks, the composition of data sets have significant different, so there is no repetition in this file;


   <!-- #### *1.1. Download The **FCS**.* -->
   <!-- :speech_balloon: The detail of CAMUS, please refer to: https://www.creatis.insa-lyon.fr/Challenge/camus/index.html/. -->

   1. Download & Unzip the dataset.

      The ***FCS dataset*** is composed as: /Hospital1 & /Hospital2 & Hospital3.

   2. The source code of loading the FCS dataset exist in path :

      ```python
      ..\data\fetus_dataset.py
      and modify the dataset path in
      ..\utils/config.py
      ```

   3. Set the parameters about GPU_id, source domain,target domain and slice etc in **utils/config.py** 
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
### *2. FCS dataset access*
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
1. Download the checkpoint in table below.
---
| Experiment      | Checkpoint |
| ----------- | ----------- |
| 4CC 1->2      | [4CC1-2](https://drive.google.com/file/d/1SXKbSGN0sGyShdu4QnphYsdNout4lNnU/view?usp=drive_link) |
| 4CC 2->1      | [4CC2-1](https://drive.google.com/file/d/1O2Py77c3DNXw5-3jm2rrOunZxPzl2X0Y/view?usp=drive_link) |
| 3VT 1->2      | [3VT1-2](https://drive.google.com/file/d/1sFtJAsCb0NZ_uUE3Xzi4grIZfsgsEbVB/view?usp=drive_link) |
| 3VT 2->1      | [3VT2-1](https://drive.google.com/file/d/1OnoCU5TrqF65fCfVGNRfli9XO294BrR5/view?usp=drive_link) |
---
2. Update the test weight path in **config.py**.
3. you only need to use the command:

   ```shell
    python test.py
    ```
#


## :feet: citation

```
@inproceedings{pu2024m3,
  title={M3-UDA: A New Benchmark for Unsupervised Domain Adaptive Fetal Cardiac Structure Detection},
  author={Pu, Bin and Wang, Liwen and Yang, Jiewen and He, Guannan and Dong, Xingbo and Li, Shengli and Tan, Ying and Chen, Ming and Jin, Zhe and Li, Kenli and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11621--11630},
  year={2024}
}
```


###### :rocket: Code Reference 
  - https://github.com/CityU-AIM-Group/SIGMA

<!-- ###### :rocket: Updates Ver 1.0（PyTorch）
###### :rocket: Project Created by Jiewen Yang : jyangcu@connect.ust.hk -->
