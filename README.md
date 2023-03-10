# PointNet.pytorch
This repo offer a patch for official PointNet.PyTorch ( https://github.com/charlesq34/pointnet ).
Patch for ModelNet40 .off format to .ply format.
convert.py including ModelNet40 folder.

It is tested with pytorch-1.0.

# Download data and running

```
git clone https://github.com/fxia22/pointnet.pytorch
cd pointnet.pytorch
pip install -e .
```

Download and build visualization tool
```
cd script
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
```

Convert ModelNet40 Dataset for training (.off to .ply)
```
cd ModelNet40
python convert.py --MutiThreads=True # if u want use single process, pls disable here.
                                     # On my intel i7-11700, muti take 11 mins, 
                                                             single take about 27 mins.
```

Training 
```
cd utils
python train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | shapenet>
python train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> 
```

Use `--feature_transform` to use feature transform.

   
      

# Data tree

.
├── misc  
├── ModelNet40  
│   ├── <font color=DodgerBlue>convert.py</font>    
│   ├── <font color=DodgerBlue>test.txt</font>  
│   ├── <font color=DodgerBlue>train.txt</font>  
│   ├── <font color=DodgerBlue>trainval.txt</font>  
│   ├── <font color=DodgerBlue>val.txt</font>  
│   ├── airplane    
│   │   ├── test  
│   │   └── train  
│   ├── bathtub  
│   │   ├── test  
│   │   └── train  
│   ├── bed  
│   │   ├── test  
│   │   └── train  
│   ├── bench  
│   │   ├── test  
│   │   └── train  
|   └──── .   
|     └── .  
|     └── .  
├── pointnet  
├── pointnet.egg-info  
├── scripts  
└── utils  


# Performance

## Classification performance

On ModelNet40:

|  | Overall Acc | 
| :---: | :---: | 
| Original implementation | 89.2 | 
| this implementation(w/o feature transform) | 86.4 | 
| this implementation(w/ feature transform) | 87.0 | 

On [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html)

|  | Overall Acc | 
| :---: | :---: | 
| Original implementation | N/A | 
| this implementation(w/o feature transform) | 98.1 | 
| this implementation(w/ feature transform) | 97.7 | 

## Segmentation performance

Segmentation on  [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

| Class(mIOU) | Airplane | Bag| Cap|Car|Chair|Earphone|Guitar|Knife|Lamp|Laptop|Motorbike|Mug|Pistol|Rocket|Skateboard|Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Original implementation |  83.4 | 78.7 | 82.5| 74.9 |89.6| 73.0| 91.5| 85.9| 80.8| 95.3| 65.2| 93.0| 81.2| 57.9| 72.8| 80.6| 
| this implementation(w/o feature transform) | 73.5 | 71.3 | 64.3 | 61.1 | 87.2 | 69.5 | 86.1|81.6| 77.4|92.7|41.3|86.5|78.2|41.2|61.0|81.1|
| this implementation(w/ feature transform) |  |  |  |  | 87.6 |  | | | | | | | | | |81.0|

Note that this implementation trains each class separately, so classes with fewer data will have slightly lower performance than reference implementation.

Sample segmentation result:
![seg](https://raw.githubusercontent.com/fxia22/pointnet.pytorch/master/misc/show3d.png?token=AE638Oy51TL2HDCaeCF273X_-Bsy6-E2ks5Y_BUzwA%3D%3D)

# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)


# Murmur

這包的主要貢獻是把 ModelNet40 的 .off 檔案換成 .ply 格式，  
因為我不知道為什麼原作者沒給，所以就自己寫了一份轉檔程式。

親測可用！
還附上多執行緒版本（因為單執行緒世界慢）。

如果有問題歡迎聯絡我。

## Contact

Further information please contact me.

wuyiulin@gmail.com