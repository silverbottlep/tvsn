# Transformation-Grounded Image Generation Network for Novel 3D View Synthesis
[Eunbyung Park](http://www.cs.unc.edu/~eunbyung/), [Jimei Yang](https://eng.ucmerced.edu/people/jyang44/), [Ersin Yumer](http://www.meyumer.com/), [Duygu Ceylan](http://www.duygu-ceylan.com/), [Alexander C. Berg](http://acberg.com/), CVPR 2017

[[Paper]](http://www.cs.unc.edu/~eunbyung/tvsn/) [[Project Homepage]](http://www.cs.unc.edu/~eunbyung/tvsn/)

Follow below instructions to run and test the codes used in the paper.

## 0. Prerequisites
[Torch](http://torch.ch/) and [stnbhwd](https://github.com/qassemoquab/stnbhwd)

## 0. ShapeNet dataset download
You should have ShapeNetCore.v1 dataset in your local $(SHAPENET_DATA) directory via [shapenet.org](https://shapenet.org/) in your local directory. We will use entire models for car category. For chair category, we used train/test split suggested by appearance flow network paper[[link]](https://github.com/tinghuiz/appearance-flow)(They picked the models that have rich textures).
```bash
$(tvsn_root)/tvsn/data$>./make_new_chair.sh $(SHAPENET_DATA)
$(tvsn_root)/tvsn/data$>ln -s $(SHAPENET_DATA)/02958343 ./car
$(tvsn_root)/tvsn/data$>ln -s $(SHAPENET_DATA)/new_chair ./chair
```

## 1. Dataset Preparation (Rendering multiple view images)
I adopted [rendering engine](https://github.com/sunweilun/ObjRenderer) used in the [appearance flow network](https://github.com/tinghuiz/appearance-flow), and modified original code a little bit to get the surface normals and object coordinates, which will be used for generating visibility maps. You can download from [here](https://github.com/silverbottlep/ObjRenderer) and edit the 'config.txt' file to tune the engine for your purpose. For example,
```bash
$(tvsn_root)$> git clone git@github.com:silverbottlep/ObjRenderer.git
$(tvsn_root)$> cd ObjRenderer
$(tvsn_root)/ObjRenderer$> cat config.txt
folder_path = $(SHAPENET_DATA)/02958343 "e.g. 'car' category"
envmap_path = envmaps/envmap2.hdr
theta_inc = 20
phi_inc = 10
phi_max = 20
output_coord = 1
output_norm = 1
render_size = 1024
output_size = 256
reverse_normals = 1
brightness = 0.7
```
Build and execute. It will take long time, and requires a lot of space(~10GB)
```bash
$(tvsn_root)/ObjRenderer$> make
$(tvsn_root)/ObjRenderer$> ./dist/Release/GNU-Linux-x86/objrenderer
```
It will generate multiple view images, 3D object coordinates, and normals under 'model_views' directory for each 3D models.
```bash
$(tvsn_root)/ObjRenderer$> ls ($SHAPENET_DATA)/02958343/56c80ed5da821cd0179005454847728d/model_views
0_0_coord.exr   
0_0_norm.exr    
0_0.png         
0_10_coord.exr  
0_10_norm.exr   
0_10.png        
0_20_coord.exr  
0_20_norm.exr   
0_20.png        
100_0_coord.exr 
100_0_norm.exr  
100_0.png       
100_10_coord.exr
100_10_norm.exr 
100_10.png      
100_20_coord.exr
100_20_norm.exr 
.....
```

## 2. Dataset Preparation (Generating visibility maps)
Now, we are going to make visibility maps.  For convinience, we provide precomputed visibilty maps. You can download them from following links, and locate them in $(tvsn_root)/tvsn/data directory

[maps_car.t7](https://drive.google.com/open?id=0B-r7apOz1BHAVEI1RURZYUl4Tlk) (~26G)

[maps_chair.t7](https://drive.google.com/open?id=0B-r7apOz1BHANGlsY1k3Z29yVEU) (~2G)

You can also run the code to compute visibility maps. First we need to have simple library for reading .exr file. You can download and install from [here](http://www.mit.edu/~kimo/software/matlabexr/). You also need to have matlab and torch installed.
```bash
$(tvsn_root)$> cd gen_vis_maps
$(tvsn_root)/gen_vis_maps$> wget http://www.mit.edu/~kimo/software/matlabexr/MatlabEXR.zip
$(tvsn_root)/gen_vis_maps$> unzip MatlabEXR.zip
$(tvsn_root)/gen_vis_maps$> cd MatlabEXR
$(tvsn_root)/gen_vis_maps/MatlabEXR$> mex exrinfo.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/include/OpenEXR/
$(tvsn_root)/gen_vis_maps/MatlabEXR$> mex exrread.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/include/OpenEXR/
$(tvsn_root)/gen_vis_maps/MatlabEXR$> mex exrwrite.cpp -lIlmImf -lIex -lImath -lHalf -I/usr/include/OpenEXR/
```
Once you have exrread library, you can run the script we provide, it will save visibility maps in '$(tvsn_root)/tvsn/data' directory, e.g. '$(tvsn_root)/tvsn/data/maps_car.t7'
```bash
$(tvsn_root)/gen_vis_maps$>./gen_vis_maps.sh $(SHAPENET_DATA)/02958343 car
$(tvsn_root)/gen_vis_maps$>./gen_vis_maps.sh $(SHAPENET_DATA)/new_chair chair
$(tvsn_root)/gen_vis_maps$>ls ../tvsn/data/
maps_car.t7  maps_chair.t7
```
Rendering images and computing visibility maps are time consuming jobs, so I highly recommend you to parallelize it across multiple cpus. It can be easily done by modifying the code or splitting the data into different directories.

## 3. Training DOAFN(Disocclusion aware appearance flow network)
```bash
$(tvsn_root)/tvsn/code$>./script_train_doafn.sh
```

## 4. Training TVSN(Transformation-grounded view synthesis network)
First, we need to prepare pretrained vgg16 network. We imported [caffemodel](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) and translated into torch nngraph format. You can download translated version with provided script.
```bash
$(tvsn_root)/tvsn/code/lossnet$>./download_lossnet.sh
```
Now, you can train!
```bash
$(tvsn_root)/tvsn/code$>./script_train_tvsn.sh
```

## 5. Downloading pretrained models
We provide pretrained models for car and chair category. You can download it from following links.

[tvsn_car_epoch220.t7](https://drive.google.com/open?id=0B-r7apOz1BHAQVVXR0JXcTh5MUk) (~134M)

[tvsn_chair_epoch200.t7](https://drive.google.com/open?id=0B-r7apOz1BHAWmQtdEZ6ZG5udW8) (~134M)

[doafn_car_epoch200.t7](https://drive.google.com/open?id=0B-r7apOz1BHAR1RKWXM1c1NBekk) (~351M)

[doafn_chair_epoch200.t7](https://drive.google.com/open?id=0B-r7apOz1BHAaWh4N1Vnc3hKdE0)(~351M)


## 6. Testing TVSN
```bash
$(tvsn_root)/tvsn/code/$>th test_tvsn.lua --category car --doafn_path ../snapshots/pretrained/doafn_car_epoch200.t7 --tvsn_path ../snapshots/pretrained/tvsn_car_epoch220.t7
```
You will get some of qualitative results in $(tvsn_root)/tvsn/code/result_car directory.

## Acknowledgments
Many parts of this code are adopted from other projects ([DCGAN](https://github.com/soumith/dcgan.torch), [Perceptual Loss](https://github.com/jcjohnson/fast-neural-style), [attr2img](https://github.com/xcyan/eccv16_attr2img), [Video Prediction](https://github.com/coupriec/VideoPredictionICLR2016))
