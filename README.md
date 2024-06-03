# mega-nvdiffrast  
  
**将可微渲染应用于更大的模型上。**  
  
由于可微渲染需要保存大量中间变量用于反向传播，因此可微渲染通常需要不小的显存要求。在实时渲染领域，我们通常可以利用视角相关性实现剔除，如虚拟纹理、Nanite等技术那样。因此，我们第一次将虚拟纹理扩展到了可微渲染上，对于Mesh，我们则将其打散为 mesh cluster（"GPU-Driven Rendering Pipelines"，SIGGRAPH 2015）。我们基于[`nvdiffrast`](https://github.com/NVlabs/nvdiffrast)实现以上功能，因为这是目前最快的光栅化可微渲染器。  
**不过，目前mega-nvdiffrast对三维模型几乎没有优化效果，只是能够在比较大的三维模型上把可微渲染跑起来而已。如何改善可微渲染对比较大的三维模型的优化效果，依然是一个亟待研究的课题。**

## Features
- 可微的虚拟纹理
- 可微的虚拟化几何
- 用虚拟纹理实现的虚拟阴影贴图
- 单机多卡训练
- IBL光照（来自[`nvdiffrec`](https://github.com/NVlabs/nvdiffrec)）

## 技术细节
- 虚拟纹理与虚拟化几何的实现主要位于modules/mesh.py与modules/texture.py
- 虚拟阴影贴图位于virtual_shadow_mapping.py

## 运行环境
`mega-nvdiffrast`目前只能运行在linux系统上。所有的测试于Ubuntu 18.04系统上进行，使用的硬件为4张NVIDIA TITAN RTX，256 GB内存  
- GCC-9及以上版本
- CUDA 11.7  
- Python 3.11.5 
- 修改后的nvdiffrast
```
git clone https://github.com/Steelwall2014/nvdiffrast
cd nvdiffrast
pip install .
``` 
- Pytorch  
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
``` 
- 一些其他库  
```
pip install ninja imageio PyOpenGL glfw slangpy loguru psutil
```
注：可能需要修改vm.max_memory_count、fs.file-max和/etc/security/limits.conf中的nofile，因为系统对共享内存、最大文件描述符数量之类的可能有一些限制。

## 运行
- 下载[`测试数据`](https://pan.baidu.com/s/1taKpWfcP37jGTJ3jU9aIxg?pwd=2z3k)。将数据复制到项目路径下，文件夹结构为./data/UrbanScene3D
- python example.py

## 测试数据
数据是[`UrbanScene3D`](https://vcc.tech/UrbanScene3D/)数据集的PolyTech场景，使用Reality Capture进行三维重建、相机位姿重建、图像去畸变与模型简化。模型经过了简化，三角形数量为原始重建结果的5%（现在有大约800万个三角形）。模型具有一张16384×16384的基础色纹理。原始无人机影像有685张，分辨率为4864×3648