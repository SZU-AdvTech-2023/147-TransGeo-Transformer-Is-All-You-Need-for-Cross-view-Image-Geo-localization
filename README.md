代码环境：
	- Python >= 3.6, numpy, matplotlib, pillow, ptflops, timm
    - PyTorch >= 1.8.1, torchvision >= 0.11.1

数据集：
    [CVUSA](http://mvrl.cs.uky.edu/datasets/cvusa/)

运行代码：
1、python data_preparation.py --- 将航拍图像进行极坐标变换
2、sh run_CVUSA.sh --- 训练模型

补充：
1、注意将代码中数据集路径替换成你自己实际的路径；
2、如果要测试模型，只需要在run_CVUSA.sh脚本的两行代码中添加“-e”参数即可。

特别说明：
    本代码是在论文："TransGeo  Transformer Is All You Need for Cross-view Image 
    Geo-localization"基础上进行的修改和改进，原论文请参考：https://ieeexplore.ieee.org/document/9880284
