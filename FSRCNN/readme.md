## FSRCNN一种快速的图像超分辨率方法 ##


----------
整理了下FSRCNN

https://github.com/66wangxuewen99/Super-Resolution/tree/master/FSRCNN



**FSRCNN** 是在2016年在CVPR上发表的一片论文

> [Accelerating the Super-Resolution Convolutional Neural Network](http://arxiv.org/pdf/1608.00367v1.pdf)

中提出的一种加速的基于卷积神经网络的图像超分辨率方法。

----------
**本项目**使用caffe+matlab实现了FSRCNN的训练以及应用过程，提供了样本产生和一些如**PSNR曲线绘制**的工具。

### **网络结构** ###
训练网络：[FSRCNN_net.prototxt](https://github.com/66wangxuewen99/Super-Resolution/blob/master/FSRCNN/FSRCNN_net.prototxt)
![FSRCNN_net.prototxt](http://img.blog.csdn.net/20160902102246837)
应用网络： [FSRCNN_mat.prototxt](https://github.com/66wangxuewen99/Super-Resolution/blob/master/FSRCNN/FSRCNN_mat.prototxt)
![这里写图片描述](http://img.blog.csdn.net/20160902102659972)

###**网络训练**###

 1. 样本准备
	 从91张训练图片中提取图像块作为样本并写入hdf5文件。使用generate_fsrcnn_train.m & generate_fsrcnn_test.m 根据自己的网络参数设定更改settings，运行得到fsrcnn_train.h5 & fsrcnn_test.h5两个文件。
 2. 开始训练
	 运行start.bat开始训练。
 3. 恢复训练
	 编辑restore.bat中--snapshot= ?.solverstate后的文件名，运行restore.bat该次迭代状态中恢复训练。


###**应用**###
本项目中[sr_demo.m](https://github.com/66wangxuewen99/Super-Resolution/blob/master/FSRCNN/sr_demo.m)实现了使用caffe的matlab接口来实现图片超分辨率。

caffe matlab接口的使用可以参考
http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html

**效果** 
![这里写图片描述](http://img.blog.csdn.net/20160902102837877)

![这里写图片描述](http://img.blog.csdn.net/20160902102902534)

![这里写图片描述](http://img.blog.csdn.net/20160902102919253)

###其他###
draw_psnr.m：  绘制指定测试图像的PSNR随迭代次数变化的曲线

![这里写图片描述](http://img.blog.csdn.net/20160902102943676)

注：本项目中训练得到的网络不可直接使用opencv或者使用c++接口来做应用，因为matlab中图像是列序优先，而opencv或c++中通常为行序优先，直接应用会得到一个效果比较差的结果。
