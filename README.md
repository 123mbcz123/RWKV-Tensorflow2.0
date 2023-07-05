# RWKV-Tensorflow2.0
 A RWKV written in TensorFlow 2.0,only tested under FP32 precision in the World model.  
 RWKV_MODEL.py is the TensorFlow version implementation of RWKV, while the remaining files were copied from chatRWKV for testing accuracy with this repository.  
 
 一个使用Tensorflow2.0编写的RWKV模型加载器.目前只支持FP32下加载模型,模型方面我只测试了World 0.1B/0.4B模型

RWKV_MODEL.py是RWKV的完整实现,其余的文件由ChatRWKV库拷贝编辑而得.其余文件仅仅是为了测试tf版本实现的正确性,一般情况下无需下载.

此仓库与ChatRWKV v2(经过一点修改)后误差平均值小于1.3238086e-05

这是我翻译模型的一部分,如果没有咕咕的话会继续更新.

下一步我会为RWKV添加训练部分,然后再尝试把注意力机制带回RWKV


