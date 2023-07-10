import gc
import json
import math
import os.path
import time
from typing import Dict, Union, Tuple, List
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import *
from keras.backend import conv1d
tf.debugging.enable_check_numerics()
"""
  获取模型一共有多少层
"""

def get_model_layers_count(model_dict: Dict):
    max_layer = 0
    for var_name in model_dict.keys():
        if 'blocks' in var_name:
            max_layer = max(max_layer, int(var_name.split('.')[1]))
    return max_layer + 1



class RWKVModelError(Exception):
    def __init__(self, error_msg):
        self.error_msg = error_msg


    def __str__(self):
        return self.error_msg


class CustomLayerNormalization(Layer):
    def __init__(self, scale_weight, center_weight, axis=-1, epsilon=1e-5, name="layerNorm"):
        super(CustomLayerNormalization, self).__init__(name=name)

        self.scale = scale_weight  # tf.expand_dims(tf.expand_dims(scale_weight, axis=0), axis=0)
        self.center = center_weight  # tf.expand_dims(tf.expand_dims(center_weight, axis=0), axis=0)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=self.axis, keepdims=True)
        outputs = tf.nn.batch_normalization(inputs, mean, variance, self.center, self.scale, self.epsilon)
        return outputs


class MultiQueryAttention(Layer):
    def __init__(self, layer_idx, weights_dict, max_kv_length=512, dropout_rate=0.2):
        super(MultiQueryAttention, self).__init__(name=f"attention_{layer_idx}")
        self.mem_max_length = max_kv_length

        self.weight_q = weights_dict[f'blocks.{layer_idx}.dense_q.weight']
        self.weight_k = weights_dict[f'blocks.{layer_idx}.dense_k.weight']
        self.weight_v = weights_dict[f'blocks.{layer_idx}.dense_v.weight']
        self.weight_o = weights_dict[f'blocks.{layer_idx}.dense_o.weight']
        self.num_q_heads = int(tf.shape(self.weight_q)[-1] // tf.shape(self.weight_k)[-1])
        self.dim_per_head = int(tf.shape(self.weight_k)[-1])
        self.value_dims = int(tf.shape(self.weight_v)[-1])
        self.scale = 1. / math.sqrt(self.dim_per_head)
        tf.debugging.assert_equal(self.num_q_heads * self.dim_per_head, tf.shape(self.weight_q)[-1],
                                  "dense_q的宽度必须是dense_k,v的整数倍")
        self.dropout1 = Dropout(dropout_rate, name="kv_drop")
        self.dropout2 = Dropout(dropout_rate, name="out_drop")

    def split_heads(self, inputs, is_query):
        bz, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        outputs = tf.reshape(inputs, (bz, seq_len, self.num_q_heads if is_query else 1, self.dim_per_head))
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        return outputs

    """
        att_mask 1 for attention, 0 for not attention
    """
    def look_ahead_mask(self,qkT):
        q_seq_length,k_seq_length = tf.shape(qkT)[-2],tf.shape(qkT)[-1]
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, q_seq_length, k_seq_length), tf.bool), -1, 0
        )
    def call(self, q, k, v, kv_cache=None,att_mask=None, training=False):
        q = tf.matmul(q, tf.cast(self.weight_q, q.dtype))
        k = tf.matmul(k, tf.cast(self.weight_k, k.dtype))
        v = tf.matmul(v, tf.cast(self.weight_v, v.dtype))

        q = self.split_heads(q, is_query=True)
        k = self.split_heads(k, is_query=False)  # bz,n,k_seq_len,dim
        v = self.split_heads(v, is_query=False)

        if kv_cache is not None:
            k = tf.concat([kv_cache['att_cache_k'], k], axis=2)  # bz,n
            v = tf.concat([kv_cache['att_cache_v'], v], axis=2)
            k = k[:, :, -self.mem_max_length:, :]
            v = v[:, :, -self.mem_max_length:, :]

        k = self.dropout1(k, training=training)
        v = self.dropout2(v, training=training)

        qkT = tf.matmul(q, k, transpose_b=True) * self.scale

        look_ahead = self.look_ahead_mask(qkT)
        if att_mask is not None:
            att_mask = tf.logical_and(att_mask,look_ahead)
        else:
            att_mask = look_ahead

        qkT -= (1. - tf.cast(att_mask, qkT.dtype)) * tf.float16.max

        att_scores = tf.nn.softmax(qkT, axis=-1)  # bz,n,q_seq_len,k_seq_len
        att_values = tf.transpose(tf.matmul(att_scores, v), [0, 2, 1, 3])
        att_values = tf.reshape(att_values, (tf.shape(att_values)[0], tf.shape(att_values)[1], self.value_dims))

        att_values = self.dropout2(att_values, training=training)
        att_outputs = tf.matmul(att_values, tf.cast(self.weight_o, att_values.dtype))

        return att_outputs, k, v


class AttentionBlock(Layer):
    def __init__(self, layer_idx, weights_dict, max_kv_length=512, dropout_rate=0.2):
        super(AttentionBlock, self).__init__(name=f"att_block_{layer_idx}")

        self.att_layer = MultiQueryAttention(layer_idx, weights_dict, max_kv_length, dropout_rate)

        ln_scale_var = weights_dict[f'blocks.{layer_idx}.ln3.weight']
        ln_center_var = weights_dict[f'blocks.{layer_idx}.ln3.bias']
        self.input_norm = CustomLayerNormalization(ln_scale_var, ln_center_var, axis=-1, epsilon=1e-5, name="ln3")

    def call(self, inputs, kv_cache=None, att_mask=None, training=False):
        inputs_norm = self.input_norm(inputs)
        att_outputs, cache_k, cache_v = self.att_layer(inputs_norm, inputs_norm, inputs_norm, kv_cache, att_mask,
                                                       training)
        outputs = att_outputs + inputs_norm

        return outputs, cache_k, cache_v


class TimeMixFirst(Layer):
    def __init__(self, layer_idx, weights_dict, trainable=True):
        super(TimeMixFirst, self).__init__(name=f"time_mix_first_{layer_idx}", trainable=trainable)

        ln_scale_var = weights_dict[f'blocks.{layer_idx}.ln1.weight']
        ln_center_var = weights_dict[f'blocks.{layer_idx}.ln1.bias']
        self.input_norm = CustomLayerNormalization(ln_scale_var, ln_center_var, axis=-1, epsilon=1e-5, name="ln1")

        self.time_mix_r = weights_dict[f'blocks.{layer_idx}.att.time_mix_r']
        self.time_mix_k = weights_dict[f'blocks.{layer_idx}.att.time_mix_k']
        self.time_mix_v = weights_dict[f'blocks.{layer_idx}.att.time_mix_v']

        self.key_weight = weights_dict[f'blocks.{layer_idx}.att.key.weight']
        self.value_weight = weights_dict[f'blocks.{layer_idx}.att.value.weight']
        self.receptance_weight = weights_dict[f'blocks.{layer_idx}.att.receptance.weight']

        self.hidden_size = int(tf.shape(self.time_mix_r)[-1])

    def get_initial_state(self, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = 1
        if dtype is None:
            dtype = tf.float32
        """
        生成初始的state cell,其中上一状态的输入state_x,分子state_a,分母state_b使用全零初始化
        缩放因子state p初始化到负无穷 经过maximum基本上不影响输入的值,在这里初始化到负-1e8
        """

        state_x = tf.zeros(shape=(batch_size, 1, self.hidden_size), dtype=dtype)
        return state_x

    def call(self, inputs, input_state=None):

        x_norm = self.input_norm(inputs)
        bz = tf.shape(inputs)[0]
        if input_state is None:
            input_state = self.get_initial_state(bz, x_norm.dtype)

        input_state = tf.concat([input_state, x_norm[:, :-1, :]], axis=1)

        kx_mixed = x_norm * self.time_mix_k + input_state * (1. - self.time_mix_k)
        vx_mixed = x_norm * self.time_mix_v + input_state * (1. - self.time_mix_v)
        rx_mixed = x_norm * self.time_mix_r + input_state * (1. - self.time_mix_r)

        r = tf.nn.sigmoid(tf.matmul(rx_mixed, tf.cast(self.receptance_weight, dtype=rx_mixed.dtype)))
        k = tf.matmul(kx_mixed, tf.cast(self.key_weight, dtype=kx_mixed.dtype))
        v = tf.matmul(vx_mixed, tf.cast(self.value_weight, dtype=vx_mixed.dtype))

        return tf.stack([k, v], axis=2), r, x_norm


class TimeMixCell(AbstractRNNCell):
    def __init__(self, layer_idx, weights_dict, trainable=True):
        super(TimeMixCell, self).__init__(name=f"time_mix_cell_{layer_idx}", trainable=trainable)

        self.time_first = weights_dict[f'blocks.{layer_idx}.att.time_first']
        self.time_decay = weights_dict[f'blocks.{layer_idx}.att.time_decay']
        self.hidden_size = int(tf.shape(self.time_first)[-1])  # 获取隐藏层宽度

    def call(self, inputs, states):  # states = (bz,4,hidden_size) states返回的是一个只有一个元素的tuple 使用states[]
        k, v = tf.unstack(inputs, axis=1)

        state_a, state_b, state_p = states[0], states[1], states[2]

        """
        state_x 是上一个时间步的输入的inputs
        state_a是上一个WKVt的分子 state_b是上一个WKVt的分母 需要注意的是上述的state_a和state_b都是进过除以exp(state_p)来进行缩放的
        整个WKVt都是进过缩放的等式 state_p是缩放因子,保证exp的指数永远小于0 防止指数爆炸
        """

        """
        下面是时间混合,kv vx rx都包含了当前时间步和过去时间步的信息
        """

        uk = k + tf.cast(self.time_first, dtype=k.dtype)

        current_p = tf.maximum(uk, state_p)

        e1 = tf.exp(state_p - current_p)
        e2 = tf.exp(uk - current_p)

        wkv = tf.math.divide_no_nan(e1 * state_a + e2 * v, e1 * state_b + e2)

        """
        下面是计算当前时间步为下一时间步生成的states信息.
        其中current_p state_p是缩放因子.state_a,state_b都被除以exp(current_p)进行缩放
        """
        current_w = -tf.exp(self.time_decay) + state_p
        current_p = tf.maximum(current_w, k)
        current_e1 = tf.exp(current_w - current_p)
        current_e2 = tf.exp(k - current_p)

        current_a = current_e1 * state_a + current_e2 * v
        current_b = current_e1 * state_b + current_e2

        return wkv, (current_a, current_b, current_p)

    @property
    def state_size(self):
        return (tf.TensorShape((self.hidden_size,)),
                tf.TensorShape((self.hidden_size,)),
                tf.TensorShape((self.hidden_size,)))

    @property
    def output_size(self):
        return self.hidden_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = 1
        if dtype is None:
            dtype = tf.float32
        """
        生成初始的state cell,其中上一状态的输入state_x,分子state_a,分母state_b使用全零初始化
        缩放因子state p初始化到负无穷 经过maximum基本上不影响输入的值,在这里初始化到负-1e8
        """

        state_a = tf.zeros(shape=(batch_size, self.hidden_size), dtype=tf.float32)
        state_b = tf.zeros(shape=(batch_size, self.hidden_size), dtype=tf.float32)
        state_p = tf.ones(shape=(batch_size, self.hidden_size), dtype=tf.float32) - 1e30

        return (state_a, state_b, state_p)

class TimeMixParallel(Layer):
    def __init__(self,layer_idx, weights_dict, trainable=True):
        super(TimeMixParallel, self).__init__(name=f"time_mix_parallel_{layer_idx}", trainable=trainable)
        self.time_first = weights_dict[f'blocks.{layer_idx}.att.time_first']#1,n_hidden
        self.time_decay = weights_dict[f'blocks.{layer_idx}.att.time_decay']#1,n_hidden
        self.hidden_size = int(tf.shape(self.time_first)[-1])  # 获取隐藏层宽度
    @staticmethod
    def convolution(inputs,kernel):
        bz, seq_len, _ = tf.TensorShape(inputs.shape).as_list()
        kernel_len,hidden_size = tf.TensorShape(kernel.shape).as_list()

        inputs_pad = tf.pad(inputs, [[0, 0], [kernel_len - 1, 0], [0, 0]], constant_values=0.)  # shape=(bz,)
        inputs_pad = tf.expand_dims(inputs_pad,axis=1)#bz,1,time_step,hidden_size

        kernel = tf.reshape(kernel,(1,kernel_len,hidden_size,1))

        outputs = tf.nn.depthwise_conv2d(inputs_pad,kernel,[1,1,1,1],'VALID')
        outputs = tf.squeeze(outputs,axis=1)#bz,timestep,hidden_size
        return outputs


    def call(self, inputs,initial_state=None):
        k, v = tf.unstack(inputs, axis=2)
        bz,seq_len,_ = tf.TensorShape(k.shape).as_list()
        zero_pad = tf.zeros(shape=(1,self.hidden_size),dtype=k.dtype)


        time_decay =  -tf.exp(self.time_decay)#(hidden,)

        reversed_id = tf.expand_dims(tf.constant(tf.range(seq_len -1,limit=-1,delta=-1,dtype=time_decay.dtype)),axis=1)[1:]

        uk = k + tf.expand_dims(self.time_first,axis=1)
        max_k = tf.reduce_max(k,axis=1,keepdims=True)
        max_uk = tf.reduce_max(uk,axis=1,keepdims=True)

        exp_k = tf.exp(k - max_k)
        exp_uk = tf.exp(uk - max_uk)

        scale = tf.exp(max_k - max_uk)

        time_curve = tf.exp(time_decay * reversed_id)#exp(-nw),exp(-(n-1)w),....,exp(-w),1

        kv = exp_k * v

        time_curve = tf.concat([time_curve,zero_pad],axis=0)
        print('curve:',tf.reduce_max(time_curve))
        print('first:',tf.reduce_max(exp_uk))
        print('expK:',tf.reduce_max(exp_k))

        num_a = TimeMixParallel.convolution(kv,time_curve)
        num = num_a * scale + exp_uk * v
        den_b = TimeMixParallel.convolution(exp_k,time_curve)
        den = den_b * scale + exp_uk

        wkv = tf.math.divide_no_nan(num,den)

        return wkv,None,None,None

class TimeMixLast(Layer):
    def __init__(self, layer_idx, weights_dict, trainable=True):
        super(TimeMixLast, self).__init__(name=f"time_mix_last_{layer_idx}", trainable=trainable)
        self.output_weight = weights_dict[f'blocks.{layer_idx}.att.output.weight']

    def call(self, inputs_x, inputs_r, inputs_wkv):
        rwkv = inputs_r * inputs_wkv
        rwkvo = tf.matmul(rwkv, tf.cast(self.output_weight, dtype=rwkv.dtype))
        outputs = inputs_x + rwkvo
        return outputs

kv_cache = None
class TimeMix(Layer):
    def __init__(self, layer_idx, weights_dict,parallel_mode=False, trainable=True):
        super(TimeMix, self).__init__(name=f"time_mix_{layer_idx}", trainable=trainable)
        self.parallel_mode = parallel_mode

        self.time_mix_first = TimeMixFirst(layer_idx, weights_dict, trainable)
        self.time_mix_last = TimeMixLast(layer_idx, weights_dict, trainable)
        if self.parallel_mode:
            self.time_mix_rnn = TimeMixParallel(layer_idx,weights_dict,trainable)
        else:
            self.time_mix_cell = TimeMixCell(layer_idx, weights_dict, trainable)
            self.time_mix_rnn = RNN(self.time_mix_cell, return_sequences=True, return_state=True,
                                    name=f"time_mix_rnn_{layer_idx}")

    def call(self, inputs, initial_state_dict=None):
        if initial_state_dict is None:
            input_state_x = None
            input_state_rnn = None
        else:
            input_state_x = initial_state_dict['time_mix_state_x']
            input_state_rnn = (initial_state_dict['time_mix_state_a'], initial_state_dict['time_mix_state_b'],
                               initial_state_dict['time_mix_state_p'])

        kv, r, output_state_x = self.time_mix_first(inputs, input_state=input_state_x)
        global  kv_cache
        if kv_cache is None:
            kv_cache = kv
        else:
            print(tf.reduce_mean(tf.abs(kv_cache - kv)))
        wkv, output_state_a, output_state_b, output_state_p = self.time_mix_rnn(kv, initial_state=input_state_rnn)
        outputs = self.time_mix_last(inputs, inputs_r=r, inputs_wkv=wkv)

        return outputs, (output_state_x, output_state_a, output_state_b, output_state_p)


class ChannelMix(Layer):
    def __init__(self, layer_idx, weights_dict, trainable=True):
        super(ChannelMix, self).__init__(name=f'channel_mix_{layer_idx}', trainable=trainable)
        ln_scale_var = weights_dict[f'blocks.{layer_idx}.ln2.weight']
        ln_center_var = weights_dict[f'blocks.{layer_idx}.ln2.bias']
        self.input_norm = CustomLayerNormalization(ln_scale_var, ln_center_var, axis=-1, epsilon=1e-5, name="ln2")
        self.time_mix_k = weights_dict[f'blocks.{layer_idx}.ffn.time_mix_k']
        self.time_mix_r = weights_dict[f'blocks.{layer_idx}.ffn.time_mix_r']

        self.key_weight = weights_dict[f'blocks.{layer_idx}.ffn.key.weight']
        self.value_weight = weights_dict[f'blocks.{layer_idx}.ffn.value.weight']
        self.receptance_weight = weights_dict[f'blocks.{layer_idx}.ffn.receptance.weight']
        self.hidden_width = int(tf.shape(ln_center_var)[0])

    def call(self, inputs, initial_state_dict=None):
        x_norm = self.input_norm(inputs)
        bz = tf.shape(x_norm)[0]
        if initial_state_dict is None:
            initial_state = self.get_initial_state(bz, x_norm.dtype)
        else:
            initial_state = initial_state_dict['channel_mix_state_x']

        initial_state = tf.concat([initial_state, x_norm[:, :-1, :]], axis=1)  # 在timestamp维度上拼接

        kx_mixed = x_norm * self.time_mix_k + initial_state * (1. - self.time_mix_k)
        rx_mixed = x_norm * self.time_mix_r + initial_state * (1. - self.time_mix_r)

        r = tf.nn.sigmoid(tf.matmul(rx_mixed, self.receptance_weight))
        vx = tf.matmul(kx_mixed, self.key_weight)
        vx = tf.square(tf.nn.relu(vx))

        out_v = r * tf.matmul(vx, self.value_weight)

        outputs = out_v + inputs

        return outputs, x_norm[:, -1:, :]

    def get_initial_state(self, batch_size=None, dtype=None):

        if batch_size is None:
            batch_size = 1
        if dtype is None:
            dtype = tf.float32
        """
        生成初始的state cell,其中上一状态的输入state_x 使用全零初始化
        """

        state_x = tf.zeros(shape=(batch_size, 1, self.hidden_width), dtype=dtype)

        return state_x


class CustomEmbedding(Layer):
    def __init__(self, weights_dict, name="embeddingLayer"):
        super(CustomEmbedding, self).__init__(name=name)
        self.embedding_weight = weights_dict['emb.weight']
        ln_scale_var = weights_dict['blocks.0.ln0.weight']
        ln_center_var = weights_dict['blocks.0.ln0.bias']
        self.embedding_layerNorm = CustomLayerNormalization(ln_scale_var, ln_center_var, axis=-1, epsilon=1e-5,
                                                            name="ln0")

    def call(self, inputs):
        x = tf.nn.embedding_lookup(self.embedding_weight, inputs)
        outputs = self.embedding_layerNorm(x)
        return outputs


class OutputLayer(Layer):
    def __init__(self, weights_dict, name="outputLayer"):
        super(OutputLayer, self).__init__(name=name)

        ln_scale_var = weights_dict[f'ln_out.weight']
        ln_center_var = weights_dict[f'ln_out.bias']
        self.output_norm = CustomLayerNormalization(ln_scale_var, ln_center_var, axis=-1, epsilon=1e-5,
                                                    name="output_ln")
        self.head = weights_dict['head.weight']

    def call(self, inputs, *args, **kwargs):
        inputs = tf.cast(inputs, dtype=tf.float32)

        x_norm = self.output_norm(inputs)
        outputs = tf.matmul(x_norm, self.head)

        return outputs


class RWKVModelOutput:
    def __init__(self, rwkv_outputs, rwkv_states=None, kv_cache_states=None):
        self.rwkv_outputs = rwkv_outputs
        self.rwkv_states = rwkv_states
        self.kv_cache_states = kv_cache_states

    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise KeyError


class RWKV(Model):
    def __init__(self, model_config=None, rwkv_model_pth=None, attention_layers=None,parallel_mode=False, model_name="RWKV"):
        super(RWKV, self).__init__(name=model_name)
        self.num_layers = 0
        self.parallel_mode = parallel_mode
        if self.parallel_mode:
            print("当前TimeMix处于并行模式下,并行模式下不支持传递state状态参数....")

        layers_and_dicts = None
        if model_config is not None:
            layers_and_dicts = self._build_model_from_config(config=model_config)
        if rwkv_model_pth is not None:
            print('开始从pytorch的pth模型权重文件上加载模型..')
            start_time = time.time()
            layers_and_dicts = self._load_model_from_pytorch(rwkv_model_pth)
            print('模型权重读取完成,耗时： %.2fs' % (time.time() - start_time))

        if layers_and_dicts is not None:
            self.num_layers, self.weights_dict = layers_and_dicts
        else:
            raise RWKVModelError("必须选择从config构建模型或者从pth上加载模型")
        start_time = time.time()
        print('开始构建模型')
        self._build_model()
        print('模型构建完成,耗时： %.2fs' % (time.time() - start_time))
        self.checkpoint = tf.train.Checkpoint(**{key.replace('.', '_'): var for key, var in self.weights_dict.items()})

    def get_checkpoint(self):
        return self.checkpoint

    def _build_model(self, build_batch_size=2, test_seq_len=64):
        self.embedding_layer = CustomEmbedding(self.weights_dict)

        self.output_layer = OutputLayer(self.weights_dict)

        self.time_mix_layers = [TimeMix(idx, self.weights_dict,self.parallel_mode) for idx in range(self.num_layers)]

        self.channel_mix_layers = [ChannelMix(idx, self.weights_dict) for idx in range(self.num_layers)]

        self.attention_layers = [
            AttentionBlock(idx, self.weights_dict) if f'blocks.{idx}.ln3.weight' in self.weights_dict else None for idx
            in range(self.num_layers)]

    def _load_model_from_pytorch(self, rwkv_pth_filepath: str) -> Tuple[int, Dict[str, tf.Variable]]:
        tf_variables_dict = {}
        try:
            import torch
            ckpt_dict = torch.load(rwkv_pth_filepath, map_location="cpu")
            gc.collect()

            num_layers = get_model_layers_count(ckpt_dict)
            var_keys = set(ckpt_dict.keys())

            for var_name in var_keys:
                tensor = ckpt_dict[var_name]
                tensor = tensor.detach().to('cpu', dtype=torch.float32).numpy()

                if 'time_decay' in var_name or 'time_first' in var_name:  # 这两个的shape=(hidden_size,)需要在前面补充一个batch_size维度
                    tensor = np.expand_dims(tensor, axis=0)

                """
                if 'att.time_mix' in var_name:
                    # attention的tensor shape=(1,1,hidden_size) 我在这里使用循环展开处理rwkv 因此中间的时间维度需要去除
                    # FFN的mix保留原始形状
                    tensor = tensor.squeeze(axis=1)
                """

                if 'key.weight' in var_name or 'value.weight' in var_name \
                        or 'receptance.weight' in var_name or 'output.weight' in var_name or 'head.weight' in var_name:
                    tensor = tensor.T

                tensor = self.add_weight(name=var_name, shape=tensor.shape,
                                         initializer=tf.initializers.constant(tensor), dtype=tf.float32)
                del ckpt_dict[var_name]
                tf_variables_dict[var_name] = tensor
        except ModuleNotFoundError as e:
            raise RWKVModelError('如果要从pth载入模型,请安装pytorch,cpu版本即可')

        return num_layers, tf_variables_dict

    # def _load_attention_from_checkpoint(self,att_ckpt_filepath:str):

    def export_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


    def _build_model_from_config(self, config: Union[str, Dict[str, Union[int, List[bool]]]]) -> Tuple[
        int, Dict[str, tf.Variable]]:
        if type(config) == str:
            if not os.path.exists(config):
                raise RWKVModelError("如果传入的config是一个字符串类型，则默认这个config是一个配置文件路径,代码会使用尝试json打开. 但是您传入的文件路径不存在")
            with open(config, mode="r", encoding="utf-8") as fi:
                config = json.load(fi)
        elif type(config) == dict:
            pass
        else:
            raise RWKVModelError("您传入的config必须为一个json格式的配置文件路径或者一个字典类型的配置对象")

        if 'num_layers' not in config: raise "您的config必须包含一个num_layers字段,来标记模型的层数"
        if 'hidden_size' not in config: raise "您的config必须包含一个hidden_size字段,来标记模型的隐藏层宽度"
        if 'vocabulary_size' not in config: raise "您的config必须包含一个vocabulary_size字段,来标记模型的模型的词汇表大小"
        if 'expand_size' not in config:
            config['expand_size'] = 4 * config['hidden_size']
            print(f'没有在config里找到expand_size字段,我们默认ffn里隐藏层宽度扩大四倍,则expand_size={config["expand_size"]}')
        if 'attention_layers' not in config: config['attention_layers'] = [False] * config['num_layers']

        if True in config['attention_layers']:
            if 'attention_query_dims' not in config and 'attention_key_dims' not in config:
                print('您启用了attention,但是没有在config里找到attention_query_dims字段,我们默认query的宽度等于rwkv隐藏层宽度(rwkv隐藏层宽度会向下取整为8的倍数)')
                print('您启用了attention,但是没有在config里找到attention_key_dims字段,我们默认key的宽度等于query的隐藏层宽度的1/8')

                config['attention_query_dims'] = (config['hidden_size'] // 8) * 8
                config['attention_key_dims'] = config['attention_query_dims'] // 8
            elif 'attention_query_dims' not in config:
                print('您启用了attention,但是没有在config里找到attention_query_dims字段,我们默认query的宽度等于key宽度的8倍')
                config['attention_query_dims'] = config['attention_key_dims'] * 8
            elif 'attention_key_dims' not in config:
                if config['attention_query_dims'] % 8 ==0:
                    print('您启用了attention,但是没有在config里找到attention_key_dims字段,我们默认key的宽度等于query的隐藏层宽度的1/8')
                    config['attention_key_dims'] = config['attention_query_dims'] // 8
                else:
                    raise RWKVModelError('您启用了attention,设置了attention_query_dims,但是没有设置attention_key_dims,我们要求query的数值是key的整数倍.')



        tf_variables_dict = {"emb.weight": self.add_weight(shape=(config['vocabulary_size'], config['hidden_size']),
                                                           dtype=tf.float32, name="emb.weight"),
                             "head.weight": self.add_weight(shape=(config['hidden_size'], config['vocabulary_size']),
                                                            dtype=tf.float32, name="head.weight")}

        for var_name in ["ln_out.weight", "ln_out.bias","blocks.0.ln0.weight","blocks.0.ln0.bias"]:
            tf_variables_dict[var_name] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                                 name=var_name)

        num_layers = config['num_layers']
        for idx in range(num_layers):
            # layerNorm 1
            for var_name in ['ln1.weight','ln1.bias','ln2.weight','ln2.bias']:
                layer_name = f"blocks.{idx}.{var_name}"
                tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                                name=layer_name)
            for var_name in ['time_decay','time_first']:
                layer_name = f"blocks.{idx}.att.{var_name}"
                tf_variables_dict[layer_name] = self.add_weight(shape=(1, config['hidden_size'],), dtype=tf.float32,
                                                                name=layer_name)

            for var_name in ['time_mix_k','time_mix_r']:
                layer_name = f"blocks.{idx}.ffn.{var_name}"
                tf_variables_dict[layer_name] = self.add_weight(shape=(1, 1, config['hidden_size'],), dtype=tf.float32,
                                                                name=layer_name)

            for var_name in ['time_mix_k', 'time_mix_v', 'time_mix_r']:
                layer_name = f"blocks.{idx}.att.{var_name}"
                tf_variables_dict[layer_name] = self.add_weight(shape=(1, 1, config['hidden_size'],), dtype=tf.float32,
                                                                name=layer_name)

            for var_name in ['key','value','receptance','output']:
                layer_name = f"blocks.{idx}.att.{var_name}.weight"
                tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'], config['hidden_size']),
                                                                dtype=tf.float32, name=layer_name)

            for var_name in ['key', 'receptance']:
                layer_name = f"blocks.{idx}.ffn.{var_name}.weight"
                tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'], config['expand_size']),
                                                                dtype=tf.float32, name=layer_name)

            layer_name = f"blocks.{idx}.ffn.value.weight"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['expand_size'], config['hidden_size']),
                                                            dtype=tf.float32, name=layer_name)

            if config['attention_layers'][idx]:
                for var_name in ['ln3.weight','ln3.bias']:

                    layer_name = f"blocks.{idx}.{var_name}"
                    tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                                    name=layer_name)

                layer_name = f"blocks.{idx}.dense_q.weight"
                tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'],config['attention_query_dims']), dtype=tf.float32,
                                                                name=layer_name)

                for var_name in ['dense_k','dense_v']:
                    layer_name = f"blocks.{idx}.{var_name}.weight"
                    tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'],config['attention_key_dims']), dtype=tf.float32,
                                                                    name=layer_name)

                layer_name = f"blocks.{idx}.dense_o.weight"
                tf_variables_dict[layer_name] = self.add_weight(shape=(config['attention_query_dims'],config['hidden_size']), dtype=tf.float32,
                                                                name=layer_name)


        return num_layers, tf_variables_dict

    def forward_sequence(self, inputs, rwkv_states=None, att_kv_caches=None, training=False):  # inputs=batch,timestamp

        outputs_sequence = inputs
        if rwkv_states is None:
            rwkv_states = [None for _ in range(self.num_layers)]
        if att_kv_caches is None:
            att_kv_caches = [None for _ in range(self.num_layers)]

        next_states = []
        next_caches = []
        for idx, (time_mix_layer, channel_mix_layer, attention_layer, layer_states, layer_caches) in enumerate(
                zip(self.time_mix_layers, self.channel_mix_layers, self.attention_layers, rwkv_states, att_kv_caches)):
            outputs_sequence, (
                time_output_state_x, time_output_state_a, time_output_state_b, time_output_state_p) = time_mix_layer(
                outputs_sequence, initial_state_dict=layer_states)
            """
            outputs_sequence, channel_output_state_x = channel_mix_layer(outputs_sequence,
                                                                         initial_state_dict=layer_states)
            """
            channel_output_state_x = None
            output_states = {
                'time_mix_state_x': time_output_state_x,
                'time_mix_state_a': time_output_state_a,
                'time_mix_state_b': time_output_state_b,
                'time_mix_state_p': time_output_state_p,
                'channel_mix_state_x': channel_output_state_x
            }
            next_states.append(output_states)

            if attention_layer is not None:
                outputs_sequence, cache_k, cache_v = attention_layer(outputs_sequence, kv_cahe=att_kv_caches,
                                                                     att_mask=None, training=training)
                output_caches = {
                    'att_cache_k': cache_k,
                    'att_cache_v': cache_v

                }
            else:
                output_caches = None
            next_caches.append(output_caches)
            break

        return outputs_sequence, next_states, next_caches

    # @tf.function
    def call(self, inputs, rwkv_states=None, att_kv_caches=None, return_states=False, training=False):
        """
        :param inputs: 输入维度,如果输入维度为1,即只有batch一个维度,则认为是循环模式,如果inputs是二维即batch,timestamp则认为是并行模式(fake
        :return:outputs,final_states
         outputs的形状与inputs相同
         final_states是四元组结构 由time_mix产生,channel_mix借用四元组里的last_input
        """

        assert len(tf.shape(inputs)) == 2
        embedded_inputs = self.embedding_layer(inputs)

        rwkv_outputs, next_rwkv_states, next_kv_caches = self.forward_sequence(embedded_inputs, rwkv_states=rwkv_states,
                                                                               att_kv_caches=att_kv_caches,
                                                                               training=training)

        outputs = self.output_layer(rwkv_outputs)

        if return_states:
            return RWKVModelOutput(rwkv_outputs=outputs, rwkv_states=rwkv_states, kv_cache_states=next_kv_caches)
        else:
            return outputs

if __name__ == '__main__1':

    hidden = 1
    seq_len = 16
    layer_idx = 0
    weight_dict = {
        f"blocks.{layer_idx}.att.time_first":tf.constant(value=2.,shape=(1,hidden),dtype=tf.float32),
        f"blocks.{layer_idx}.att.time_decay":tf.constant(value=-1.,shape=(1,hidden),dtype=tf.float32)
    }
    cell = TimeMixCell(layer_idx,weights_dict=weight_dict)
    rnn = RNN(cell,return_sequences=True)
    parallel,_,_,_ = TimeMixParallel(layer_idx,weights_dict=weight_dict)

    inputs_v = tf.ones(shape=(1,seq_len,1),dtype=tf.float32)
    inputs_k = tf.zeros(shape=(1,seq_len,1),dtype=tf.float32)
    inputs = tf.stack([inputs_k,inputs_v],axis=2)
    outputs1 = rnn(inputs)
    outputs2 = parallel(inputs)
    print(outputs1)
    print(outputs2)
    print(tf.reduce_mean(tf.abs(outputs1-outputs2),axis=[0,2]).numpy())


if __name__ == '__main__2':

    hidden = 384
    seq_len = 128
    layer_idx = 0
    bz = 2
    weight_dict = {
        f"blocks.{layer_idx}.att.time_first":tf.random.uniform(shape=(1,hidden),minval=-10,maxval=10,dtype=tf.float32),
        f"blocks.{layer_idx}.att.time_decay":tf.random.uniform(shape=(1,hidden),minval=-10,maxval=10,dtype=tf.float32)
    }
    cell = TimeMixCell(layer_idx,weights_dict=weight_dict)
    rnn = RNN(cell,return_sequences=True)
    parallel = TimeMixParallel(layer_idx,weights_dict=weight_dict)

    inputs = tf.random.uniform(shape=(bz,seq_len,2,hidden),minval=-10,maxval=10,dtype=tf.float32)
    outputs1 = rnn(inputs)
    outputs2,_,_,_ = parallel(inputs)

    print(tf.reduce_mean(tf.abs(outputs1-outputs2),axis=[0,2]).numpy())
    print(tf.reduce_mean(tf.abs(outputs1 - outputs2)))


if __name__ == '__main__':
    rwkv1 = RWKV(rwkv_model_pth=r"C:\Users\a1313\Desktop\RWKV-4-World-0.4B-v1-20230529-ctx4096.pth",parallel_mode=False,model_name="rnn")
    rwkv2 = RWKV(rwkv_model_pth=r"C:\Users\a1313\Desktop\RWKV-4-World-0.4B-v1-20230529-ctx4096.pth",parallel_mode=True,model_name="par")
    inputs = tf.random.uniform(shape=(4,32),minval=0,maxval=1000,dtype=tf.int32)
    start = time.time()
    outputs1 = rwkv1(inputs)
    print(time.time()-start)

    start = time.time()
    outputs2 = rwkv2(inputs)
    print(time.time()-start)
    print(tf.reduce_mean(tf.abs(outputs1 - outputs2)))