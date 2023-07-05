import gc
import json
import os.path
import time
from typing import Dict, Union, Tuple
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import *


"""
  获取模型一共有多少层
"""
def mae(x,y):
    return tf.reduce_mean(tf.abs(x-y))


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

    def get_initial_state(self,batch_size=None, dtype=None):
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
            input_state = self.get_initial_state(bz,x_norm.dtype)

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
        我无法理解current_w 为什么要加上state_p 源代码有 就抄上了
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


        return (state_a,state_b,state_p)


class TimeMixLast(Layer):
    def __init__(self, layer_idx, weights_dict, trainable=True):
        super(TimeMixLast, self).__init__(name=f"time_mix_last_{layer_idx}", trainable=trainable)
        self.output_weight = weights_dict[f'blocks.{layer_idx}.att.output.weight']

    def call(self, inputs_x, inputs_r, inputs_wkv):
        rwkv = inputs_r * inputs_wkv
        rwkvo = tf.matmul(rwkv, tf.cast(self.output_weight, dtype=rwkv.dtype))
        outputs = inputs_x + rwkvo
        return outputs

class TimeMix(Layer):
    def __init__(self, layer_idx, weights_dict, trainable=True):
        super(TimeMix, self).__init__(name=f"time_mix_{layer_idx}",trainable=trainable)
        self.time_mix_cell = TimeMixCell(layer_idx,weights_dict,trainable)
        self.time_mix_first = TimeMixFirst(layer_idx,weights_dict,trainable)
        self.time_mix_last = TimeMixLast(layer_idx,weights_dict,trainable)
        self.time_mix_rnn = RNN(self.time_mix_cell, return_sequences=True, return_state=True,name=f"time_mix_rnn_{layer_idx}")

    def call(self, inputs,initial_state_dict=None):
        if initial_state_dict is None:
            input_state_x = None
            input_state_rnn = None
        else:
            input_state_x = initial_state_dict['time_mix_state_x']
            input_state_rnn = (initial_state_dict['time_mix_state_a'],initial_state_dict['time_mix_state_b'],initial_state_dict['time_mix_state_p'])

        kv,r,output_state_x = self.time_mix_first(inputs,input_state=input_state_x)
        wkv,output_state_a, output_state_b, output_state_p = self.time_mix_rnn(kv,initial_state=input_state_rnn)
        outputs = self.time_mix_last(inputs,inputs_r=r,inputs_wkv=wkv)

        return outputs,(output_state_x,output_state_a,output_state_b,output_state_p)



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
            initial_state = self.get_initial_state(bz,x_norm.dtype)
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


class RWKV(Model):
    def __init__(self, model_config=None, model_pth=None, model_name="RWKV"):
        super(RWKV, self).__init__(name=model_name)
        self.num_layers = 0
        layers_and_dicts = None
        if model_config is not None:
            layers_and_dicts = self._build_model_from_config(config=model_config)
        if model_pth is not None:
            print('开始从pytorch的pth模型权重文件上加载模型..')
            start_time = time.time()
            layers_and_dicts = self._load_model_from_pth(model_pth)
            print('模型权重读取完成,耗时： %.2fs' % (time.time() - start_time))
        if layers_and_dicts is not None:
            self.num_layers, self.weights_dict = layers_and_dicts
        else:
            raise RWKVModelError("必须选择从config构建模型或者从pth上加载模型")
        start_time = time.time()
        print('开始构建模型')
        self._build_model()
        print('模型构建完成,耗时： %.2fs' % (time.time() - start_time))

    def _build_model(self, build_batch_size=2, test_seq_len=64):
        self.embedding_layer = CustomEmbedding(self.weights_dict)

        self.output_layer = OutputLayer(self.weights_dict)


        self.time_mix_layers = [TimeMix(idx,self.weights_dict) for idx in range(self.num_layers)]

        self.channel_mix_layers = [ChannelMix(idx, self.weights_dict) for idx in range(self.num_layers)]

        #x_inputs = tf.zeros(shape=(build_batch_size, test_seq_len), dtype=tf.int32)
        #self(x_inputs)

    def _load_model_from_pth(self, filepath: str) -> Tuple[int, Dict[str, tf.Variable]]:
        try:
            import torch
            ckpt_dict = torch.load(filepath, map_location="cpu")
            gc.collect()

            num_layers = get_model_layers_count(ckpt_dict)
            var_keys = set(ckpt_dict.keys())
            tf_variables_dict = {}
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

            return num_layers, tf_variables_dict

        except ModuleNotFoundError as e:
            print('如果要从pth载入模型,请安装pytorch,cpu版本即可')

    def _build_model_from_config(self, config: Union[str, Dict[str, int]]) -> Tuple[int, Dict[str, tf.Variable]]:
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

        tf_variables_dict = {}

        tf_variables_dict["emb.weight"] = self.add_weight(shape=(config['vocabulary_size'], config['hidden_size']),
                                                          dtype=tf.float32, name="emb.weight")

        tf_variables_dict["head.weight"] = self.add_weight(shape=(config['hidden_size'], config['vocabulary_size']),
                                                           dtype=tf.float32, name="head.weight")

        tf_variables_dict["ln_out.weight"] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                             name="ln_out.weight")
        tf_variables_dict["ln_out.bias"] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                           name="ln_out.bias")

        tf_variables_dict["blocks.0.ln0.weight"] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                                   name="ln_out.weight")
        tf_variables_dict["blocks.0.ln0.bias"] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                                 name="ln_out.bias")

        for idx in range(self.num_layers):
            # layerNorm 1
            layer_name = f"blocks.{idx}.ln1.weight"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)
            layer_name = f"blocks.{idx}.ln1.bias"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)

            # layerNorm 2
            layer_name = f"blocks.{idx}.ln2.weight"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)
            layer_name = f"blocks.{idx}.ln2.bias"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)

            layer_name = f"blocks.{idx}.att.time_decay"
            tf_variables_dict[layer_name] = self.add_weight(shape=(1, config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)

            layer_name = f"blocks.{idx}.att.time_first"
            tf_variables_dict[layer_name] = self.add_weight(shape=(1, config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)

            layer_name = f"blocks.{idx}.att.time_mix_k"
            tf_variables_dict[layer_name] = self.add_weight(shape=(1, config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)

            layer_name = f"blocks.{idx}.att.time_mix_v"
            tf_variables_dict[layer_name] = self.add_weight(shape=(1, config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)

            layer_name = f"blocks.{idx}.att.time_mix_r"
            tf_variables_dict[layer_name] = self.add_weight(shape=(1, config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)

            layer_name = f"blocks.{idx}.att.key.weight"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'], config['hidden_size']),
                                                            dtype=tf.float32, name=layer_name)

            layer_name = f"blocks.{idx}.att.value.weight"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'], config['hidden_size']),
                                                            dtype=tf.float32, name=layer_name)

            layer_name = f"blocks.{idx}.att.receptance.weight"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'], config['hidden_size']),
                                                            dtype=tf.float32, name=layer_name)

            layer_name = f"blocks.{idx}.att.output.weight"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'], config['hidden_size']),
                                                            dtype=tf.float32, name=layer_name)

            layer_name = f"blocks.{idx}.ffn.time_mix_k"
            tf_variables_dict[layer_name] = self.add_weight(shape=(1, config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)

            layer_name = f"blocks.{idx}.ffn.time_mix_r"
            tf_variables_dict[layer_name] = self.add_weight(shape=(1, config['hidden_size'],), dtype=tf.float32,
                                                            name=layer_name)

            layer_name = f"blocks.{idx}.ffn.key.weight"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'], config['expand_size']),
                                                            dtype=tf.float32, name=layer_name)

            layer_name = f"blocks.{idx}.ffn.receptance.weight"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'], config['expand_size']),
                                                            dtype=tf.float32, name=layer_name)

            layer_name = f"blocks.{idx}.ffn.value.weight"
            tf_variables_dict[layer_name] = self.add_weight(shape=(config['hidden_size'], config['expand_size']),
                                                            dtype=tf.float32, name=layer_name)
        """
          需要补全
        """

    def forward_sequence(self, inputs,rwkv_states=None):  # inputs=batch,timestamp

        embedded_inputs = self.embedding_layer(inputs)
        outputs_sequence = embedded_inputs
        if rwkv_states is None:
            rwkv_states = [None for _ in range(self.num_layers)]

        next_states = []
        for idx, (time_mix_layer, channel_mix_layer,layer_states) in enumerate(zip(self.time_mix_layers, self.channel_mix_layers,rwkv_states)):
            outputs_sequence, (time_output_state_x,time_output_state_a,time_output_state_b,time_output_state_p) = time_mix_layer(outputs_sequence, initial_state_dict=layer_states)
            outputs_sequence, channel_output_state_x = channel_mix_layer(outputs_sequence,initial_state_dict=layer_states)

            output_states = {
                'time_mix_state_x': time_output_state_x,
                'time_mix_state_a': time_output_state_a,
                'time_mix_state_b': time_output_state_b,
                'time_mix_state_p': time_output_state_p,
                'channel_mix_state_x': channel_output_state_x
            }
            next_states.append(output_states)


        return outputs_sequence,next_states

    # @tf.function
    def call(self, inputs, rwkv_states=None, return_states=False):
        """
        :param inputs: 输入维度,如果输入维度为1,即只有batch一个维度,则认为是循环模式,如果inputs是二维即batch,timestamp则认为是并行模式(fake
        :return:outputs,final_states
         outputs的形状与inputs相同
         final_states是四元组结构 由time_mix产生,channel_mix借用四元组里的last_input
        """

        assert len(tf.shape(inputs)) == 2

        rwkv_outputs,next_rwkv_states = self.forward_sequence(inputs,rwkv_states=rwkv_states)

        outputs = self.output_layer(rwkv_outputs)

        if return_states:
            return outputs, next_rwkv_states
        else:
            return outputs


if __name__ == '__main__':
    """
    在chatRWKV中,layernormalization是直接对embedding做的.这个操作会尝试在bf16下完成.
    当把chatRWKV的embedding转换为fp32以后 误差到达了1.78e-5级别
    """
    from chat import inputs_token, outputs as pytorch_outputs

    inputs = tf.expand_dims(inputs_token, axis=0)
    rwkv = RWKV(model_pth=r"C:\Users\a1313\Desktop\RWKV-4-World-0.1B-v1-20230520-ctx4096.pth")
    outputs, _ = rwkv(inputs, return_states=True)
    print('Tensorflow与Pytorch最终测试误差: ',tf.reduce_mean(tf.abs(outputs - pytorch_outputs)).numpy())
