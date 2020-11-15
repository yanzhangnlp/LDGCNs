import mxnet as mx

import sockeye.constants as C
from sockeye import utils
from sockeye.config import Config


import logging
logger = logging.getLogger(__name__)


def get_gcn(config, prefix):
    gcn = GCNCell(input_dim=config.input_dim,
                  output_dim=config.output_dim,
                  direction_num=config.direction_num,
                  num_blocks=config.num_blocks,
                  adj_norm=config.adj_norm,
                  dropout=config.dropout,
                  prefix=prefix)
    return gcn


class GCNConfig(Config):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 direction_num: int,
                 num_blocks: int,
                 adj_norm: bool = True,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.direction_num = direction_num
        self.num_blocks = num_blocks
        self.adj_norm = adj_norm
        self.activation = activation
        self.dropout = dropout
        self.dtype = dtype


class GCNCell(object):
    def __init__(self,
                 input_dim,
                 output_dim,
                 direction_num,
                 num_blocks,
                 adj_norm=True,
                 prefix='gcn_',
                 activation='relu',
                 dropout=0.0):

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._direction_num = direction_num
        self._num_blocks = num_blocks
        self._layers = []
        self._prefix = prefix

        self._activation = activation
        self._dropout = dropout
        self._norm = adj_norm

        self.reset()
        self._modified = False
        self._own_params = True

        if self._input_dim != self._output_dim:
            self._input_W = mx.symbol.Variable(self._prefix + '_input_weight',
                                               shape=(input_dim, output_dim))
            self._input_b = mx.symbol.Variable(self._prefix + '_input_bias',
                                               shape=(output_dim,))

        self._layers.append(GraphConvolution(prefix="%s_6_" % self._prefix,
                                             heads=6,
                                             output_dim=self._output_dim,
                                             direction_num=self._direction_num,
                                             dropout=self._dropout,
                                             norm=self._norm,
                                             activation=self._activation))
        self._layers.append(GraphConvolution(prefix="%s_3_" % self._prefix,
                                             heads=3,
                                             output_dim=self._output_dim,
                                             direction_num=self._direction_num,
                                             dropout=self._dropout,
                                             norm=self._norm,
                                             activation=self._activation))

        # Layer Aggregation Params
        self._aggregate_W = mx.symbol.Variable(self._prefix + '_aggregate_weight',
                                               shape=(self._num_blocks * 2 * self._output_dim, self._output_dim))
        self._aggregate_b = mx.symbol.Variable(self._prefix + '_aggregate_bias',
                                               shape=(self._output_dim,))

    def convolve(self, adj, inputs, seq_len):
        layer_list = []

        if self._input_dim != self._output_dim:
            inputs = mx.sym.dot(inputs, self._input_W)
            inputs = mx.sym.broadcast_add(inputs, self._input_b)

        outputs = inputs
        for _ in range(self._num_blocks):
            outputs = self._layers[0](adj=adj, inputs=outputs, seq_len=seq_len)
            layer_list.append(outputs)
            outputs = self._layers[1](adj=adj, inputs=outputs, seq_len=seq_len)
            layer_list.append(outputs)

        aggregate_output = mx.sym.concat(*layer_list, dim=2)
        aggregate_output = mx.sym.dot(aggregate_output, self._aggregate_W)
        aggregate_output = mx.sym.broadcast_add(aggregate_output, self._aggregate_b)

        return aggregate_output

    def reset(self):
        pass


class GraphConvolution:
    def __init__(self,
                 prefix: str,
                 heads: int,
                 output_dim: int,
                 direction_num: int,
                 dropout: float,
                 norm: bool,
                 activation: str = 'relu'):

        self._prefix = prefix
        self._heads = heads
        self._output_dim = output_dim
        self._direction_num = direction_num
        utils.check_condition(output_dim % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, output_dim))
        self._hidden_dim = self._output_dim // self._heads
        self._dropout = dropout
        self._norm = norm
        self._activation = activation
        self._weight_list = []
        self._bias_list = []

        # Graph Convolution Params
        for i in range(heads):
            self._weight_list.append([mx.symbol.Variable(self._prefix + "_dense_" + str(i) + "_" + str(j) + "_weight",
                                                         shape=(self._output_dim + self._hidden_dim * i, self._hidden_dim))
                                      for j in range(self._direction_num)])
            self._bias_list.append([mx.symbol.Variable(self._prefix + "_dense_" + str(i) + "_" + str(j) + "_bias",
                                                       shape=(self._hidden_dim,))
                                    for j in range(self._direction_num)])

        # Direction Params
        self._direct_W = [mx.symbol.Variable(self._prefix + str(i) + '_direct_weight',
                                             shape=(self._direction_num * self._hidden_dim, self._hidden_dim))
                          for i in range(self._heads)]
        self._direct_b = [mx.symbol.Variable(self._prefix + str(i) + '_direct_bias',
                                             shape=(self._hidden_dim,))
                          for i in range(self._heads)]

        # Linear Transform Params
        self._linear_W = mx.symbol.Variable(self._prefix + '_linear_weight',
                                            shape=(self._output_dim, self._output_dim))
        self._linear_b = mx.symbol.Variable(self._prefix + '_linear_bias',
                                            shape=(self._output_dim,))

    def __call__(self, adj, inputs, seq_len):
        outputs = inputs
        cache_list = [outputs]
        output_list = []
        for i in range(self._heads):
            convolved = self._convolve(adj, outputs, i, seq_len)
            cache_list.append(convolved)
            outputs = mx.sym.concat(*cache_list, dim=2)
            output_list.append(convolved)

        outputs = mx.sym.concat(*output_list, dim=2)
        if self._dropout != 0.0:
            outputs = mx.sym.Dropout(outputs, p=self._dropout)

        outputs = mx.sym.broadcast_add(outputs, inputs)
        outputs = mx.sym.dot(outputs, self._linear_W)
        outputs = mx.sym.broadcast_add(outputs, self._linear_b)

        return outputs

    def _convolve(self, adj, inputs, i, seq_len):
        direct_list = []
        for j in range(self._direction_num):
            # k = i * self._directions + j

            weight = self._weight_list[i][j]
            bias = self._bias_list[i][j]

            output = mx.sym.dot(inputs, weight)
            output = mx.sym.broadcast_add(output, bias)

            label_id = j + 2
            mask = mx.sym.ones_like(adj) * label_id
            adji = (mask == adj)
            adji = mx.sym.Dropout(adji, p=self._dropout)

            mix_hop_adji = mx.sym.batch_dot(adji, adji)

            lambda_p = 0.7

            output_a = mx.sym.batch_dot(adji, output)
            output_b = mx.sym.batch_dot(mix_hop_adji, output)
            gate_b = mx.sym.Activation(data=output_b, act_type="sigmoid") * ( 1 - lambda_p)

            output = mx.sym.broadcast_mul(output_a, 1.0-gate_b) +  mx.sym.broadcast_mul(output_b, gate_b) 
            if j == 0:
                total_mix_hop_adji = adji
            else:
                total_mix_hop_adji = total_mix_hop_adji + adji

            direct_list.append(output)

            output = mx.sym.batch_dot(adji, output)
            direct_list.append(output)

        outputs = mx.sym.concat(*direct_list, dim=2)
        direct_W = self._direct_W[i]
        direct_b = self._direct_b[i]

        outputs = mx.sym.dot(outputs, direct_W)
        outputs = mx.sym.broadcast_add(outputs, direct_b)

        if self._norm:
            norm_adj = mx.sym.broadcast_not_equal(adj, mx.sym.zeros_like(adj))
            norm_factor = mx.sym.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.sym.broadcast_div(outputs, norm_factor)

        final_output = mx.sym.Activation(outputs, act_type=self._activation)
        return final_output