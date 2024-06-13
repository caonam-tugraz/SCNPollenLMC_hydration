'''
    Source: Learning Convolutions from Scratch: https://arxiv.org/pdf/2007.13657.pdf
'''

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import List


######## SConv no bias
class SConv(nn.Module):
    def __init__(self, n_layers, n_units, n_channels, n_classes=10):
        super(SConv, self).__init__()

        mid_layers = []
        mid_layers.extend([
            nn.Conv2d(n_channels, n_units, kernel_size=25, stride=5, padding=1, bias=False),
            nn.ReLU(inplace=True)
            ])

        for _ in range(n_layers-2):
            mid_layers.extend([
                nn.Conv2d(n_units, n_units, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(n_units, momentum=0.9),
                nn.ReLU(inplace=True),
            ])

        mid_layers.extend([
            nn.Conv2d(n_units, n_units, kernel_size=15, stride=3, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(n_units, n_classes, bias=False)
        ])

        self.linear_stack = nn.Sequential(*mid_layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


######## SConv with bias
class SConvB(nn.Module):
    def __init__(self, n_layers, n_units, n_channels, n_classes=10):
        super(SConvB, self).__init__()

        mid_layers = []
        mid_layers.extend([
            nn.Conv2d(n_channels, n_units, kernel_size=25, stride=5, padding=1),
            nn.ReLU(inplace=True)
            ])

        for _ in range(n_layers-2):
            mid_layers.extend([
                nn.Conv2d(n_units, n_units, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(n_units, momentium=0.9),
                nn.ReLU(inplace=True),
            ])

        mid_layers.extend([
            nn.Conv2d(n_units, n_units, kernel_size=15, stride=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(n_units, n_classes)
        ])

        self.linear_stack = nn.Sequential(*mid_layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def linearcurve(self, e0, e1, t, inputs):
        for count, layer in enumerate(self.linear_stack):
            if isinstance(layer, nn.Linear):
                inputs = (1 - t) * e0.linear_stack[count](inputs) + \
                         t * e1.linear_stack[count](inputs)
            elif isinstance(layer, nn.Conv2d):
                inputs = (1 - t) * e0.linear_stack[count](inputs) + \
                         t * e1.linear_stack[count](inputs)
            else:
                inputs = self.linear_stack[count](inputs)
                
        # Print the data type of inputs before returning
        # print(f"Data type of inputs before return: {type(inputs)}")
        
        return inputs

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

######## SConv with bias
class SConvB_pollen(nn.Module):
    def __init__(self, n_layers, n_units, n_channels, n_classes=10):
        super(SConvB_pollen, self).__init__()

        mid_layers = []
        mid_layers.extend([
            nn.Conv2d(n_channels, n_units, kernel_size=25, stride=5, padding=1),
            nn.ReLU(inplace=True)
            ])

        for _ in range(n_layers-2):
            mid_layers.extend([
                nn.Conv2d(n_units, n_units, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(n_units, momentium=0.9),
                nn.ReLU(inplace=True),
            ])

        mid_layers.extend([
            nn.Conv2d(n_units, n_units, kernel_size=15, stride=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(n_units, n_classes)
        ])

        self.linear_stack = nn.Sequential(*mid_layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def linearcurve(self, e0, e1, t, inputs):
        for count, layer in enumerate(self.linear_stack):
            if isinstance(layer, nn.Linear):
                inputs = (1-t) * e0.linear_stack[count](inputs) + \
                         t * e1.linear_stack[count](inputs)
            elif isinstance(layer, nn.Conv2d):
                inputs = (1-t) * e0.linear_stack[count](inputs) + \
                         t * e1.linear_stack[count](inputs)
            else:
                inputs = self.linear_stack[count](inputs)
        return inputs

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits
    
######## HHN-SConv no bias
class HHN_SConv(nn.Module):
    def __init__(self, hin, dimensions, n_layers, n_units, n_channels, n_classes=10):
        super(HHN_SConv, self).__init__()
        self.hyper_stack = nn.Sequential(
            nn.Linear(hin, 64),
            nn.ReLU(),
            nn.Linear(64, dimensions),
            nn.Softmax(dim=0)
        )

        self.dimensions = dimensions
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.weight_list_conv1 = self.create_param_combination_conv(in_channels=n_channels,
                                                                    out_channels=n_units, kernel=25)

        #self.running_mu = torch.zeros(self.n_units).to(self.device)  # zeros are fine for first training iter
        #self.running_std = torch.ones(self.n_units).to(self.device)  # ones are fine for first training iter

        self.mus = []
        self.stds = []

        self.weight_and_biases = nn.ParameterList()
        for _ in range(n_layers - 2):
            w = self.create_param_combination_conv(in_channels=n_units,
                                                   out_channels=n_units, kernel=3)
            self.weight_and_biases += w

            self.mus.append(torch.zeros(self.n_units).to(self.device))
            self.stds.append(torch.ones(self.n_units).to(self.device))

        self.weight_list_conv2 = self.create_param_combination_conv(in_channels=n_units,
                                                                    out_channels=n_units, kernel=15)
        self.weight_list_fc3 = self.create_param_combination_linear(in_features=n_units,
                                                                    out_features=n_classes)

    def create_param_combination_conv(self, in_channels, out_channels, kernel):
        weight_list = nn.ParameterList()
        for _ in range(self.dimensions):
            weight = Parameter(torch.empty((out_channels, in_channels, kernel, kernel)))
            init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)
        return weight_list

    def create_param_combination_linear(self, in_features, out_features):
        weight_list = nn.ParameterList()
        for _ in range(self.dimensions):
            weight = Parameter(torch.empty((out_features, in_features)))
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)
        return weight_list

    def calculate_weighted_sum(self, param_list: List, factors: Tensor):
        weighted_list = [a * b for a, b in zip(param_list, factors)]
        return torch.sum(torch.stack(weighted_list), dim=0)

    def forward(self, x, hyper_x, training=True):
        hyper_output = self.hyper_stack(hyper_x)

        w_conv1 = self.calculate_weighted_sum(self.weight_list_conv1, hyper_output)
        w_conv2 = self.calculate_weighted_sum(self.weight_list_conv2, hyper_output)
        w_fc3 = self.calculate_weighted_sum(self.weight_list_fc3, hyper_output)

        logits = F.conv2d(x, weight=w_conv1, stride=5, padding=1, bias=None)
        logits = torch.relu(logits)

        it = iter(self.weight_and_biases)
        for (w, m, s) in zip(*[it] * self.dimensions, self.mus, self.stds):
            w = nn.ParameterList(w)
            w = self.calculate_weighted_sum(w.to(self.device), hyper_output)
            logits = F.conv2d(logits, weight=w, stride=1, padding=1, bias=None)
            logits = F.batch_norm(logits, m, s, training=training)
            logits = torch.relu(logits)

        logits = F.conv2d(logits, weight=w_conv2, stride=3, padding=0, bias=None)
        logits = torch.relu(logits)
        logits = torch.flatten(logits, start_dim=1)
        logits = F.linear(logits, weight=w_fc3, bias=None)
        return logits


######## HHN-SConv with bias
class HHN_SConvB(nn.Module):
    def __init__(self, hin, dimensions, n_layers, n_units, n_channels, n_classes=10):
        super(HHN_SConvB, self).__init__()
        self.hyper_stack = nn.Sequential(
            nn.Linear(hin, 64),
            nn.ReLU(),
            nn.Linear(64, dimensions),
            nn.Softmax(dim=0)
        )

        self.dimensions = dimensions
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #self.running_mu = torch.zeros(self.n_units).to(self.device)  # zeros are fine for first training iter
        #self.running_std = torch.ones(self.n_units).to(self.device)  # ones are fine for first training iter

        self.weight_list_conv1, self.bias_list_conv1 = \
            self.create_param_combination_conv(in_channels=n_channels,
                                               out_channels=n_units, kernel=25)

        self.mus = []
        self.stds = []

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for _ in range(n_layers - 2):
            w, b = self.create_param_combination_conv(in_channels=n_units,
                                                      out_channels=n_units, kernel=3)
            self.weights += w
            self.biases += b

            self.mus.append(torch.zeros(self.n_units).to(self.device))
            self.stds.append(torch.ones(self.n_units).to(self.device))

        self.weight_list_conv2, self.bias_list_conv2 = \
            self.create_param_combination_conv(in_channels=n_units,
                                               out_channels=n_units, kernel=15)
        self.weight_list_fc3, self.bias_list_fc3 = \
            self.create_param_combination_linear(in_features=n_units, out_features=n_classes)

    def create_param_combination_conv(self, in_channels, out_channels, kernel):
        weight_list = nn.ParameterList()
        bias_list = nn.ParameterList()
        for _ in range(self.dimensions):
            weight = Parameter(torch.empty((out_channels, in_channels, kernel, kernel)))
            init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)

            bias = Parameter(torch.empty(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)
            bias_list.append(bias)
        return weight_list, bias_list

    def create_param_combination_linear(self, in_features, out_features):
        weight_list = nn.ParameterList()
        bias_list = nn.ParameterList()
        for _ in range(self.dimensions):
            weight = Parameter(torch.empty((out_features, in_features)))
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)

            bias = Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
            bias_list.append(bias)
        return weight_list, bias_list

    def calculate_weighted_sum(self, param_list: List, factors: Tensor):
        weighted_list = [a * b for a, b in zip(param_list, factors)]
        return torch.sum(torch.stack(weighted_list), dim=0)

    def forward(self, x, hyper_x, training=True):
        hyper_output = self.hyper_stack(hyper_x)

        w_conv1 = self.calculate_weighted_sum(self.weight_list_conv1, hyper_output)
        w_conv2 = self.calculate_weighted_sum(self.weight_list_conv2, hyper_output)
        w_fc3 = self.calculate_weighted_sum(self.weight_list_fc3, hyper_output)

        b_conv1 = self.calculate_weighted_sum(self.bias_list_conv1, hyper_output)
        b_conv2 = self.calculate_weighted_sum(self.bias_list_conv2, hyper_output)
        b_fc3 = self.calculate_weighted_sum(self.bias_list_fc3, hyper_output)

        logits = F.conv2d(x, weight=w_conv1, bias=b_conv1, stride=5, padding=1)
        logits = torch.relu(logits)

        it_w = iter(self.weights)
        it_b = iter(self.biases)
        for (w, b, m, s) in zip(zip(*[it_w] * self.dimensions), zip(*[it_b] * self.dimensions), self.mus, self.stds):
            w = nn.ParameterList(w)
            b = nn.ParameterList(b)
            w = self.calculate_weighted_sum(w.to(self.device), hyper_output)
            b = self.calculate_weighted_sum(b.to(self.device), hyper_output)
            logits = F.conv2d(logits, weight=w, bias=b, stride=1, padding=1)
            #logits = F.batch_norm(logits, m, s, training=training)
            logits = torch.relu(logits)

        logits = F.conv2d(logits, weight=w_conv2, bias=b_conv2, stride=3, padding=0)
        logits = torch.relu(logits)
        logits = torch.flatten(logits, start_dim=1)
        logits = F.linear(logits, weight=w_fc3, bias=b_fc3)
        return logits
