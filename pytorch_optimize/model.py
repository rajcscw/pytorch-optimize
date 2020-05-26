import numpy as np
import torch
from functools import reduce
from enum import Enum
from collections import OrderedDict
from typing import List, Dict, Tuple
from torch import nn


class SamplingStrategy(Enum):
    RANDOM = 1
    BOTTOM_UP = 2
    TOP_DOWN = 3
    ALL = 4


class Model:
    """
    A wrapper over pytorch model to sample layer wise parameters and to update gradients
    """
    def __init__(self, net: nn.Module, strategy=SamplingStrategy.RANDOM):

        self.net = net
        self.sampling_strategy = strategy

        # collect layer wise parameters
        self.layer_wise_parameters = self._get_layer_params(net)
        self.total_layers = len(self.layer_wise_parameters)
        self.all_layers = list(self.layer_wise_parameters.keys())
        self.layer_dimensions = {}
        self._initialize_counters()

    def _initialize_counters(self):
        if self.sampling_strategy == SamplingStrategy.BOTTOM_UP:
            self.last = -1
        else:
            self.last = len(self.layer_wise_parameters)

    @staticmethod
    # TBD: may have to take care of cuda tensors
    def _str_to_type(str_type: str):
        if str_type == "torch.float32":
            return torch.float32
        elif str_type == "torch.float64":
            return torch.float64

    def _get_param_value_type(self, name: str) -> Tuple[torch.Tensor, torch.dtype]:
        """
        Gets value and type of a parameter
        """
        refs = name.split(".")
        last = self.net
        for ref in refs:
            last = getattr(last, ref)

        param_value = last.cpu().data
        param_type = last.data.dtype
        return param_value, param_type

    def _set_param_value(self, name: str, type: str, value: torch.Tensor):
        """
        Sets value of the specified parameter
        """
        obj = self.net
        parts = name.split(".")
        for attr in parts[:-1]:
            obj = getattr(obj, attr)
        type = self._str_to_type(str_type=type)
        setattr(obj, parts[-1], value.type(type))

    def _get_layer_params(self, net: nn.Module) -> Dict[str, Dict[str, Tuple]]:
        """
        Gets parameters which are grouped layer wise for the given network
        """
        layer_params = OrderedDict()
        parameters = list(torch.nn.Module.named_parameters(net))
        for param in parameters:

            # split into layer name and parameter name
            param_name = param[0]
            refs = param_name.split(".")
            layer_name = refs[0]

            # get parameter details
            data, type = self._get_param_value_type(param[0])
            type = str(type)
            shape = data.shape

            # store it in the layer information
            if layer_name in layer_params.keys():
                layer_params[layer_name][param_name] = (shape, type)
            else:
                layer_params[layer_name] = dict()
                layer_params[layer_name][param_name] = (shape, type)

        return layer_params

    def _sample_random_layer(self) -> Tuple[str, torch.Tensor]:
        """
        Samples a random layer and its value
        """
        choice = np.random.choice(np.arange(len(self.all_layers)))
        random_layer_name = self.all_layers[choice]
        random_layer_value = self._get_layer_value_by_name(random_layer_name)
        return random_layer_name, random_layer_value

    def sample(self) -> Tuple[str, torch.Tensor]:
        """
        Samples a layer and its value

        Returns:
            Tuple[str, torch.Tensor] -- a tuple of name of layer and its current value
        """
        if self.sampling_strategy == SamplingStrategy.RANDOM:
            return self._sample_random_layer()
        elif self.sampling_strategy == SamplingStrategy.BOTTOM_UP:
            return self._sample_bottom_up()
        elif self.sampling_strategy == SamplingStrategy.TOP_DOWN:
            return self._sample_top_down()
        else:  # default is all layers at once
            return self._get_layer_value("all")

    def _sample_bottom_up(self) -> Tuple[str, torch.Tensor]:
        """
        Samples layer from bottom to top
        """
        self.last = self.last + 1
        self.last = self.last % (self.total_layers)
        layer_name = self.all_layers[self.last]
        layer_value = self._get_layer_value(layer_name)
        return layer_name, layer_value

    def _sample_top_down(self) -> Tuple[str, torch.Tensor]:
        """
        Samples layer from top to bottom
        """
        self.last = self.last - 1
        if self.last < 0:
            self.last = self.total_layers - 1
        layer_name = self.all_layers[self.last]
        layer_value = self._get_layer_value(layer_name)
        return layer_name, layer_value

    def _get_layer_value(self, layer_name: str) -> torch.Tensor:
        """
        Gets a layer value by its name
        """
        layer_value = None
        if layer_name == "all":
            for layer in self.all_layers:
                layer_v = self._get_layer_value_by_name(layer)
                # set layer dimensions for later unpacking
                self.layer_dimensions[layer] = layer_v.shape[0]
                if layer_value is None:
                    layer_value = layer_v
                else:
                    layer_value = torch.cat((layer_value, layer_v))
        else:
            layer_value = self._get_layer_value_by_name(layer_name)

        return layer_value

    def _get_layer_value_by_name(self, layer_name: str) -> torch.Tensor:
        """
        Gets layer value by its name
        """

        layer_parameters = list(self.layer_wise_parameters[layer_name])

        # packs all the parameter values
        value = None
        for param in layer_parameters:
            param_value, _ = self._get_param_value_type(param)
            if value is None:
                value = param_value.reshape((-1, 1))
            else:
                value = torch.cat((value, param_value.reshape((-1, 1))))
        return value

    def set_layer_value(self, layer_name: str, layer_value: torch.Tensor, value_type: str = ".data"):
        """
        Sets layer value

        Arguments:
            layer_name {str} -- layer name
            layer_value {torch.Tensor} -- layer value
        """
        if layer_name == "all":
            # unpack into separate values
            last = 0
            for layer_name, layer_dim in self.layer_dimensions.items():
                value = layer_value[last:last+layer_dim]
                last += layer_dim
                self._set_layer_value_by_name(layer_name, value, value_type)
        else:
            # just call the underlying method
            self._set_layer_value_by_name(layer_name, layer_value, value_type)

    def _set_layer_value_by_name(self, layer_name: str, layer_value: torch.Tensor, value_type: str):
        """
        Sets the value of individual parameters in the layer by properly unpacking and setting them one at a time
        """
        layer_parameters = list(self.layer_wise_parameters[layer_name])
        last = 0

        # unpack individual parameters and set its value
        for param in layer_parameters:
            param_shape, param_type = self.layer_wise_parameters[layer_name][param]
            param_size = reduce(lambda x, y: x*y, list(param_shape))
            param_value = layer_value[last:last+param_size].reshape(param_shape)
            last += param_size

            self._set_param_value(param+value_type, param_type, param_value)

    def set_gradients(self, layer_name: str, gradients: torch.Tensor):
        self._set_layer_value_by_name(layer_name, gradients, ".grad")

    def forward(self, *args):
        """
        Just forwards the request to underlying model
        """
        return self.net.forward(*args)
