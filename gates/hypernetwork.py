
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
## load mnist dataset
import matplotlib.pyplot as plt
# from hypernn.torch import TorchLinearHyperNetwork
from typing import Any, Optional
from typing import Any, Dict, List, Optional, Tuple, Type, Union  # noqa
import tqdm
import numpy as np

from typing import Any, Dict, List, Optional, Tuple  # noqa
from functorch import vmap  # noqa
from typing import Any, Dict, List, Optional, Tuple  # noqa
import math
from functorch import make_functional, make_functional_with_buffers
import abc




class HyperNetwork(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def count_params(
        cls,
        target,
        target_input_shape: Optional[Any] = None,
    ):
        """
        Counts parameters of target nn.Module

        Args:
            target (Union[torch.nn.Module, flax.linen.Module]): _description_
            target_input_shape (Optional[Any], optional): _description_. Defaults to None.
        """

    @classmethod
    @abc.abstractmethod
    def from_target(cls, target, *args, **kwargs) -> HyperNetwork:
        """
        creates hypernetwork from target

        Args:
            cls (_type_): _description_
        """

    @abc.abstractmethod
    def generate_params(
        self, inp: Optional[Any] = None, *args, **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate a vector of parameters for target network

        Args:
            inp (Optional[Any], optional): input, may be useful when creating dynamic hypernetworks

        Returns:
            Any: vector of parameters for target network and a dictionary of extra info
        """

    @abc.abstractmethod
    def forward(
        self,
        *args,
        generated_params=None,
        has_aux: bool = True,
        **kwargs,
    ):
        """
        Computes a forward pass with generated parameters or with parameters that are passed in

        Args:
            inp (Any): input from system
            generated_params (Optional[Union[torch.tensor, jnp.array]], optional): Generated params. Defaults to None.
            has_aux (bool): flag to indicate whether to return auxiliary info
        Returns:
            returns output and generated params and auxiliary info if has_aux is provided
        """


def get_weight_chunk_dims(num_target_parameters: int, num_embeddings: int):
    weight_chunk_dim = math.ceil(num_target_parameters / num_embeddings)
    if weight_chunk_dim != 0:
        remainder = num_target_parameters % weight_chunk_dim
        if remainder > 0:
            diff = math.ceil(remainder / weight_chunk_dim)
            num_embeddings += diff
    return weight_chunk_dim


def count_params(module: nn.Module, input_shape=None, inputs=None):
    return sum([np.prod(p.size()) for p in module.parameters()])


class FunctionalParamVectorWrapper(nn.Module):
    """
    This wraps a module so that it takes params in the forward pass
    """

    def __init__(self, module: nn.Module):
        super(FunctionalParamVectorWrapper, self).__init__()
        self.custom_buffers = None
        param_dict = dict(module.named_parameters())
        self.target_weight_shapes = {k: param_dict[k].size() for k in param_dict}

        try:
            _functional, named_params = make_functional(module)
        except Exception:
            _functional, named_params, buffers = make_functional_with_buffers(module)
            self.custom_buffers = buffers
        self.named_params = [named_params]
        self.functional = [_functional]  # remove params from being counted

    def forward(self, param_vector: torch.Tensor, *args, **kwargs):
        params = []
        start = 0
        for p in self.named_params[0]:
            end = start + np.prod(p.size())
            params.append(param_vector[start:end].view(p.size()))
            start = end
        if self.custom_buffers is not None:
            return self.functional[0](params, self.custom_buffers, *args, **kwargs)
        return self.functional[0](params, *args, **kwargs)




def create_functional_target_network(target_network: nn.Module):
    func_model = FunctionalParamVectorWrapper(target_network)
    return func_model


class TorchHyperNetwork(nn.Module, HyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        num_target_parameters: Optional[int] = None,
    ):
        super().__init__()

        self.functional_target_network = create_functional_target_network(
            target_network
        )
        self.target_weight_shapes = self.functional_target_network.target_weight_shapes

        self.num_target_parameters = num_target_parameters
        if num_target_parameters is None:
            self.num_target_parameters = count_params(target_network)

        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    def assert_parameter_shapes(self, generated_params):
        assert generated_params.shape[-1] >= self.num_target_parameters

    def generate_params(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError("Generate params not implemented!")

    def target_forward(
        self,
        *args,
        generated_params: torch.Tensor,
        assert_parameter_shapes: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if assert_parameter_shapes:
            self.assert_parameter_shapes(generated_params)

        return self.functional_target_network(generated_params, *args, **kwargs)

    def forward(
        self,
        *args,
        generated_params: Optional[torch.Tensor] = None,
        has_aux: bool = False,
        assert_parameter_shapes: bool = True,
        generate_params_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """
        Main method for creating / using generated parameters and passing in input into the target network

        Args:
            generated_params (Optional[torch.Tensor], optional): Generated parameters of the target network. If not provided, the hypernetwork will generate the parameters. Defaults to None.
            has_aux (bool, optional): If True, return the auxiliary output from generate_params method. Defaults to False.
            assert_parameter_shapes (bool, optional): If True, raise an error if generated_params does not have shape (num_target_parameters,). Defaults to True.
            generate_params_kwargs (Dict[str, Any], optional): kwargs to be passed to generate_params method
            *args, *kwargs, arguments to be passed into the target network (also gets passed into generate_params)
        Returns:
            output (torch.Tensor) | (torch.Tensor, Dict[str, torch.Tensor]): returns output from target network and optionally auxiliary output.
        """
        aux_output = {}
        if generated_params is None:
            generated_params, aux_output = self.generate_params(
                **generate_params_kwargs
            )

        if has_aux:
            return (
                self.target_forward(
                    *args,
                    generated_params=generated_params,
                    assert_parameter_shapes=assert_parameter_shapes,
                    **kwargs,
                ),
                generated_params,
                aux_output,
            )
        return self.target_forward(
            *args,
            generated_params=generated_params,
            assert_parameter_shapes=assert_parameter_shapes,
            **kwargs,
        )

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[Any] = None,
    ):
        return count_params(target, target_input_shape, inputs=inputs)

    @classmethod
    def from_target(
        cls,
        target_network: nn.Module,
        target_input_shape: Optional[Any] = None,
        num_target_parameters: Optional[int] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ) -> TorchHyperNetwork:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(
                target_network, target_input_shape, inputs=inputs
            )
        return cls(
            target_network=target_network,
            num_target_parameters=num_target_parameters,
            *args,
            **kwargs,
        )

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))



class TorchLinearHyperNetwork(TorchHyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        num_target_parameters: Optional[int] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        weight_chunk_dim: Optional[int] = None,
        custom_embedding_module: Optional[nn.Module] = None,
        custom_weight_generator: Optional[nn.Module] = None,
    ):
        super().__init__(target_network, num_target_parameters)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.weight_chunk_dim = weight_chunk_dim
        if weight_chunk_dim is None:
            self.weight_chunk_dim = get_weight_chunk_dims(
                self.num_target_parameters, num_embeddings
            )

        self.custom_embedding_module = custom_embedding_module
        self.custom_weight_generator = custom_weight_generator
        self.setup()

    def setup(self):
        if self.custom_embedding_module is None:
            self.embedding_module = self.make_embedding_module()
        else:
            self.embedding_module = self.custom_embedding_module

        if self.custom_weight_generator is None:
            self.weight_generator = self.make_weight_generator()
        else:
            self.weight_generator = self.custom_weight_generator_module

    def make_embedding_module(self) -> nn.Module:
        return nn.Embedding(self.num_embeddings, self.embedding_dim)

    def make_weight_generator(self) -> nn.Module:
        return nn.Linear(self.embedding_dim, self.weight_chunk_dim)

    def generate_params(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embedding = self.embedding_module(
            torch.arange(self.num_embeddings, device=self.device)
        )
        generated_params = self.weight_generator(embedding).view(-1)
        return generated_params, {"embedding": embedding}



class GATEHyperNetwork(TorchLinearHyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        *args,
        **kwargs
    ):
        super().__init__(
                    target_network = target_network,
                    *args,
                    **kwargs
                )

    def make_weight_generator(self):
        return nn.Sequential(
            nn.Linear(self.embedding_dim, 32),
            nn.Tanh(),
            nn.Linear(32, self.weight_chunk_dim)
        )
