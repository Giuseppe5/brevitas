from typing import Any, Optional, Tuple

from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.quant_tensor import GroupwiseIntQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseInt


class GroupwiseActQuantProxyFromInjector(ActQuantProxyFromInjector):

    def __init__(self, quant_layer, quant_injector):
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOGroupwiseInt

    @property
    def group_dim(self):
        return self.quant_injector.group_dim

    @property
    def group_size(self):
        return self.quant_injector.group_size

    def create_quant_tensor(
            self,
            qt_args: Tuple[Any],
            x: Optional[GroupwiseIntQuantTensor] = None) -> GroupwiseIntQuantTensor:
        if x is None:
            value, scale, zero_point, bit_width, = qt_args
            out = GroupwiseIntQuantTensor(
                value,
                scale,
                zero_point,
                self.group_size,
                self.group_dim,
                bit_width,
                self.is_signed,
                self.training)
        else:
            out = GroupwiseIntQuantTensor(
                qt_args,
                x.scale,
                x.zero_point,
                self.group_size,
                self.group_dim,
                x.bit_width,
                x.signed,
                self.training)
        return out
