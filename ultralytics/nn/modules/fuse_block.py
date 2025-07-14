import torch
from torch import nn
from typing import Optional, List, Union, Tuple
from .conv import Conv


def autopad(k):
    assert len(k) == 2
    assert all(isinstance(_k, int) for _k in k)
    return ((k[0]-1)//2, (k[1]-1)//2)


class ScaleMergeFuse(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int=1,
        kernel_size: List[Tuple[int, int]] = [(3, 3)],
        hidden_channels: Optional[int] = None,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.split_kernel = nn.ModuleList([
            Conv(in_channels, hidden_channels, k, p=autopad(k), g=groups)
            for k in kernel_size
        ])
        self.merge_kernel = Conv(hidden_channels * len(kernel_size), out_channels)
    
    def forward(self, x):
        xs = torch.cat(
            [m(x) for m in self.split_kernel],
            dim=1
        )
        return self.merge_kernel(xs)


class ScaleConvFuse(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]] = 3,
        seperate_pw_dw: bool = False,
        strip_conv: str = 'None',
        hidden_channels: Optional[int] = None,
    ):
        super().__init__()
        in_channels *= 2
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        assert all(isinstance(k, int) for k in kernel_size)

        if hidden_channels is None:
            hidden_channels = in_channels
        
        assert strip_conv in ['None', 'Serial', 'Parallel']
        if strip_conv in ['None', 'Parallel']:
            if strip_conv == 'None':
                kernel_size = [(k, k) for k in kernel_size]
            else:
                kernel_size = [(k, 1) for k in kernel_size] + [(1, k) for k in kernel_size]
            self.m = ScaleMergeFuse(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=(hidden_channels if seperate_pw_dw else 1),
                kernel_size=kernel_size,
                hidden_channels=hidden_channels
            )
        else:
            mk1 = ScaleMergeFuse(
                in_channels=in_channels,
                out_channels=hidden_channels,
                groups=(hidden_channels if seperate_pw_dw else 1),
                kernel_size=[(k, 1) for k in kernel_size],
                hidden_channels=hidden_channels
            )
            m1k = ScaleMergeFuse(
                in_channels=hidden_channels,
                out_channels=out_channels,
                groups=(hidden_channels if seperate_pw_dw else 1),
                kernel_size=[(1, k) for k in kernel_size],
                hidden_channels=hidden_channels
            )
            self.m = nn.Sequential(mk1, m1k)
    
    def forward(self, x):
        return self.m(torch.cat(x, dim=1))

