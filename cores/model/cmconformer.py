
# # check the root later
# import sys
# sys.path.append("/vol/research/VS-Work/PW00391/D-CMCM")

from typing import Optional, Tuple
from utils.utils import CustomException
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from torchvision.models import r


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask

class _ConvolutionModule(nn.Module):
    r"""
        Convolution Module
        Args:
            in_dim (int): input dimension
            num_channels (int): number of channels
            depthwise_kernel_size (int): depthwise kernel size
            dropout (float, optional): dropout probability (default: 0.0)
            bias (bool, optional): whether to use bias (default: False)
            use_bn (bool, optional): whether to use group batch normalization (default: False)
    """
    def __init__(
        self, in_dim:int, 
        num_channels:int, 
        depthwise_kernel_size:int, 
        dropout=0.0, 
        bias=False, 
        use_bn=False):
        super(_ConvolutionModule, self).__init__()
        try:
            if (depthwise_kernel_size - 1) % 2 != 0:
                raise CustomException(error_code=1,message="depthwise_kernel_size must be odd to achieve 'SAME' padding")
        except CustomException as ce:
            print("Error code: {}, {}".format(ce.args[0], ce.args[1]))
            sys.exit()

        self.ln = nn.LayerNorm(in_dim)
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim, 
                out_channels=num_channels * 2, 
                kernel_size=1, 
                padding=0, 
                bias=bias
            ), # Pointwise Convolution
            nn.GLU(dim=1), # GLU
            nn.Conv1d(
                in_channels=num_channels, 
                out_channels=num_channels, 
                kernel_size=depthwise_kernel_size, 
                padding=(depthwise_kernel_size - 1) // 2, 
                groups=num_channels, 
                bias=bias
            ), # Depthwise Convolution
            nn.GroupNorm(num_groups=1, num_channels=num_channels) if use_bn else nn.BatchNorm1d(num_features=num_channels), # Group or Batch Normalization
            nn.SiLU(), # Swish
            nn.Conv1d(
                in_channels=num_channels, 
                out_channels=in_dim, 
                kernel_size=1, 
                padding=0, 
                bias=bias
            ), # Pointwise Convolution
            nn.Dropout(p=dropout), # Dropout
        )

    def forward(self, input:torch.Tensor):
        r"""
            Forward propagation
            Args:
                input (torch.Tensor): input tensor with shape B, T, D (batch_size, in_dim, seq_len)
            Returns:
                output (torch.Tensor): output tensor with shape B, T, D (batch_size, in_dim, seq_len)
        """
        x = self.ln(input)
        x = x.transpose(1, 2)
        x = self.block(x)
        return x.transpose(1, 2)

class _FeedForwardModule(nn.Module):
    r"""
        Pointwise Feed Forward Module
        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension
            dropout (float, optional): dropout probability (default: 0.0)
    """
    def __init__(self, in_dim:int, hidden_dim:int, dropout=0.0):
        super(_FeedForwardModule, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim), # Layer Normalization
            nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=True), # Linear
            nn.SiLU(), # Swish
            nn.Dropout(p=dropout), # Dropout
            nn.Linear(in_features=hidden_dim, out_features=in_dim, bias=True), # Linear
            nn.Dropout(p=dropout), # Dropout
        )

    def forward(self, input:torch.Tensor):
        r"""
            Forward propagation
            Args:
                input (torch.Tensor): input tensor with shape * , D
            Returns:
                output (torch.Tensor): output tensor with shape * , D
        """
        return self.block(input)

    
class ConformerLayer(nn.Module):
    r"""
        Original Conformer layer
        Args:
            in_dim (int): input dimension
            ffn_dim (int): dimension of feed forward module
            num_heads (int): number of attention heads
            depthwise_kernel_size (int): depthwise kernel size
            dropout (float, optional): dropout probability (default: 0.0)
            use_gn (bool, optional): whether to use group normalization (default: False)
            conv_first (bool, optional): whether to use convolution ahead of the attention (default: False)
    """
    def __init__(
        self, 
        in_dim:int, 
        ffn_dim:int, 
        num_heads:int, 
        depthwise_kernel_size:int, 
        dropout:float = 0.0, 
        use_gn:bool = False, 
        conv_first:bool = False):
        super(ConformerLayer, self).__init__()

        self.ffn_1 = _FeedForwardModule(in_dim=in_dim, hidden_dim=ffn_dim, dropout=dropout) # feed forward module 1

        # self_attention
        self.self_atten_norm = nn.LayerNorm(in_dim)
        self.self_atten = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout)
        self.self_atten_dropout = nn.Dropout(p=dropout)

        self.conv_module = _ConvolutionModule(
            in_dim=in_dim, 
            num_channels=2 * in_dim, 
            depthwise_kernel_size=depthwise_kernel_size, 
            dropout=dropout,
            bias=True,
            use_bn=use_gn
        ) # convolution module

        self.ffn_2 = _FeedForwardModule(in_dim=in_dim, hidden_dim=ffn_dim, dropout=dropout) # feed forward module 2
        self.ln = nn.LayerNorm(in_dim)
        self.conv_first = conv_first

    def _apply_convolution(self, input:torch.Tensor):
        r"""
            Apply convolution module        
        """
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input:torch.Tensor, key_padding_mask:Optional[torch.Tensor]=None):
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        # Step 1: Feed Forward 1
        residual = input
        x = self.ffn_1(input)
        x = x * 0.5 + residual 

        # Step 2: Self Attention & Convolution
        if self.conv_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.self_atten_norm(x)
        x, _ = self.self_atten(query=x, key=x, value=x, need_weights=False, key_padding_mask=key_padding_mask)
        x = self.self_atten_dropout(x)
        x = x + residual

        if not self.conv_first:
            x = self._apply_convolution(x)

        # Step 3: Feed Forward 2
        residual = x
        x = self.ffn_2(x)
        x = x * 0.5 + residual

        # Step 4: Layer Normalization
        x = self.ln(x)
        return x

class Conformer(nn.Module):
    r"""
        Originial Conformer model based on pytorch official source code
        Args:
            in_dim (int): input dimension
            ffn_dim (int): dimension of feed forward module
            num_heads (int): number of attention heads
            num_layers (int): number of Conformer layers
            depthwise_kernel_size (int): depthwise kernel size
            dropout (float, optional): dropout probability (default: 0.0)
            use_gn (bool, optional): whether to use group normalization (default: False)
            conv_first (bool, optional): whether to use convolution ahead of the attention (default: False)
        

        Examples:
        conformer = Conformer(
             input_dim=80,
             num_heads=4,
             ffn_dim=128,
             num_layers=4,
             depthwise_conv_kernel_size=31,
        )
        lengths = torch.randint(1, 400, (10,))  # (batch,)
        input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        output = conformer(input, lengths)
    """
    def __init__(
        self, 
        in_dim:int, 
        ffn_dim:int, 
        num_heads:int, 
        num_layers:int, 
        depthwise_kernel_size:int, 
        dropout:float = 0.0, 
        use_gn:bool = False, 
        conv_first:bool = False
    ):
        super(Conformer, self).__init__()
        self.layers = nn.ModuleList([
            ConformerLayer(
                in_dim=in_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                depthwise_kernel_size=depthwise_kernel_size,
                dropout=dropout,
                use_gn=use_gn,
                conv_first=conv_first,
            )
            for _ in range(num_layers)
        ])


    def forward(self, input:torch.Tensor, lengths:Optional[torch.Tensor]=None)-> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """

        # Step 1: Calculate padding mask
        if lengths is not None:
            encoder_padding_mask = _lengths_to_padding_mask(lengths)
        else:
            encoder_padding_mask = None

        # Step 2: Forward propagation
        x = input.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
        x = x.transpose(0, 1)
        return x, lengths


class CMConformerLayer(nn.Module):
    r"""
        CrossModal-Conformer layer
        Args:
            in_dim (int): input dimension
            ffn_dim (int): dimension of feed forward module
            num_heads (int): number of attention heads
            depthwise_kernel_size (int): depthwise kernel size
            dropout (float, optional): dropout probability (default: 0.0)
            use_gn (bool, optional): whether to use group normalization (default: False)
            conv_first (bool, optional): whether to use convolution ahead of the attention (default: False)
    """
    def __init__(
        self, 
        in_dim:int, 
        ffn_dim:int, 
        num_heads:int, 
        depthwise_kernel_size:int, 
        dropout:float = 0.0, 
        use_gn:bool = False, 
        conv_first:bool = False
    ):
        super(CMConformerLayer, self).__init__()

        self.ffn_1 = _FeedForwardModule(in_dim=in_dim, hidden_dim=ffn_dim, dropout=dropout) # feed forward module 1

        # self_attention
        self.self_atten_norm = nn.LayerNorm(in_dim)
        self.self_atten = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout)
        self.self_atten_dropout = nn.Dropout(p=dropout)

        # cross_attention
        self.cross_atten_norm = nn.LayerNorm(in_dim)
        self.cross_atten = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout)
        self.cross_atten_dropout = nn.Dropout(p=dropout)

        self.conv_module = _ConvolutionModule(
            in_dim=in_dim, 
            num_channels=2 * in_dim, 
            depthwise_kernel_size=depthwise_kernel_size, 
            dropout=dropout,
            bias=True,
            use_bn=use_gn
        )   # convolution module

        self.ffn_2 = _FeedForwardModule(in_dim=in_dim, hidden_dim=ffn_dim, dropout=dropout) # feed forward module 2
        self.ln = nn.LayerNorm(in_dim)
        self.conv_first = conv_first


    def _apply_convolution(self, input:torch.Tensor):
        r"""
            Apply convolution module        
        """
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input:torch.Tensor, key_padding_mask:Optional[torch.Tensor]=None, second_input:torch.Tensor=None):
        r"""
        Args:
            input (Tuple [torch.Tensor, torch.Tensor]): input tensors [interal, external], each with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        
        # Step 1: Feed Forward 1
        residual = input
        x = self.ffn_1(input)
        x = x * 0.5 + residual
        if second_input is not None: # process the second input if it exists
            residual_y = second_input
            y = self.ffn_1(second_input) 
            y = residual_y + y * 0.5

        # Step 2: Attention & Convolution
        if self.conv_first:
            x = self._apply_convolution(x)


        residual = x
        # self attention
        x = self.self_atten_norm(x)
        x, _ = self.self_atten(query=x, key=x, value=x, need_weights=False, key_padding_mask=key_padding_mask)
        x = self.self_atten_dropout(x)

        # cross attention if second input exists
        if second_input is not None:
            x_ = residual
            x_ = self.cross_atten_norm(x_)
            y  = self.cross_atten_norm(y)
            x_, _ = self.cross_atten(query=x_, key=y, value=y, need_weights=False, key_padding_mask=key_padding_mask)
            x_ = self.cross_atten_dropout(x_)

            x = (x + x_) * 0.5 # fuse the self attention and cross attention
            
        x = x + residual

        if not self.conv_first:
            x = self._apply_convolution(x)

        # Step 4: Feed Forward 2
        residual = x
        x = self.ffn_2(x)
        x = x * 0.5 + residual

        # Step 5: Layer Normalization
        x = self.ln(x)
        return x

class CMConformer(nn.Module):
    r"""
        CrossModal-Conformer model
        Args:
            in_dim (int): input dimension
            ffn_dim (int): dimension of feed forward module
            num_heads (int): number of attention heads
            num_layers (int): number of Conformer layers
            depthwise_kernel_size (int): depthwise kernel size
            dropout (float, optional): dropout probability (default: 0.0)
            use_gn (bool, optional): whether to use group normalization (default: False)
            conv_first (bool, optional): whether to use convolution ahead of the attention (default: False)
    """
    def __init__(
        self, 
        in_dim:int, 
        ffn_dim:int, 
        num_heads:int, 
        num_layers:int, 
        depthwise_kernel_size:int, 
        dropout:float = 0.0, 
        use_gn:bool = False, 
        conv_first:bool = False,
    ):
        super(CMConformer, self).__init__()
        self.layers = nn.ModuleList([
            CMConformerLayer(
                in_dim=in_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                depthwise_kernel_size=depthwise_kernel_size,
                dropout=dropout,
                use_gn=use_gn,
                conv_first=conv_first,
            )
            for _ in range(num_layers)
        ])


    def forward(self, input:torch.Tensor, lengths:Optional[torch.Tensor]=None, second_input:torch.Tensor=None)-> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """

        # Step 1: Calculate padding mask
        if lengths is not None:
            encoder_padding_mask = _lengths_to_padding_mask(lengths)
        else:
            encoder_padding_mask = None

        # Step 2: Forward propagation
        x = input.transpose(0, 1)
        if second_input is not None:
            y = second_input.transpose(0, 1)
            for layer in self.layers:
                x = layer(x, encoder_padding_mask, y)
        else:
            for layer in self.layers:
                x = layer(x, encoder_padding_mask)
        x = x.transpose(0, 1)
        return x, lengths

# # Understanding the padding mask
# lengths = torch.randint(1, 10, (4,))
# input = torch.rand(4, int(lengths.max()), 3)
# mask = _lengths_to_padding_mask(lengths)
# print(mask)

# mask can be None if the input length is fixed (as what we do it (only consider seperate chunks and some can be unfull))