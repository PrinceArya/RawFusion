import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table



class Channel_attn(nn.Module):
    def __init__(self, num_channel):
        super(Channel_attn, self).__init__()
        self.channel = num_channel
        self.pool1 = nn.AdaptiveAvgPool2d(1) 

        self.fc = nn.Sequential(nn.Conv2d(num_channel, num_channel//4, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(num_channel//4, num_channel, kernel_size=1),
                                nn.Sigmoid()
                                )
    def forward(self, x):
        wt1 = self.pool1(x)
        wt1 = self.fc(wt1)
        return wt1 

class SimpleChannelAttention(nn.Module):
    def __init__(self, channels):
        super(SimpleChannelAttention, self).__init__()
        # Global Average Pooling is implicit in AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (N, C, H, W)
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=0, bias=True)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        # Calculate avg and max along the channel axis (dim 1)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
       
        concat = torch.cat([avg_pool, max_pool], dim=1)
       
        # Pad manually to match 'valid' convolution in TF with explicit pad
        net = self.pad(concat)
        attention = torch.sigmoid(self.conv(net))
       
        return x * attention

class ConvBlock(nn.Module):
    """
    Helper block for: ReflectionPad -> Conv2d -> LeakyReLU
    Matches the pattern used repeatedly in the TF encoder.
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super(ConvBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0)
        self.relu = nn.LeakyReLU(0.3) # TF default alpha is 0.3

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class MidBlock(nn.Module):
    def __init__(self, filters):
        super(MidBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU(0.3)
        self.chan_att = SimpleChannelAttention(filters)
       
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(0.3)
        self.spat_att = SpatialAttention()

    def forward(self, input_tensor):
        net = self.pad1(input_tensor)
        net = self.conv1(net)
        net = self.relu1(net)
        net = self.chan_att(net)
       
        net2 = net + input_tensor
       
        net = self.pad2(net2)
        net = self.conv2(net)
        net = self.relu2(net)
        net = self.spat_att(net)
       
        return net + net2

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # In TF: Conv2DTranspose(k=3, s=2, padding='same')
        # In PyTorch: we need output_padding to match 'same' behavior with stride 2
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.LeakyReLU(0.3)

    def forward(self, conv_tensor_1, conv_tensor_2, split_tensor):
        concatenated_tensor = torch.cat([conv_tensor_1, conv_tensor_2], dim=1)
        x = self.upconv(concatenated_tensor)
        x = self.relu(x)
        return x + split_tensor

class DYNUnet(nn.Module):
    def __init__(self, in_channels=16, num_filters=64):
        super(DYNUnet, self).__init__()
       
        # Initial Convolution
        self.chann_attn0 = Channel_attn(in_channels*3)
        self.conv_0 = nn.Conv2d(in_channels*3, in_channels, kernel_size=3, padding=1)

        self.chann_attn1 = Channel_attn(in_channels*3)
        self.conv_1 = nn.Conv2d(in_channels*3, in_channels, kernel_size=3, padding=1)

        self.chann_attn2 = Channel_attn(in_channels*3)
        self.conv_2 = nn.Conv2d(in_channels*3, in_channels, kernel_size=3, padding=1)

        self.conv_3 = nn.Conv2d(in_channels*3, in_channels*2, kernel_size=3, padding=1)

        self.act_ = nn.ReLU()

        self.initial_conv = nn.Conv2d(in_channels*2, num_filters, kernel_size=3, padding=1)
       
        # --- Encoder Layers ---
        # We use ModuleDict or ModuleList to store the tree nodes.
        # The indices here match the TF variable names (e.g., conv_tensor_1, conv_tensor_3...)
        self.encoder_convs = nn.ModuleList([None] * 32) # Placeholder for 0-31
       
        # Stage 1 (Index 1-2)
        # Input splits into 2, each gets a conv block
        self.encoder_convs[1] = ConvBlock(num_filters//2, num_filters)
        self.encoder_convs[2] = ConvBlock(num_filters//2, num_filters)
       
        # Stage 2 (Index 3-6)
        # Inputs from 1,2 split.
        for i in range(3, 7):
            self.encoder_convs[i] = ConvBlock(num_filters//2, num_filters)
           
        # Stage 3 (Index 8-15) - Note: 7 is skipped in TF logic
        for i in range(8, 16):
            self.encoder_convs[i] = ConvBlock(num_filters//2, num_filters)
           
        # Stage 4 (Index 16-31)
        for i in range(16, 32):
            self.encoder_convs[i] = ConvBlock(num_filters//2, num_filters)

        # --- Mid Blocks ---
        # 16 MidBlocks for the bottleneck tensors (16 to 31)
        self.mid_blocks = nn.ModuleList([None] * 32)
        for i in range(16, 32):
            self.mid_blocks[i] = MidBlock(num_filters)

        # --- Decoder Layers ---
        # Fusion tensors follow a reverse tree.
        # We need specific decoder blocks matching the concatenation size.
        # Concatenation size = num_filters + num_filters = 2 * num_filters
        self.decoders = nn.ModuleList([None] * 16) # Indices 1-15
       
        for i in range(1, 16):
            self.decoders[i] = DecoderBlock(in_channels=num_filters*2, out_channels=num_filters)

        # Final Reconstruction
        self.skip = nn.Conv2d(num_filters, num_filters//2, kernel_size=3, padding=1)
        self.up_ = nn.PixelShuffle(2)
        self.final_conv = nn.Conv2d(num_filters//2, 12, kernel_size=3, padding=1)

    def split_tensor(self, x):
        # Simulating TF split_using_slice
        # Slicing along channel dim (dim 1 in PyTorch)
        c = x.shape[1]
        half_c = c // 2
        return x[:, :half_c, :, :], x[:, half_c:, :, :]

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        
        low_exp = torch.cat((x1, x2, x3), 1)
        mid_exp = torch.cat((x4, x5, x6), 1)
        high_exp = torch.cat((x7, x8, x9), 1)

        attn_0 = self.chann_attn0(low_exp)
        low_exp = low_exp*attn_0
        low_exp = self.act_(self.conv_0(low_exp))

        attn_1 = self.chann_attn1(mid_exp)
        mid_exp = mid_exp*attn_1
        mid_exp = self.act_(self.conv_1(mid_exp))

        attn_2 = self.chann_attn2(high_exp)
        high_exp = high_exp*attn_2
        high_exp = self.act_(self.conv_2(high_exp))

        fused = self.act_(self.conv_3(torch.cat((low_exp, mid_exp, high_exp), 1)))

        x = self.initial_conv(fused)
       
        # --- Encoder ---
       
        # Stage 1
        s1, s2 = self.split_tensor(x)
        c1 = self.encoder_convs[1](s1)
        c2 = self.encoder_convs[2](s2)
       
        # Stage 2
        s3, s4 = self.split_tensor(c1)
        s5, s6 = self.split_tensor(c2)
        c3 = self.encoder_convs[3](s3)
        c4 = self.encoder_convs[4](s4)
        c5 = self.encoder_convs[5](s5)
        c6 = self.encoder_convs[6](s6)
       
        # Stage 3
        s8, s9 = self.split_tensor(c3)
        s10, s11 = self.split_tensor(c4)
        s12, s13 = self.split_tensor(c5)
        s14, s15 = self.split_tensor(c6)
       
        c8 = self.encoder_convs[8](s8)
        c9 = self.encoder_convs[9](s9)
        c10 = self.encoder_convs[10](s10)
        c11 = self.encoder_convs[11](s11)
        c12 = self.encoder_convs[12](s12)
        c13 = self.encoder_convs[13](s13)
        c14 = self.encoder_convs[14](s14)
        c15 = self.encoder_convs[15](s15)
       
        # Stage 4 (Bottleneck inputs)
        s16, s17 = self.split_tensor(c8)
        s18, s19 = self.split_tensor(c9)
        s20, s21 = self.split_tensor(c10)
        s22, s23 = self.split_tensor(c11)
        s24, s25 = self.split_tensor(c12)
        s26, s27 = self.split_tensor(c13)
        s28, s29 = self.split_tensor(c14)
        s30, s31 = self.split_tensor(c15)
       
        # Create dictionary to hold bottleneck tensors for easier loop access if needed
        # But explicit variable names match your request better.
        c16 = self.encoder_convs[16](s16)
        c17 = self.encoder_convs[17](s17)
        c18 = self.encoder_convs[18](s18)
        c19 = self.encoder_convs[19](s19)
        c20 = self.encoder_convs[20](s20)
        c21 = self.encoder_convs[21](s21)
        c22 = self.encoder_convs[22](s22)
        c23 = self.encoder_convs[23](s23)
        c24 = self.encoder_convs[24](s24)
        c25 = self.encoder_convs[25](s25)
        c26 = self.encoder_convs[26](s26)
        c27 = self.encoder_convs[27](s27)
        c28 = self.encoder_convs[28](s28)
        c29 = self.encoder_convs[29](s29)
        c30 = self.encoder_convs[30](s30)
        c31 = self.encoder_convs[31](s31)

        # --- MidBlocks ---
        c16 = self.mid_blocks[16](c16)
        c17 = self.mid_blocks[17](c17)
        c18 = self.mid_blocks[18](c18)
        c19 = self.mid_blocks[19](c19)
        c20 = self.mid_blocks[20](c20)
        c21 = self.mid_blocks[21](c21)
        c22 = self.mid_blocks[22](c22)
        c23 = self.mid_blocks[23](c23)
        c24 = self.mid_blocks[24](c24)
        c25 = self.mid_blocks[25](c25)
        c26 = self.mid_blocks[26](c26)
        c27 = self.mid_blocks[27](c27)
        c28 = self.mid_blocks[28](c28)
        c29 = self.mid_blocks[29](c29)
        c30 = self.mid_blocks[30](c30)
        c31 = self.mid_blocks[31](c31)

        # --- Decoder ---
       
        # 4th Stage Reconstruction
        f1 = self.decoders[1](c16, c17, c8)
        f2 = self.decoders[2](c18, c19, c9)
        f3 = self.decoders[3](c20, c21, c10)
        f4 = self.decoders[4](c22, c23, c11)
        f5 = self.decoders[5](c24, c25, c12)
        f6 = self.decoders[6](c26, c27, c13)
        f7 = self.decoders[7](c28, c29, c14)
        f8 = self.decoders[8](c30, c31, c15)
       
        # 3rd Stage Reconstruction
        f9  = self.decoders[9](f1, f2, c3)
        f10 = self.decoders[10](f3, f4, c4)
        f11 = self.decoders[11](f5, f6, c5)
        f12 = self.decoders[12](f7, f8, c6)
       
        # 2nd Stage Reconstruction
        f13 = self.decoders[13](f9, f10, c1)
        f14 = self.decoders[14](f11, f12, c2)
       
        # 1st Stage Reconstruction
        f15 = self.decoders[15](f13, f14, x)

        f16 = self.skip(f15) + fused
       
        # Final Output
        final = self.up_(self.final_conv(f16))
       
        # Residual Connection (Global)
        return nn.functional.sigmoid(final)
   