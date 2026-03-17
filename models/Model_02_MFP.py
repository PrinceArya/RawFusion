import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
from .alignhead import *
from .splitter_net import *
from fvcore.nn import FlopCountAnalysis, flop_count_table

Domain_transfer = lambda x, mu, exp: torch.pow(x, mu) / torch.tensor(2**exp, dtype=x.dtype, device=x.device)


class Merging_Net(nn.Module):
    def __init__(self, in_channel = 1):
        super(Merging_Net, self).__init__()

        nFeat = 16
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        #Alignment 
        self.align_head = MultiCrossAlign_head_atttrans_res1sepalign(in_c=4, dim_align=nFeat)
        self.att1 = MultiScaleGatedModule(nFeat)
        self.att2 = MultiScaleGatedModule(nFeat)
        self.att3 = MultiScaleGatedModule(nFeat)
        self.att4 = MultiScaleGatedModule(nFeat)
        self.att5 = MultiScaleGatedModule(nFeat)
        self.att6 = MultiScaleGatedModule(nFeat)
        self.att7 = MultiScaleGatedModule(nFeat)
        self.att8 = MultiScaleGatedModule(nFeat)
        #self.fuse = Fusion_net()
        self.fuse = DYNUnet(in_channels=nFeat, num_filters=64)

          
        
    def forward(self, x):

        x1, x2, x3, x4, x5, x6, x7, x8, x9 = torch.split(x, 1, dim=1)

        x1 = self.pixel_unshuffle(x1)
        x2 = self.pixel_unshuffle(x2)
        x3 = self.pixel_unshuffle(x3)
        x4 = self.pixel_unshuffle(x4)
        x5 = self.pixel_unshuffle(x5)
        x6 = self.pixel_unshuffle(x6)
        x7 = self.pixel_unshuffle(x7)
        x8 = self.pixel_unshuffle(x8)
        x9 = self.pixel_unshuffle(x9)


        f1_att, f2, f3_att, f4_att, f5_att, f6_att, f7_att, f8_att, f9_att = self.align_head(x2, x1, x3, x4, x5, x6, x7, x8, x9)
        f1_att = f2 + f1_att
        f3_att = f2 + f3_att
        f4_att = f2 + f4_att
        f5_att = f2 + f5_att
        f6_att = f2 + f6_att
        f7_att = f2 + f7_att
        f8_att = f2 + f8_att
        f9_att = f2 + f9_att

        f1_att = self.att1(f1_att, f2)
        f3_att = self.att2(f3_att, f2)
        f4_att = self.att3(f4_att, f2)
        f5_att = self.att4(f5_att, f2)
        f6_att = self.att5(f6_att, f2)
        f7_att = self.att6(f7_att, f2)
        f8_att = self.att7(f8_att, f2)
        f9_att = self.att8(f9_att, f2)

        out = self.fuse(f1_att, f2, f3_att, f4_att, f5_att, f6_att, f7_att, f8_att, f9_att)  #Fuse all exposure

        return out
        

        


        

if __name__ == '__main__':
    inp = torch.rand(1, 9, 768, 1536)
    model = Merging_Net()   
    checkpoint = torch.load("checkpoint_dir/model_best.pth.tar", map_location='cpu')
    # Handle DataParallel wrapper
    state_dict = checkpoint['state_dict']
    if hasattr(model, 'module') and not list(state_dict.keys())[0].startswith('module.'):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif not hasattr(model, 'module') and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)


    flops = FlopCountAnalysis(model, torch.ones(1, 9, 768, 1536))
    print(flop_count_table(flops))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total # of model parameters : {num_params / 1000 / 1000 :.3f}(M)")
    print(f"Total FLOPs of the model : {flops.total() / (1000**4) :.3f}(T)")

    out = model(inp)  
    print(out.shape) 