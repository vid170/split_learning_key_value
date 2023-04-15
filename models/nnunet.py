from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from torch import nn
import torch
import torch.fx.experimental.optimization as optimization

print("model model : ", Generic_UNet)
print("ConvDropoutNormNonlin: " , ConvDropoutNormNonlin)
print("initial weights: ", InitWeights_He)

class Baseline(Generic_UNet):
    def __init__(self):
        pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
        conv_kernel_sizes = [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ]
        super(Baseline, self).__init__(
            input_channels=1,
            base_num_features=32,
            num_classes=3,
            num_pool=5,
            num_conv_per_stage=2,
            feat_map_mul_on_downscale=2,
            conv_op=nn.Conv3d,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0, "inplace": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
            deep_supervision=False,
            dropout_in_localization=False,
            final_nonlin=lambda x: x,
            weightInitializer=InitWeights_He(1e-2),
            pool_op_kernel_sizes=pool_op_kernel_sizes,
            conv_kernel_sizes=conv_kernel_sizes,
            upscale_logits=False,
            convolutional_pooling=True,
            convolutional_upsampling=True,
            max_num_features=None,
            basic_block=ConvDropoutNormNonlin,
        )


def get_model():
    global total_model
    total_model=Baseline()
    idx=1
    # print("momdel: ", total_model)
    model_children = list(total_model.children())
   
    return total_model

class front(nn.Module):
    def __init__(self, input_channels=1, pretrained=False):
        super(front, self).__init__()
        model = get_model()
        model_children = list(model.children())
        idx=0
        for child in model_children:
            idx+=1
            if(idx==2):
                child2=list(child.children())
                self.front_model=nn.Sequential(*child2[:1])

    def forward(self, x):
       
        x = self.front_model(x)
      
        return x

class center(nn.Module):
    def __init__(self, pretrained=False, skips=[]):
        super(center, self).__init__()
        current_model = get_model()
        self.skips=skips
        device="cuda"
        # self.tu=
        model_children = list(current_model.children())
        self.tu=nn.ModuleList(*model_children[3:4])
        self.conv_blocks_localization=nn.ModuleList(*model_children[:1])
        self.seg_outputs=nn.ModuleList(*model_children[4:])
        self.final_nonlin=lambda x: x
        self.tu.to(device)
        self.conv_blocks_localization.to(device)
        self.seg_outputs.to(device)
        # self.final_nonlin.to(device)

        idx=0
        for child in model_children:
            idx+=1
            if(idx==2):
                self.center_model=nn.ModuleList(list(child.children())[1:])
               

    def forward(self, x):
        # self.seg_outputs=[]
        for d in range(len(self.center_model)-1):
            x=self.center_model[d](x)
            self.skips.append(x)
        x=self.center_model[-1](x)

        for u in range(len(self.tu)-1):
            # print("self.tu shape: ",x.shape, " self.skips[] shape: ", (self.skips[-(u+1)]).shape)
            x=self.tu[u](x)
            x = torch.cat((x, self.skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            # self.seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

    
        return x

class back(nn.Module):
    def __init__(self, pretrained=False,  skips=[]):
        super(back, self).__init__()
        current_model = get_model()
        self.skips=skips
        # print("length of skips is : ", len(skips), "kkkkkk", len(self.skips))
        device="cuda"
        model_children = list(current_model.children())
        self.tu=nn.ModuleList(*model_children[3:4])
        self.conv_blocks_localization=nn.ModuleList(*model_children[:1])
        self.seg_outputs=nn.ModuleList(*model_children[4:])
        self.final_nonlin=lambda x: x
        self.tu.to(device)
        self.conv_blocks_localization.to(device)
        self.seg_outputs.to(device)
      
        

    def forward(self, x):
        # print("self.tu shape: ", len(self.tu), ";;;;", len(self.skips))
        x=self.tu[4](x)
        # for s in self.skips:
        #     print("****", s.shape)
        # p
        # print(x.shape, "llllll", self.skips[-5].shape)
        x = torch.cat((x, self.skips[-(4+ 1)]), dim=1)
        x = self.conv_blocks_localization[4](x)
        x = self.final_nonlin(self.seg_outputs[4](x))
        return x
        
if __name__ == '__main__':
    # mm=get_model()
    model = front(pretrained=True)
    print("FRONT MODEL")
    print("model1: ", model.model1)
    print(f'{model.front_model}\n\n')
    model = center(pretrained=True)
    print("CENTER MODEL")
  
    print(f'{model.center_model}\n\n')
    model = back(pretrained=True)
    print("BACK MODEL")
    print(f'{model.back_model}')
    model=get_model()
    
