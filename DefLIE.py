import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, c1, c2, act_layer=nn.ReLU):
        super(Conv, self).__init__()
        self.c = nn.Sequential(
            nn.Conv2d(c1,c2,3,1,1,bias=True),
            act_layer()
        )
    def forward(self, x):
        return self.c(x)

class Convlog(nn.Module):
    def __init__(self, c1, c2):
        super(Convlog, self).__init__()
        self.c = nn.Conv2d(c1,c2,3,1,1,bias=True)
    def forward(self, x):
        return torch.log1p(self.c(x))
        
class GsConv2d(nn.Module):
    #GaussianConv2d
    def __init__(self,in_channels,out_channels, scale=3, stride=1, dilation=1,groups=1,bias=True):  
        super(GsConv2d, self).__init__()  
        self.scale = scale     
        self.groups = groups
        self.stride = stride
        self.dilation = dilation
        self.padding = scale//2
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *(scale,scale))) 
        self.weight_sclce = nn.Parameter(torch.Tensor(scale))
        self.weight_sigma = nn.Parameter(torch.Tensor(scale))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-self.scale//2, self.scale//2)
        nn.init.constant_(self.weight_sclce, 1.)
        nn.init.constant_(self.weight_sigma, 1.)
        nn.init.constant_(self.bias, 0.) if self.bias is not None else None
        
    def forward(self, x):  
        gaussian_weight = self.weight_sclce*torch.exp(-self.weight**2 / (2 * self.weight_sigma**2))
        gc = F.conv2d(x, gaussian_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return torch.log1p(gc)

        
class HCEA(nn.Module):
    #Hourglass
    def __init__(self, dim, h_dim=3,droprate=0.2):
        super(HCEA, self).__init__()
        self.d = Convlog(dim,h_dim)
        self.u = Conv(h_dim,dim)
        self.drop = nn.Dropout(droprate)
        self.h_dim = h_dim
        
    def forward(self, x):
        a = self.d(x)
        p = self.u(a)
        p=p+x
        p = self.drop(p)
        return p
        
class DeftLIE(nn.Module):
    def __init__(self,dim=32, h_dim=3,droprate=0.2):
        super(DeftLIE, self).__init__()
        
        self.c1 = nn.Sequential(
            Conv(3,dim),
            HCEA(dim, h_dim,droprate),
            HCEA(dim, h_dim,droprate),
            HCEA(dim, h_dim,droprate),
            Convlog(dim,3) 
        )
        self.c2 = nn.Sequential(
            Conv(3,dim),
            HCEA(dim, h_dim,droprate),
            HCEA(dim, h_dim,droprate),
            Convlog(dim,3) 
        )
        self.gc1 = GsConv2d(3,3,scale=3)
        self.gc2 = GsConv2d(3,3,scale=3)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        i = self.c1(x)
        c1 = x+i*self.gc1(x)
        gamma = self.c2(c1)**2
        ex = x+c1+gamma*self.gc2(x)
        return ex, i, c1

        
    
def DeftLIE_sice():
    model = DeftLIE(dim=3, h_dim=3,droprate=0.)
    return model


    
    
