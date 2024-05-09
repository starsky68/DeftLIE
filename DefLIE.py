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
    def __init__(self,in_channels,out_channels, scale=3, stride=1, dilation=1,groups=1,bias=True,sigma=1):  
        super(GsConv2d, self).__init__()  
        self.scale = scale     
        self.groups = groups
        self.stride = stride
        self.dilation = dilation
        self.padding = scale//2
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *(scale,scale))) 
        self.sigma = sigma
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-self.scale//2, self.scale//2)
        nn.init.constant_(self.bias, 0.) if self.bias is not None else None
        
    def forward(self, x):  
        gaussian_weight = torch.exp(-self.weight**2 / (2 * self.sigma**2))
        gc = F.conv2d(x, gaussian_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return torch.log1p(gc)

        
class HCEA(nn.Module):
    #Hourglass
    def __init__(self, dim, h_dim=3,droprate=0.2):
        super(HCEA, self).__init__()
        self.d = Convlog(dim,h_dim)
        self.u = Conv(h_dim,dim)
        #self.drop = nn.Dropout(droprate)
        self.h_dim = h_dim
        
    def forward(self, x):
        a = self.d(x)
        p = self.u(a)
        #p = self.drop(p)
        return p

class FoE(nn.Module):
    #First order enhancement
    def __init__(self, dim, h_dim=3,droprate=0.2):
        super(FoE, self).__init__()
        self.c1 = Conv(3,dim)
        self.h1 = HCEA(dim, h_dim)
        self.h2 = HCEA(dim, h_dim)
        self.h3 = HCEA(dim, h_dim)
        self.c2 = Convlog(dim,3) 
    def forward(self, x):
        c1 = self.c1(x)
        h1 = self.h1(c1)
        h2 = self.h2(h1)
        h3 = self.h3(h2+h1)
        out = self.c2(h3+c1)
        return out

class SoE(nn.Module):
    #Second order enhancement
    def __init__(self, dim, h_dim=3,droprate=0.2):
        super(SoE, self).__init__()
        self.c1 = Conv(3,dim)
        self.h1 = HCEA(dim, h_dim)
        self.h2 = HCEA(dim, h_dim)
        self.c2 = Convlog(dim,3) 
    def forward(self, x):
        c1 = self.c1(x)
        h1 = self.h1(c1)
        h2 = self.h2(h1+c1)
        out = self.c2(h2)
        return out
        
class DeftLIE(nn.Module):
    def __init__(self,dim=32, h_dim=3,droprate=0.2):
        super(DeftLIE, self).__init__()
        
        self.c1 = FoE(dim)
        
        self.c2 = SoE(dim)
        
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


    
    
