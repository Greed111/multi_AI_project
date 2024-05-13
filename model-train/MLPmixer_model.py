import torch
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, in_C=36, LAT=30, LON=150, expansion_factor = 3,act_layer = nn.GELU):
        super(Block,self).__init__()
        self.norm = nn.BatchNorm2d(in_C)
        self.C_MLP1 = nn.Linear(in_C, in_C*expansion_factor)
        self.C_MLP2 = nn.Linear(in_C * expansion_factor, in_C)
        self.LAT_MLP1 = nn.Linear(LAT, LAT*expansion_factor)
        self.LAT_MLP2 = nn.Linear(LAT * expansion_factor, LAT)
        self.LON_MLP1 = nn.Linear(LON, LON*expansion_factor)
        self.LON_MLP2 = nn.Linear(LON * expansion_factor, LON)
        self.act1 = act_layer()
        self.act2 = act_layer()
        self.act3 = act_layer()

    def forward(self,x):
        x = self.norm(x)
        x = self.C_MLP1(x.permute(0,2,3,1))
        x = self.act1(x)
        x = self.C_MLP2(x)
        x = self.LAT_MLP1(x.permute(0,3,2,1))
        x = self.act2(x)
        x = self.LAT_MLP2(x)
        x = self.LON_MLP1(x.permute(0,1,3,2))
        x = self.act3(x)
        x = self.LON_MLP2(x)
        return x

class Head(nn.Module):
    def __init__(self,in_C=36,out_C=3,actlayer=nn.GELU):
        super(Head,self).__init__()
        self.C_MLP1 = nn.Linear(in_C,in_C*2)
        self.C_MLP2 = nn.Linear(in_C*2, out_C)
        self.act = actlayer()
    def forward(self,x):
        x = self.C_MLP1(x)
        x = self.act(x)
        x = self.C_MLP2(x)
        return x

class MLPM(nn.Module):
    def __init__(self, lat=30, lon=120, in_C=36, out_C=3, depth= 2):
        super(MLPM,self).__init__()
        self.blocks = nn.Sequential(*[
            Block(in_C=in_C,LAT=lat,LON=lon)
            for i in range(depth)
        ])
        self.head = Head(in_C=in_C,out_C=out_C)

    def forward(self,x):
        x = self.blocks(x)
        x = self.head(x.permute(0,2,3,1)).permute(0,3,1,2)
        return x



if __name__ == '__main__':
    x = torch.rand([6, 36, 30, 120])
    model = MLPM(in_C=36,out_C=3)
    x = model(x)
    print(x.shape)
    print(model)