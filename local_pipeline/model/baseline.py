import torch.nn as nn
import torch
import torch.nn.functional as F
class DHN(nn.Module):
    def __init__(self, args, first_stage, ue_method="none"):
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args
        self.ue_method = ue_method
        self.first_stage = first_stage
        self.layer1 = nn.Sequential(nn.Conv2d(6,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
                                    
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())        
        self.layer6 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.fc1 = nn.Linear(128*32*32,1024)
        if self.args.ue_method == "single" and self.first_stage:
            self.fc2 = nn.Linear(1024,16)
        else:
            self.fc2 = nn.Linear(1024,8)

    def forward(self, image1, image2, iters_lev0 = 6, iters_lev1=3, corr_level=2, corr_radius=4, early_stop=-1):
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        x = torch.cat((image1, image2), 1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(-1,128* 32* 32)
        out = self.fc1(out)
        out = self.fc2(out)
        if self.ue_method == "single" and self.first_stage:
            out_ue = out[:, 8:]
            out = out[:, :8]
            out = out.view(-1, 2, 2, 2)
            out_ue = out_ue.view(-1, 2, 2, 2)
            return [out], out, [out_ue]
        else:
            out = out.view(-1, 2, 2, 2)
            return [out], out