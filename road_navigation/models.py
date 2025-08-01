import torch 
from torchvision import transforms
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),# Converts the image to a tensor and scales pixel values to [0, 1]
])

def conv_block(input_channels,output_channels):
   block = nn.Sequential(
   nn.Conv2d(in_channels=input_channels,out_channels = output_channels,kernel_size=3,stride=1,padding=1),
   nn.ReLU(),
   nn.Conv2d(in_channels=output_channels,out_channels = output_channels,kernel_size=3,stride=1,padding=1),
   nn.ReLU()
   )
   return block

class Encoder_Block(nn.Module):
  def __init__(self,input_channels,output_channels):
    super().__init__()
    self.conv = conv_block(input_channels,output_channels)
    self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

  def forward(self,x):
    x = self.conv(x)
    p = self.maxpool(x)
    return x ,p
  
class Decoder_block(nn.Module):

  def __init__(self,input_channels,output_channels):
    super().__init__()
    self.conv_transpose = nn.ConvTranspose2d(input_channels,output_channels,kernel_size=2,stride=2,padding=0)
    self.conv = conv_block(input_channels,output_channels)

  def forward(self,x,skip_features):
    x = self.conv_transpose(x)
    x = torch.cat((x,skip_features),dim=1)
    x = self.conv(x)
    return x

class Encoder_block(nn.Module):

  def __init__(self,input_channels,output_channels):
    super().__init__()
    self.conv = conv_block(input_channels,output_channels)
    self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

  def forward(self,x):
    x = self.conv(x)
    p = self.maxpool(x)
    return x ,p
  
class UNET(nn.Module):
 def __init__(self):
    super().__init__()
    self.encoder1 = Encoder_block(3,64)
    self.encoder2 = Encoder_block(64,128)
    self.encoder3 = Encoder_block(128,256)
    self.encoder4 = Encoder_block(256,512)

    self.base_layer = conv_block(512,1024)

    self.decoder1 = Decoder_block(1024,512)
    self.decoder2 = Decoder_block(512,256)
    self.decoder3 = Decoder_block(256,128)
    self.decoder4 = Decoder_block(128,64)

    self.final_layer =nn.Sequential(nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1), nn.Sigmoid())

 def forward(self,X):

    x1, p1 = self.encoder1(X)

    x2, p2 = self.encoder2(p1)

    x3, p3 = self.encoder3(p2)

    x4, p4 = self.encoder4(p3)

    base = self.base_layer(p4)

    dec1 = self.decoder1(base,x4)

    dec2 = self.decoder2(dec1,x3)

    dec3 = self.decoder3(dec2,x2)

    dec4 = self.decoder4(dec3,x1)

    final_output = self.final_layer(dec4)
    return final_output
 