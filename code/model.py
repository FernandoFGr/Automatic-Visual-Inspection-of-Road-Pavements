import torch.nn as nn
import torch
from torch import autograd

class ConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class SpatialPyramidPoolingBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SpatialPyramidPoolingBlock, self).__init__()
        self.maxpooling2 = nn.MaxPool2d(2)
        self.maxpooling4 = nn.MaxPool2d(4)
        self.maxpooling8 = nn.MaxPool2d(8)
        self.maxpooling16 = nn.MaxPool2d(16)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.compact = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, input):
        maxPool_2 = self.maxpooling2(input)
        maxPool_4 = self.maxpooling4(input)
        maxPool_8 = self.maxpooling8(input)
        maxPool_16 = self.maxpooling16(input)

        maxPool_4_upsampled = self.upsample2(maxPool_4)
        maxPool_8_upsampled = self.upsample4(maxPool_8)
        maxPool_16_upsampled = self.upsample8(maxPool_16)
        poolCat = torch.cat([maxPool_2, maxPool_4_upsampled, maxPool_8_upsampled, maxPool_16_upsampled], dim=1)
        out = self.compact(poolCat)
        return out

class Unet_SpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch):
        super(Unet_SpatialPyramidPooling, self).__init__()
        self.Convolution1 = ConvolutionBlock(in_ch, 64)  
        self.maxpooling1 = nn.MaxPool2d(2)
        self.Convolution2 = ConvolutionBlock(64, 128)
        self.maxpooling2 = nn.MaxPool2d(2)
        self.Convolution3 = ConvolutionBlock(128, 256)
        self.maxpooling3 = nn.MaxPool2d(2)
        self.Convolution4 = ConvolutionBlock(256, 512)
        self.maxpooling4 = nn.MaxPool2d(2)
        self.Convolution5 = ConvolutionBlock(512, 1024)
        
        self.SpatialPyramidPoolingBlock = SpatialPyramidPoolingBlock(1024,1024) #test out 2048
        self.Convolution_bottom = ConvolutionBlock(1024, 2048)
        self.maxpooling_bottom = nn.ConvTranspose2d(2048, 1024, 2, stride=2)

        self.maxpooling6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.Convolution6 = ConvolutionBlock(1536, 512)
        self.maxpooling7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.Convolution7 = ConvolutionBlock(512, 256)
        self.maxpooling8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.Convolution8 = ConvolutionBlock(256, 128)
        self.maxpooling9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.Convolution9 = ConvolutionBlock(128, 64)
        
    def forward(self, x):
        conv1 = self.Convolution1(x)
        pool1 = self.maxpooling1(conv1)
        conv2 = self.Convolution2(pool1)
        pool2 = self.maxpooling2(conv2)
        conv3 = self.Convolution3(pool2)
        pool3 = self.maxpooling3(conv3)
        conv4 = self.Convolution4(pool3)
        pool4 = self.maxpooling4(conv4)
        conv5 = self.Convolution5(pool4)

        compactPooling = self.SpatialPyramidPoolingBlock(conv5)
        compactPooling_conv = self.Convolution_bottom(compactPooling)
        compactPooling_conv_upsampled = self.maxpooling_bottom(compactPooling_conv)
        conv5_and_pollings = torch.cat([conv5, compactPooling_conv_upsampled], dim=1)

        up_6 = self.maxpooling6(conv5_and_pollings)
        merge6 = torch.cat([up_6, conv4], dim=1)
        conv6 = self.Convolution6(merge6)
        up_7 = self.maxpooling7(conv6)
        merge7 = torch.cat([up_7, conv3], dim=1)
        conv7 = self.Convolution7(merge7)
        up_8 = self.maxpooling8(conv7)
        merge8 = torch.cat([up_8, conv2], dim=1)
        conv8 = self.Convolution8(merge8)
        up_9 = self.maxpooling9(conv8)
        merge9 = torch.cat([up_9, conv1], dim=1)
        conv9 = self.Convolution9(merge9)
        return conv9

class Unet(nn.Module):
    def __init__(self, in_ch):
        super(Unet, self).__init__()
        self.Convolution1 = ConvolutionBlock(in_ch, 64)  
        self.maxpooling1 = nn.MaxPool2d(2)
        self.Convolution2 = ConvolutionBlock(64, 128)
        self.maxpooling2 = nn.MaxPool2d(2)
        self.Convolution3 = ConvolutionBlock(128, 256)
        self.maxpooling3 = nn.MaxPool2d(2)
        self.Convolution4 = ConvolutionBlock(256, 512)
        self.maxpooling4 = nn.MaxPool2d(2)
        self.Convolution5 = ConvolutionBlock(512, 1024)

        self.maxpooling6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.Convolution6 = ConvolutionBlock(1024, 512)
        self.maxpooling7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.Convolution7 = ConvolutionBlock(512, 256)
        self.maxpooling8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.Convolution8 = ConvolutionBlock(256, 128)
        self.maxpooling9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.Convolution9 = ConvolutionBlock(128, 64)
        
    def forward(self, x):
        conv1 = self.Convolution1(x)
        pool1 = self.maxpooling1(conv1)
        conv2 = self.Convolution2(pool1)
        pool2 = self.maxpooling2(conv2)
        conv3 = self.Convolution3(pool2)
        pool3 = self.maxpooling3(conv3)
        conv4 = self.Convolution4(pool3)
        pool4 = self.maxpooling4(conv4)
        conv5 = self.Convolution5(pool4)
        up_6 = self.maxpooling6(conv5)
        merge6 = torch.cat([up_6, conv4], dim=1)
        conv6 = self.Convolution6(merge6)
        up_7 = self.maxpooling7(conv6)
        merge7 = torch.cat([up_7, conv3], dim=1)
        conv7 = self.Convolution7(merge7)
        up_8 = self.maxpooling8(conv7)
        merge8 = torch.cat([up_8, conv2], dim=1)
        conv8 = self.Convolution8(merge8)
        up_9 = self.maxpooling9(conv8)
        merge9 = torch.cat([up_9, conv1], dim=1)
        conv9 = self.Convolution9(merge9)
        return conv9

class classifier(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(classifier, self).__init__()
        self.Convolution10 = nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, x):
        conv10 = self.Convolution10(x)
        out = nn.Softmax2d()(conv10)
        return out