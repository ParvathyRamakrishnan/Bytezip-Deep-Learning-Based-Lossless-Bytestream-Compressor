import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gdn import GDN

#torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()   

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
l=2**5
###################
### FULL MODEL#####
###################

# Define the residual block with GDN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(5,1), padding=(2,0)),
        GDN(64)
        )
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):               
        residual = x
        for i in range(2):
            x1 = self.conv1(x)           
            x1 = self.relu(x1)
        out = x1.mul(0.1) + residual
        return out

# Define the tail block
class TailBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TailBlock, self).__init__()
        #self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(5,1), padding=(2,0))
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(5,5), padding=(2,2))
        self.relu = nn.ReLU(True)
        #self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(5,1), padding=(2,0))
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(5,5), padding=(2,2))
        #self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=(5,1), padding=(2,0))
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=(5,5), padding=(2,2))
        self.leaky_relu = nn.LeakyReLU()
        #self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=(5,1), padding=(2,0))
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=(5,5), padding=(2,2))
        
        
    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2 = self.relu(x1)
        x3 = self.conv2(x2)
        x4 = x3 + residual
        x5 = self.conv3(x4)
        x6 = self.leaky_relu(x5)
        x7 = self.conv4(x6)
        #x8=  nn.functional.softmax(x7, dim=1)
        return x7

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64,  kernel_size=(5,1), padding=(2,0))
        self.conv2 = nn.Conv2d(64, 64,  kernel_size=(5,1), padding=(2,0), stride=(2,1))
        self.conv3 = nn.Conv2d(64, 64,  kernel_size=(5,1), padding=(2,0))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(5,1), padding=(2,0), stride=(2,1))
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(8)]
        )
        
        self.conv5 = nn.Conv2d(64, 1, kernel_size=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)
    def forward(self, x):
        #print(x.shape)
        x1 = self.conv1(x)       
        #print(x1.shape)
        x2 = self.conv2(x1)
        #print(x2.shape)
        x3 = self.conv3(x2)
        #print(x3.shape)
        x4 = self.conv4(x3)
        #print(x4.shape)
        residual = x4        
        #print(residual.shape)
        x5 = self.residual_blocks(x4)                
        #print(x5.shape)
        x6 = x5 + residual
        #print(x6.shape)
        x7 = self.conv5(x6)                 
        #print(x7.shape)
        x8=self.relu(x7)
        return x8
        
class PixelShuffle(torch.nn.Module):
    """
    Upscales sample length, downscales channel 
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel = x.shape[1]
        short_length = x.shape[2]
        width = x.shape[3]

        long_channel = short_channel // self.upscale_factor
        long_length = self.upscale_factor * short_length

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel, short_length, width])
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        x = x.view(batch_size, long_channel, long_length, width)
        return x

class Atrous_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Atrous_conv, self).__init__()
        self.atrous_rates=(1,2,4,5,7)
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, (5,1), dilation=d, padding=((d * (5 - 1)) // 2, 0)) for d in self.atrous_rates
        ])
        
    def forward(self, x):
        #print("atrous",x.shape)
        for atrous_conv in self.conv:
            x = atrous_conv(x)
            #print("atrous",x.shape)
        return x
                
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5,1), padding=(2,0))
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(8)]
        )
        '''
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=(5,1), padding=(2,0), stride=(2,1), output_padding=(1,0))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5,1), padding=(2,0))
        '''
        self.conv2 = nn.Conv2d(64, 256, kernel_size=(5,1), padding=(2,0))
        self.upsampling = PixelShuffle(2)
        self.atrous_convs1 = Atrous_conv(128, 128)
        self.atrous_convs2 = Atrous_conv(64, 64)
        
        
    def forward(self, x, features_to_fuse):#features_to_fuse
        x1 = self.conv1(x) 
        if features_to_fuse is not None:
            x1 =  x1 + features_to_fuse   
        x2 = self.residual_blocks(x1)                
        x3 = x1 + x2
        '''
        x4 = self.conv2(x3)   
        x5 = self.conv3(x4) 
        x6 = self.conv2(x5) 
        x7 = self.conv3(x6)               
        return x7        
        '''
        #print(x3.shape)
        x4 = self.conv2(x3) 
        #print("x4",x4)
        x5 = self.upsampling(x4)
        #print(x5.shape)
        x6 = self.atrous_convs1(x5)
        #print(x6.shape)
        x7 = self.upsampling(x6) 
        #print(x7.shape)
        x8 = self.atrous_convs2(x7)  
        #print("x8",x8)
        return x8     

class ProbabilityEstimator(nn.Module):
    def __init__(self):
        super(ProbabilityEstimator, self).__init__()
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=(5,1), padding=(2,0))
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(5,5), padding=(2,2))
        self.relu = nn.ReLU(True)
        
        self.tail1 = TailBlock(64, 3)        
        self.tail2 = TailBlock(64, 3)        
        self.tail3 = TailBlock(64, 3)        
               
                
    def forward(self, x):
        x1 = self.conv1(x)     
        x2 = self.relu(x1)
                       
        pi = self.tail1(x2) #nn.functional.softmax(self.tail1(x9), dim=1)
        mu = self.tail2(x2)
        sigma = self.tail3(x2) 
        return pi,mu,sigma     

      
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.scales = list(range(2))
        self.encoders = nn.ModuleList([Encoder() for scale in self.scales])    
        self.decoders = nn.ModuleList([Decoder() for scale in self.scales])  
        self.prob_clfs = nn.ModuleList([ProbabilityEstimator() for scale in self.scales]  )
        
    def forward(self, x):
        enc_inps = []
        enc_inps.append(x)
        for fwd_scale in self.scales:
            encoder= self.encoders[fwd_scale]            
            x = encoder(x)            
            enc_inps.append(x)
            
        pi_outs, mu_outs, sigma_outs = [],[],[]
        for i, scale in reversed(list(enumerate(self.scales))):  # from coarse to fine
            #print("decoder scale:",scale)
            decoder= self.decoders[scale]
            prob_clf = self.prob_clfs[scale]
            
            if scale==max(self.scales):
                features_to_fuse = None
            else:
                features_to_fuse = dec_out #dec_outs[scale+1]   
            
            Q_x = (enc_inps[scale+1]  * (l-1)).to(torch.int16)
            Q_N_x = Q_x/(l-1)
            dec_inp= Q_N_x 
            dec_out= decoder(dec_inp,features_to_fuse)
            pi,mu,sigma = prob_clf(dec_out)
            pi_outs.insert(0,pi)
            mu_outs.insert(0,mu)
            sigma_outs.insert(0,sigma)
        return  enc_inps, pi_outs, mu_outs, sigma_outs
        
                
                
# Create an instance of the model
'''
model = CombinedModel().to(device)
sample_input = torch.randn(1, 1, 64, 1504) 
enc_inps,  pi_outs, mu_outs, sigma_outs= model(sample_input)
for i in range(3):
    print(enc_inps[i].shape, pi_outs[i].shape)


# Print the model architecture
print(model)

# Create a random input tensor with the same shape as your desired input (100x100)
#sample_input = torch.randn(1, 1, 100, 100)  # Batch size of 1, 1 channel, 100x100 input
sample_input = torch.randint(0, 256, (1, 1, 1000, 1504), dtype=torch.float32)
sample_input= sample_input.round()

# Forward pass to get the outputs
#pi, sigma, mu, lambda_ = model(sample_input)
pdf = model(sample_input)
print("pdf shape:", pdf.shape)

# Print the shapes of the output tensors
print("pi shape:", pi.shape)
print("sigma shape:", sigma.shape)
print("mu shape:", mu.shape)
print("lambda shape:", lambda_.shape)

weighted_pdf= model.calculate_weighted_pdf(pi, sigma, mu, lambda_ )
print(weighted_pdf)
predicted_character = weighted_pdf.argmax()
print(predicted_character)
'''
