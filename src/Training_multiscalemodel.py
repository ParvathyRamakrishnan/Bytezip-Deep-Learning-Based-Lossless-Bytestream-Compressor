import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.distributions as dist

import os
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import torchac
import pickle
import filecmp
import subprocess
import multiscalemodel as model
import glob
from gdn import GDN
from quantized_tensor import NormalizedTensor
from torch.utils.data import DataLoader

#export CUDA_LAUNCH_BLOCKING=1

from torch.optim.lr_scheduler import StepLR

torch.cuda.empty_cache()   

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##########################################
###########Logistic Mixtures##############
##########################################
       
        
def get_cdf( logit_probs, means, log_scales, L):
    """
    :param logit_probs: NCKHW
    :param means: Updated w.r.t. some x! NCKHW
    :param log_scales: NCKHW
    :return:
    """
    #logit_probs= logit_probs.unsqueeze(dim=1)
    #means= means.unsqueeze(dim=1)
    #log_scales= log_scales.unsqueeze(dim=1)
    #L = 256
    #Adapted bounds for our case.
    bin_width = 2 / (L-1)
    # Lp = L+1
    targets = torch.linspace(-1 - bin_width / 2,
                                  1  + bin_width / 2,
                                  L + 1, dtype=torch.float32, device=logit_probs.device)
                                  
    log_scales = torch.clamp(log_scales, min=-7)
    #print("targets",targets)
    # NCKHW1
    inv_stdv = torch.exp(-log_scales).unsqueeze(-1)
    #print("inv_stdv",inv_stdv[0][0][0][0][0])
    #print("means",means,means.max(), means.min() )
    # NCKHWL'
    centered_targets = (targets - means.unsqueeze(-1))
    #print("centered_targets",centered_targets[0][0][0][0][0])
    # NCKHWL'
    cdf_k = centered_targets.mul(inv_stdv).sigmoid()
    #print("cdf_k",cdf_k[0][0][0][0][0])
    # NCKHW1, pi_k
    logit_probs_softmax = F.softmax(logit_probs, dim=2).unsqueeze(-1)
    #print("logit_probs_softmax",logit_probs_softmax[0][0][0][0][0])
    # NCHWL'
    cdf = cdf_k.mul(logit_probs_softmax).sum(2)
    #print("cdf",cdf[0][0][0][0])
    return cdf
    
def compute_loss( x, pi, mu, sigma, L, scale=0):
    """
    :param x: labels, i.e., NCHW, float
    :param l: predicted distribution, i.e., NKpHW, see above
    :return: log-likelihood, as NHW if shared, NCHW if non_shared pis
    """
    # Extract ---
    #L = 256
    #Adapted bounds for our case.
    bin_width = 2 / (L-1)
    # Lp = L+1
        
    x_raw = x#.get()
    N, C, H, W = x_raw.shape
    logit_pis = pi  # NCKHW
    means = mu  # NCKHW
    log_scales = torch.clamp(sigma, min=-7)  # NCKHW, is >= -MIN_SIGMA
    x_raw = x_raw.reshape(N, C, 1, H, W)
    
    x_raw = NormalizedTensor(x_raw, L)
    x_raw = x_raw.get()
    
    
    bitcost = forward_raw(x_raw, log_scales, logit_pis, means, L)
    return bitcost


def forward_raw( x_raw, log_scales, logit_pis, means, L):

    # Extract ---
    #L = 256
    x_lower_bound = -1 + 0.001
    x_upper_bound =  1 - 0.001
        
    #Adapted bounds for our case.
    bin_width = 2 / (L-1)
    # Lp = L+1
    

    centered_x = x_raw - means  # NCKHW
    # Calc P = cdf_delta
    # all of the following is NCKHW
    inv_stdv = torch.exp(-log_scales)  # <= exp(7), is exp(-sigma), inverse std. deviation, i.e., sigma'
    plus_in = inv_stdv * (centered_x + bin_width / 2)  # sigma' * (x - mu + 0.5)
    # exp(log_scales) == sigma -> small sigma <=> high certainty
    cdf_plus = torch.sigmoid(plus_in)  # S(sigma' * (x - mu + 1/255))
    min_in = inv_stdv * (centered_x - bin_width / 2)  # sigma' * (x - mu - 1/255)
    cdf_min = torch.sigmoid(min_in)  # S(sigma' * (x - mu - 1/255)) == 1 / (1 + exp(sigma' * (x - mu - 1/255))
    # the following two follow from the definition of the logistic distribution
    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255
    # NCKHW, P^k(c)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases, essentially log_cdf_plus + log_one_minus_cdf_min
    mid_in = inv_stdv * centered_x  # sigma' * x
    # log probability in the center of the bin, to be used in extreme cases
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    # NOTE: the original code has another condition here:
    #   tf.where(cdf_delta > 1e-5,
    #            tf.log(tf.maximum(cdf_delta, 1e-12)),
    #            log_pdf_mid - np.log(127.5)
    #            )
    # which handles the extremly low porbability case.
    #
    # so, we have the following if, where I put in the X_UPPER_BOUND and X_LOWER_BOUND values for RGB
    # if x < 0.001:                         cond_C
    #       log_cdf_plus                    out_C
    # elif x > 254.999:                     cond_B
    #       log_one_minus_cdf_min           out_B
    # else:
    #       log(cdf_delta)                  out_A
    out_A = torch.log(torch.clamp(cdf_delta, min=1e-12))
    
    #print(print("cdf_delta",cdf_delta[0][0][0][0],cdf_delta.shape))
    
    out_A_pdf = log_pdf_mid + np.log(bin_width)
    # self.summarizer.register_scalars(
    #         'val',
    #         {f'dmll/{scale}/pdf_cdf': lambda: (out_A - out_A_pdf).mean()})
    cond_A = (cdf_delta > 1e-5).float()
    cond_B = (x_raw > x_upper_bound).float()
    cond_C = (x_raw < x_lower_bound).float()
    out_A = (cond_A * out_A) + \
            (1. - cond_A) * out_A_pdf
    out_B = (cond_B * log_one_minus_cdf_min) + \
            (1. - cond_B) * out_A
    log_probs = (cond_C * log_cdf_plus) + \
                (1. - cond_C) * out_B  # NCKHW, =log(P^k(c))
    # combine with pi, NCKHW, (-inf, 0]
    log_probs_weighted = log_probs.add(
            log_softmax(logit_pis, dim=2))  # (-inf, 0]
    #print("log_probs_weighted",log_probs_weighted)
    # TODO:
    # for some reason, this somehow can become negative in some elements???
    # final log(P), NCHW
    bitcost = -log_sum_exp(log_probs_weighted, dim=2)  # NCHW
    #print("bitcost",bitcost[0][0])
    return bitcost
        

# TODO: replace with pytorch internal in 1.0, there is a bug in 0.4.1
def log_softmax(logit_probs, dim):
    """ numerically stable log_softmax implementation that prevents overflow """
    m, _ = torch.max(logit_probs, dim=dim, keepdim=True)
    return logit_probs - m - torch.log(torch.sum(torch.exp(logit_probs - m), dim=dim, keepdim=True))

def log_sum_exp(log_probs, dim):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    m, _        = torch.max(log_probs, dim=dim)
    m_keep, _   = torch.max(log_probs, dim=dim, keepdim=True)
    # == m + torch.log(torch.sum(torch.exp(log_probs - m_keep), dim=dim))
    return log_probs.sub_(m_keep).exp_().sum(dim=dim).log_().add(m)       

         
######################################################
####### Train combined_model parameters ######
######################################################

t1=datetime.now()
input_dir='/home/iitp/Desktop/Autoencoder/Trainfiles'
#input_dir='/media/iitp/NRB_IIT/Files'

filecount= 0
length = 1504
height= 64
headersize= 48
scale= 2

padded_data=[]
extension = "gdr"
files_with_extension = glob.glob(os.path.join(input_dir, f"*.{extension}"))


for filename in files_with_extension: #os.listdir(input_dir):
    print(filename)
    filecount= filecount+1
      
    fp = open(filename, 'rb')       
    
    header= fp.read(headersize)
    
    while header:
        packetsize= np.frombuffer(header[24:28], dtype=np.uint32) 
        packet= fp.read(int(packetsize))
        #data.append(header+packet)
        padding_size = length - len(header+packet)
        padded_element = header+ packet + bytes([0] * padding_size)
        padded_element= np.frombuffer(padded_element, dtype=np.uint8)
        padded_data.append(padded_element)    
        header= fp.read(headersize)
    fp.close()
    print(len(padded_data))

print(len(padded_data))
model= model.CombinedModel().to(device)
#model.load_state_dict(torch.load('./checkpoints/Encoder_model_epoch_100_Trainloss_10.500291971059946_Compress_85.13393370882773.pth'))

pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print( "Total:", pytorch_total_params, "Trainable:", pytorch_total_train_params)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.9)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=3, verbose=True)
#lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
max_grad_norm = 1.0
nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
num_epochs = 500
batch_size = 1
torch.cuda.empty_cache()

batch_num = len(padded_data)//(batch_size*height) 
padded_data = np.array(padded_data[0:(batch_num*batch_size*height)] )
print(padded_data.shape)       
train_data = padded_data.reshape(padded_data.shape[0]//height,1, height, padded_data.shape[1])
print(train_data.shape)   
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    i=0
    total_train_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        torch.cuda.empty_cache()
        batch_x = torch.tensor(batch_data, dtype=torch.float32).to(device)
        batch_x_norm = batch_x/255
        enc_inps, pi_outs, mu_outs, sigma_outs= model(batch_x_norm)
        weight=0
        for j in range(scale):
            weight+= enc_inps[j].shape[2]
        Train_loss=0
        for j in range(scale):
            if j==0:
                l= 2**8
                nll= compute_loss(enc_inps[j], pi_outs[j], mu_outs[j], sigma_outs[j], l)
                print("\t",enc_inps[j].shape,np.prod(enc_inps[j].shape) )
                conversion = np.log(2.) * np.prod(enc_inps[j].shape) #144000
                costs_bpsp = nll.sum() / (conversion) 
                print("\t", j,"costs_bpsp:",costs_bpsp.item())
                Train_loss += costs_bpsp* enc_inps[j].shape[2]
            else:
                l= 2**5
                nll= compute_loss(enc_inps[j], pi_outs[j], mu_outs[j], sigma_outs[j], l)
                print("\t",enc_inps[j].shape,np.prod(enc_inps[j].shape) )
                conversion = np.log(2.) * np.prod(enc_inps[j].shape) #144000
                costs_bpsp = nll.sum() / (conversion) 
                print("\t", j,"costs_bpsp:",costs_bpsp.item())
                Train_loss += costs_bpsp* enc_inps[j].shape[2]
    
        #Train_loss += 8 * (enc_inps[j].shape[2]//4)
        #weight +=  (enc_inps[j].shape[2]//4)
        
        Train_loss = Train_loss/weight
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  
        optimizer.zero_grad()
        Train_loss.backward()
        optimizer.step()              
        i += 1
        total_train_loss += Train_loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{batch_num}, Loss: {Train_loss}" )
       
    torch.cuda.empty_cache()         
    mean_train_loss = total_train_loss / i
    print("mean_train_loss", mean_train_loss)
    mean_percent=0
    if epoch %1==0 :#and epoch!=0:
    
        totalpercent=0 
        for batch_idx, batch_data in enumerate(train_loader):
            percentstr=''
            total_bytes=0 
            torch.cuda.empty_cache()
            batch_x = torch.tensor(batch_data, dtype=torch.float32).to(device)
            batch_x_norm = batch_x/255
            print(batch_x_norm, batch_x)
            enc_inps,  pi_outs, mu_outs, sigma_outs= model(batch_x_norm)
            for j in range(scale):            
                if j==0:
                    l= 2**8
                    sym = (enc_inps[j]  * (l-1)).to(torch.int16)        
                    pi_outs[j] = pi_outs[j].unsqueeze(dim=1)
                    mu_outs[j] = mu_outs[j].unsqueeze(dim=1)
                    sigma_outs[j] = sigma_outs[j].unsqueeze(dim=1)
                    cdf=get_cdf(pi_outs[j], mu_outs[j], sigma_outs[j], l)
                else:
                    l= 2**5
                    sym = (enc_inps[j] * (l-1)).to(torch.int16)        
                    pi_outs[j] = pi_outs[j].unsqueeze(dim=1)
                    mu_outs[j] = mu_outs[j].unsqueeze(dim=1)
                    sigma_outs[j] = sigma_outs[j].unsqueeze(dim=1)
                    cdf=get_cdf(pi_outs[j], mu_outs[j], sigma_outs[j], l)
                
                print(j,sym)
                cdf=torch.clamp(cdf, 0.0, 1.0).squeeze(dim=1).cpu()                  
                sym = sym.squeeze(dim=1).cpu()
                byte_stream = torchac.encode_float_cdf(cdf, sym, check_input_bounds=True) 
                percent= (len(byte_stream)/(len(sym.flatten())))*100
                percentstr += str(round(percent))+"_"
                print("\t",j,len(sym.flatten()),len(byte_stream), percent,len(byte_stream)*8/(len(sym.flatten())) )
                total_bytes += len(byte_stream)
        
            torch.cuda.empty_cache()
            l= 2**5
            sym= enc_inps[j+1] * (l-1)
            sym = sym.squeeze(dim=1).cpu().to(torch.int16)
            prior = torch.ones(l) / l
            cdf = torch.zeros(l+1, dtype= torch.float32)
            cdf[1:] = torch.cumsum(prior, dim=0)
            output_cdf = cdf.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(sym.shape[0],sym.shape[1], sym.shape[2], 1)    
            byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True) 
            percent= (len(byte_stream)/(len(sym.flatten())))*100
            print("\t",j+1,len(sym.flatten()),len(byte_stream), percent)  
            total_bytes += len(byte_stream)
            #print("\t",j+1,len(enc_inps[j+1].flatten()))
            #total_bytes +=  len(enc_inps[j+1].flatten())
            percent= (total_bytes/(len(batch_x.flatten())))*100  
            print(f"Batch {batch_idx}/{batch_num}: ",total_bytes, percentstr, percent) 
            del sym, cdf,  enc_inps, pi_outs, mu_outs, sigma_outs
            torch.cuda.empty_cache()
            totalpercent+= percent
            if batch_idx==25:
                break
    mean_percent = totalpercent / 26 #batch_num
    print("mean_percent", totalpercent, mean_percent)
    torch.cuda.empty_cache()
    print("mean_train_loss", mean_train_loss, "total_train_loss", total_train_loss, "len(train_data)", len(train_data))
    checkpoint_path = os.path.join(checkpoint_dir, f"Encoder_model_epoch_{epoch + 1}_Trainloss_{mean_train_loss}_Compress_{str(mean_percent)}.pth")
    torch.save(model.state_dict(), checkpoint_path)                           
    lr_scheduler.step(mean_percent)                          
        
torch.save(model.state_dict(), 'Encoder_model.pth')

print("Model saved.")
t2=datetime.now()
print("Total Encoder Training time:",str(t2-t1))
 

