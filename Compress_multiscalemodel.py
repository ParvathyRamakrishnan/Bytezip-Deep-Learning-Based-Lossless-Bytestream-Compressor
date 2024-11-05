import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
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
import shutil

from gdn import GDN
from quantized_tensor import NormalizedTensor

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
    #print("L",l)
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
######################
####### Compress######
######################

t1=datetime.now()
inputfile='/home/iitp/Desktop/Autoencoder/Trainfiles/Altas_2020Nov10_19_00_35.gdr'
#input_dir='/media/iitp/NRB_IIT/Files'

length =1504
height= 64
headersize=48

fp= open(inputfile, 'rb')
header= fp.read(headersize)
padded_data=[]      
while header:
    packetsize= np.frombuffer(header[24:28], dtype=np.uint32) 
    packet= fp.read(int(packetsize))
    padding_size = length - len(header+packet)
    padded_element = header+ packet + bytes([0] * padding_size)
    padded_element= np.frombuffer(padded_element, dtype=np.uint8)
    padded_data.append(padded_element)    
    header= fp.read(headersize)
fp.close()

print(len(padded_data))
model= model.CombinedModel().to(device)
#model.load_state_dict(torch.load('./checkpoints/Encoder_model_epoch_61_Trainloss_10.31418826029851_Compress_83.79235899780073.pth'))

compressed_dir = inputfile.replace("gdr", "compressed")
if os.path.exists(compressed_dir):
    shutil.rmtree(compressed_dir)
os.makedirs(compressed_dir, exist_ok=True)

batch_size=1
scale=2
batch_num = len(padded_data)//(batch_size*height) 
rem = np.array(padded_data[(batch_num*batch_size*height):])
padded_data = padded_data[0:(batch_num*batch_size*height)] 

torch.cuda.empty_cache()

print("Iterations", batch_num)
i=1
for start_idx in range(0, len(padded_data), batch_size*height):
    torch.cuda.empty_cache()
    end_idx = start_idx+(batch_size*height)
    train_data= np.array(padded_data[start_idx:end_idx])
    batch_x = train_data.reshape(batch_size,height,length)
    batch_x = batch_x[:, np.newaxis,:, :]    
    batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
    batch_x_norm = batch_x/255
    #print(batch_x, batch_x_norm)
    #dec_inp, 
    enc_inps,  pi_outs, mu_outs, sigma_outs= model(batch_x_norm)
    total_bytes=0
    for j in range(scale):
        torch.cuda.empty_cache()
        if j==0:
            l= 2**8
        else:
            l= 2**5
        sym = (enc_inps[j]  * (l-1)).to(torch.int16)  
        #print(sym)   
        cdf=get_cdf(pi_outs[j].unsqueeze(dim=1), mu_outs[j].unsqueeze(dim=1), sigma_outs[j].unsqueeze(dim=1),l)
        output_cdf=torch.clamp(cdf, 0.0, 1.0).squeeze(dim=1).cpu()
        #output_cdf1= dec_inp #output_cdf
        sym = sym.squeeze(dim=1).cpu()
        #print(j, sym, sym.shape) 
        byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
        percent= (len(byte_stream)/len(sym.flatten()))*100
        print("\t",j,len(sym.flatten()),len(byte_stream), percent)
        total_bytes+=len(byte_stream)
        #output_cdf1= dec_inp    
        with open(compressed_dir+'/'+str(i)+".scale"+str(j), 'wb') as fout:
                #print(compressed_dir+'/'+str(i)+".scale"+str(j))
                fout.write(byte_stream)
    torch.cuda.empty_cache()
    l= 2**5
    sym= enc_inps[j+1] * (l-1)        
    sym = sym.squeeze(dim=1).cpu().to(torch.int16)
    
    prior = torch.ones(l) / l
    cdf = torch.zeros(l+1, dtype= torch.float32) 
    cdf[1:] = torch.cumsum(prior, dim=0)
    output_cdf = cdf.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(sym.shape[0],sym.shape[1], sym.shape[2], 1)    
    byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True) 
    #print(j+1, sym,  len(byte_stream))
    percent= (len(byte_stream)/(len(sym.flatten())))*100
    print("\t",j+1,len(sym.flatten()),len(byte_stream), percent)    
    with open(compressed_dir+'/'+str(i)+".scale"+str(j+1), 'wb') as fout:
        #print(compressed_dir+'/'+str(i)+".scale"+str(j+1))
        fout.write(byte_stream)                         
    
    total_bytes+= len(byte_stream)
    print(i,total_bytes)
    i=i+1
    del enc_inps,  pi_outs, mu_outs, sigma_outs, output_cdf
    

if len(rem)>0:
    with open(compressed_dir+'/compressed.last', 'wb') as fout:
        pickle.dump(rem, fout)    



t2=datetime.now()
print("Total Compression time:",str(t2-t1))
#exit()       
  
########################
####### Decompress######
########################

#t1=datetime.now()
#compressed_dir ='/home/iitp/Desktop/Autoencoder/smallTrainfiles/Altas_10mb.compressed'

outputfile = compressed_dir.replace("compressed", "decompressed")
tempfile = compressed_dir.replace("compressed", "temp")
if os.path.exists(outputfile):
    os.remove(outputfile)
if os.path.exists(tempfile):
    os.remove(tempfile)
    
i=1
scale=2
width =1504
height=64
headersize=48

out= open(tempfile, 'wb')
while os.path.exists(compressed_dir+'/'+str(i)+".scale0"):
    torch.cuda.empty_cache()
    with open(compressed_dir+'/'+str(i)+".scale"+str(scale), 'rb') as fout:
        #print(compressed_dir+'/'+str(i)+".scale"+str(scale))
        byte_stream = fout.read()
    l=2**5
    prior = torch.ones(l) / l
    cdf = torch.zeros(l+1, dtype= torch.float32) 
    cdf[1:] = torch.cumsum(prior, dim=0)
    output_cdf = cdf.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(batch_size,height//(4**scale), width, 1) 
    sym = torchac.decode_float_cdf(output_cdf, byte_stream)
    del output_cdf,prior
    #print(scale, sym, len(byte_stream))
    sym = sym.unsqueeze(dim=1)
    features_to_fuse = None
    for j in range(scale-1, -1, -1):
        torch.cuda.empty_cache()
        sym= sym.cuda() #.to(torch.float32)
        N_sym = sym/(l-1)
        decoder= model.decoders[j]
        prob_clf = model.prob_clfs[j]
        dec_out= decoder(N_sym,features_to_fuse) # features_to_fuse)
        
        pi,mu,sigma = prob_clf(dec_out)
        features_to_fuse = dec_out
        
        if j==0:
            l=2**8            
        else:
            l=2**5
            
        cdf=get_cdf(pi.unsqueeze(dim=1), mu.unsqueeze(dim=1), sigma.unsqueeze(dim=1),l)       
        output_cdf=torch.clamp(cdf, 0.0, 1.0).squeeze(dim=1).cpu()
        #output_cdf2= N_sym #output_cdf
        with open(compressed_dir+'/'+str(i)+".scale"+str(j), 'rb')  as fout:
            #print(compressed_dir+'/'+str(i)+".scale"+str(j))
            byte_stream = fout.read()
        sym = torchac.decode_float_cdf(output_cdf, byte_stream)
        sym = sym.unsqueeze(dim=1)
        #print(j,output_cdf,output_cdf.shape , sym.shape, sym)    
    sym_=sym.detach().numpy().astype(np.uint8).flatten()
    sym_.tofile(out)
    print(i) #sym
    i=i+1 
out.close()
'''
print("\n", output_cdf1,output_cdf2 , output_cdf1.shape,output_cdf2.shape , output_cdf1.dtype,output_cdf2.dtype      )
   
print(torch.eq(output_cdf1,output_cdf2) )

diff_locations = torch.nonzero(output_cdf1.cuda() != output_cdf2.cuda())
print("Differences at locations:", diff_locations)
print("Values in output_cdf1:", output_cdf1.cuda()[diff_locations])
print("Values in output_cdf2:", output_cdf2.cuda()[diff_locations])
'''

if os.path.exists(compressed_dir+'/compressed.last'):
    with open(compressed_dir+'/compressed.last', 'rb') as fout:
        rem= pickle.load(fout)
    print(rem.shape)    
    with open(tempfile, 'ab') as out:
        out.write(rem) 



fp= open(tempfile, 'rb')
print(len(fp.read()))

print("temp")
fp= open(tempfile, 'rb')
packet= fp.read(width)
packetsize= width-headersize 
with open(outputfile, 'wb') as fout:   
    while packet:
        size= np.frombuffer(packet[24:28], dtype=np.uint32)[0] 
        print(size)
        if size<packetsize:
            print("less",size)
            packet= packet[:headersize+size]
        fout.write(packet)
        packet= fp.read(width)
fp.close()


are_equal= filecmp.cmp(outputfile,inputfile)  
if are_equal:
    print("The files are the same.")
else:
    print("The files are not the same.")
 
t3=datetime.now()
print("Total Compression time:",str(t2-t1))
print("Total Decompression time:",str(t3-t2))
exit()     
    
    
   
