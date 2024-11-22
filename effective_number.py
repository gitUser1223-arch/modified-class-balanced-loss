# this code set here is used for EFL
import torch  

samples_per_cls = torch.tensor(torch.load('cocolt_category_sample_counts.pkl')).float()
beta = 0.99 #可调参数
effective_num = 1.0 + (-1.0 * beta) * torch.pow((beta + 1.0) / 2, samples_per_cls-1) #modified
#effective_num = 1.0 - torch.pow(beta, samples_per_cls) #origin version
effective_num = effective_num / (1.0 - beta) 


min1 = effective_num.min()
max1 = effective_num.max()
#20 为可调参数
#effective_num = effective_num * (samples_per_cls >= 20) + samples_per_cls * (samples_per_cls < 20) # new add code
weights = 1.0  / effective_num
class_weights =(weights - weights.min()) / (weights.max() - weights.min())

# Save tensors as .pth files
torch.save(class_weights, 'cocolt_class_weights_0.99.pth')
#torch.save(effective_num, 'effective_num_0.999.pth')
torch.save(samples_per_cls, 'cocolt_samples_per_cls.pth')

# Load tensors from .pth files
# class_weights = torch.load('class_weights.pth')
# effective_num = torch.load('effective_num.pth')
