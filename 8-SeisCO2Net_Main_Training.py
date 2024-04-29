from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from numpy.matlib import repmat
from sklearn.model_selection import train_test_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(device); print(torch.cuda.get_device_name())

## Load ##
X_train_np = np.load('data/crop_DLdatasets/exF_comb_X_train.npy')
X_valid_np = np.load('data/crop_DLdatasets/exF_comb_X_valid.npy') 
# y_train_np = np.load('data/crop_DLdatasets/exF_comb_y_train.npy') #this is full dimension
# y_valid_np = np.load('data/crop_DLdatasets/exF_comb_y_valid.npy') #this is full dimension
y_train_np = np.load('data/crop_DLdatasets/exF_comb_y_train_enc.npy') #encoded
y_valid_np = np.load('data/crop_DLdatasets/exF_comb_y_valid_enc.npy') #encoded


### Reduce training size
_,X_train_np,_,y_train_np = train_test_split(X_train_np,y_train_np,test_size=0.480769,random_state=155)
_,X_valid_np,_,y_valid_np = train_test_split(X_valid_np,y_valid_np,test_size=0.40,random_state=155)

print(X_train_np.shape); print(y_train_np.shape)
print(X_valid_np.shape); print(y_valid_np.shape)

## NN
class AutoEncoder(nn.Module): 
    def __init__(self, neurons=1, kernsz = 4):
        super().__init__()
        #original data is #10,300,200 (nz,nx)
        #output is 15,10
        
        self.encoder = nn.Sequential(
            nn.Conv2d(10,neurons*4,stride=5,kernel_size=kernsz,padding=1), #60,40
            nn.Conv2d(neurons*4,neurons*4,stride=1,kernel_size=kernsz,padding='same'),
            nn.ReLU(),
            nn.Conv2d(neurons*4,neurons*8,stride=2,kernel_size=kernsz,padding=1), #30,20
            nn.Conv2d(neurons*8,neurons*8,stride=1,kernel_size=kernsz,padding='same'),
            nn.ReLU(),
            nn.Conv2d(neurons*8,neurons*16,stride=2,kernel_size=kernsz,padding=1), #15,10
            nn.Conv2d(neurons*16,neurons*16,stride=1,kernel_size=kernsz,padding='same'),
            nn.ReLU(),
            nn.Conv2d(neurons*16,10,stride=1,kernel_size=kernsz,padding='same'), #15,10
            nn.Tanh(),
            )    
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10,neurons*16,stride=2,kernel_size=kernsz,padding=1), #30,20
            nn.Conv2d(neurons*16,neurons*16,stride=1,kernel_size=kernsz,padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(neurons*16,neurons*8,stride=2,kernel_size=kernsz,padding=1), #60,40
            nn.Conv2d(neurons*8,neurons*8,stride=1,kernel_size=kernsz,padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(neurons*8,neurons*4,stride=5,kernel_size=kernsz,padding=0,output_padding=1), #300,200
            nn.Conv2d(neurons*4,neurons*4,stride=1,kernel_size=kernsz,padding='same'),
            nn.ReLU(),
            nn.Conv2d(neurons*4,10,stride=1,kernel_size=kernsz,padding='same'),
            )
        
    def forward(self,x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    
class InvNet(nn.Module): 
    # takes in 1600,13
    def __init__(self, neurons=1):
        super().__init__()
        
        self.up1 = nn.Sequential(
            nn.Conv2d(10,neurons*2,stride=(5,1),kernel_size=(3,3),padding=(1,2)), #320,13
            nn.Tanh(),)
        self.up2 = nn.Sequential(
            nn.Conv2d(neurons*2,neurons*4,stride=(5,1),kernel_size=(3,3),padding=(1,1)), #64,13
            nn.Tanh(),)
        self.up3 = nn.Sequential(
            nn.Conv2d(neurons*4,neurons*8,stride=(2,1),kernel_size=(3,3),padding=(1,1)), #32,13
            nn.Tanh(),)
        self.up4 = nn.Sequential(
            nn.Conv2d(neurons*8,neurons*16,stride=(2,2),kernel_size=(3,3),padding=(1,1)), #16,13
            nn.Tanh(),)
        self.up5 = nn.Sequential(
            nn.Conv2d(neurons*16,neurons*32,stride=(2,1),kernel_size=(3,3),padding=(1,1)), #8,13
            nn.Tanh(),)
        self.up6 = nn.Sequential(
            nn.Conv2d(neurons*32,neurons*64,stride=(2,2),kernel_size=(3,3),padding=(1,1)), #4,4
            nn.Tanh(),)
        self.up7 = nn.Sequential(
            nn.Conv2d(neurons*64,neurons*64,stride=(2,2),kernel_size=(3,3),padding=(1,1)), #2,2
            nn.Tanh(),)
        self.up8 = nn.Sequential(
            nn.Conv2d(neurons*64,neurons*64,stride=(2,2),kernel_size=(3,3),padding=(1,1)), #1,1
            nn.Tanh(),)

        self.down1 = nn.Sequential(
            nn.ConvTranspose2d(neurons*64,neurons*64,stride=(2,2),kernel_size=(4,4),padding=(1,1)), #2,2
            nn.Tanh(),)
        self.down2 = nn.Sequential( 
            nn.ConvTranspose2d(neurons*64*2,neurons*64,stride=(2,2),kernel_size=(4,4),padding=(1,1)), #4,4
            nn.Tanh(),)
        self.down3 = nn.Sequential( 
            nn.ConvTranspose2d(neurons*64*2,neurons*32,stride=(2,2),kernel_size=(4,4),padding=(1,1)), #8,8
            nn.Tanh(),)
        self.out = nn.Sequential( 
            nn.ConvTranspose2d(neurons*32*2,neurons*16,stride=(2,1),kernel_size=(3,3),padding=(1,0)), #15,10
            nn.Conv2d(neurons*16,10,stride=(1,1),kernel_size=(4,4),padding='same'), 
            nn.Tanh(),)

    def forward(self,x):
        #Inversion   
        u1 = self.up1(x)
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        u4 = self.up4(u3)
        u5 = self.up5(u4)
        u6 = self.up6(u5)
        u7 = self.up7(u6)
        u8 = self.up8(u7)
        d1 = self.down1(u8)
        m1 = torch.cat([u7,d1],axis=1)
        d2 = self.down2(m1)
        m2 = torch.cat([u6,d2],axis=1)
        d3 = self.down3(m2)
        m3 = torch.cat([u5,d3],axis=1)
        out = self.out(m3)
        
        return out


def weights_init(m):
    '''Ensure weights initialization same as tensorflow'''
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)
        
def to_np(input):
    return input.detach().cpu().numpy()

def load_batch(dataset, batch_size=1,random=True):
    '''
    dataset is (X_train,y_train)
    '''
    X_train = dataset[0]
    y_train = dataset[1]
    
    total_samples = min(len(X_train), len(y_train))
    n_batches = int(total_samples/ batch_size)
    
    if random==True:
        indices = np.random.choice(total_samples,size=total_samples,replace=False)
    else:
        indices = np.arange(0,total_samples)
        
    for i in range(0, total_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_x, batch_y = X_train[batch_indices], y_train[batch_indices]
        
        yield batch_x, batch_y

#torch format
X_train = torch.tensor(X_train_np).to(device).double()
X_valid = torch.tensor(X_valid_np).to(device).double()
y_train = torch.tensor(y_train_np).to(device).double()
y_valid = torch.tensor(y_valid_np).to(device).double()

X_train = torch.swapaxes(X_train,2,3) #N,10,1600,13
X_valid = torch.swapaxes(X_valid,2,3) #N,10,1600,13


print(X_train.shape)
print(X_valid.shape)

# Preparing training
dataset = (X_train,y_train)
print(dataset[0].shape)
print(dataset[1].shape)


BS = 40
total_samples = len(X_train)
#total number of batches
n_batches = int(total_samples/ BS); print(n_batches)
#print out progress after every x batches (e.g., after 10 batches)
print_batches = 4

def SgMetricLoss(pred, GT):
    refzeros = torch.zeros(pred.shape).to(device)
    mse_blank = loss_fn_mse(GT,refzeros)
    mse_pred = loss_fn_mse(pred,GT)
    SGMLoss = (mse_pred/mse_blank)
    return SGMLoss

lr = 0.001 #0.001 original
loss_fn_mse = nn.MSELoss()
num_epochs = 300

# AE_model = AutoEncoder(neurons=2,kernsz=4).to(device).double()
# AE_model.load_state_dict(torch.load('data/weights_logs/SeisCO2Net_AE.pth.pth'))

# ####Freezing decoder
# for param in AE_model.parameters():
#     param.requires_grad = False


good_counter = 0
run = 0
while good_counter < 20:
    
    model = InvNet(neurons=1).to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    record_NNLoss = []
    record_valLoss = []
    main_log=dict()
    
    
    print('''Start Training Loop! for RUN {}'''.format(run))
    
    for epoch in range(num_epochs):
        print('\n')
        print('''Run {} Epoch {} '''.format(run,epoch))
        
        #Rerun if bad initializations
        if epoch > 2:
            if record_NNLoss[-1] > 0.95:
                print('RESETING !!!')
                break_check = 1
                break
            
        break_check = 0
        
        batch_NNLoss = []
        for batch_i, (batch_x,batch_y) in enumerate(load_batch(dataset,batch_size=BS)):

            #zero the parameter gradients
            optimizer.zero_grad()
            #predict Sg from Seismic Gather
            pred1 = model(batch_x) #output is #BS, 10, 15, 10

            #compute loss
            Loss = loss_fn_mse(pred1,batch_y)
            #Backpropagate
            Loss.backward()
            #Apply optimizer
            optimizer.step()

            #Print out batch progress
            if batch_i % print_batches == 0:
                print('[%d/%d], Batch:[%d/%d]\t NN_Loss: %.9f \t '
                      % (epoch,num_epochs,batch_i,n_batches,Loss.item()))
            #Save batch losses
            batch_NNLoss.append(Loss.item())

        #Compute epoch losses
        epoch_NNLoss = np.mean(batch_NNLoss)

        #Print out Epoch Progress
        print('EPOCH %d Training -> NN_Loss: %.9f \t ' 
               %(epoch,epoch_NNLoss))

        ###############################
        ######### Validation ##########
        ###############################
        with torch.no_grad():

            #predict
            val_prediction1 = model(X_valid)
            #decoder
            #calculate loss
            val_Loss = SgMetricLoss(val_prediction1, y_valid)

            #Print validation progress
            print('EPOCH %d Validation -> NN_Loss: %.9f \t '
                   %(epoch, val_Loss.item()))


        #Save only the lowest loss score
        if epoch > 0:
            if val_Loss.item() < np.min(record_valLoss):
                print('Saving Lowest Loss!!')
                torch.save(model.state_dict(),'data/weights_logs/SeisCO2Net_Main/ex6_mm_run{}_LV.pth'.format(run))
            else:
                print('Nawh.... ')
        # torch.save(model.state_dict(),'data/weights_logs/SeisCO2Net_Main/ex6_mm_run{}_C.pth'.format(run))
        

        #save epoch progress
        record_NNLoss.append(epoch_NNLoss)
        record_valLoss.append(val_Loss.item())

        #save into dictionary
        main_log['NNLoss'] = record_NNLoss
        main_log['ValLoss'] = record_valLoss

        #save
        np.save('data/weights_logs/SeisCO2Net_Main/ex6_mm_run{}_logs.npy'.format(run),main_log)
        
    
    if break_check == 0:
        
        #update run
        run = run+1
        
        #save good counter
        good_counter = good_counter+1
        
        
        

    
