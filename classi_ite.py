import copy, numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.stats

#%%
def data_generation(sigma_x, thresh, size):
    z=np.random.normal(0, 1, size+1)
    x=np.random.normal(0, sigma_x, size)     
    zp=np.random.normal(0, np.sqrt(1-sigma_x*sigma_x), size+1)
    y=np.zeros(size+1)
    
    for i in range(size):
        if y[i]<thresh:
            y[i+1]=z[i+1]
        else:
            y[i+1]=x[i]+zp[i+1]  
            
    y_ts=y[0:size]
    y_ts=y_ts.reshape(-1,1)
  
    y_t=y[1:size+1]
    y_t=y_t.reshape(-1,1)
    x=x.reshape(-1,1)
    
    data = np.concatenate((x,y_t,y_ts),axis=1)
    return data

#%%
def data_gen_zbar(data, size_g, data_net):
    data_z = data[:,2].reshape(-1,1)  
    data_z_ten = torch.FloatTensor(data_z)
    data_zbar = data_net(data_z_ten)
    sample_dataxyzbar = np.concatenate((data[:,[0,1]],data_zbar.detach().numpy()),axis=1)
    Jacobian_weight = torch.zeros((data_net.fc1.weight.shape[0], size_g)) 
    for i in range(size_g):
        output = torch.zeros(size_g,1)
        output[i] = 1
         #   each column is dz_1/dw_1, \cdots, dz_1/dw_n
        Jacobian_weight[:,i:i+1] = torch.autograd.grad(data_zbar,data_net.fc1.weight,
                                                           grad_outputs = output, 
                                                           retain_graph = True)[0]
    
    return sample_dataxyzbar, Jacobian_weight

#%% 
def recons_data(rho_data, size, train_size):  

    total_size = size
    train_index = np.random.choice(range(total_size), size=train_size, replace=False)
    test_index =  np.delete(np.arange(total_size), train_index, 0)
    joint_train = rho_data[train_index][:]
    joint_test = rho_data[test_index][:]

    marg_data, joint_index, marginal_index= sample_batch(rho_data, size, 
                                                         batch_size=size, 
                                                         sample_mode='marginal')
    
    marg_train = marg_data[train_index][:]
    marg_test = marg_data[test_index][:]
    
    train_data = np.vstack((joint_train,marg_train))
 
    joint_label = np.ones(train_size)
    marg_label = np.zeros(train_size)  
    label = np.vstack((joint_label,marg_label)).flatten()
    return train_data, joint_test, marg_test, label, train_index, marginal_index
#%%
def sample_batch(data, input_size, batch_size, sample_mode='joint'):
    joint_index=0
    marginal_index2=0
    if input_size==2:
        if sample_mode == 'joint':
            joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            batch = np.concatenate((data[joint_index][:,0].reshape(-1,1),data[joint_index][:,-1].reshape(-1,1)),axis=1)
        else:
            marginal_index1 = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            marginal_index2 = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            batch = np.concatenate((data[marginal_index1][:,0].reshape(-1,1),data[marginal_index2][:,-1].reshape(-1,1)),axis=1)
    else:
        if sample_mode == 'joint':
            joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            batch =data[joint_index]
        else:
            marginal_index1 = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            marginal_index2 = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            batch = np.concatenate((data[marginal_index1][:,0].reshape(-1,1),data[marginal_index2][:,[1,2]]),axis=1)
    return batch, joint_index, marginal_index2
#%%
# define varitional auto encoder 
class VAE(nn.Module):
    def __init__(self,VAE_input_size=1, VAE_hidden_size=200):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(VAE_input_size, VAE_hidden_size)
        self.fc21 = nn.Linear(VAE_hidden_size, 1)
        self.fc22 = nn.Linear(VAE_hidden_size, 1)

        nn.init.normal_(self.fc1.weight,std=0.2)
        self.fc1.weight.requires_grad = True
        nn.init.normal_(self.fc21.weight,std=0.2)
        nn.init.constant_(self.fc21.bias, 0)
        nn.init.normal_(self.fc22.weight,std=0.2)
        nn.init.constant_(self.fc22.bias, 0)
            
    def encode(self, x):
        h1 = F.elu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
#       std is equal to exp(0.5*logvar)
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        zbar = self.reparametrize(mu, logvar)
        return zbar 
#%%
class Class_Net(nn.Module):
    def __init__(self, input_size, hidden_size, std):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.fc1.weight, std=std)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=std)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=std)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self,input):
        m = nn.Sigmoid()
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        logit = self.fc3(output)
        prob = m(logit)
        return logit, prob
#%% 
def train(rho_data, size, train_size, mine_net, optimizer, iteration, input_size, tau):
    criterion = nn.BCEWithLogitsLoss()
    diff_et = torch.tensor(0.0)
    data, test_p0, test_q0, label, train_index, marg_index = recons_data(rho_data, size, 
                                                                           train_size) 
    for i in range(iteration):   

        batch_size = int(len(data)/4)
        if input_size == 2:  
            test_p = torch.FloatTensor(test_p0[:,[0,2]])
            test_q = torch.FloatTensor(test_q0[:,[0,2]])
            
        else: 
            test_p = torch.FloatTensor(test_p0)
            test_q = torch.FloatTensor(test_q0)
        
        train_batch, index1, index2 = sample_batch(data, input_size, 
                                                   batch_size = batch_size, 
                                                   sample_mode = 'joint')
        label_batch = label[index1]
        train_batch = torch.autograd.Variable(torch.FloatTensor(train_batch), requires_grad=True)
        label_batch = torch.FloatTensor(label_batch)
        
        logit = mine_net(train_batch)[0]
        loss = criterion(logit.reshape(-1), label_batch)
        
        if i < iteration-1:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()       
            train_batch.grad.zero_()
            loss.backward()
            grads = train_batch.grad
        
        if i >= iteration-101:            
            prob_p = mine_net(test_p)[1]
            rn_est_p = prob_p/(1-prob_p)
            finp_p = torch.log(torch.abs(rn_est_p))
            
            prob_q = mine_net(test_q)[1]
            rn_est_q = prob_q/(1-prob_q)
            a = torch.abs(rn_est_q)
            clip = torch.max(torch.min(a,torch.exp(tau)), torch.exp(-tau))        
            diff_et = diff_et+torch.max(torch.mean(finp_p)-torch.log(torch.mean(clip)), torch.tensor(0.0))
            
    return (diff_et/100).detach().cpu().numpy(), grads, index1, train_index, marg_index

#%%
def mi(rho_data, size, train_size, model, optimizer, repo, tau, input_size):    
    mi, grad, index, train_index, marg_index = train(rho_data, size, train_size,
                                                     model, optimizer, repo, 
                                                     input_size, tau=tau)
    
    return mi, grad, index, train_index, marg_index
#%%
def ma(a, window_size=20):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]    

#%%   
rho = 0.9
repi = int(200)
repo = int(200)
rep = int(1000)
alpha = 0.001
tau = torch.tensor(0.9)
print("tau", tau)
quan=7

thresh_vec = np.linspace(-3,3,quan)
size = 4000
train_size = 3000
test_size = size-train_size

realization = 1
print("ite, realization", realization)

total_te = np.zeros(shape=(realization,quan))
total_ite = np.zeros(shape=(realization,quan))
total_ste = np.zeros(shape=(realization,quan))

for i in range(realization):
    condiMI=[]
    ground_truth=[]
    result_ITE=[]

    for thresh in thresh_vec:
        rho_data = data_generation(rho, thresh, size)    
       
        modelP = Class_Net(input_size = 3, hidden_size = 130, std = 0.08)
        modelQ = Class_Net(input_size = 2, hidden_size = 100, std = 0.02)
        
        optimizerP = torch.optim.Adam(modelP.parameters(), lr=1e-3)
        optimizerQ = torch.optim.Adam(modelQ.parameters(), lr=1e-3)
       # conditional mi  
        mi_p = mi(rho_data, size, train_size, modelP, optimizerP, rep, tau, input_size=3)[0]
        mi_q = mi(rho_data, size, train_size, modelQ, optimizerQ, rep, tau, input_size=2)[0]
       
        condi_mi = mi_p - mi_q
        condiMI.append(condi_mi*1.4427)
      # ground truth  
        p=scipy.stats.norm(0, 1).cdf(thresh)         
        ground_value=-(1-p)*0.5*np.log(1-rho*rho)*1.4427
        ground_truth.append(ground_value)
        print("Conditional TE", condiMI)        
        print("ground_truth", ground_truth)
        
      # ite
        vae_net=VAE()
        diff_ite=[]
        modelA = Class_Net(input_size=3, hidden_size=130, std=0.08)
        modelB = Class_Net(input_size=2, hidden_size=100, std=0.02)
        optimizerA = torch.optim.Adam(modelA.parameters(), lr=1e-3)
        optimizerB = torch.optim.Adam(modelB.parameters(), lr=1e-3)
        for j in range(repo):          
            vae_data, Jacobian_joint=data_gen_zbar(rho_data, size, vae_net)
    #jacobian matrix is equal to Jacobian_joint together with Jacobian_joint by mar_index
          
            miA, gradsA, indexA, t_indexA, m_indexA = mi(vae_data, size, train_size, 
                                                         modelA, optimizerA, repi,
                                                         tau,input_size=3)
            J_reorderA = torch.index_select(Jacobian_joint, 1, torch.LongTensor(m_indexA))
            J_Am1 = torch.index_select(J_reorderA, 1, torch.LongTensor(t_indexA))
            J_Aj1 = torch.index_select(Jacobian_joint, 1, torch.LongTensor(t_indexA))           
            Jacobian_jm = torch.cat ((J_Aj1, J_Am1), 1)
            Jacobian_A = torch.index_select(Jacobian_jm, 1, torch.LongTensor(indexA))
       
            miB, gradsB, indexB, t_indexB, m_indexB = mi(vae_data, size, train_size, 
                                                         modelB, optimizerB, repi, 
                                                         tau, input_size=2)        
            J_reorderB = torch.index_select(Jacobian_joint, 1, torch.LongTensor(m_indexB))
            J_Bm1 = torch.index_select(J_reorderB, 1, torch.LongTensor(t_indexB))
            J_Bj1 = torch.index_select(Jacobian_joint, 1, torch.LongTensor(t_indexB))           
            Jacobian_jmB = torch.cat ((J_Bj1, J_Bm1), 1)
            Jacobian_B = torch.index_select(Jacobian_jmB, 1, torch.LongTensor(indexB))
            
            grads_j_A=gradsA[:,-1].reshape(-1,1) 
            grads_j_B=gradsB[:,-1].reshape(-1,1) 
            #calculate gradient
                     #     calculate the gradient wrt the weights of network vae
            grads_A=torch.mm(torch.t(grads_j_A),torch.t(Jacobian_A))     
            grads_B=torch.mm(torch.t(grads_j_B),torch.t(Jacobian_B))
            
            diff_grads=grads_A-grads_B
            with torch.no_grad():
                vae_net.fc1.weight -=alpha*torch.t(diff_grads)       
            
            diff_ite.append(miA-miB)
        
        result_ITE.append(ma(diff_ite)[-1]*1.4427)     
        print("result_ITE", result_ITE)
       
    total_te[i,:] = condiMI
    total_ite[i,:] = result_ITE
    final_result_STE=[a_i - b_i for a_i, b_i in zip(condiMI, result_ITE)]
    total_ste[i,:]=final_result_STE
    
plt.figure(1)        
max_te=np.amax(total_te, axis=0) 
min_te=np.min(total_te, axis=0) 
total=[sum(x) for x in zip(max_te, min_te)]   
mid_te=[x / 2 for x in total]  


max_ite=np.amax(total_ite, axis=0) 
min_ite=np.min(total_ite, axis=0) 
totali=[sum(x) for x in zip(max_ite, min_ite)]   
mid_ite=[x / 2 for x in totali]  

max_ste=np.amax(total_ste, axis=0) 
min_ste=np.min(total_ste, axis=0) 
totals=[sum(x) for x in zip(max_ste, min_ste)]   
mid_ste=[x / 2 for x in totals] 
 
plt.plot(thresh_vec, mid_te, color='orange',alpha=.9, label='TE') 
plt.fill_between(thresh_vec, max_te,min_te, color='orange',alpha=.9) 

plt.plot(thresh_vec, mid_ite, color='cyan',alpha=.9, label='ITE') 
plt.fill_between(thresh_vec, max_ite, min_ite, color='cyan',alpha=.9)

plt.plot(thresh_vec, mid_ste, color='magenta',alpha=.9, label='STE') 
plt.fill_between(thresh_vec, max_ste, min_ste, color='magenta',alpha=.9)     
#plt.plot(thresh_vec, final_result_STE, marker='*', color='m',label='STE') 
plt.plot(thresh_vec,ground_truth, 'b--', label='Ground Truth of TE')  

plt.xlabel("threshold $\lambda$")
plt.legend() 
plt.savefig("comte_variance.pdf") 


plt.figure(1)        
#plt.plot(thresh_vec,ground_truth)
plt.plot(thresh_vec, result_ITE, 'r', label='ITE')   
plt.plot(thresh_vec, condiMI, marker='^', color='g',label='TE') 
plt.plot(thresh_vec, condiMI-result_ITE, marker='*', color='m',label='STE') 
plt.plot(thresh_vec,ground_truth, 'b--', label='Ground Truth')  

plt.xlim(-3,3)
plt.xlabel("threshold $\lambda$")
plt.legend() 
plt.savefig("comq.pdf") 
plt.show()
    
