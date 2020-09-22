#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd /home/dinghj/dinghj/MathModel


# In[3]:


import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pywt

import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import xgboost as xgb

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sns.set_style('darkgrid')


# ## Preprocessing

# In[26]:


with open('./whole_data.json', 'r') as f:
    data_dict = json.load(f)

list_2, list_3, list_4, list_5, list_6 = data_dict['2'], data_dict['3'], data_dict['4'], data_dict['5'], data_dict['6']
x_2, x_3, x_4, x_5, x_6 = np.array(list_2), np.array(list_3), np.array(list_4), np.array(list_5), np.array(list_6)
y_2, y_3, y_4, y_5, y_6 = np.zeros(len(x_2)), np.ones(len(x_3)), np.ones(len(x_4))*2, np.ones(len(x_5))*3, np.ones(len(x_6))*4

x = np.concatenate((x_2, x_3, x_4, x_5, x_6), axis=0)
y = np.concatenate((y_2, y_3, y_4, y_5, y_6), axis=0)


def wavelet_transform(origin_data):
    coeffs = pywt.wavedec(origin_data, 'db4', level=7)
    ya7 = pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0, 0, 0, 0]).tolist(), 'db4')
    yd7 = pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0, 0, 0, 0]).tolist(), 'db4')
    yd6 = pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0, 0, 0, 0]).tolist(), 'db4')
    yd5 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0, 0, 0, 0]).tolist(), 'db4')
    yd4 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 1, 0, 0, 0]).tolist(), 'db4')
    yd3 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 0, 1, 0, 0]).tolist(), 'db4')
    yd2 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 0, 0, 1, 0]).tolist(), 'db4')
    yd1 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 0, 0, 0, 1]).tolist(), 'db4')

    return ya7

alpha, beta, theta, delta = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

p_alpha = wavelet_transform(alpha)
p_beta = wavelet_transform(beta)
p_theta = wavelet_transform(theta)
p_delta = wavelet_transform(delta)


# In[27]:


plt.figure(figsize=(12,9))
plt.subplot(311)
plt.plot(range(len(alpha)), alpha)
plt.title('original signal')

plt.subplot(312)
plt.plot(range(len(p_alpha)), p_alpha)
plt.title('approximated component')

plt.tight_layout()
# plt.savefig('./imgs/wave.png', dpi=200)
plt.show()


# In[12]:


x[:, 0], x[:, 1], x[:, 2], x[:, 3] = p_alpha, p_beta, p_theta, p_delta

# shuffle dataset with same order
state = np.random.get_state()
np.random.shuffle(x)
np.random.set_state(state)
np.random.shuffle(y)


# # Hyper Params

# In[13]:


# hyper paramaters
max_epochs = 100
batch_size = 32
input_dim = 4
hidden_dim = 128
output_dim = 5
dropout_rate = 0.5
learning_rate = 0.0003
weight_decay = 0.0
grad_clip = 5.0
seed = 1

# self learning
iters = 5
threshold = 0.92


# In[14]:


def standardization(data_x):
    new_x = data_x.copy()
    new_x[:, 0] = (data_x[:, 0] - np.mean(data_x[:, 0], axis=0)) / (np.std(data_x[:, 0], axis=0) + 1e-7)
    new_x[:, 1] = (data_x[:, 1] - np.mean(data_x[:, 1], axis=0)) / (np.std(data_x[:, 1], axis=0) + 1e-7)
    new_x[:, 2] = (data_x[:, 2] - np.mean(data_x[:, 2], axis=0)) / (np.std(data_x[:, 2], axis=0) + 1e-7)
    new_x[:, 3] = (data_x[:, 3] - np.mean(data_x[:, 3], axis=0)) / (np.std(data_x[:, 3], axis=0) + 1e-7)
    return new_x

x_standarded = standardization(x)

# split dataset
train_x, test_x, train_y, test_y = train_test_split(x_standarded, y, test_size=0.7, random_state=seed)
# train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.1, random_state=seed)

print(f'train_x.shape: {train_x.shape}')
# print(f'dev_x.shape: {dev_x.shape}')
print(f'test_x.shape: {test_x.shape}')
print(f'test_y.shape: {test_y.shape}')


# In[24]:


plt.figure(figsize=(10, 12))

plt.subplot(311)
plt.plot(x[:200, 0])
plt.title('origin alpha')

plt.subplot(312)
plt.plot(x_standarded[:200, 0])
plt.title('alpha after standardization')

# plt.savefig('./imgs/standard.png', dpi=200)
plt.show()


# ### Deep Learning

# In[ ]:


# build model
class CnnModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(CnnModel, self).__init__()
        
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.fc_2 = nn.Linear(88, int(hidden_dim / 2))
        self.fc_3 = nn.Linear(int(hidden_dim / 2), output_dim)
        
    def forward(self, in_tensor):
        res = self.dropout(torch.relu(self.fc_1(in_tensor)))
        res = res.unsqueeze(dim=1)
        for net in range(10):
            res = self.pool(torch.relu(self.conv(res)))
        res = res.squeeze(dim=1)
        res = torch.relu(self.fc_2(res))
        out_tensor = torch.log_softmax(self.fc_3(res), dim=1)
        return out_tensor


class DnnModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(DnnModel, self).__init__()
        
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, in_tensor):
        res = self.dropout(torch.relu(self.fc_1(in_tensor)))
        res = torch.relu(self.fc_2(res))
        out_tensor = torch.log_softmax(self.fc_3(res), dim=1)
        return out_tensor


def weights_init(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0.0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def get_data_loader(x, y, shuffle=False):
    x_tensor = torch.tensor(x, dtype=torch.float, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    dataset = Data.TensorDataset(x_tensor, y_tensor)
    data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
    
    
def plot(to_plot, mode):
    plt.figure()
    plt.plot(to_plot)
    plt.ylabel(mode)
    plt.xlabel('epoch')
    plt.show()

    
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


model = DnnModel(input_dim, hidden_dim, output_dim, dropout_rate).to(device)
model.apply(weights_init)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.NLLLoss()

# train and evaluate
def train(data_loader):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in data_loader:
        optimizer.zero_grad()
        train_output = model(batch_x)
        loss = criterion(train_output, batch_y.reshape(-1))
        loss.backward()
        epoch_loss += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return epoch_loss / len(data_loader)

def validate(data_loader):
    model.eval()
    true_num = 0
    for batch_x, batch_y in data_loader:
        dev_out = model(batch_x)
        dev_out = dev_out.max(1)[1]
        true_num += torch.sum((dev_out == batch_y) * 1).item()
    return true_num / len(dev_x)


def evaluate(infer_x):
    model.eval()
    with torch.no_grad():
        infer_x = torch.tensor(infer_x, dtype=torch.float, device=device)
        infer_y = model(infer_x.unsqueeze(0))
#         infer_y = infer_y.max(1)[1]
    return infer_y


# In[ ]:


# self learning

tmp_x, tmp_y = train_x, train_y

for i in tqdm(range(5)):
    
    train_data_loader = get_data_loader(x = tmp_x, y = tmp_y)
#     dev_data_loader = get_data_loader(x = dev_x, y = dev_y)
    
    loss_train = []
    for epoch in range(max_epochs):
        train_loss = train(train_data_loader)
#         dev_acc = validate(dev_data_loader)
        loss_train.append(train_loss)
#         acc_dev.append(dev_acc)
#     plot(loss_train, mode='train_loss')
#     plot(acc_dev, mode='dev_acc')
    
    add_x, add_y = [], []
    for idx_data in test_x:
        infer_out = torch.exp(evaluate(idx_data)).squeeze().cpu().numpy()
        infer_out = np.where((infer_out >= threshold), infer_out, 0)
        if np.sum(infer_out) == 0.0:
            continue
        else:
            add_x.append(idx_data)
            add_y.append(infer_out.argmax())
    if len(add_x) == 0:
        print('no samples be added .')
        continue
    else:
#         print(f'{len(add_y)} samples be added .')
        tmp_x = np.concatenate((train_x, np.array(add_x)), axis=0)
        tmp_y = np.concatenate((train_y, np.array(add_y)), axis=0)

infer_test = []
for idx_data in test_x:
    infer_out = evaluate(idx_data)
    infer_out = infer_out.max(1)[1]
    infer_test.append(infer_out.item())

accuracy = metrics.accuracy_score(test_y, np.array(infer_test))
print(f'accuracy: {accuracy}')


# ### XGBoosting

# In[ ]:


params = {
    'objective': 'multi:softmax',
    'num_class': 5,
    'gamma': 0.1,
    'max_depth': 12,
    'lambda': 1,
    'eta': 0.001,
    'seed': seed,
    'nthread': 4 
}

xg_train = xgb.DMatrix(train_x, label=train_y)
xg_test = xgb.DMatrix(test_x, label=test_y)

model = xgb.train(params, xg_train, num_boost_round=10)
predict = model.predict(xg_test)

acc = metrics.accuracy_score(predict, test_y)
print(f'test_acc: {acc:.4f}')
print(type(xg_train))


# In[ ]:


# self learning with XGBoosting

params = {
    'objective': 'multi:softprob',
    'num_class': output_dim,
    'gamma': 0.1,
    'max_depth': 12,
    'lambda': 1,
    'eta': learning_rate,
    'seed': seed,
    'nthread': 4
}

tmp_x, tmp_y = train_x.copy(), train_y.copy()
xg_dev = xgb.DMatrix(dev_x, label=dev_y)
xg_test = xgb.DMatrix(test_x, label=test_y)

for i in tqdm(range(iters)):
    
    xg_train = xgb.DMatrix(tmp_x, label=tmp_y)
    model = xgb.train(params, xg_train, num_boost_round=10)
    
    dev_pred_prob = model.predict(xg_dev)
    dev_pred = np.argmax(dev_pred_prob, axis=1)
#     print(dev_pred)
    dev_acc = np.sum((dev_pred == dev_y) * 1) / len(dev_y)
    print(f'dev_acc: {dev_acc}')
    
    add_x, add_y = [], []
    test_pred = model.predict(xg_test)
    for idx, i_test_pred in enumerate(test_pred):
        tmp = np.where((i_test_pred >= threshold), i_test_pred, 0)
        if np.sum(tmp) == 0:
            continue
        else:
            add_x.append(test_x[idx])
            add_y.append(i_test_pred.argmax())
    if len(add_x) == 0:
        print('no samples be added .')
        continue
    else:
        print(f'{len(add_y)} samples be added .')
        tmp_x = np.concatenate((train_x, np.array(add_x)), axis=0)
        tmp_y = np.concatenate((train_y, np.array(add_y)), axis=0)


test_pred = model.predict(xg_test)
test_pred = test_pred.argmax(axis=1)
print(f'\ntest_acc: {(metrics.accuracy_score(test_pred, test_y)):.4f}')


# ### SVM

# In[ ]:


clf = svm.SVC(probability=True)
clf.fit(train_x, train_y)


# In[ ]:


test_pred = clf.predict(test_x)
print(test_pred[:10])
# test_pred_prob = clf.predict_proba(test_x)

test_acc = metrics.accuracy_score(test_pred, test_y)
print(f'test_acc: {test_acc:.4f}')


# In[22]:


# experiments for spliting data

params = {
    'objective': 'multi:softmax',
    'num_class': 5,
    'gamma': 0.1,
    'max_depth': 12,
    'lambda': 1,
    'eta': learning_rate,
    'seed': seed,
    'nthread': 4 
}

svm_acc, xgb_acc = [], []
for split in np.linspace(0.9, 0.1, 15):
    
    train_x, test_x, train_y, test_y = train_test_split(x_standarded, y, test_size=split, random_state=seed)
    
    # SVM
    clf = svm.SVC(probability=True)
    clf.fit(train_x, train_y)
    test_pred = clf.predict(test_x)
    test_acc_svm = metrics.accuracy_score(test_pred, test_y)
    svm_acc.append(test_acc_svm)
    
    XGBoosting
    xg_train = xgb.DMatrix(train_x, label=train_y)
    xg_test = xgb.DMatrix(test_x, label=test_y)
    model = xgb.train(params, xg_train, num_boost_round=10)
    predict = model.predict(xg_test)
    test_acc_xgb = metrics.accuracy_score(predict, test_y)
    xgb_acc.append(test_acc_xgb)


plt.figure(figsize=(10,15))
plt.subplot(311)
plt.plot(np.linspace(0.1, 0.9, 15), svm_acc)
plt.ylabel('test_acc')
plt.xlabel('epoch')
plt.title('Accuracy on test_set with SVM')

plt.subplot(312)
plt.plot(np.linspace(0.1, 0.9, 15), xgb_acc)
plt.ylabel('test_acc')
plt.xlabel('epoch')
plt.title('Accuracy on test_set with XGBoosting')

plt.tight_layout()
plt.savefig('./imgs/svm_xgb.png', dpi=200)
plt.show()


# In[23]:


test_acc_dnn = [0.9625925925925926, 0.9913009094503756, 0.9944868532654793, 0.9954254345837146, 0.9950372208436724, 0.9940314704286489,  0.9946172248803827, 0.9973333333333333, 0.9977426636568849, 0.9965457685664939, 0.9959432048681541, 0.996319018404908, 0.9937791601866252, 0.9978813559322034, 0.9933333333333333]

plt.figure(figsize=(10,15))
plt.subplot(311)
plt.plot(np.linspace(0.1, 0.9, 15), test_acc_dnn)
plt.ylabel('test_acc')
plt.xlabel('epoch')
plt.title('Accuracy on test_set with Semi-Supervised Learning')

# plt.savefig('./imgs/semi.png', dpi=200)
plt.show()


# In[ ]:




