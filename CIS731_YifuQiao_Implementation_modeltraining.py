###This code is for model construction and training.
###A word embedding model is built and stored
###A LSTM-CNN model is also built and stored.
### The new version of it includes model comparison, by running multiple times we can have an average estimate

import torch as tr
import numpy as np
from gensim.models import word2vec_inner
from gensim.models import word2vec
from gensim.test.utils import common_texts, get_tmpfile
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import random as rd
batchsize = 250
learning_rate = 0.1

class CNNNet(nn.Module): #CNN model
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=300, out_channels=32, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64*2*2,out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.reshape((batchsize,300,10,10))
        x = self.maxpool(nn.functional.relu(self.conv1(x)))
        x = self.maxpool(nn.functional.relu(self.conv2(x)))
        #print(x.shape)
        x = x.reshape(batchsize,64*2*2)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x) #To avoid overfitting
        x = self.fc2(x)
        return x

class LSTMNet(nn.Module): #LSTM model

    def __init__(self, input_size):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=256,num_layers=1,batch_first=True)
        self.out = nn.Sequential(nn.Linear(256, 2))

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1])
        #print(out.shape)
        return out


class LSTM_CNNNet(nn.Module): #The model to be trained
    def __init__(self, input_size):
        super(LSTM_CNNNet,self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True)
        self.conv1 = nn.Conv2d(in_channels=300, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = r_out.reshape(batchsize, 300, 16, 16)
        out = self.maxpool(nn.functional.relu(self.conv1(out)))
        out = self.maxpool(nn.functional.relu(self.conv2(out)))
        out = out.reshape(batchsize, 32 * 4 * 4)
        out = nn.functional.relu(self.fc1(out))
        out = self.dropout(out)  # To avoid overfitting
        out = self.fc2(out)
        return out



with open('C:/Users/28529/Desktop/SU/Courses/ANN/Dataset/imdb.pkl','rb') as fp: #Load the data
    all_data = pickle.load(fp)


for i in range(len(all_data['pos'])):
    for j in range(len(all_data['pos'][i])):
        all_data['pos'][i][j] = all_data['pos'][i][j].lower()

for i in range(len(all_data['neg'])):
    for j in range(len(all_data['neg'][i])):
        all_data['neg'][i][j] = all_data['neg'][i][j].lower()


sentences = all_data['pos'].copy()
sentences.extend(all_data['neg'])

#model = word2vec.Word2Vec(sentences,sg=0,size=50,window=5,min_count=5,negative=3,sample=0.001,hs=1,workers=4) #Build word embedding
#model.save('C:/Users/28529/Desktop/SU/Courses/ANN/Dataset/imdb.model') #Store the model
model = word2vec.Word2Vec.load('C:/Users/28529/Desktop/SU/Courses/ANN/Dataset/imdb.model')

hit_count = 0
err_count = 0
hit_word = {}
err_word = {}
#hit is to check the times when the visited word is included in the embedding
#err is to check the times when the visited word is of the minority

pos_mat = []
longest = 300 #Max length
count = 0
for pos_sent in all_data['pos']:
    onesent = []
    count += 1
    for k in range(len(pos_sent) - 1):
        try:
            bigram = []
            bigram.extend(model[pos_sent[k]]) #Create bigrams
            bigram.extend(model[pos_sent[k+1]])
            bigram = np.array(bigram)
            hit_count += 1
            if(pos_sent[k] not in hit_word.keys()):
                hit_word[pos_sent[k]] = 0
            if(pos_sent[k+1] not in hit_word.keys()):
                hit_word[pos_sent[k+1]] = 0
            onesent.append(bigram)
        except: #Word not in the embedding model
            bigram = [0] * 100
            onesent.append(bigram)
            err_count += 1
            if (pos_sent[k] not in err_word.keys()):
                err_word[pos_sent[k]] = 0
            if (pos_sent[k + 1] not in err_word.keys()):
                err_word[pos_sent[k + 1]] = 0

        if(len(onesent) == longest):
            break

    for i in range(longest - len(onesent)):
        replace = [0] * 100
        onesent.append(replace)
    onesent = np.array(onesent)
    pos_mat.append([onesent,'pos',0])
    if(count == 12500):
        break


for neg_sent in all_data['neg']:
    onesent = []
    count += 1
    for k in range(len(neg_sent) - 1):
        try:
            bigram = []
            bigram.extend(model[neg_sent[k]])
            bigram.extend(model[neg_sent[k + 1]])
            bigram = np.array(bigram)
            hit_count += 1
            if(neg_sent[k] not in hit_word.keys()):
                hit_word[neg_sent[k]] = 0
            if(neg_sent[k+1] not in hit_word.keys()):
                hit_word[neg_sent[k+1]] = 0
            onesent.append(bigram)
        except:
            err_count += 1
            bigram = [0] * 100
            onesent.append(bigram)
            if (neg_sent[k] not in err_word.keys()):
                err_word[neg_sent[k]] = 0
            if (neg_sent[k + 1] not in err_word.keys()):
                err_word[neg_sent[k + 1]] = 0
        if (len(onesent) == longest):
            break
    for i in range(longest - len(onesent)):
        replace = [0] * 100
        onesent.append(replace)
    onesent = np.array(onesent)
    pos_mat.append([onesent,'neg',1])
    if (count == 25000):
        break


rd.shuffle(pos_mat)
datamat = []
labels = []
for onesentence in pos_mat:
    datamat.append(onesentence[0])
    labels.append(onesentence[-1])

print(np.shape(datamat))
datamat = np.array(datamat)

#print(hit_count,err_count)
#print(len(hit_word),len(err_word))

X_train = datamat[:20000]
Y_train = labels[:20000]
X_test = datamat[20000:]
Y_test = labels[20000:]
#print(np.shape(X_train),np.shape(X_test),len(Y_train),len(Y_test))



### Choose the model you want to train

#net = LSTM_CNNNet(np.shape(X_train)[2])
net = CNNNet()
#net = LSTMNet(np.shape(X_train)[2])



criterion = nn.CrossEntropyLoss()
optimizer = tr.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
randupper = np.shape(X_train)[0] - batchsize - 1
randupper_test = np.shape(X_test)[0] - batchsize - 1
xaxis = []
yaxis = []
xaxis_t = []
yaxis_t = []

for epoch in range(7):  ## Model Training
    for i in range(80):
        batch_input = X_train[i*batchsize:(i+1)*batchsize,:,:]
        batch_output = Y_train[i*batchsize:(i+1)*batchsize]
        X_var = Variable(tr.tensor(batch_input, dtype=tr.float32))
        Y_var = Variable(tr.tensor(batch_output, dtype=tr.float32))
        out = net(X_var)

        loss = criterion(out, Y_var.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        xaxis.append(epoch*100 + i)
        yaxis.append(loss.item())
        print(epoch, i, out.shape, loss.item())

    accuracy = []
    for j in range(10):
        test_input = X_test[j * batchsize:(j + 1) * batchsize, :, :]
        test_output = Y_test[j * batchsize:(j + 1) * batchsize]
        X_t = Variable(tr.tensor(test_input, dtype=tr.float32))
        Y_t = test_output
        out_t = net(X_t)
        outnp = out_t.detach().numpy()
        correct = 0
        for r in range(np.shape(outnp)[0]):
            if (np.argmax(outnp[r, :]) == int(Y_t[r])):
                correct += 1
        acc = correct / np.shape(outnp)[0]
        accuracy.append(acc)
    xaxis_t.append(epoch)
    yaxis_t.append(np.mean(accuracy))
    print(epoch, loss.item(), np.mean(accuracy))



#tr.save(net.state_dict(), 'C:/Users/28529/Desktop/SU/Courses/ANN/Dataset/LSTM_CNN.pth')

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)
ax.plot(xaxis, yaxis, c = 'r')
plt.xlabel('Number of weight updates')
plt.ylabel('Loss')
plt.title('')
plt.show()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)
ax.plot(xaxis_t, yaxis_t, c = 'b')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('')
plt.show()
