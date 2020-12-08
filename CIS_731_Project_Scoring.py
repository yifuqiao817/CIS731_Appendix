###This code is to train the scoring model based on IMDb online data.




import torch as tr
import numpy as np
from gensim.models import word2vec_inner
from gensim.models import word2vec
from gensim.test.utils import common_texts, get_tmpfile
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import random as rd
import re
import string
import nltk
from nltk.corpus import stopwords

batchsize = 250

def illegal_char(s): #By using regular expression, to compile the unicodes and eliminate the illegal characters
    s = re \
        .compile( \
        u"[^"
        u"\u4e00-\u9fa5"
        u"\u0041-\u005A"
        u"\u0061-\u007A"
        u"\u0030-\u0039"
        u"\u3002\uFF1F\uFF01\uFF0C\u3001\uFF1B\uFF1A\u300C\u300D\u300E\u300F\u2018\u2019\u201C\u201D\uFF08\uFF09\u3014\u3015\u3010\u3011\u2014\u2026\u2013\uFF0E\u300A\u300B\u3008\u3009"
        u"\!\@\#\$\%\^\&\*\(\)\-\=\[\]\{\}\\\|\;\'\:\"\,\.\/\<\>\?\/\*\+"
        u"]+") \
        .sub('', s)
    return s


class LSTM_CNNNet(nn.Module):
    def __init__(self, input_size):
        super(LSTM_CNNNet,self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True)
        self.conv1 = nn.Conv2d(in_channels=300, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
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

#model = word2vec.Word2Vec.load('C:/Users/28529/Desktop/SU/Courses/ANN/Dataset/imdb.model') #Word embedding should be reconstructed
imdb_crawlerpath = 'C:/Users/28529/Desktop/SU/Courses/ANN/imdb_crawler/movie_review_info.csv' #Load online data

csvfile = pd.read_csv(imdb_crawlerpath)
names = list(csvfile.columns)
ind_label = names.index('rating')
ind_corpus = names.index('userReview')
csvfile = csvfile.iloc[0:]
sentences = []
old_labels = []
for i in range(csvfile.shape[0]):
    row = csvfile.iloc[i, :].values
    label = int(row[ind_label])
    txt = row[ind_corpus]
    if(label != 0):
        txt = txt[txt.find('>') + 1:-1]
        startpoint = txt.find('</div>')  # Delete the string we do not need
        while (startpoint > 0):
            txt = list(txt)
            for i in range(6):
                txt[i + startpoint] = ' '
            txt = ''.join(txt)
            startpoint = txt.find('</div>')

        startpoint = txt.find('<br/>')
        while (startpoint > 0):
            txt = list(txt)
            for i in range(5):
                txt[i + startpoint] = ''
            txt = ''.join(txt)
            startpoint = txt.find('<br/>')
        tt = nltk.TweetTokenizer(txt)
        tokens = tt.tokenize(txt)  # Tokenize the text
        stems = tokens
        stop = stopwords.words('english')  # Delete the stopwords
        # Compile the unicodes, delete punctuations, web links, and @, # marks.
        tokens_filtered = [illegal_char(w) for w in stems if
                           w.lower() not in stop and w.lower() not in string.punctuation and not (
                               w.lower().startswith(('http', '#', '@')))]
        sentences.append(tokens_filtered)
        del tokens_filtered
        old_labels.append(label)

model = word2vec.Word2Vec(sentences,sg=0,size=50,window=5,min_count=5,negative=3,sample=0.001,hs=1,workers=4) #Build word embedding
model.save('C:/Users/28529/Desktop/SU/Courses/ANN/Dataset/imdb_scoring.model') #Store the model

hit_count = 0
err_count = 0
hit_word = {}
err_word = {}
pos_mat = []
longest = 300
count = 0

for pos_sent in sentences:
    onesent = []
    for k in range(len(pos_sent) - 1):
        try:
            bigram = []
            bigram.extend(model[pos_sent[k]])
            bigram.extend(model[pos_sent[k+1]])
            bigram = np.array(bigram)
            hit_count += 1
            if(pos_sent[k] not in hit_word.keys()):
                hit_word[pos_sent[k]] = 0
            if(pos_sent[k+1] not in hit_word.keys()):
                hit_word[pos_sent[k+1]] = 0
            onesent.append(bigram)
        except:
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
    pos_mat.append([onesent,old_labels[count] - 1])
    count += 1

rd.shuffle(pos_mat)
datamat = []
labels = []
for onesentence in pos_mat:
    datamat.append(onesentence[0])
    labels.append(onesentence[-1])


datamat = np.array(datamat)

X_train = datamat[:5000,:,:]
Y_train = labels[:5000]
X_test = datamat[5000:,:,:]
Y_test = labels[5000:]
net = LSTM_CNNNet(np.shape(X_test)[2])

criterion = nn.CrossEntropyLoss()
optimizer = tr.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
randupper = np.shape(X_train)[0] - batchsize - 1
randupper_test = np.shape(X_test)[0] - batchsize - 1
xaxis = []
yaxis = []
xaxis_t = []
yaxis_t = []

for epoch in range(50):  ## Model Training
    for i in range(20):
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
    for j in range(7):
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


