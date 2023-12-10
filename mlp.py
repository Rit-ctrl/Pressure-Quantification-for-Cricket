import pandas as pd
import torch
import numpy as np
from utils import process_df
import os
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def NormalizeTensor(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def create_pressure_dataset(dataset,lookback):

    X  = []
    n,d = dataset.shape

    for i in range(lookback):
        true_values = dataset[:i].values
        pad_array = np.zeros((lookback-i,d))
        X.append(np.vstack((pad_array,true_values)))
    for i in range(lookback,len(dataset)):
        feature = dataset[i-lookback:i].values
        # target = dataset[i+1:i+lookback+1].values
        X.append(feature)
        # y.append(target)
    X = np.array(X)
    # y = np.array(y)
    return X

def get_data(train_cols,state_train_cols,target_cols):
    '''
    Get second innings data for training and test
    '''
    data_dir = "ipl_csv2"
    file_list = os.listdir(data_dir)
    file_list = [x for x in file_list if 'info' not in x and 'csv' in x]
    file_list.remove('all_matches.csv')

    df_list = []
    for matches in tqdm(file_list):
        # print(matches)
        df_list.append(process_df(pd.read_csv(os.path.join(data_dir,matches))))
    
    seasons = list(set([df.season.unique()[0] for df in df_list]))

    train = []
    test = []
    for df in df_list:
        if df.season.unique()[0] > 2021:
            test.append(df)
        else:
            train.append(df)
    
    train_data = pd.concat(train)
    train_data= train_data.query("innings == 2")
    test_data = pd.concat(test)
    test_data = test_data.query("innings == 2")

    train_X,test_X = train_data[train_cols],test_data[train_cols]
    train_Y,test_Y = train_data[target_cols],test_data[target_cols]

    return train_data,test_data

class custom_dataset(torch.utils.data.Dataset):

    def __init__(self,lstm_train,state_train,target):
        super().__init__()
        self.labels = torch.Tensor(target)
        self.lstm_train = torch.Tensor(lstm_train)
        self.state_train = torch.Tensor(state_train)

    def __len__(self):

        # print((self.lstm_train).shape)
        return len(self.labels)

    def __getitem__(self,index):

        return self.lstm_train[index],self.state_train[index],self.labels[index]

class xR_Model(torch.nn.Module):
    
    def __init__(self,state_input_size = 4,seq_len = 6,lstm_input_size = 4,lstm_hidden_size = 1) -> None:
        super().__init__()
        
        self.pressure_layer = torch.nn.LSTM(input_size = lstm_input_size,hidden_size = lstm_hidden_size,num_layers = 1,batch_first = True)
        self.pressure_linear = torch.nn.Linear(lstm_hidden_size * seq_len,1)
        # self.state_layer = torch.nn.Linear(state_input_size,1)
        self.comb_layer = torch.nn.Linear(state_input_size+1,2) #1 is pressure
        self.output_layer_reg = torch.nn.Linear(2,1)
        torch.nn.init.xavier_uniform_(self.pressure_linear.weight)

        # self.output_layer_class = torch.nn.Linear(4,1)

    def forward(self,lstm_x,state_x):

        b,seq,_ = lstm_x.shape
        lstm_output, _ = self.pressure_layer(lstm_x)
        # print(lstm_output.shape)
        lstm_output = lstm_output.reshape(b,-1)
        # print(lstm_output.shape)


        lstm_output = self.pressure_linear(lstm_output)
        # print(lstm_output.shape)
        # lstm_output = torch.sigmoid(lstm_output)

        # state_output = self.state_layer(state_x)
        # state_output = F.relu(state_output)

        # print(lstm_output.shape,state_output.shape)

        comb_output = self.comb_layer(torch.hstack((state_x,lstm_output)))
        comb_output = F.tanh(comb_output)

        reg_out = self.output_layer_reg(comb_output)
        # reg_out = F.relu(reg_out)
        # class_out = self.output_layer_class(comb_output)

        # class_out = F.sigmoid(class_out)

        return reg_out,lstm_output

class linear_reg(torch.nn.Module):

    def __init__(self, input_size,output_size = 1) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(input_size,output_size)
    
    def forward(self,x):

        return self.linear_1(x)




if __name__ == "__main__":
    train_cols = ['innings', 'ball', 'wicket','runs_off_bat','wickets_left','runs_left']
    state_train_cols = ['ball','wickets_left','runs_left']
    target_cols = ['runs_off_bat']

    lookback = 1
    train_data,test_data = get_data(train_cols,state_train_cols,target_cols)

    lstm_train = []
    current_state_train = []
    target = []

    
    for id in train_data.match_id.unique():
        x = create_pressure_dataset(train_data.loc[train_data['match_id'] == id,['total_runs','wicket']],lookback=lookback)
        current_state_train.append(train_data.loc[train_data['match_id']==id,state_train_cols].values)
        target.append(train_data.loc[train_data['match_id']==id,target_cols].values)
        lstm_train.append(x)
    
    # train_dataset = torch.utils.data.Dataset(torch.Tensor(np.vstack(lstm_train)),torch.Tensor(np.vstack(current_state_train),torch.Tensor(np.vstack(target))))
    train_dataset = custom_dataset(np.vstack(lstm_train),np.vstack(current_state_train),np.vstack(target))
    train_loader = DataLoader(train_dataset,batch_size=len(train_dataset),shuffle=True)   

    sk_lm = LinearRegression()
    print(np.vstack(current_state_train).shape)

    sk_lm.fit(np.vstack(current_state_train),np.vstack(target))
    # t1 = torch.nn.MSELoss()
    # sk_pred  = sk_lm.predict(np.vstack(current_state_train))
    # sk_targets = np.vstack(target)
    # print("loss for sk : ",mean_squared_error(sk_targets,sk_pred))
    # t_p = torch.Tensor(sk_pred)
    # t_t = torch.Tensor(sk_pred)

    # t_p.requires_grad_()

    # print("loss pytorch ",t1(t_p,t_t))

    device = torch.device('cpu')
    epochs = 40
    toy_model = linear_reg(3,1)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(toy_model.parameters(),lr= 0.01)

    for iter in range(epochs):

        total_loss = 0
        for batch,data in enumerate(train_loader):
            
            optimizer.zero_grad()
            x1,x2,y = data

            x2 = x2.to(device)
            y = y.to(device)

            # x2 = NormalizeTensor(x2)

            preds = toy_model(x2)
            # print(preds.shape,y.shape)

            loss = loss_fn(preds,y)
            total_loss += loss.item()

            loss.backward()

            if batch == 2:
                print(x2[0],y[0],preds[0])

                print(toy_model.linear_1.weight.grad)
                print(toy_model.linear_1.bias.grad)

            optimizer.step()

        print(f" EPOCH {iter} t_loss : {total_loss}")
    



    