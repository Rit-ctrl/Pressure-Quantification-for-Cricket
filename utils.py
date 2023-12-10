import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder



def process_df(df:pd.DataFrame,useful_columns = ['match_id', 'season', 'venue', 'innings', 'ball',
       'striker', 'non_striker', 'bowler',
       'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes',
       'penalty', 'wicket_type', 'player_dismissed', 'other_wicket_type',
       'other_player_dismissed']):

    extra_cols = ['wides','noballs','byes','legbyes','penalty','runs_left']

    df_cleaned = df[useful_columns].copy()
    df_cleaned['wicket_type'].fillna(0,inplace=True)
    df_cleaned['wicket'] = 0
    df_cleaned.loc[df_cleaned.wicket_type !=0,'wicket'] = 1
    df_cleaned['wickets_left'] = 10 - df_cleaned.groupby(['innings'])['wicket'].cumsum()
#    df_cleaned.query('wicket == 1') 
    df_cleaned.loc[df_cleaned['wicket'] == 1,'wickets_left'] += 1

    df_cleaned['total_runs'] = df_cleaned['runs_off_bat'] + df_cleaned['extras']
    df_cleaned['runs_left'] = df_cleaned.groupby('innings')['total_runs'].cumsum() #let us accumulate runs grouped by innings in runs_left for now
    inn_1_total = df_cleaned.groupby('innings')['runs_left'].max().iloc[0] #gets total runs scored by first innings team (gets the target needed)
    target = inn_1_total + 1
    
    df_cleaned.loc[df_cleaned['innings'] == 1,'runs_left'] = df_cleaned.loc[df_cleaned['innings'] == 1,'runs_left'].shift(1)
    df_cleaned.loc[df_cleaned['innings'] == 2,'runs_left'] = df_cleaned.loc[df_cleaned['innings'] == 2,'runs_left'].shift(1)

    df_cleaned.fillna(dict.fromkeys(extra_cols,0),inplace=True)


    df_cleaned.loc[df_cleaned['innings'] == 1,'runs_left'] = inn_1_total - df_cleaned.loc[df_cleaned['innings'] == 1,'runs_left']
    df_cleaned.loc[df_cleaned['innings'] == 2,'runs_left'] = target - df_cleaned.loc[df_cleaned['innings'] == 2,'runs_left']

    # df_cleaned = df_cleaned.loc[(df_cleaned.noballs == 0) & (df_cleaned.wides == 0)] #select only those that are not noballs or wides

    season = df_cleaned.season.unique()[0] 
    if season == '2020/21':
        df_cleaned['season'] = 2020
    elif season ==  '2007/08':
        df_cleaned['season'] = 2008
    elif season ==  '2009/10':
        df_cleaned['season'] = 2010


    return df_cleaned

def get_data():
    '''
    Get second innings data for training and test
    '''
    data_dir = "ipl_csv2"
    file_list = os.listdir(data_dir)
    file_list = [x for x in file_list if 'info' not in x and 'csv' in x]
    file_list.remove('all_matches.csv')

    df_list = []
    for matches in (file_list):
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

    # train_X,test_X = train_data[train_cols],test_data[train_cols]
    # train_Y,test_Y = train_data[target_cols],test_data[target_cols]

    return train_data,test_data

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

def prep_lstm_dataset(train_data,lstm_train_cols,state_train_cols,target_cols,lookback):

# lookback = 6
    lstm_train = []
    current_state_train = []
    target = []

    for id in train_data.match_id.unique():
        # x = create_pressure_dataset(train_data.loc[train_data['match_id'] == id,['total_runs','wicket']],lookback=lookback)
        x = create_pressure_dataset(train_data.loc[train_data['match_id'] == id,lstm_train_cols],lookback=lookback)

        current_state_train.append(train_data.loc[train_data['match_id']==id,state_train_cols].values)
        target.append(train_data.loc[train_data['match_id']==id,target_cols].values)
        lstm_train.append(x)

    return lstm_train,current_state_train,target

class xR_Model(torch.nn.Module):
    
    def __init__(self,state_input_size = 4,seq_len = 1,lstm_input_size = 4,lstm_hidden_size = 1) -> None:
        super().__init__()
        
        self.pressure_layer = torch.nn.LSTM(input_size = lstm_input_size,hidden_size = lstm_hidden_size,num_layers = 1,batch_first = True)
        self.pressure_linear = torch.nn.Linear(lstm_hidden_size * seq_len,1)
        # self.state_layer = torch.nn.Linear(state_input_size,1)
        self.comb_layer = torch.nn.Linear(state_input_size+1,1) #+1 is pressure
        # self.output_layer_reg = torch.nn.Linear(2,1)
        # torch.nn.init.xavier_uniform_(self.pressure_linear.weight)

        # self.output_layer_class = torch.nn.Linear(4,1)

    def forward(self,lstm_x,state_x):

        b,seq,_ = lstm_x.shape
        lstm_output, _ = self.pressure_layer(lstm_x)
        # print(lstm_output.shape)
        lstm_output = lstm_output.reshape(b,-1)
        # print(lstm_output.shape)


        lstm_output = self.pressure_linear(lstm_output)
        # print(lstm_output.shape)
        # lstm_output = torch.tanh(lstm_output)
        lstm_output = torch.sigmoid(lstm_output)

        # state_output = self.state_layer(state_x)
        # state_output = F.relu(state_output)

        # print(lstm_output.shape,state_output.shape)

        comb_output = self.comb_layer(torch.hstack((state_x,lstm_output)))
        reg_out = F.relu(comb_output)

        # reg_out = self.output_layer_reg(comb_output)
        # reg_out = F.relu(reg_out)
        # reg_out = torch.min(reg_out,torch.ones_like(reg_out)*6)
        # class_out = self.output_layer_class(comb_output)

        # class_out = F.sigmoid(class_out)

        return reg_out,lstm_output

def encode_venue(train_data,test_data):
    enc = OneHotEncoder(sparse = False,handle_unknown='ignore')
    train_data[[f'venue_{no}' for no in range(train_data.venue.nunique())]] = enc.fit_transform(train_data[['venue']])
    test_data[[f'venue_{no}' for no in range(train_data.venue.nunique())]] = enc.transform(test_data[['venue']])