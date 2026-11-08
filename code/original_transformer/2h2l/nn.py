import datetime
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Import and Read Data
data_path = os.environ.get('QCML_DATA_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'legacy_raw17')))
h2_ucc = pd.read_csv(f'{data_path}/ucc/ucc_H2_data.txt', sep='\t')
h2_uccg = pd.read_csv(f'{data_path}/uccg/uccg_H2_data.txt', sep='\t')
heh_ucc = pd.read_csv(f'{data_path}/ucc/ucc_HeH+_data.txt', sep='\t')
heh_uccg = pd.read_csv(f'{data_path}/uccg/uccg_HeH+_data.txt', sep='\t')
lih_ucc = pd.read_csv(f'{data_path}/ucc/ucc_LiH_data.txt', sep='\t')
lih_uccg = pd.read_csv(f'{data_path}/uccg/uccg_LiH_data.txt', sep='\t')
h4_uccg = pd.read_csv(f'{data_path}/uccg/uccg_H4_data.txt', sep='\t')
beh2_uccg = pd.read_csv(f'{data_path}/uccg/uccg_BeH2_data.txt', sep='\t')
h2o_uccg = pd.read_csv(f'{data_path}/uccg/uccg_H2O_data.txt', sep='\t')
lih_1k = pd.read_csv(f'{data_path}/1kupccgsd/1kupccgsd_LiH_data.txt', sep='\t')
lih_2k = pd.read_csv(f'{data_path}/2kupccgsd/2kupccgsd_LiH_data.txt', sep='\t')
lih_3k = pd.read_csv(f'{data_path}/3kupccgsd/3kupccgsd_LiH_data.txt', sep='\t')
h4_2k = pd.read_csv(f'{data_path}/2kupccgsd/2kupccgsd_H4_data.txt', sep='\t')
h4_3k = pd.read_csv(f'{data_path}/3kupccgsd/3kupccgsd_H4_data.txt', sep='\t')
beh2_2k = pd.read_csv(f'{data_path}/2kupccgsd/2kupccgsd_BeH2_data.txt', sep='\t')
beh2_3k = pd.read_csv(f'{data_path}/3kupccgsd/3kupccgsd_BeH2_data.txt', sep='\t')
h2o_3k = pd.read_csv(f'{data_path}/3kupccgsd/3kupccgsd_H2O_data.txt', sep='\t')

# Step 2: Concatenate both datasets
data = pd.concat([h2_ucc, h2_uccg, heh_ucc, heh_uccg, lih_ucc, lih_uccg, 
                  h4_uccg, beh2_uccg, h2o_uccg, lih_1k, lih_2k, lih_3k, 
                  h4_2k, h4_3k, beh2_2k, beh2_3k, h2o_3k], ignore_index=True)

# Step 3: Preprocessing
input_features = ['bondlengths', 'no_qubits', 'no_parameters', 'no_gates', 
                  'no_Pauli_strings', 'no_electron', 'HOMO (eV)', 'LUMO (eV)']
output_features = [col for col in data.columns if 'VQE_param' in col]
num_all_params = len(output_features)
print('num_all_params', num_all_params)
# Convert categorical data to one-hot encoding
data = pd.get_dummies(data, columns=['system'])
system_columns = [col for col in data.columns if 'system_' in col]  # Save system columns for later use
input_features += system_columns  # Include generated one-hot columns

# Normalize the non-categorical data
scaler = StandardScaler()
numeric_features = ['bondlengths', 'no_qubits', 'no_parameters', 'no_gates', 
                    'no_Pauli_strings', 'no_electron', 'HOMO (eV)', 'LUMO (eV)']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Splitting data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Ensure all data is numerical and there are no missing values
#train_data = train_data.dropna()  # Remove rows with NaN values if any
#test_data = test_data.dropna()    # Remove rows with NaN values if any
#train_data = train_data.fillna(0) # replace NAN with 0
#test_data = test_data.fillna(0)
print("The number of train data :", len(train_data))
print("The number of test data :", len(test_data))

# Convert DataFrame to tensors
try:
    X_train = torch.tensor(train_data[input_features].values.astype(np.float32), dtype=torch.float32)
    y_train = torch.tensor(train_data[output_features].values.astype(np.float32), dtype=torch.float32)
    X_test = torch.tensor(test_data[input_features].values.astype(np.float32), dtype=torch.float32)
    y_test = torch.tensor(test_data[output_features].values.astype(np.float32), dtype=torch.float32)
    train_data_inversed = train_data.copy()
    test_data_inversed = test_data.copy()
    train_data_inversed[numeric_features] = scaler.inverse_transform(train_data_inversed[numeric_features])
    test_data_inversed[numeric_features] = scaler.inverse_transform(test_data_inversed[numeric_features])
   # print(train_data_inversed[system_columns].head())
   # print("train_data_inversed['no_parameters'].head()\n",train_data_inversed['no_parameters'].head())
   # print("isnull().sum \n",train_data_inversed.isnull().sum(axis=1).head())
    #print(train_data['no_parameters'].head())
   # print(test_data_inversed[system_columns].head())
   # print("test_data_inversed['no_parameters'].head()\n",test_data_inversed['no_parameters'].head())
   # print("isnull().sum \n",test_data_inversed.isnull().sum(axis=1).head())
    #print(test_data['no_parameters'].head())
    weights_train = torch.tensor((num_all_params-train_data_inversed.isnull().sum(axis=1)).values.astype(np.float32), dtype=torch.float32)
    print(weights_train[0:5,])
    weights_test = torch.tensor((num_all_params-test_data_inversed.isnull().sum(axis=1)).values.astype(np.float32), dtype=torch.float32)
    print(weights_test[0:5,])
except Exception as e:
    print("Error in converting data to tensors:", e)
    # Optionally, log or handle the specific error further if necessary
#import sys
#sys.exit(0)
# Dataloader
train_dataset = TensorDataset(X_train, y_train, weights_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(X_test, y_test, weights_test)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# Step 4: Define the Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim_model, num_heads, num_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=dim_model, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dropout=dropout
        )
        self.fc_in = nn.Linear(input_dim, dim_model)
        self.fc_out = nn.Linear(dim_model, output_dim)

    def forward(self, src):
        src = self.fc_in(src)
        src = src.unsqueeze(1)  # Add sequence length dimension
        tgt = src.clone()
        output = self.transformer(src, tgt)
        output = self.fc_out(output.squeeze(1))
        return output

# Step 5: Instantiate the Model, Define Loss and Optimizer
model = TransformerModel(input_dim=X_train.shape[1], output_dim=y_train.shape[1], dim_model=64, num_heads=2, num_layers=2)
if os.path.exists('transformer_model_params.pth'):
    print('load previous transformer model params')
    model.load_state_dict(torch.load('transformer_model_params.pth'))
loss_function = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, verbose=True)
max_grad_norm = 1e-1

def loss_function_ignore_nan(outputs, targets):
    mask = ~torch.isnan(targets)
    valid_outputs = outputs[mask]
    valid_targets = targets[mask]
    #print(valid_outputs.numel())
    if valid_outputs.numel() > 0:
        loss = loss_function(valid_outputs, valid_targets)
    else:
        loss = torch.tensor(0.0, requires_grad=True)
    return loss

# Step 6: Train the Model
def train_model(model, epochs):
    model.train()
    loss_list = [100]
    save_list = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs, targets, weights = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            ###
            batch_loss = 0
            for i in range(inputs.size(0)):
                loss_i = loss_function_ignore_nan(outputs[i], targets[i]) * weights[i]
                batch_loss += loss_i
            batch_loss /= inputs.size(0)
            batch_loss.backward()
            ###
            #loss = loss_function(outputs, targets)
            #loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            print("Batch Gradient Norm:", total_norm)
            optimizer.step()
            total_loss += batch_loss.item()
      #  loss_i = total_loss/len(train_loader)
      #  if loss_i < min(loss_list):
      #      torch.save(model.state_dict(), f'transformer_model_params.pth')
      #      save_list.append(epoch+1)
      #      loss_list.append(total_loss/len(train_loader))
        #print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, LR: {optimizer.param_groups[0]["lr"]}', flush=True)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets, val_weights in val_loader:
                val_outputs = model(val_inputs)
                ###
                batch_loss = 0
                for i in range(val_inputs.size(0)):
                  #  print(i,val_weights[i])
                    loss_i = loss_function_ignore_nan(val_outputs[i], val_targets[i]) * val_weights[i]
                    batch_loss += loss_i
                batch_loss /= val_inputs.size(0)
                val_loss += batch_loss.item()
                ###
                #val_loss += loss_function(val_outputs, val_targets).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        if val_loss < min(loss_list):
            torch.save(model.state_dict(), f'transformer_model_params.pth')
            save_list.append(epoch+1)
            loss_list.append(val_loss)
        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader)}, Val Loss: {val_loss}, LR: {optimizer.param_groups[0]["lr"]}', flush=True)
    print(f"save epoch {save_list[-1]} loss is {loss_list[-1]}")
    with open('epoch_loss.txt', 'w') as e:
        e.write('epoch'+'\t'+'loss'+'\n')
        for line_i in range(1,len(save_list)+1):
            e.write(str(save_list[line_i-1]) + '\t' + str(loss_list[line_i]) + '\n')

# Training and save the model
t1 = datetime.datetime.now() 
train_model(model, epochs=3000)
t2 = datetime.datetime.now()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ", total_params)

#torch.save(model.state_dict(), 'transformer_model_params.pth')

# Load model parameters
model = TransformerModel(input_dim=X_train.shape[1], output_dim=y_train.shape[1], dim_model=64, num_heads=2, num_layers=2)
model.load_state_dict(torch.load('transformer_model_params.pth'))
print("input_dim :",X_train.shape[1], "output_dim :",y_train.shape[1])
print("Input features :", input_features)

# Model in evaluation mode
model.eval()

# Prediction
def prediction(t):
    print(f"Prediction for {t} set :")
    t3 = datetime.datetime.now()
    if t == "test":
        pre_data = test_data
        X_data = X_test
    elif t == "train":
        pre_data = train_data
        X_data = X_train

    with torch.no_grad():
        # Extract the one-hot encoded column names for 'system'
        system_columns = [col for col in pre_data.columns if 'system_' in col]
    
        numeric_features_reversed = scaler.inverse_transform(pre_data[numeric_features])
        # Identify the original category from one-hot encoded vectors
        def get_original_category(row):
            for col in system_columns:
                if row[col] == 1:
                    return col.split('_')[1]  # Assuming the column names are in the form 'system_categoryvalue'
            return np.nan  # In case no category is active (this should not happen normally)
        
        # Apply to DataFrame to create the original categorical column
        original_system = pre_data[system_columns].apply(get_original_category, axis=1)
        
        # Incorporate the reversed numeric features back into DataFrame
        pre_data[numeric_features] = numeric_features_reversed
        
        # Add the reversed categorical feature
        pre_data['system'] = original_system
        
        # Now remove the one-hot encoded columns as they are no longer needed
        pre_data = pre_data.drop(columns=system_columns)
    
        # Now test_data should look like the original data before preprocessing
    #    print(test_data.head())
        predictions = model(X_data)
    #    print(predictions)
    t4 = datetime.datetime.now()
    print(f"{t} prediction time :", t4 - t3)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    data_rows = pre_data._stat_axis.values.tolist()
    data_columns = list(pre_data) #or pre_data.columns.values.tolist())
    #print(f"{t} data index :", data_rows)
    #print(f"{t} data columns :", data_columns)
#    print(pre_data[['system', 'bondlengths'] + output_features].head()) #pre_data[one column or a list of columns]
    print(pre_data[['system', 'bondlengths'] + output_features[0:4]].head())
    print(predictions.size())
#    print(predictions[0:5,:])
    print(predictions[0:5,0:4])

#    print(f"{t} data index :", data_rows)
    data_rows_sorted = sorted(data_rows)
    pre_index = [data_rows.index(i) for i in data_rows_sorted]
    #print(data_rows_sorted)
    #print(pre_index)
    #print(pre_data.loc[data_rows_sorted, ['system', 'bondlengths']])
    #print(predictions[pre_index,:])

    data_columns_sorted = [data_columns[-1]] + data_columns[:-1]
#    print(data_columns_sorted)
    pre_columns = [f"pre_{i}" for i in data_columns if 'VQE_param' in i]
#    print(pre_columns)
    err_columns = [f"err_{i}" for i in data_columns if 'VQE_param' in i]

#    txtfile = f"{data_path}/pre/{t}_mix.txt"
    txtfile = f"pre/{t}.txt"
    with open(txtfile, "w") as d:
        for line_i in range(len(data_rows) + 1):
            if line_i == 0:
                for title_i in range(len(data_columns_sorted)):
                    d.write(data_columns_sorted[title_i] + "\t") 
                for title_j in range(len(pre_columns)):
                    d.write(pre_columns[title_j] + "\t")
                for title_k in range(len(pre_columns)):
                    d.write(err_columns[title_k])
                    #d.write(f"Error_param{title_k+1}")
                    if title_k < len(pre_columns)-1:
                        d.write("\t")
            else:
                for column_i in range(len(data_columns_sorted)):
                    d.write(str(pre_data.loc[data_rows_sorted[line_i-1], data_columns_sorted[column_i]]) + "\t")
                for column_j in range(len(pre_columns)):
                    d.write(str(predictions[pre_index[line_i-1],column_j].item()) + "\t")
                for column_k in range(len(pre_columns)):
                    data_y = pre_data.loc[data_rows_sorted[line_i-1], data_columns_sorted[column_k+len(data_columns_sorted)-len(pre_columns)]]
                    pre_y = predictions[pre_index[line_i-1],column_k].item()
                    e_k = pre_y - data_y
                    d.write(str(e_k))
                    if column_k < len(pre_columns)-1:
                        d.write("\t")
            d.write("\n")

prediction("test")
prediction("train")
print("t1 :", t1)
print("t2 :", t2)
print("t2 - t1 :", t2 - t1, flush=True)
