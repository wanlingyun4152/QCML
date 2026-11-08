import random
#import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def test(file_name):
    torch.manual_seed(42)
    
    # Step 1: Import and Read Data
    data_path = '/home/huangdq_group/liujie/data/taoyx/trans/data'
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
    pre_param = pd.read_csv("./nn_input.txt", sep="\t")
    
    # Step 2: Concatenate both datasets
    data = pd.concat([h2_ucc, h2_uccg, heh_ucc, heh_uccg, lih_ucc, lih_uccg, 
                      h4_uccg, beh2_uccg, h2o_uccg, lih_1k, lih_2k, lih_3k, 
                      h4_2k, h4_3k, beh2_2k, beh2_3k, h2o_3k], ignore_index=True)
    
    pre_param_bond = pre_param.loc[0,'bondlengths']
#    print("pre_param_bond", pre_param_bond)
    replace_bond = round(pre_param_bond, 2)
#    print("replace_bond", replace_bond)
    bond_index = int(1155 + (replace_bond - 0.8) / 0.01)
#    print("bond_index", bond_index)
    replace_columns = ['bondlengths', 'HOMO (eV)', 'LUMO (eV)', 'FCI_energy', 'VQE_energy']
#    print(data.loc[bond_index,replace_columns])
    data.loc[bond_index, replace_columns] = pre_param.loc[0,replace_columns]
#    print(data.loc[bond_index,replace_columns])
    
    # Step 3: Preprocessing
    input_features = ['bondlengths', 'no_qubits', 'no_parameters', 'no_gates', 
                      'no_Pauli_strings', 'no_electron', 'HOMO (eV)', 'LUMO (eV)']
    output_features = [col for col in data.columns if 'VQE_param' in col]
    num_all_params = len(output_features)
#    print('num_all_params', num_all_params)
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
    if bond_index in train_data.index.values.tolist() and bond_index not in test_data.index.values.tolist():
#        print("pre_set = train_data")
        pre_set = train_data
    elif bond_index not in train_data.index.values.tolist() and bond_index in test_data.index.values.tolist():
        pre_set = test_data
#        print("pre_set = test_data")
    #print(pre_set.loc[bond_index,replace_columns])
    # Convert DataFrame to tensors
    X_pre = torch.tensor(pre_set[input_features].values.astype(np.float32), dtype=torch.float32)
    y_pre = torch.tensor(pre_set[output_features].values.astype(np.float32), dtype=torch.float32)

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
    
    # Load model parameters
    model = TransformerModel(input_dim=X_pre.shape[1], output_dim=y_pre.shape[1], dim_model=64, num_heads=2, num_layers=2)
    model.load_state_dict(torch.load('transformer_model_params.pth'))
#    print("input_dim :",X_pre.shape[1], "output_dim :",y_pre.shape[1])
#    print("Input features :", input_features)
    
    # Model in evaluation mode
    model.eval()
    
    # Prediction
    def prediction(t):
#        print(f"Prediction for {t} set :")
       # t3 = datetime.datetime.now()
        pre_data = pre_set
        X_data = X_pre
    
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
            pre_data.loc[:,numeric_features] = numeric_features_reversed
            
            # Add the reversed categorical feature
            pre_data.loc[:,'system'] = original_system
            
            # Now remove the one-hot encoded columns as they are no longer needed
            pre_data = pre_data.drop(columns=system_columns)
        
            # Now test_data should look like the original data before preprocessing
            predictions = model(X_data)
       # t4 = datetime.datetime.now()
#        print(f"{t} prediction time :", t4 - t3)
    
#        pd.set_option("display.max_rows", None)
#        pd.set_option("display.max_columns", None)
        data_rows = pre_data.index.values.tolist()
        data_columns = pre_data.columns.values.tolist()
#        print(pre_data[['system', 'bondlengths'] + output_features[0:4]].head())
#        print(predictions.size())
#        print(predictions[0:5,0:4])
        #p = np.array(predictions)
        #np.savetxt("predictions.txt",p)
    
        pre_bond_index = data_rows.index(bond_index) #index of bond in prediction
        data_columns_sorted = [data_columns[-1]] + data_columns[:-1]
        pre_columns = [f"pre_{i}" for i in data_columns if 'VQE_param' in i]
        err_columns = [f"err_{i}" for i in data_columns if 'VQE_param' in i]
    
        txtfile = f"{t}.txt"
        with open(txtfile, "w") as d:
            for line_i in range(2):
                if line_i == 0:
                    for title_i in range(len(data_columns_sorted)):
                        d.write(data_columns_sorted[title_i] + "\t") 
                    for title_j in range(len(pre_columns)):
                        d.write(pre_columns[title_j] + "\t")
                    for title_k in range(len(pre_columns)):
                        d.write(err_columns[title_k])
                        if title_k < len(pre_columns)-1:
                            d.write("\t")
                else:
                    for column_i in range(len(data_columns_sorted)):
                        d.write(str(pre_data.loc[bond_index, data_columns_sorted[column_i]]) + "\t")
                    for column_j in range(len(pre_columns)):
                        d.write(str(predictions[pre_bond_index, column_j].item()) + "\t")
                    for column_k in range(len(pre_columns)):
                        data_y = pre_data.loc[bond_index, data_columns_sorted[column_k+len(data_columns_sorted)-len(pre_columns)]]
                        pre_y = predictions[pre_bond_index,column_k].item()
                        e_k = pre_y - data_y
                        d.write(str(e_k))
                        if column_k < len(pre_columns)-1:
                            d.write("\t")
                d.write("\n")
    
        txtdata = pd.read_csv(f"{t}.txt", sep='\t')
        txtdata = txtdata.fillna(0)
        params_columns = [col for col in txtdata.columns if 'VQE_param' in col and 'pre' not in col and 'err' not in col]
        zeros_columns = []
        for j in params_columns:
            if not txtdata.loc[:,j].any(): #all is 0
                zeros_columns.append(j)
        #print(len(zeros_columns))
        delete_columns = zeros_columns + [f"pre_{col}" for col in zeros_columns] + [f"err_{col}" for col in zeros_columns]
        #print(len(delete_columns))
        for col in delete_columns:
            del txtdata[col]
        
        err_cols = [int(col.split('m')[-1]) for col in txtdata.columns if 'err_VQE_param' in col]
        err_cols_sort = sorted(err_cols)
        feature_cols = [col for col in txtdata.columns if 'VQE_param' not in col] + \
                       [f'VQE_param{col}' for col in err_cols_sort] + \
                       [f'pre_VQE_param{col}' for col in err_cols_sort] + \
                       [f'err_VQE_param{col}' for col in err_cols_sort]
        txtdata_sort = txtdata[feature_cols]
      #  txtdata_sort.to_csv(f'{t}_clean_{pre_param_bond}.txt', sep='\t')
        txtdata_sort.to_csv(f'{t}_clean.txt', sep='\t')
    
    prediction(file_name)

def output_f(file_name):
    filename1 = f"{file_name}.txt"
    filename2 = f"{file_name}_clean.txt"
    os.system("rm -rf %s" % filename1)
    os.system("rm -rf %s" % filename2)
    test(file_name)
