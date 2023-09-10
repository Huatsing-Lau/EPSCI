# -*- coding: utf-8 -*-
# 导入第三方包
import os
import pandas as pd
import numpy as np

def write_lines_to_file(lines, file_path):
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def fill_missing_data(data):
    from fancyimpute import KNN
    data = pd.read_csv(path,encoding='gbk')
    columns = data.columns
    data = pd.DataFrame(KNN(k=6).fit_transform(data)) 
    data.columns = columns  # fancyimpute填补缺失值时会自动删除列名
    return data

def SCI_data_process(fname,
                     sheet_name, 
                     Y_feature_name, 
                     delete_columns,
                     dummy_columns,
                     multi_class = False,
                     n_features_to_select=None, 
                     base_model=None):
    assert os.path.isfile(fname), 'no such file:'+fname
    if os.path.splitext(fname)[-1] == '.xlsx':
        data = pd.read_excel(fname,sheet_name=sheet_name,header=0,index_col=False)#,encoding='gb18030')
    if os.path.splitext(fname)[-1] == '.csv':
        data = pd.read_csv(fname,header=0,index_col=False)#,encoding='gb18030')

#     # 填补缺失值
#     data = fill_missing_data(data)
    # 删除用户要求删除的变量
    data.drop(columns=delete_columns,inplace=True)
    # 确保各变量无缺失情况
#     assert sum(data.isnull().sum())==0, print( "存在数据缺失：",sum(data.isnull().sum()) )
    # 所有哑变量的处理
    dummy_columns = [col if col not in delete_columns else [] for col in dummy_columns]
    dummy_columns = dummy_columns.remove([]) if [] in dummy_columns else dummy_columns

    
    dataX = data.drop(columns=Y_feature_name,inplace=False)
    dataX = pd.get_dummies(dataX,
                            drop_first=True,
                            columns=dummy_columns,
                            prefix=dummy_columns,
                            prefix_sep=': ')
    dataY = data[Y_feature_name]
    if multi_class:
        dataY = pd.get_dummies(dataY,
                               drop_first=False,
                               columns=[Y_feature_name],
                               prefix='Y',
                               prefix_sep=': '
                              )

    X_feature_names = dataX.columns.values.tolist()
    Y_feature_name = dataY.columns.values.tolist() if multi_class else Y_feature_name
    dataYX = pd.concat([dataY,dataX],axis=1)
#     import pdb
#     pdb.set_trace()     
    return dataYX, X_feature_names, Y_feature_name

def select_features(dataYX,Y_feature_name,n_features_to_select,base_model):
    #特征删选/降维
    X = dataYX.drop(Y_feature_name,axis=1,inplace=False)
    Y = dataYX[Y_feature_name]
    index_selected_features = features_select(X,Y,base_model=base_model,
                                              n_features_to_select=n_features_to_select)
#     import pdb
#     pdb.set_trace()
    X_feature_names = (np.array(X.columns)[index_selected_features]).tolist()#X_feature_names
    dataYX = dataYX[Y_feature_name+X_feature_names]
    return dataYX



def features_select(X,Y,base_model,n_features_to_select=20):
# =============================================================================
#     from sklearn.feature_selection import SelectKBest, chi2
#     test = SelectKBest(score_func=chi2, k=4) 
#     fit = test.fit(X, Y)
#     features = fit.transform(X)
#     return features
# ============================================================================= 
    from sklearn.feature_selection import RFE
    rfe = RFE(base_model, n_features_to_select=n_features_to_select)
    fit = rfe.fit(X, Y)
    print("Num Features: %d"% fit.n_features_) 
    print("Selected Features: %s"% fit.support_)
    print("Feature Ranking: %s"% fit.ranking_)
    index_selected_features = np.where(fit.support_==True)[0].tolist()
    return index_selected_features

def uncensored(data):
    #data[np.isnan(data)]=0
    #return data
    for k in range(data.shape[1]):
        feat_data = data[:,k]
        isnan_idx = np.where(np.isnan(feat_data))
        feat_data = np.delete(feat_data,isnan_idx)
        feat_mean = feat_data.mean()
        data[isnan_idx,k] = feat_mean
    return data

def stdScal(X):
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X = ss.fit_transform(X)
# =============================================================================
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.scatter(X[:,0],X[:,1])
#     plt.show()
# =============================================================================
    return X


if __name__=="__main__":
    fname = './data/first-first.csv'
    sheet_name = "first-first"
    dataYX, X_feature_names, Y_feature_name = SCI_data_process(fname,sheet_name)
