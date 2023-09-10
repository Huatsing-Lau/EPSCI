# -*- coding: utf-8 -*-
# +
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn import metrics
import xgboost
from imblearn.over_sampling import SMOTE  # 选取少数类样本插值采样
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:37:14 2019
采用FCN方法做数据挖掘的分析
p次k折交叉验证
决策边界可视化
参考：
[1] https://www.jianshu.com/p/97557b463df8
@author: liuhuaqing
"""
# import umap
import matplotlib.pyplot as plt 
# import tensorflow.compat.v1 as tf
import tensorflow as tf #tensorflow version should be > 2
import tensorflow.keras as keras

import numpy as np
from sklearn.model_selection import cross_validate

try:
    from sklearn.neighbors.classification import KNeighborsClassifier
except:
    from sklearn.neighbors import KNeighborsClassifier
try:
    from sklearn.manifold.t_sne import TSNE
except:
    from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.metrics import confusion_matrix

import scipy.stats as stats

import os

# 设置全局随机数种子
from global_var import seed, random_state
import random
random.seed(seed)
np.random.seed(seed)

# 画混淆矩阵
import matplotlib.pyplot as plt
import seaborn as sns
def plot_confusion_matrix(cm,target_names,
                          title='Confusion Matrix',
                          fn='Confusion Matrix.png',
                          fmt='.2g',
                          center=None,
                          show=False):
    '''
    example:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true = truelabel, y_pred = predClasses)
    sns.set()
    plot_cm(cm,target_names,
            title='Confusion Matrix Model',
            fn='Confusion Matrix Model.png',
            fmt='.20g',
            center=cm.sum()/num_classes
           )
    '''

    fig,ax = plt.subplots(dpi=300,figsize=(6,6))
    ax = sns.heatmap(cm,annot=True,fmt=fmt,center=center,annot_kws={'size':20,'ha':'center','va':'center'})#fmt='.20g',center=250
    ax.set_title(title,fontsize=20)#图片标题文本和字体大小
    ax.set_xlabel('Predict',fontsize=20)#x轴label的文本和字体大小
    ax.set_ylabel('Ground-Truth',fontsize=20)#y轴label的文本和字体大小
    ax.set_xticklabels(target_names,fontsize=20)#x轴刻度的文本和字体大小
    ax.set_yticklabels(target_names,fontsize=20)#y轴刻度的文本和字体大小
    #设置colorbar的刻度字体大小
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)
    ax.set_aspect('equal')
    plt.savefig(fn, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return

def plot_3cm(cm,target_names,num_classes,clf_name,dir_result):
    plot_confusion_matrix(cm,target_names,
            title='Confusion Matrix\n'+clf_name,
            fn=os.path.join(dir_result,'Confusion Matrix '+clf_name+'.png'),
            fmt='.20g',
            #center=cm.sum()/num_classes
           )
    cm_norm_recall = cm.astype('float') / cm.sum(axis=0) 
    cm_norm_recall = np.around(cm_norm_recall, decimals=3)
    plot_confusion_matrix(cm_norm_recall,target_names,
            title='Column-Normalized Confusion Matrix\n'+clf_name,
            fn=os.path.join(dir_result,'Column-Normalized Confusion Matrix '+clf_name+'.png'),
            fmt='.3g',
            center=0.5
           )
    
    cm_norm_precision = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
    cm_norm_precision = np.around(cm_norm_precision, decimals=3)
    plot_confusion_matrix(cm_norm_precision,target_names,
            title='Row-Normalized Confusion Matrix\n'+clf_name,
            fn=os.path.join(dir_result,'Row-Normalized Confusion Matrix '+clf_name+'.png'),
            fmt='.3g',
            center=0.5
           )

def onehot2label(Y_onehot):
    one_hot_label = np.array(Y_onehot,dtype=int).tolist()
    Y = np.array([one_label.index(1) for one_label in one_hot_label]) # 找到下标是1的位置
    return Y

def multi_class_roc(y_label,y_proba,avr_type='macro',to_csv=False,fn='multi_class_roc.csv'):
    """
    计算多类别的平均roc曲线
    y_label: ground-truth y(onehot encoded)
    y_proba: predicted y(probability)
    """
    from sklearn.metrics import roc_curve, auc
    n_classes = y_label.shape[1]
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    
    columns = ['y_true']+\
    ['y_proba_%d'%i for i in range(n_classes)]+\
    ['fpr_%d'%i for i in range(n_classes)]+\
    ['tpr_%d'%i for i in range(n_classes)]+\
    ['threshold_%d'%i for i in range(n_classes)]+\
    ['fpr_micro','tpr_micro']+['fpr_macro','tpr_macro']

    
    index_len = len(y_label.ravel())
    results = pd.DataFrame(columns=columns, index=np.arange(index_len,dtype=int).astype('str'))
    results.loc[np.arange(y_label.shape[0]).astype('str'),'y_true'] = np.argmax(y_label, axis=1)
    for i in range(n_classes):
        results.loc[np.arange(y_label.shape[0]).astype('str'),'y_proba_%d'%i] = y_proba[:, i]

    # Compute roc for each class:
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(y_label[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        results.loc[np.arange(len(fpr[i])).astype('str'),'fpr_%d'%i] = fpr[i]
        results.loc[np.arange(len(tpr[i])).astype('str'),'tpr_%d'%i] = tpr[i]
        results.loc[np.arange(len(threshold[i])).astype('str'),'threshold_%d'%i] = threshold[i]
    
    # Compute micro-average ROC curve and ROC area（micro方法）
    fpr["micro"], tpr["micro"], threshold['micro'] = roc_curve(y_label.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    results.loc[np.arange(len(fpr['micro'])).astype('str'),'fpr_micro'] = fpr['micro']
    results.loc[np.arange(len(tpr['micro'])).astype('str'),'tpr_micro'] = tpr['micro']
    
    # Compute macro-average ROC curve and ROC area（macro方法）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    results.loc[np.arange(len(fpr['macro'])).astype('str'),'fpr_macro'] = fpr['macro']
    results.loc[np.arange(len(tpr['macro'])).astype('str'),'tpr_macro'] = tpr['macro']
    
    if to_csv:
        results.to_csv(fn)
        
    return fpr,tpr,roc_auc

def plot_multi_class_roc_auc(classifier,
                             X,
                             y_true,
                             target_names,
                             ax,
                             clf_name,
                             title,
                             fn,
                             show=False,
                            ):
    """绘制多类别roc曲线"""
    import sklearn
#     from sklearn.ensemble import StackingClassifier
#     if classifier.__class__ is sklearn.ensemble._stacking.StackingClassifier:
    try:
        y_proba = classifier.predict_proba(X) 
    except:
        y_proba = classifier.predict_proba(X.values)
#     roc_auc = roc_auc_score(y_true,y_proba,average='macro',multi_class='ovr')#多类别平均auc
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    y_onehot = enc.fit_transform(y_true.reshape(len(y_true), 1)).toarray()

    fpr,tpr,roc_auc = multi_class_roc(y_label=y_onehot,y_proba=y_proba,to_csv=True,fn=fn+'.csv')#多类类别平均
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    # 画出
    colors = ['red','green','blue','orange','cyan','']
    for k,key in enumerate( fpr.keys() ):
        if key.__class__ is int:
            label=r'ROC of '+target_names[key]+', AUC: %0.3f' % (roc_auc[key])
            lw, alpha = 1.5, 0.8
        else:
            label=str(key)+r'-average ROC '+', AUC: %0.3f' % (roc_auc[key])
            lw, alpha = 2, 0.8
        ax.plot(fpr[key], tpr[key], color=colors[k],
                label=label,
                lw=lw, alpha=alpha)
        
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=title)
    ax.legend(loc="lower right", fontsize='small')
    ax.set_aspect('equal')
    if fn:
        plt.savefig(fn, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    ret=dict()   
    ret['fpr'],ret['tpr'] = fpr, tpr
    ret['roc_auc'] = roc_auc
    return ret

from sklearn.preprocessing import OneHotEncoder
def run_RepeatedKFold(n_splits,n_repeats,classifier,X,Y,title,dir_result,show=False):
    """
    注意：Y是(n_sample,)形式的numpy数组
    X是dataframete或ndarray
    Y是ndarray
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.metrics import auc
    from sklearn.model_selection import RepeatedKFold
    
    num_cls = np.unique(Y)
    fold = RepeatedKFold(n_splits=n_splits,n_repeats=n_repeats)
    tprs = {'micro':[],'macro':[]}
    aucs = {'micro':[],'macro':[]}
    thresholds = {'micro':[],'macro':[]}
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = {'micro':[],'macro':[]}
    mean_auc = {'micro':[],'macro':[]}
    std_tpr = {'micro':[],'macro':[]}
    ci = {'micro':[],'macro':[]}
    
    fig_micro, ax_micro = plt.subplots(figsize=(6,6),dpi=300)
    fig_macro, ax_macro = plt.subplots(figsize=(6,6),dpi=300)
    for i,(train_index, test_index) in enumerate(fold.split(Y)):
        print("RepeatedKFold ",i)
        if X.__class__ == pd.core.frame.DataFrame:
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
        elif X.__class__ == np.ndarray:
            X_train = X[train_index]
            X_test = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]  
        
        classifier.fit(X_train,Y_train)
        y_proba = classifier.predict_proba(X_test)
        
        enc = OneHotEncoder()
        y_onehot = enc.fit_transform(Y_test.reshape(len(test_index), 1)).toarray()
        fn = os.path.join(dir_result,title.replace('\n','')+'_%d'%i+'.csv')
        fpr,tpr,roc_auc = multi_class_roc(y_label=y_onehot,y_proba=y_proba,to_csv=True,fn=fn)#多类类别平均      
        
        interp_tpr = np.interp(mean_fpr, fpr['micro'], tpr['micro'])
        interp_tpr[0] = 0.0
        tprs['micro'].append(interp_tpr)
        ax_micro.plot(mean_fpr, interp_tpr, color='gray', label=None, lw=1, alpha=.3)
        
        interp_tpr = np.interp(mean_fpr, fpr['macro'], tpr['macro'])
        interp_tpr[0] = 0.0
        tprs['macro'].append(interp_tpr)
        ax_macro.plot(mean_fpr, interp_tpr, color='gray', label=None,lw=1, alpha=.3)
        
        roc_auc = roc_auc_score(y_onehot,y_proba,average='micro',multi_class='ovr')#多类别平均auc
        aucs['micro'].append(roc_auc)
        roc_auc = roc_auc_score(Y_test,y_proba,average='macro',multi_class='ovr')#多类别平均auc
        aucs['macro'].append(roc_auc)


    # roc micro
    ax_micro.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr['micro'] = np.mean(tprs['micro'], axis=0)
    mean_tpr['micro'][-1] = 1.0
    mean_auc['micro'] = np.mean(aucs['micro'])#auc(mean_fpr, mean_tpr['micro'])
    ci['micro'] = stats.t.interval(alpha=0.95, df=len(aucs['micro']) - 1, loc=np.mean(aucs['micro']), scale=stats.sem(aucs['micro']) )
    ax_micro.plot(mean_fpr, mean_tpr['micro'], color='b',
                  label=r'micro-average ROC, AUC: %0.3f, 95%% CI: %0.3f~%0.3f' % (mean_auc['micro'], ci['micro'][0],ci['micro'][1]),
                  lw=2, alpha=.8)
    std_tpr['micro'] = np.std(tprs['micro'], axis=0)
    tprs_upper = np.minimum(mean_tpr['micro'] + std_tpr['micro'], 1)
    tprs_lower = np.maximum(mean_tpr['micro'] - std_tpr['micro'], 0)   
    ax_micro.fill_between(mean_fpr, mean_tpr['micro'], tprs_lower, tprs_upper, color='gray', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax_micro.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax_micro.legend(loc="lower right", fontsize='small')
    ax_micro.set_aspect('equal')
    fig_micro.savefig( os.path.join(dir_result,title.replace('\n','')+'_micro') )
    if show:
        plt.show()
    else:
        plt.close()
                                   
    # roc macro
    ax_macro.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr['macro'] = np.mean(tprs['macro'], axis=0)
    mean_tpr['macro'][-1] = 1.0
    mean_auc['macro'] = np.mean(aucs['macro'])#auc(mean_fpr, mean_tpr['macro'])
    ci['macro'] = stats.t.interval(alpha=0.95, df=len(aucs['macro']) - 1, loc=np.mean(aucs['macro']), scale=stats.sem(aucs['macro']) )
    ax_macro.plot(mean_fpr, mean_tpr['macro'], color='b',
                  label=r'macro-average ROC, AUC: %0.3f, 95%% CI: %0.3f~%0.3f' % (mean_auc['macro'],ci['macro'][0],ci['macro'][1]),
                  lw=2, alpha=.8)
    std_tpr['macro'] = np.std(tprs['macro'], axis=0)
    tprs_upper = np.minimum(mean_tpr['macro'] + std_tpr['macro'], 1)
    tprs_lower = np.maximum(mean_tpr['macro'] - std_tpr['macro'], 0)
    ax_macro.fill_between(mean_fpr, mean_tpr['macro'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax_macro.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax_macro.legend(loc="lower right", fontsize='small')
    ax_macro.set_aspect('equal')
    fig_macro.savefig( os.path.join(dir_result,title.replace('\n','')+'_macro') )
    if show:
        plt.show()
    else:
        plt.close()
        
    ret=dict({})   
#     ret['mean_fpr'],ret['mean_tpr'],ret['mean_threshold'] = mean_fpr, mean_tpr, mean_threshold
    ret['cv_mean_fpr'],ret['cv_mean_tpr'] = mean_fpr, mean_tpr
    ret['cv_mean_auc'],ret['cv_mean_aucs_95ci'],ret['cv_aucs'] = mean_auc, ci, aucs
    return ret

def visualize_data_reduced_dimension(data,reducer='TNSE',
                                     n_dim = 2,
                                     title="t-NSE Projection",
                                     dir_result='./',
                                     show=True):
    '''
    数据降维可视化
        data:字典，包含：
            X：输入特征
            Y:真实类别
    '''
    target_names = data['target_names']#类别名称
    if reducer == 'TNSE':
        reducer = TSNE(n_components = n_dim, random_state=random_state)
        X_embedding = reducer.fit_transform(data['X'])
    elif reducer == 'UMAP':
        reducer = umap.UMAP(n_components=n_dim, random_state=random_state)
        X_embedding = reducer.fit_transform(data['X']) 
        
    X2d_xmin,X2d_xmax = np.min(X_embedding[:,0]), np.max(X_embedding[:,0])
    X2d_ymin,X2d_ymax = np.min(X_embedding[:,1]), np.max(X_embedding[:,1])
    if n_dim==3:
        X2d_zmin,X2d_zmax = np.min(X_embedding[:,2]), np.max(X_embedding[:,2])
        
    #plot 
    fig = plt.figure(dpi=300,figsize=(6,6))
    colors = ['red','green','blue','orange','cyan']
    markers = ['o','s','^','*']

    if n_dim == 2:
        for i in range(len(target_names)):
            idx = np.where(data['Y']==i)[0].tolist()#根据真实类别标签来绘制散点
            plt.scatter(X_embedding[idx, 0], X_embedding[idx, 1], 
                        c=colors[i],# cmap='Spectral', 
                        marker=markers[i],s=10,
                        label=target_names[i],
                        alpha=0.5, 
                        edgecolors='none'
                       )
        plt.xlabel('t-NSE 1')
        plt.ylabel('t-NSE 2') 
    elif n_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('t-NSE 1')
        ax.set_ylabel('t-NSE 2') 
        ax.set_zlabel('t-NSE 3')
#         plt.zlim(X2d_zmin,X2d_zmax)
        for i in range(len(target_names)):
            idx = np.where(data['Y']==i)[0].tolist()#根据真实类别标签来绘制散点
            ax.scatter(X_embedding[idx, 0].squeeze(), 
                       X_embedding[idx, 1].squeeze(), 
                       X_embedding[idx, 2].squeeze(),
                       c=colors[i],
                       s=15,
                       marker=markers[i],
                       label=target_names[i],
                       alpha=0.5, 
                      )
                 
     
#     plt.xlim(X2d_xmin,X2d_xmax)
#     plt.ylim(X2d_ymin,X2d_ymax)       
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    fn = os.path.join(dir_result,title.replace('/n','_')+".png")
    plt.savefig(fn, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return  

def visualize_reduced_decision_boundary(clf,data,reducer='TNSE',
                                        title="t-NSE Projection",
                                        dir_result='./',
                                        show=False):
    '''
    分类器的决策边界可视化
    clf:分类器
    X：输入特征
    Y:真实类别
    '''
    if reducer == 'TNSE':
        reducer = TSNE(n_components = 2, random_state=random_state)
        X_embedding = reducer.fit_transform(data['X'])
    elif reducer == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        X_embedding = reducer.fit_transform(data['X']) 

    y_predicted = clf.predict(data['X'])#根据y_predicted 结合KNeighborsClassifier来确定决策边界

    # 生成新的数据, 并调用meshgrid网格搜索函数帮助我们生成矩阵
    # 创建meshgrid 
    resolution = 400 #100x100背景像素
    X2d_xmin,X2d_xmax = np.min(X_embedding[:,0]), np.max(X_embedding[:,0])
    X2d_ymin,X2d_ymax = np.min(X_embedding[:,1]), np.max(X_embedding[:,1])
    xx,yy = np.meshgrid(np.linspace(X2d_xmin,X2d_xmax,resolution), np.linspace(X2d_ymin,X2d_ymax,resolution))

    #使用1-NN 
    #在分辨率x分辨率网格上近似Voronoi镶嵌化
    background_model = KNeighborsClassifier(n_neighbors = 1).fit(X_embedding,y_predicted)
    voronoiBackground = background_model.predict( np.c_[xx.ravel(),yy.ravel()] )
    voronoiBackground = voronoiBackground.reshape((resolution,resolution))

    #plot 
    plt.figure(figsize=(6,6),dpi=300)#figsize=(8,6))
    plt.contourf(xx,yy,voronoiBackground,alpha=0.2)
    idx_0 = np.where(data['Y']==0)[0].tolist()#根据真实类别标签来绘制散点
    idx_1 = np.where(data['Y']==1)[0].tolist()#根据真实类别标签来绘制散点
    plt.scatter(X_embedding[idx_0, 0], X_embedding[idx_0, 1], 
                c='blue',# cmap='Spectral', 
                marker='o',s=20,
                label=data['classname'][0],
                )
    plt.scatter(X_embedding[idx_1, 0], X_embedding[idx_1, 1], 
                c='orange',# cmap='Spectral', 
                marker='s',s=20,
                label=data['classname'][1],
                )
    plt.legend()
    plt.xlim(X2d_xmin,X2d_xmax)
    plt.ylim(X2d_ymin,X2d_ymax)
    #plt.gca().set_aspect('equal', 'datalim')
    plt.title(title)
    plt.savefig( os.path.join(dir_result,title.replace('/n','_')+".png") )
    if show:
        plt.show()
    else:
        plt.close()
    return

def UMAP_visualize_decision_boundary(clf,data,show=False):
    '''
    分类器的决策边界可视化
    clf:分类器
    X：输入特征
    Y:真实类别
    '''
    reducer = umap.UMAP(n_components=2,random_state=random_state)
    X_embedding = reducer.fit_transform(data['X'])

    # 生成新的数据, 并调用meshgrid网格搜索函数帮助我们生成矩阵
    #创建meshgrid 
    resolution = 100 #100x100背景像素
    X2d_xmin,X2d_xmax = np.min(X_embedding[:,0]), np.max(X_embedding[:,0])
    X2d_ymin,X2d_ymax = np.min(X_embedding[:,1]), np.max(X_embedding[:,1])
    xx,yy = np.meshgrid(np.linspace(X2d_xmin,X2d_xmax,resolution), np.linspace(X2d_ymin,X2d_ymax,resolution))

    # 有了新的数据, 我们需要将这些数据输入到分类器获取预测结果
    Z = clf.predict( reducer.inverse_transform(np.c_[xx.ravel(), yy.ravel()]) )
    # 这个时候得到的是Z还是一个向量, 将这个向量转为矩阵即可
    Z = Z.reshape(xx.shape)
    # 分解的时候有背景颜色    
    plt.figure(figsize=(6,6),dpi=300)
    #plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdYlBu)
    plt.contourf(xx, yy, Z, alpha=0.2)
    idx_0 = np.where(data['Y']==0)[0].tolist()#根据真实类别标签来绘制散点
    idx_1 = np.where(data['Y']==1)[0].tolist()
    plt.scatter(X_embedding[idx_0, 0], X_embedding[idx_0, 1], 
                c='blue',# cmap='Spectral', 
                marker='o',
                label='benign',
                s=20)
    plt.scatter(X_embedding[idx_1, 0], X_embedding[idx_1, 1], 
                c='orange',# cmap='Spectral', 
                marker='s',
                label='malignant',
                s=20
                )
    plt.legend()
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('Decision-Boundary of the Classifier in TNSE-projection')
    plt.savefig('Decision-Boundary of the Classifier in TNSE-projection.png')
    if show:
        plt.show()
    else:
        plt.close()
    return

"""
1. Filter方法(逐个特征分析，没有考虑到特征之间的关联作用，可能把有用的关联特征误踢掉。)
    1.1 移除低方差的特征 (Removing features with low variance)
    1.2 单变量特征选择 (Univariate feature selection)
        1.2.1 卡方(Chi2)检验
        1.2.2 互信息和最大信息系数 Mutual information and maximal information coefficient (MIC)
        1.2.3 基于模型的特征排序 (Model based ranking)
2. Wrapper
    2.1 RFE
3.Embedding
    3.1 使用SelectFromModel选择特征 (Feature selection using SelectFromModel)
        3.1.1 基于L1的特征选择 (L1-based feature selection)
"""

# Filter
def select_features_Chi2(X,Y,kbest=10):
    """
    采用卡方检验(Chi2)方法(SelectKBest)选择特征
    注意：经典的卡方检验是检验定性自变量对定性因变量的相关性。
    注意：Input X must be non-negative.
    """
    from sklearn.feature_selection import SelectKBest, chi2
    fit = SelectKBest(score_func=chi2, k=kbest).fit(X, Y)
    index_selected_features = np.where(fit.support_==True)[0].tolist()
    return index_selected_features

def selelct_features_MIC(X,Y,kbest=10):
    """采用互信息指标，来筛选特征"""
    from minepy import MINE#minepy包——基于最大信息的非参数估计
    m = MINE()
    # 单独采用每个特征进行建模
    scores=[]
    for i in range(X.shape[1]):
        m.compute_score( X[:, i], Y)
        scores.append( (round(m.mic(),3), int(i)) )
    scores = sorted(scores, reverse=True)#由大到小排序
    index_selected_features = np.asarray(scores)[:kbest,1]
    return index_selected_features

def selelct_features_MIC(X,Y,kbest=15,dir_result='./'):
    """采用互信息指标，来筛选特征"""
    from minepy import MINE#minepy包——基于最大信息的非参数估计
    m = MINE()
    # 单独采用每个特征进行建模
    feature_names = X.columns
    importance = []
    for i in range(X.shape[1]):
        m.compute_score( X.iloc[:, i], Y)
        importance.append( round(m.mic(),3) )
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)
    feat_imp = feat_imp.iloc[:kbest]
    
    # 画图   
    plt_feature_importance(feat_imp,dir_result,show=False)
    
    return feat_imp

# Wrapper（RFE是包装法的一种）
def select_features_RFE(X,Y,target_names=None,base_model=None,kbest=10,dir_result='./'):
    """采用RFE方法选择特征，用户可以指定base_model"""   
    from sklearn.feature_selection import RFE
    rfe = RFE(base_model, n_features_to_select=kbest)

    fit = rfe.fit(X, Y)
    
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    importance_all = fit.estimator_.coef_.squeeze().tolist()
    index_selected_features = np.where(fit.support_==True)[0].tolist()
    feature_names_all = [feature_names[k] for k in index_selected_features]
    index_selected_features = np.arange(0,kbest).tolist()#更新一下
    
    # 画图
    feature_names_all = X.columns.values.tolist()
    feat_imp = plot_featue_importance_multiclass(importance_all,index_selected_features,feature_names_all,
                                            target_names,dir_result)

    return feat_imp

def select_features_LSVC(X,Y,target_names=None,dir_result='./'):
    """采用LSVC方法选择特征""" 
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC

    my_model = LinearSVC(C=0.01, penalty="l1", dual=False).fit( X, Y )
    selector = SelectFromModel(my_model,prefit=True)
    
    importance_all = selector.estimator.coef_.squeeze().tolist()
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    feature_names_all = X.columns.values.tolist()

    feat_imp = plot_featue_importance_multiclass(importance_all,index_selected_features,feature_names_all,
                                            target_names,dir_result)

    return feat_imp

def select_features_LR(X,Y,target_names=None,dir_result='./'):
    """采用带L1和L2惩罚项的逻辑回归作为基模型的特征选择,
    参数threshold为权值系数之差的阈值""" 
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression as LR

    my_model = LR(C=0.1).fit( X, Y )
    selector = SelectFromModel(my_model,prefit=True)
    
    importance_all = selector.estimator.coef_.squeeze().tolist()
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    feature_names_all = X.columns.values.tolist()
    
    feat_imp = plot_featue_importance_multiclass(importance_all,index_selected_features,feature_names_all,
                                            target_names,dir_result)
    
    return feat_imp

def select_features_Tree(X,Y,target_names=None,dir_result='./'):
    """采用Tree的方法选择特征""" 
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
    
    my_model = ExtraTreesClassifier().fit( X, Y )
    selector = SelectFromModel(my_model,prefit=True)

    importance_all = selector.estimator.feature_importances_.squeeze().tolist()
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    feature_names_all = X.columns.values.tolist()
    
    feat_imp = plot_featue_importance_multiclass(importance_all,index_selected_features,feature_names_all,
                                            target_names,dir_result)
    
    return feat_imp

def select_features_RF(X,Y,target_names=None,dir_result='./'):
    """采用RF的方法选择特征""" 
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestRegressor

    my_model = RandomForestRegressor(n_estimators=20, max_depth=4).fit( X, Y )
    selector = SelectFromModel(my_model,prefit=True)

    importance_all = selector.estimator.feature_importances_.squeeze().tolist()
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    feature_names_all = X.columns.values.tolist()
    
    feat_imp = plot_featue_importance_multiclass(importance_all,index_selected_features,feature_names_all,
                                            target_names,dir_result)
    return feat_imp

def select_features_mrmr(X,method='MID',kbest=10,dir_result='./'):
    """
    采用mRMR方法筛选特征(该方法不考虑应变量)、无特征重要性。
    X是dataframe
    """
    import pymrmr       
    selected_features_names = pymrmr.mRMR(X, method, kbest)
    feat_imp = pd.DataFrame( data=np.zeros([kbest,1]), index=selected_features_names, columns=['feature importance'])

    # 画图  
    fig = plt.figure(dpi=300)#,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
    return feat_imp

def plt_feature_importance(feat_imp,dir_result='./',show=False):
    # 画特征重要性bar图
    y_pos = np.arange(feat_imp.shape[0])
    fig = plt.figure(dpi=300)
    plt.barh(y_pos, feat_imp.values.squeeze(), align='center', alpha=0.8)
    plt.yticks(y_pos, feat_imp.index.values)
    plt.xlabel('Feature Importance')
    plt.title('Feature Selection')
    fn = os.path.join(dir_result,'Feature Importance.png')
    plt.savefig(fn, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return

def plot_featue_importance_multiclass(importance_all,
                                      index_selected_features,
                                      feature_names_all,
                                      target_names,
                                      dir_result,
                                      show=False):
    #　对于m分类问题,ovr下共有m个子问题，因此共有m个重要性排序
    #　挑选出的特征
    
    
    if np.array(importance_all).ndim>1:
        fig = plt.figure(dpi=300,figsize=(6.4*3, 4.8*1.5))#(6.4, 4.8*5)
        num_class = len(importance_all)
        feat_imp = []
        for cls in range(num_class):
            importance = [importance_all[cls][k] for k in index_selected_features]
            feature_names = [feature_names_all[k] for k in index_selected_features]
            # 重新排序
            feat_imp_cls = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
            feat_imp_cls.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)
            # 画图
            plt.subplot(1,num_class,1+cls)
            y_pos = np.arange(feat_imp_cls.shape[0])
            plt.barh(y_pos, feat_imp_cls.values.squeeze(), align='center', alpha=0.8)
            plt.yticks(y_pos, feat_imp_cls.index.values)
            plt.xlabel('Feature Importance')
            plt.ylabel(target_names[cls])
            plt.tight_layout()
            feat_imp.append(feat_imp_cls)
        fig.suptitle('Feature Selection',x=.5,y=1.05)
    else:
        fig = plt.figure(dpi=300,figsize=(6.4, 4.8*1.5))
        importance = [importance_all[k] for k in index_selected_features]
        feature_names = [feature_names_all[k] for k in index_selected_features]
        # 重新排序
        feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
        feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)
        # 画图
        y_pos = np.arange(feat_imp.shape[0])
        plt.barh(y_pos, feat_imp.values.squeeze(), align='center', alpha=0.8)
        plt.yticks(y_pos, feat_imp.index.values)
        plt.xlabel('Feature Importance')
        plt.tight_layout()
    fn = os.path.join(dir_result,'Feature Importance.png')
    plt.savefig(fn, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return feat_imp

def select_features(X,Y,target_names,method,dir_result,**kwargs):
    """多种特征选择方法的封装函数"""
    if 'kbest' in kwargs.keys():
        kbest = kwargs['kbest']
    else:
        kbest = 20
        
    if method == 'MIC':
        feat_imp = selelct_features_MIC(X,Y,kbest=kbest,dir_result=dir_result)
    elif method == 'RFE':
        from sklearn.linear_model import LogisticRegression as LR
        LogR = LR(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,
           intercept_scaling=1,class_weight=None,random_state=None,
           solver='liblinear',max_iter=1000,multi_class='ovr',
           verbose=0,warm_start=False,n_jobs=1)
        feat_imp = select_features_RFE(X,Y,target_names,base_model=LogR,kbest=kbest,dir_result=dir_result)
    elif method == 'EmbeddingLSVC':
        feat_imp = select_features_LSVC(X,Y,target_names,dir_result)
    elif method == 'EmbeddingLR':
        feat_imp = select_features_LR(X,Y,target_names,dir_result)
    elif method == 'EmbeddingTree':
        feat_imp = select_features_Tree(X,Y,target_names,dir_result)
    elif method == 'EmbeddingRF':
        feat_imp = select_features_RF(X,Y,target_names,dir_result)
    elif method == 'mRMR':
        feat_imp = select_features_mrmr(X,'MID',kbest=kbest,dir_result=dir_result)
#         import pandas as pd
#         columns = [str(i) for i in range(X.shape[1])]
#         dfX = pd.DataFrame(data=X,columns=columns)
#         index_selected_features = select_features_mrmr(dfX,'MID',kbest=15)

    #Data = scratch_data(index_selected_features,{'X':X,'Y':Y})
    
    if feat_imp.__class__ == list:
        Data = {'X':X.loc[:,feat_imp[0].index],'Y':Y}
    elif feat_imp.__class__ == pd.core.frame.DataFrame:
        Data = {'X':X.loc[:,feat_imp.index],'Y':Y}
    return Data,feat_imp

def scratch_data(index_selected_features,data):
    """根据特征删选/降维结果，提取出指定的特征"""
    X,Y = data['X'],data['Y']
    index_selected_features = np.array(index_selected_features,'int')
    if X.__class__ == pd.core.frame.DataFrame:
        X = X.values[:,index_selected_features]
    elif X.__class__ == np.ndarray:
        X = X[:,index_selected_features]
    new_data = dict(X=X,Y=Y,features_index=index_selected_features)
    return new_data

def boxplot(x,x_names,title='AUC of different Algorithms',fn_save='AUC-of-different-Algorithms.png',show=True):
    # 参考：https://blog.csdn.net/roguesir/article/details/78249864 
    fig=plt.figure(dpi=300)
    plt.boxplot(x,patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
                showmeans=True, # 以点的形式显示均值 
                boxprops = {'color':'black','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色           
                flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
                meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色 
                medianprops = {'linestyle':'--','color':'orange'}) # 设置中位数线的属性，线的类型和颜色
    plt.xticks([y+1 for y in range(len(x))], x_names, rotation='vertical')
    plt.ylabel('Cross Validation AUC')
    plt.title(title)
    plt.savefig(fn_save, bbox_inches='tight')
    if show:
        plt.show() 
    else:
        plt.close()
    return

def violinplot(x,x_names,title='AUC of different Algorithms',fn_save='AUC-of-different-Algorithms.png',show=True):
    # 参考：https://blog.csdn.net/roguesir/article/details/78249864 
    fig = plt.figure(dpi=300)
    plt.violinplot(
        x,
        showmeans=True, 
        showmedians=True,
        showextrema=True) # 设置中位数线的属性，线的类型和颜色
    plt.xticks([y+1 for y in range(len(x))], x_names, rotation='vertical')
    plt.ylabel('Cross Validation AUC')
    plt.title(title)
    plt.savefig(fn_save, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return

def get_algorithm_test_result(
    X,Y,target_names,
    classifier_fitted,
    clf_name,
    dataset_name,
    dir_result='./'
):
    """
    # 单一算法出各类结果的封装函数 #
    外部验证，输出roc曲线、混淆矩阵
    classifier_fitted
    """
    dir_result = os.path.join(dir_result,dataset_name)
    if not os.path.isdir(dir_result):
        os.makedirs(dir_result)
     
    # roc曲线
    fig, ax = plt.subplots(figsize=(6,6),dpi=300)
    title='ROC Curve of '+clf_name+' against '+dataset_name
    fn =  os.path.join(dir_result,title.replace('\n','_'))
    ret = plot_multi_class_roc_auc(classifier_fitted,
                                   X = X,
                                   y_true = Y,
                                   target_names = target_names,
                                   ax = ax,
                                   clf_name = clf_name,
                                   title = title,
                                   fn = fn
                                  )
    
    # 画混淆矩阵
    import sklearn
#     from sklearn.ensemble import StackingClassifier
#     if classifier_fitted.__class__ is sklearn.ensemble._stacking.StackingClassifier:
    try:
        y_pred = np.array(classifier_fitted.predict(X),dtype=np.int)
    except:
        y_pred = np.array(classifier_fitted.predict(X.values),dtype=np.int)
    
    cm = confusion_matrix( y_true = Y, y_pred = y_pred )
    num_classes = len(target_names)
    plot_3cm(cm,target_names,num_classes,clf_name,dir_result)
    ret['cm'] = cm
    
    # report
    
#     # 画决策边界
#     data = dict(X=X,Y=Y,classname=['no','yes'])
#     visualize_reduced_decision_boundary(
#         clf=classifier_fitted,data=data,
#         title="Decision-Boundary of "+clf_name+" in TNSE-Projection",
#         dir_result=dir_result
#     ) 
    
    return ret

from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss,log_loss
from sklearn.preprocessing import label_binarize

def plot_calibration_curve(
    clf, 
    clf_name, 
    X_train, 
    y_train,
    X_test, 
    y_test,
    target_names:list=None, 
    cv=5, 
    dir_result='./', 
    show=False):
    
    # 将分类器模型包装在 CalibratedClassifierCV 中
    calibrator = CalibratedClassifierCV(clf, cv=5, method='isotonic')

    # 训练模型
    calibrator.fit(X_train, y_train)

    # 在测试集上进行预测，并计算 Brier 分数
    y_pred_prob = calibrator.predict_proba(X_test)
    brier_score = 0
    # 将标签进行二元化处理
    y_test_bin = label_binarize(y_test, classes=list(range(len(target_names))))
    for i in range(y_test_bin.shape[1]):
        brier_score += brier_score_loss(y_test_bin[:, i], y_pred_prob[:, i])
    brier_score /= y_test_bin.shape[1]

    # 绘制校准曲线
    fig, ax = plt.subplots(dpi=300)
    ax.plot([0, 1], [0, 1], "k:", label="Reference")

    for i in range(y_test_bin.shape[1]):
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_bin[:, i], y_pred_prob[:, i], n_bins=10)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{target_names[i]}')
    
    # 添加图表标题和标签
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right")
    ax.set_title(f'Calibration Curve')
    ax.text(0.0, 1.0, f'Overall Brier score: {brier_score:.3f}', verticalalignment='top')
    ax.set_aspect('equal')

    fn = os.path.join(dir_result,"Calibration Curve "+clf_name+'.png')
    plt.savefig(fn)
    # 显示图表
    if show:
        plt.show()
    else:
        plt.close()
        
def plot_calibration_curve_triple_class(
    clf, 
    clf_name, 
    X_train_valid, y_train_valid,
    X_test, y_test,
    target_names,
    cv=5,
    dir_result='./',
    show=False,
):
    
    # Train uncalibrated classifier on whole train and validation data and evaluate on test data
    clf.fit(X_train_valid, y_train_valid)
    clf_probs = clf.predict_proba(X_test.values)
    score = log_loss(y_test, clf_probs)#++++++++注意这里，是否可以作为评价指标++++++++++＃
#     score = brier_score_loss(y_test, clf_probs)
    
    # Calibrate the classifier on the whole train and validation data in cross validation manner\
    # and evaluate on test data   
    sig_clf = CalibratedClassifierCV(clf, method='isotonic', cv=cv)#"sigmoid"
    sig_clf.fit(X_train_valid, y_train_valid)
    sig_clf_probs = sig_clf.predict_proba(X_test.values)
    sig_score = log_loss(y_test, sig_clf_probs)
#     sig_score = brier_score_loss(y_test, sig_clf_probs)

    # Plot changes in predicted probabilities via arrows
    fig = plt.figure(dpi=300,figsize=(6,6))
#     plt.subplot(2,1,1)
    colors = ["r", "g", "b"]
    for i in range(clf_probs.shape[0]):
        plt.arrow(clf_probs[i, 0], clf_probs[i, 1],
                  sig_clf_probs[i, 0] - clf_probs[i, 0],
                  sig_clf_probs[i, 1] - clf_probs[i, 1],
                  color=colors[y_test[i]], head_width=1e-2)

    # Plot perfect predictions
    plt.plot([1.0], [0.0], 'ro', ms=10, label=target_names[0])
    plt.plot([0.0], [1.0], 'go', ms=10, label=target_names[1])
    plt.plot([0.0], [0.0], 'bo', ms=10, label=target_names[2])

    # Plot boundaries of unit simplex
    plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

    # Annotate points on the simplex
    plt.annotate(r'($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)',
                 xy=(1.0/3, 1.0/3), xytext=(1.0/3, .23), xycoords='data',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='center')
    plt.plot([1.0/3], [1.0/3], 'ko', ms=5)
    plt.annotate(r'($\frac{1}{2}$, $0$, $\frac{1}{2}$)',
                 xy=(.5, .0), xytext=(.5, .1), xycoords='data',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='center')
    plt.annotate(r'($0$, $\frac{1}{2}$, $\frac{1}{2}$)',
                 xy=(.0, .5), xytext=(.1, .5), xycoords='data',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='center')
    plt.annotate(r'($\frac{1}{2}$, $\frac{1}{2}$, $0$)',
                 xy=(.5, .5), xytext=(.6, .6), xycoords='data',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='center')
    plt.annotate(r'($0$, $0$, $1$)',
                 xy=(0, 0), xytext=(.1, .1), xycoords='data',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='center')
    plt.annotate(r'($1$, $0$, $0$)',
                 xy=(1, 0), xytext=(1, .1), xycoords='data',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='center')
    plt.annotate(r'($0$, $1$, $0$)',
                 xy=(0, 1), xytext=(.1, 1), xycoords='data',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='center')
    # Add grid
    plt.grid(False)
    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        plt.plot([0, x], [x, 0], 'k', alpha=0.2)
        plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
        plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)

    plt.title("Change of Predicted Probabilities after Calibration of "+clf_name)
    plt.xlabel("Probability "+target_names[0])
    plt.ylabel("Probability "+target_names[1])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="best")
    plt.text(x=0.3,y=0.8,
             s="uncalibrated log_loss: %.3f \n calibrated log_loss %.3f"%(score,sig_score))
    plt.gca().set_aspect('equal')
    fn = os.path.join(dir_result,"Change of Predicted Probabilities after Calibration of "+clf_name+'.png')
    plt.savefig(fn)
    if show:
        plt.show()
    else:
        plt.close()

    print("Log-loss of")
    print("uncalibrated classifier: %.3f "% score)
    print("calibrated classifier : %.3f"% sig_score)

    # Illustrate calibrator
    plt.figure(dpi=300,figsize=(6,6))
#     plt.subplot(2,1,2)
    # generate grid over 2-simplex
    p1d = np.linspace(0, 1, 20)
    p0, p1 = np.meshgrid(p1d, p1d)
    p2 = 1 - p0 - p1
    p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
    p = p[p[:, 2] >= 0]

    calibrated_classifier = sig_clf.calibrated_classifiers_[0]
    prediction = np.vstack([calibrator.predict(this_p)
                            for calibrator, this_p in
                            zip(calibrated_classifier.calibrators_, p.T)]).T
    prediction /= prediction.sum(axis=1)[:, None]

    # Plot modifications of calibrator
    for i in range(prediction.shape[0]):
        plt.arrow(p[i, 0], p[i, 1],
                  prediction[i, 0] - p[i, 0], prediction[i, 1] - p[i, 1],
                  head_width=1e-2, color=colors[np.argmax(p[i])])
    # Plot boundaries of unit simplex
    plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

    plt.grid(False)
    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        plt.plot([0, x], [x, 0], 'k', alpha=0.2)
        plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
        plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)

    plt.title("Illustration of Calibrator of "+clf_name)
    plt.xlabel("Probability "+target_names[0])
    plt.ylabel("Probability "+target_names[1])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.text(x=0.3,y=0.8,
             s="uncalibrated log_loss: %.3f \n calibrated log_loss %.3f"%(score,sig_score))
    plt.gca().set_aspect('equal')
    fn = os.path.join(dir_result,"Illustration of Calibrator of "+clf_name+".png")
    plt.savefig(fn, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return score, sig_score

def save_text(filename, contents):
    fh = open(filename, 'w', encoding='utf-8')
    fh.write(contents)
    fh.close()    

def get_algorithm_result(X_train,
                         Y_train,
                         X_test,
                         Y_test,
                         X_external,
                         Y_external,
                         target_names,
                         n_splits,
                         n_repeats,
                         classifier,
                         clf_name, 
                         dir_result='./'
                        ):
    """
    运行单一算法，给出各种结果:
        内部训练集上:
            P次K折交叉验证的roc曲线+AUC值、交叉验证校正曲线
        内部测试集上：
            roc曲线
            混淆矩阵（３个）
            report
            校正曲线
        #外部数据集上：
        #    roc曲线
        #    混淆矩阵（３个）
        X_train: numpy array, shape (n,m)
        Y_train: numpy array, shape (n,)
        
    """

    dir_result = os.path.join(dir_result,clf_name)
    if not os.path.isdir(dir_result):
        os.mkdir(dir_result)
    #　1、获取训练验证集上的roc,auc
    ret = run_RepeatedKFold(
        n_splits,
        n_repeats,
        classifier = classifier,
        X = X_train,
        Y = Y_train,
        title = 'ROC Curve of '+clf_name+' Classifier \n'+str(n_repeats)+' Times '+str(n_splits)+' Fold Cross Validation',
        dir_result = dir_result
    )
    print('run_RepeatedKFold done!')
    # 画混淆矩阵(内部数据集的测试集)
    classifier.fit(X_train,Y_train)
#     dst = np.sqrt(ret['mean_fpr']**2+(ret['mean_tpr']-1)**2)
#     best_threshold = ret['mean_threshold'][dst==dst.min()][0]
#     y_pred = np.array(classifier.predict_proba(X_test)[:,1]>best_threshold,dtype=np.int)

    # ２、获取测试集上的roc,auc,混淆矩阵
    ret_test = get_algorithm_test_result(
        X = X_test,
        Y = Y_test,
        target_names = target_names,
        classifier_fitted = classifier,
        clf_name = clf_name,
        dataset_name = 'Test Set',
        dir_result = dir_result
        )
    
    # 3、获取report： specificity,precision,recall,f1-score等
    # 1)获取原始report
    import sklearn
#     from sklearn.ensemble import StackingClassifier
#     if classifier.__class__ is sklearn.ensemble._stacking.StackingClassifier:
    try:
        y_pred = np.array(classifier.predict(X_test),dtype=np.int)
    except:
        y_pred = np.array(classifier.predict(X_test.values),dtype=np.int)
    from sklearn.metrics import classification_report
    report = classification_report(
        y_true=Y_test, 
        y_pred=y_pred, 
        labels=np.unique(Y_test).tolist(),
        target_names=target_names,
        digits=3,
        output_dict=True
    )
    report = pd.DataFrame(report)
    ## 2)补充多类别specificity
    multi_class_specificity = get_multi_class_specificity(
        y_true=Y_test,
        y_pred=y_pred,
        target_names=target_names
    )
    report = pd.concat([multi_class_specificity,report], axis=0).round(decimals=3)
    #report = multi_class_specificity.append(report).round(decimals=3)
    ## 3)补充roc_auc
    roc_auc_pd = pd.DataFrame(ret_test['roc_auc'],index=['roc-auc'])
    roc_auc_pd.columns = target_names+['micro avg','macro avg']
    report = pd.concat([roc_auc_pd, report], axis=0)
    #report = roc_auc_pd.append(report).round(decimals=3)
    ## 4)转置
    report.loc['support','micro avg'] = report.loc['support','macro avg']
    report = report.T
    report['support'] = report['support'].astype(int)
    ## 5)保存
    fn = os.path.join(dir_result,'classification report of '+clf_name+' against test set.csv')
    report.to_csv(fn)
    display(report)
    
    # 4、校正图
#     try:
    plot_calibration_curve(
        clf = classifier, 
        clf_name = clf_name, 
        X_train = X_train, 
        y_train = Y_train,
        X_test = X_test, 
        y_test = Y_test,
        target_names = target_names,
        cv = 5,
        dir_result = os.path.join(dir_result,'Test Set'),
        show=False
    )
#     except :
#         print("fail to run plot_calibration_curve")
    
    
    #　结果变量保存
    ret_test['report'] = report
    ret['test'] = ret_test

    # # 画决策边界
    # visualize_reduced_decision_boundary(
    #     clf=LogR,data=Data,
    #     title="Decision-Boundary of Logistic in t-NSE-Projection",
    #     dir_result=dir_result
    # ) 
    
    ##　外部数据集：
    if X_external and Y_external:
        ret_external = get_algorithm_test_result(
            X = X_external,
            Y = Y_external,
            target_names = target_names,
            classifier_fitted = classifier,
            clf_name = clf_name,
            dataset_name = 'External Data Set',
            dir_result = dir_result
        )
        ret['external'] = ret_external

    return ret

def get_multi_class_specificity(
    y_true,
    y_pred,
    target_names):
    """计算多类别的specificity in ovr　manner"""
    from sklearn.metrics import recall_score, accuracy_score
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    y_true_onehot = enc.fit_transform(y_true[:,np.newaxis]).toarray()
    y_pred_onehot = enc.transform(y_pred[:,np.newaxis]).toarray()
    multi_class_specificity = pd.DataFrame(data=0,index=['specificity'],columns=target_names)
    for k in range(len(target_names)):
        y_true_k = y_true_onehot[:,k]
        y_pred_k = y_pred_onehot[:,k]
        spec_recall = recall_score(y_true_k,y_pred_k,average=None)
        # Note that in binary classification, recall of the positive class 
        # is also known as “sensitivity”; recall of the negative class is “specificity”.
        multi_class_specificity.loc['specificity',target_names[k]] = spec_recall[0]   
    
    # macro
    macro = multi_class_specificity.loc['specificity',target_names].mean()
    multi_class_specificity['macro avg'] = macro
    # micro: 特异度的micro平均等价于accuracy
    micro = accuracy_score(y_true,y_pred)
    multi_class_specificity['micro avg'] = micro
    # weighted
    support = [sum(y_true==i) for i in range(len(target_names))]
    weighted = sum(multi_class_specificity.loc['specificity',target_names]*support/sum(support))
    multi_class_specificity['weighted avg'] = weighted
    return multi_class_specificity



from keras import backend as K
import scipy.stats as stats
'''
Compatible with tensorflow backend
'''
def focal_loss(gamma=2., alpha=.25):
    #经过实验，发现loss容易变inf或者NaN
    #参考：https://blog.csdn.net/m0_37477175/article/details/83004746#WCE_Loss_26
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+0.001))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0+0.001))
    return focal_loss_fixed

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed

def run_algorithms(data_internal,
                   data_external = None,
                   clf_names: list = None,
                   feature_select_method: str = 'RFE',
                   kbest: int = 25, 
                   n_splits: int = 5,
                   n_repeats: int = 3,
                   dir_result: str = './'):
    """运行多种算法"""  
    target_names = data_internal['target_names']
    
    # SMOTE数据的扩增以缓和类别不均衡的影像
    smo = SMOTE(sampling_strategy='auto',random_state=random_state)# 实例化
    X_train, Y_train = smo.fit_resample(data_internal['X_train'].values, data_internal['Y_train'].values)
    X_train = pd.DataFrame(data=X_train,columns=data_internal['X_train'].columns)
    Y_train = pd.DataFrame(data=Y_train,columns=data_internal['Y_train'].columns)
    
    
    # 特征选择
    _,feat_imp = select_features(
        X = X_train,
        Y = onehot2label(Y_train),
        target_names=target_names,
        method=feature_select_method,
        dir_result=dir_result,
        kbest=kbest)
    if feat_imp.__class__ == list:
        selected_features_names = feat_imp[0].index.values.tolist()
    elif feat_imp.__class__ == pd.core.frame.DataFrame:
        selected_features_names = feat_imp.index.values.tolist()
          

    # 获取数据，切分数据集
    X_train, Y_train = data_internal['X_train'][selected_features_names], data_internal['Y_train']
    X_test, Y_test = data_internal['X_test'][selected_features_names], data_internal['Y_test']
    if data_external:
        X_external, Y_external = data_external['X'][selected_features_names], data_external['Y']
        Y_external = onehot2label(Y_external.values)
    else:
        X_external, Y_external = None, None
    Y_train = onehot2label(Y_train.values)#由onehot转化回(n_sample,)形式的label
    Y_test = onehot2label(Y_test.values)#由onehot转化回(n_sample,)形式的label
        
    ##　数据降维可视化
    visualize_data_reduced_dimension(
        data={'X':pd.concat([X_train,X_test],axis=0), 
              'Y':np.concatenate((Y_train,Y_test),axis=0),
              'target_names':target_names },
        n_dim = 2,
        title="t-NSE Projection of Internal data selected by "+feature_select_method,
        dir_result=dir_result
    ) 
    if X_external and Y_external:
        visualize_data_reduced_dimension(
            data={'X':X_external, 
                  'Y':Y_external, 
                  'target_names':target_names },
            n_dim = 2,
            title="t-NSE Projection of External data selected by "+feature_select_method,
            dir_result=dir_result
        )  

    # ================================================模型训练和验证=======================================================
    my_scorer = "roc_auc_ovr"# "f1_macro" #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter 
    result_dict = {'selected_features_names': selected_features_names}
    for clf_name in clf_names:
        print(f'Running {clf_name}...')
        # Logistic回归
        if clf_name == 'Logistic':
            from sklearn.linear_model import LogisticRegression as LR
            Logistic_clf = LR(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,
               intercept_scaling=1, random_state=None,solver='lbfgs',#'liblinear',
               max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1)# solver='newton-cg'  'liblinear'  'lbfgs'  'sag' 'saga'

            param_grid ={ 'class_weight' : ['balanced', None] }
            search = GridSearchCV(estimator=Logistic_clf, param_grid=param_grid, scoring=my_scorer)#
            search.fit(X_train, Y_train)
            Logistic_clf = search.best_estimator_
            ret_Logistic = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=Logistic_clf, 
                clf_name = 'Logistic',
                dir_result=dir_result, )
            result_dict['Logistic'] = {'clf': Logistic_clf, 'metric': ret_Logistic}
            
        # LDA==========================
        if clf_name == 'LDA':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            LDA_clf = LinearDiscriminantAnalysis()
            ret_LDA = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=LDA_clf,
                clf_name='LDA',
                dir_result=dir_result)
            result_dict['LDA'] = {'clf': LDA_clf, 'metric': ret_LDA}
    
        # SVM==========================
        if clf_name == 'SVM':
            from sklearn.svm import SVC
            SVM_clf = SVC(decision_function_shape='ovr',probability=True)
            param_grid ={ 'kernel' : ['rbf', 'sigmoid'], 'class_weight' : ['balanced', None] }#'linear','rbf', 'poly', 'sigmoid'
            search = GridSearchCV(estimator=SVM_clf, param_grid=param_grid, scoring=my_scorer)#"f1_macro"
            search.fit(X_train, Y_train)
            SVM_clf = search.best_estimator_
            ret_SVM = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=SVM_clf, 
                clf_name='SVM',
                dir_result=dir_result, )
            result_dict['SVM'] = {'clf': SVM_clf, 'metric': ret_SVM}
    
        ## KNN分类器==========================
        if clf_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            KNN_clf = KNeighborsClassifier(metric="minkowski")
            param_grid = {'n_neighbors':[2,5,10], 'weights': ['uniform', 'distance'], 'p': [1,2]}
            search = GridSearchCV(estimator=KNN_clf, param_grid=param_grid, scoring=my_scorer)#"f1_macro"
            search.fit(X_train, Y_train)
            KNN_clf = search.best_estimator_
            ret_KNN = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=KNN_clf, 
                clf_name='KNN',
                dir_result=dir_result)
            result_dict['KNN'] = {'clf': KNN_clf, 'metric': ret_KNN}

        # GaussianNB型朴素贝叶斯分类器
        if clf_name == 'GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            GaussianNB_clf = GaussianNB()
            ret_GaussianNB = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=GaussianNB_clf, 
                clf_name='GaussianNB',
                dir_result=dir_result)
            result_dict['GaussianNB'] = {'clf': GaussianNB_clf, 'metric': ret_GaussianNB}
    
        # 决策树==========================
        if clf_name == 'Tree':
            from sklearn.tree import DecisionTreeClassifier
            Tree_clf = DecisionTreeClassifier()
            param_grid = {'max_depth': [5, 10, 20],
                            'min_samples_leaf': [2,4,8,16] ,
                            'min_samples_split': [2,4,8,16],
                            'class_weight' : ["balanced",None]
                            }
            search = GridSearchCV(estimator=Tree_clf, param_grid=param_grid, scoring=my_scorer)#"f1_macro"
            search.fit(X_train, Y_train)
            Tree_clf = search.best_estimator_
            ret_Tree = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=Tree_clf, 
                clf_name='Tree',
                dir_result=dir_result,)
            result_dict['Tree'] = {'clf': Tree_clf, 'metric': ret_Tree}
    
    
        xk = np.logspace(1,2,4,base=10).astype(int)
        pk = np.ones(xk.shape)/len(xk)
        n_estimators = stats.rv_discrete(name='custm', values=(xk, pk))
        # ExtraTrees==========================
        if clf_name == 'ExtraTrees':
            from sklearn.ensemble import ExtraTreesClassifier
            ExtraTrees_clf = ExtraTreesClassifier()
            param_grid = {'n_estimators' : xk, 'class_weight': ['balanced', None]}#[10,100,1000]
            search = GridSearchCV(estimator=ExtraTrees_clf, param_grid=param_grid, scoring=my_scorer)#"f1_macro"
            search.fit(X_train, Y_train)
            ExtraTrees_clf = search.best_estimator_
            ret_ExtraTrees = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=ExtraTrees_clf, 
                clf_name='ExtraTrees',
                dir_result=dir_result)
            result_dict['ExtraTrees'] = {'clf': ExtraTrees_clf, 'metric': ret_ExtraTrees}
    
    
        ## 随机森林========================
        if clf_name == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            RandomForest_clf = RandomForestClassifier(bootstrap=True, criterion='gini', random_state=0, verbose=0, warm_start=False)
            param_grid = dict(
                n_estimators = n_estimators,#[10, 50, 100]
                class_weight = ['balanced', None],
            )
            clf = RandomizedSearchCV(estimator=RandomForest_clf, 
                                     param_distributions=param_grid,
                                     n_iter=50,n_jobs = -1,
                                     scoring=my_scorer)#"f1_macro"
            search = clf.fit(X_train, Y_train)
            search.best_params_
            RandomForest_clf = search.best_estimator_
            ret_RandomForest = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=RandomForest_clf, 
                clf_name= 'RandomForest',
                dir_result=dir_result)
            result_dict['RandomForest'] = {'clf': RandomForest_clf, 'metric': ret_RandomForest}
    
        # Bagging========================
        if clf_name == 'Bagging':
            from sklearn.ensemble import BaggingClassifier
            Bagging_clf = BaggingClassifier()
            param_grid = {'estimator': [Logistic_clf,
                                             SVM_clf,
                                             Tree_clf],
                          'n_estimators' : xk,#[10,100,200]#,1000
                         }
            search  = GridSearchCV(estimator=Bagging_clf, param_grid=param_grid, scoring=my_scorer)#"f1_macro"
            search = search.fit(X_train, Y_train)
            Bagging_clf = search.best_estimator_
            ret_Bagging = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=Bagging_clf, 
                clf_name='Bagging',
                dir_result=dir_result)
            result_dict['Bagging'] = {'clf': Bagging_clf, 'metric': ret_Bagging}
    
        ## Adaboost算法========================
        if clf_name == 'AdaBoost':
            from sklearn.ensemble import AdaBoostClassifier
            AdaBoost_clf = AdaBoostClassifier(algorithm='SAMME.R') #base_estimator默认是决策树（深度为1），可修改
            param_dict = dict(estimator = [Logistic_clf, SVM_clf, Tree_clf], n_estimators = xk)
            search = GridSearchCV(estimator=AdaBoost_clf, param_grid=param_dict, scoring=my_scorer)
        #     search = RandomizedSearchCV(estimator=AdaBoost_clf, 
        #                                 param_distributions=param_dict, 
        #                                 n_iter=100,n_jobs = -1,
        #                                 scoring=my_scorer)
            search.fit(X_train, Y_train)
            AdaBoost_clf = search.best_estimator_
            ret_AdaBoost = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=AdaBoost_clf, 
                clf_name='AdaBoost',
                dir_result=dir_result,)
            result_dict['AdaBoost'] = {'clf': AdaBoost_clf, 'metric': ret_AdaBoost}
    
        # GBDT========================
        if clf_name == 'GBDT':
            from sklearn.ensemble import GradientBoostingClassifier
            GBDT_clf = GradientBoostingClassifier()
            n_estimators = np.logspace(1,2,4,base=10).astype(int)
            param_dict = dict( n_estimators = n_estimators, )
            search = GridSearchCV(
                estimator=GBDT_clf, 
                param_grid=param_dict, 
                n_jobs = -1,
                scoring=my_scorer)#"f1_macro"
            search = search.fit(X_train, Y_train)
            GBDT_clf = search.best_estimator_
            ret_GBDT = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=GBDT_clf, 
                clf_name='GBDT',
                dir_result=dir_result,)
            result_dict['GBDT'] = {'clf': GBDT_clf, 'metric': ret_GBDT}

    
        # XGBoost========================
        if clf_name == 'XGBoost':
            import xgboost
            param_dict = dict(learning_rate = [0.01,0.05,0.1],
                              booster = ['gbtree', 'dart'],
        #                       max_depth = [0,6],          # 树的深度
        #                       gamma=[0, 0.1, 0.01, 0.001],# 惩罚项的参数
        #                       reg_alpha = [0.1],
        #                       reg_lambda = [0.2],
        #                       subsample = [0.8],           # 随机选择80%样本建立决策树
        #                       colsample_bytree = [0.7],    # 随机选择70%特征建立决策树
        #                       min_child_weight = [1],      # 叶子节点最小权重
        #                       objective = ['multi:softproba'],#['multi:softmax'],
        #                       eval_metric = ['mlogloss'],
                              n_estimators = xk,# 树的个数 #复用GBDT的参数
                             )
            #use_label_encoder=False,
            XGBoost_clf = xgboost.XGBClassifier(objective='multi:softproba', tree_method='gpu_hist', gpu_id=0)
            search = GridSearchCV(estimator=XGBoost_clf, param_grid=param_dict, scoring=my_scorer)#"f1_macro"
        #     search = RandomizedSearchCV(estimator=XGBoost_clf, 
        #                                 param_distributions=param_dict, 
        #                                 n_iter=50,n_jobs = -1,
        #                                 scoring=my_scorer)
            search.fit(X_train, Y_train)
            XGBoost_clf = search.best_estimator_
            ret_XGBoost = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=XGBoost_clf, 
                clf_name='XGBoost',
                dir_result=dir_result)
            result_dict['XGBoost'] = {'clf': XGBoost_clf, 'metric': ret_XGBoost}
    
    
        # lightgbm========================
        if clf_name == 'lightGBM':
            import lightgbm as lgb
            xk = np.logspace(2,6,5,base=2).astype(int)
            pk = np.ones(xk.shape)/len(xk)
            num_leaves = stats.rv_discrete(name='custm', values=(xk, pk))
            param_dict = dict(learning_rate = [0.01,0.1],
                              max_bin = [150],
                              num_leaves = num_leaves,#[4,8,16,32,64], 
                              max_depth = [-1],
                              reg_alpha = [0.1],
                              reg_lambda = [0.2],
                              n_estimators = n_estimators,#复用GBDT的参数
                             )
            lightGBM_clf = lgb.LGBMClassifier(objective = 'multiclass',class_weight = 'balanced')
            #search = GridSearchCV(estimator=lightGBM_clf, param_grid=param_dict, scoring=my_scorer)#"f1_macro"
            search = RandomizedSearchCV(estimator=lightGBM_clf, 
                                     param_distributions=param_dict, 
                                     n_iter=50,n_jobs = -1,
                                     scoring=my_scorer)#"f1_macro"
            search.fit(X_train, Y_train)
            lightGBM_clf = search.best_estimator_
            ret_lightGBM = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=lightGBM_clf, 
                clf_name='lightGBM',
                dir_result=dir_result)
            result_dict['lightGBM'] = {'clf': lightGBM_clf, 'metric': ret_lightGBM}  

     
        # MLP========================
        if clf_name == 'MLP':
            from sklearn.neural_network import MLPClassifier
            MLP_clf = MLPClassifier(solver='lbfgs', max_iter=3000, early_stopping=True, random_state=random_state)
            param_grid = {'activation' : ['identity', 'logistic', 'relu'],
                          'learning_rate' : ['constant', 'invscaling', 'adaptive'],
                          'hidden_layer_sizes': [(32,32),(64,64),(128,128)]
                         }
            search = GridSearchCV(estimator=MLP_clf, param_grid=param_grid, scoring=my_scorer)#"f1_macro"
            search.fit(X_train, Y_train)
            MLP_clf = search.best_estimator_
            ret_MLP = get_algorithm_result(
                X_train,Y_train,
                X_test,Y_test,
                X_external,Y_external,
                target_names,
                n_splits,n_repeats,
                classifier=MLP_clf, 
                clf_name='MLP',
                dir_result=dir_result)
            result_dict['MLP'] = {'clf': MLP_clf, 'metric': ret_MLP}
    #各分类算法的最佳模型对象


    ##　整理输出结果
    clf_names = clf_names
    
    aucs_RepeatedKFold = dict([])
    aucs_RepeatedKFold['micro'] = np.array( [result_dict[clf_name]['metric']['cv_aucs']['micro'] for clf_name in clf_names] )  
    aucs_RepeatedKFold['macro'] = np.array( [result_dict[clf_name]['metric']['cv_aucs']['macro'] for clf_name in clf_names] )
    
    aucs_testset = dict([]) 
    aucs_testset['micro'] = np.array( [result_dict[clf_name]['metric']['test']['roc_auc']['micro'] for clf_name in clf_names] )
    aucs_testset['macro'] = np.array( [result_dict[clf_name]['metric']['test']['roc_auc']['macro'] for clf_name in clf_names] )
    
    if data_external:
        aucs_external = np.array( [result_dict[clf_name]['metric']['external']['auc'] for clf_name in clf_names] )
    else:
        aucs_external = None


    aucs_RepeatedKFold_df = {'micro':[],'macro':[]}
    aucs_RepeatedKFold_df['micro'] = pd.DataFrame(
        data=aucs_RepeatedKFold['micro'],
        index=clf_names,
        columns=np.arange(aucs_RepeatedKFold['micro'].shape[1]),)
    aucs_RepeatedKFold_df['macro'] = pd.DataFrame(
        data=aucs_RepeatedKFold['macro'],
        index=clf_names,
        columns=np.arange(aucs_RepeatedKFold['macro'].shape[1]),)
    
    aucs_testset_df = {'micro':[],'macro':[]}
    aucs_testset_df['micro'] = pd.DataFrame(
        data=aucs_testset['micro'],
        index=clf_names,
        columns=[feature_select_method],)
    aucs_testset_df['macro'] = pd.DataFrame(
        data=aucs_testset['macro'],
        index=clf_names,
        columns=[feature_select_method],)
    
    if data_external:
        aucs_external_df = pd.DataFrame(
            data=aucs_external,
            index=clf_names,
            columns=[feature_select_method],)        
    else:
        aucs_external_df = None
    
    fg = violinplot( 
        x = [result_dict[clf_name]['metric']['cv_aucs']['macro'] for clf_name in clf_names],
        x_names = clf_names,
        title='macro-AUC of Cross Validation of Different Algorithms on Training Set',
        fn_save=os.path.join(dir_result,'macro-AUC of Cross Validation of Different Algorithms on Training Set.png'), )
    fg = violinplot(
        x = [result_dict[clf_name]['metric']['cv_aucs']['micro'] for clf_name in clf_names],
        x_names=clf_names,
        title='micro-AUC of Cross Validation of Different Algorithms on Training Set',
        fn_save=os.path.join(dir_result,'micro-AUC of Cross Validation of Different Algorithms on Training Set.png'), )
    
    return result_dict, aucs_RepeatedKFold_df, aucs_testset_df, aucs_external_df

# 画图
def plot_mean_aucs_matirx(mean_aucs_matrix, fmt='.3g',center=0.5,
                          title='mean AUC Matrix',
                          fn='mean AUC Matrix.png'
                         ):
    """画平均auc矩阵：横轴是机器学习分类算法，纵轴是特征选择算法，格子内的数值是mean_auc"""

    cm = mean_aucs_matrix.values
    algorithms = mean_aucs_matrix.columns.tolist()
    feature_selection_methods = mean_aucs_matrix.index.tolist()
    
    f,ax = plt.subplots(dpi=100)
    ax = sns.heatmap(cm,annot=True,fmt=fmt,center=center,annot_kws={'size':6,'ha':'center','va':'center'})
    ax.set_title(title,fontsize=10)#图片标题文本和字体大小
    ax.set_xlabel('Classifier',fontsize=10)#x轴label的文本和字体大小
    ax.set_ylabel('Feature Selection Method',fontsize=10)#y轴label的文本和字体大小
    ax.set_xticklabels(algorithms,fontsize=10, rotation='vertical')#x轴刻度的文本和字体大小
#     plt.xticks([y+1 for y in range(len(x))], x_names, rotation='vertical')
    ax.set_yticklabels(feature_selection_methods,fontsize=10, rotation='horizontal')#y轴刻度的文本和字体大小
    #设置colorbar的刻度字体大小
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=10)
    
    plt.savefig(fn, bbox_inches='tight')
    plt.show()
    return f,ax


# +
import pycm
from scipy.stats import norm
from scipy import stats
from sklearn.metrics import roc_auc_score

def delong_test(y_true, y_scores_model1, y_scores_model2):
    '''
    对两个不同的 3 分类模型在同义测试集上的 ROC 曲线进行比较（DeLong检验）。
    
    Parameters：
    y_true: 真实标签（one-hot编码格式）
    y_scores_model1: 第一个模型的概率预测，shape为：(样本数，3)
    y_scores_model2: 第二个模型的概率预测，shape为：(样本数，3)

    Returns：
    p_value: 显著性检验 p 值
    auc1: 第一个模型的 AUC 值 
    auc2: 第二个模型的 AUC 值 
    auc_diff: 两个 AUC 值之间的差异
    '''
    # 计算每个类别的AUC值
    n_classes = y_true.shape[1]
    auc_list1 = []
    auc_list2 = []
    for i in range(n_classes):
        auc_i_1 = roc_auc_score(y_true[:, i], y_scores_model1[:, i])
        auc_i_2 = roc_auc_score(y_true[:, i], y_scores_model2[:, i])
        auc_list1.append(auc_i_1)
        auc_list2.append(auc_i_2)

    # 计算micro平均的AUC值
    auc1 = np.mean(auc_list1)
    auc2 = np.mean(auc_list2)

    # 计算标准误差和z值
    auc_diff = auc1 - auc2
    var1 = np.var(auc_list1)
    var2 = np.var(auc_list2)
    n1 = y_scores_model1.shape[0]
    n2 = y_scores_model2.shape[0]
    s = np.sqrt(var1/n1 + var2/n2)
    z_score = auc_diff / s

    # 计算显著性检验p值
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return p_value, auc1, auc2, auc_diff



# +
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_curve(y_true, y_pred_prob, class_names):
    '''
    生成三分类预测模型测试集临床决策曲线
    :param y_true: 测试集真实标签
    :param y_pred_prob: 测试集各类别的预测概率
    :param class_names: 标签类别名称
    '''
    n = len(y_true)

    # 计算各类别的正样本比例
    pos_frac = [np.mean(y_true == i) for i in range(len(class_names))]

    print(pos_frac)
    # 计算CE各类别阈值和效益
    thresh = np.linspace(0, 1, 100)
    ce = []
    ap = []
    an = []
    for t in thresh:
        ce_t = 0
        ap_t = 0
        an_t = 0
        for k in range(len(class_names)):
            y_k = (y_true == k)
            pred_k = (y_pred_prob[:, k] > t).astype(int)
            tp = np.dot(y_k, pred_k)
            fp = np.sum((pred_k == 1) & (y_k == 0))
            fn = np.sum((pred_k == 0) & (y_k == 1))
#             ce_t += 1/len(class_names) * 1/n*(tp-fp*t/(1-t))
            ce_t += pos_frac[k] * 1/n*(tp-fp*t/(1-t))
            pr = (fn+fp)/n
            ap_t += pos_frac[k] * (pr-(1-pr)*t/(1-t))
            an_t += 0.0
        ce.append(ce_t)
        ap.append(ap_t)
        an.append(an_t)

    # 绘制临床决策曲线
    fig, ax = plt.subplots(1,1,dpi=300)
    ax.plot(thresh, ce, label='overall DCA')
    ax.plot(thresh, ap, ':', label='overall all positive')
    ax.plot(thresh, an, ':', label='overall all negative')
    ax.set_xlabel('thresh')
    ax.set_ylabel('CE')
    ax.set_title('Clinical Decision Curve')
    ax.legend()
    ax.set_ylim(-0.05,max([max(ce),max(ap),max(an)]))
    plt.show()
    return 
    
    
def get_decision_curve(y_true, y_pred_prob, class_names, pos_frac=None):
    '''
    生成三分类预测模型测试集临床决策曲线
    :param y_true: 测试集真实标签
    :param y_pred_prob: 测试集各类别的预测概率
    :param class_names: 标签类别名称
    '''
    n = len(y_true)

    if pos_frac is None:
        # 计算各类别的正样本比例
        pos_frac = [np.mean(y_true == i) for i in range(len(class_names))]

    print(pos_frac)
    # 计算CE各类别阈值和效益
    thresh = np.linspace(0, 1, 100)
    ce = []
    ap = []
    an = []
    for t in thresh:
        ce_t = 0
        ap_t = 0
        an_t = 0
        for k in range(len(class_names)):
            y_k = (y_true == k)
            pred_k = (y_pred_prob[:, k] > t).astype(int)
            tp = np.dot(y_k, pred_k)
            fp = np.sum((pred_k == 1) & (y_k == 0))
            fn = np.sum((pred_k == 0) & (y_k == 1))
#             ce_t += 1/len(class_names) * 1/n*(tp-fp*t/(1-t))
            ce_t += pos_frac[k] * 1/n*(tp-fp*t/(1-t))
            pr = (fn+fp)/n
#             ap_t += 1/len(class_names)  * (pr-(1-pr)*t/(1-t))
            ap_t += pos_frac[k] * (pr-(1-pr)*t/(1-t))
            an_t += 0.0
        ce.append(ce_t)
        ap.append(ap_t)
        an.append(an_t)

    return thresh, ce, ap, an
# -


