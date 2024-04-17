import pandas as pd
import numpy as np
from numpy import *
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from datetime import datetime, timedelta
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore")
import operator
import time
import pdb
import glob
from to import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import lightgbm as lgb



def statistical_model(datas, returns, marketdata, savepath, method, methodreturn):
    global dff
    if methodreturn == 'SZ':
        returns = getexcessreturns(returns, 'SZ')
    elif methodreturn == 'SH':
        returns = getexcessreturns(returns, 'SH')
    else:
        returns = getreturns(returns, methodreturn)
    returns['tmrt'] = returns.groupby('code')['return'].shift(-1, axis=0)
    returns = returns.loc[:, ['code', 'date', 'tmrt', 'return']]
    bigdata = pd.merge(datas, returns, on=['code', 'date'], how='left')
    bigdata = pd.merge(bigdata, marketdata, on=['code', 'date'], how='left')
    bigdata = bigdata[~bigdata['tmrt'].isnull()]
    bigdata = bigdata[~bigdata['circulated_market_value'].isnull()]
    bigdata = bigdata.dropna()
    date = bigdata.loc[:, 'date'].unique()
    date = [f for f in date]
    k = len(date)
    df = pd.DataFrame(columns=['date', 'code', 'tmrt', 'predict', 'residual'])
    df.to_csv(savepath + method + methodreturn + '.csv', index=False)
    #### The data is really large. 先把每年的预测数据加在一个DATAFRAME.然后在结尾写进Excel。
    for i in range(k - 1):
        data = bigdata[(bigdata.date == date[i]) | (bigdata.date == date[i + 1])]
        ###today's factors & tomorrow's in dataframe form
        today = data[data.date == date[i]]
        X, y = data[data.date == date[i]].iloc[:, 2:-3].values, data[data.date == date[i]].loc[:, 'tmrt'].values
        ### X: factors y: tomorrow return
        Xtmw, ytmw = data[data.date == date[i + 1]].iloc[:, 2:-3].values, data[data.date == date[i + 1]].loc[:, 'tmrt'].values
        ### tomorrow's X, tomorrow's return
        if method == 'RF':
            start = time.time()
            model = RandomForestRegressor(n_estimators=500, max_depth=3, max_leaf_nodes=6, random_state=0)
            model.fit(X, y)
            yhat = model.predict(Xtmw)
            end = time.time()
            print(end - start)
        if method == 'OLS':
            model = sm.OLS(y, X)
            result = model.fit()
            params = result.params
            #pdb.set_trace()
            yhat = np.dot(Xtmw, params)
        if method == 'LASSO':
            #model = LassoLarsCV(cv=20).fit(X, y)
            model = linear_model.Lasso(fit_intercept = False)
            model.fit(X, y)
            yhat = model.predict(Xtmw)
        if method == 'ENET':
            #model = ElasticNetCV(cv=20, random_state=0)
            model = linear_model.ElasticNet(fit_intercept = False)
            model.fit(X, y)
            yhat = model.predict(Xtmw)
        if method == 'PLS':
            model = PLSRegression(n_components=2)
            model.fit(X, y)
            yhat = model.predict(Xtmw)
        if method == 'PCR':
            #model = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
            #print(model1.explained_variance_ratio_)
            model1 = PCA(n_components = 2)
            model1.fit(X)
            coefficient1 = model1.components_.dot(X.T)
            model = linear_model.LinearRegression()
            model.fit(coefficient1.T, y)
            coefficient = model.coef_ / 100
            coefficient2 = model1.components_.dot(Xtmw.T)
            yhat = coefficient2.T.dot(coefficient)
        if method == 'GLM':
            model = linear_model.TweedieRegressor(fit_intercept = False)
            model.fit(X, y)
            coefficient = model.coef_[:10] / 100
            yhat = model.predict(Xtmw)
        if method == 'WLS':
            weight = np.sqrt(today['circulated_market_value'])
            weight = weight.values / sum(weight)
            weight = pd.DataFrame(np.diag(weight))
            Xtmw = sm.add_constant(Xtmw)
            X = sm.add_constant(X)
            industry_exp = today.iloc[:, 12:40].transpose()
            total_mkt = sum(today['circulated_market_value'])
            industry_weight = industry_exp.dot(today['circulated_market_value'])
            industry_weight = industry_weight / total_mkt
            industry_weight = -industry_weight.values[:-1] / industry_weight[-1]
            ss = X.shape[1]
            diag_R = np.diag(np.ones(ss))
            R = np.delete(diag_R, -1, axis=1)
            R[-1, -27:] = industry_weight
            W = R.dot(np.linalg.inv(R.T.dot(X.T).dot(weight).dot(X).dot(R))).dot(R.T).dot(X.T).dot(weight)
            W = pd.DataFrame(W)
            f = W.dot(y).to_frame().T
            yhat = np.array(f.dot(Xtmw.T))[0]
        dff = data[data.date == date[i + 1]].loc[:, ['date', 'code', 'tmrt']]
        dff['predict'] = yhat
        dff['residual'] = dff['tmrt'] - dff['predict']
        if (i<k-2) & ((date[i] // 10000) == (date[i + 1] // 10000)):
            df = df.append(dff)
        elif i == k - 2:
            df = df.append(dff)
            df.to_csv(savepath + method + methodreturn + '.csv', mode='a', header=False, index=False)
            break
        else:
            df.to_csv(savepath + method + methodreturn + '.csv', mode='a', header=False, index=False)
            df = dff
    return df

def GDBT(datas, marketdata, returns, savepath):
    returns = getexcessreturns(returns, 'SZ')
    returns['tmrt'] = returns.groupby('code')['return'].shift(-1, axis=0)
    returns = returns.rename({'tradeday': 'date'}, axis=1)
    marketdata = marketdata.rename({'tradeday': 'date'}, axis=1)
    returns = returns.loc[:, ['code', 'date', 'tmrt', 'return']]
    bigdata = pd.merge(datas, returns, on=['code', 'date'], how='left')
    bigdata = pd.merge(bigdata, marketdata, on=['code', 'date'], how='left')
    bigdata = bigdata[~bigdata['tmrt'].isnull()]
    bigdata = bigdata[~bigdata['circulated_market_value'].isnull()]
    bigdata = bigdata.dropna(axis=0).reset_index(drop=True)
    tempdata = bigdata.copy()
    tempdata = tempdata.drop(columns='code')
    training = tempdata[tempdata.date <= 20201230]
    test = tempdata[tempdata.date > 20201231]
    # training = tempdata[tempdata.date == 20170901]
    # test = tempdata[tempdata.date == 20170901]
    dates = training.date.unique()
    parameter_grid = {'n_estimators': np.arange(10, 100, 20), 'max_features': [1, 2, 3, 4]}
    cross_validation = StratifiedKFold(n_splits=10)
    random_forest_classifier = RandomForestClassifier(n_estimators=50)
    random_forest_regressor = RandomForestRegressor(n_estimators=50)
    grid_search = GridSearchCV(random_forest_regressor, param_grid=parameter_grid, cv=cross_validation)
    dff = pd.DataFrame()
    for i in dates:
        data = bigdata[(bigdata.date == i)]
        X, y = data.iloc[:, 2:-3].values, data.loc[:, 'tmrt'].values
        mean = np.mean(y)
        std_dev = np.std(y)
        Var_35 = norm.ppf(1 - 0.35, mean, std_dev)
        Var_75 = norm.ppf(1 - 0.75, mean, std_dev)
        tempp = np.where(data['tmrt'] > Var_35, 1, np.where((data['tmrt'] > Var_75) & (data['tmrt'] <= Var_35), 0, -1))
        dff = dff.append(pd.DataFrame(tempp))
    # data = bigdata[(bigdata.date == 20170901)]
    # X, y = data.iloc[:, 2:-3].values, data.loc[:, 'tmrt'].values
    # mean = np.mean(y)
    # std_dev = np.std(y)
    # Var_35 = norm.ppf(1-0.35, mean, std_dev)
    # Var_75 = norm.ppf(1-0.75, mean, std_dev)
    # Var_50 = norm.ppf(0.5, mean, std_dev)
    # tempp = np.where(data['tmrt'] > Var_35, 1, np.where((data['tmrt'] > Var_75) & (data['tmrt'] <= Var_35), 0, -1))
    # #tempp = np.where(data['tmrt'] > Var_50, 1, -1)
    # dff = pd.DataFrame(tempp)
    X, y = training.loc[:, ~training.columns.isin(
        ['tmrt', 'date', 'circulated_market_value', 'codenum', 'return', 'target',
         'codez'])].values, dff.values.flatten()
    test_X = test.loc[:,
             ~test.columns.isin(['tmrt', 'return', 'date', 'circulated_market_value', 'codenum', 'codez'])].values
    # random_forest_classifier.fit(X, y)
    # yhat = random_forest_classifier.predict(test_X)
    # #get importance
    # importance = random_forest_classifier.feature_importances_
    # #summarize feature importance
    # for i,v in enumerate(importance):
    #     print('Classifier Feature: %0d, Score: %.5f' % (i,v))
    # #plot feature importance
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.show()
    # pdb.set_trace()
    #
    # random_forest_regressor.fit(X, training.loc[:, 'tmrt'])
    # ybar = random_forest_regressor.predict(test_X)
    # importance = random_forest_classifier.feature_importances_
    # # summarize feature importance
    # for i, v in enumerate(importance):
    #     print('Regressor Feature: %0d, Score: %.5f' % (i, v))
    # # plot feature importance
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.show()
    model = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=2000, num_leaves=2000, reg_alpha=0, reg_lambda=0,
                               max_depth=100, colsample_bytree=0.8, bagging_fraction=0.7, bagging_freq=1,
                               learning_rate=1, min_child_samples=500, random_state=2021, n_jobs=-1,
                               importance_type='gain')
    model.fit(X, y)
    predicted_target = model.predict(test_X)
    predicted_prob = model.predict_proba(test_X)
    model = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=2000, reg_alpha=0, reg_lambda=0, max_depth=100,
                              n_estimators=100, colsample_bytree=0.8, bagging_fraction=0.7, subsample=0.8,
                              bagging_freq=1, learning_rate=0.05, min_child_samples=500, random_state=2021, n_jobs=-1,
                              importance_type='gain')
    model.fit(X, training.loc[:, 'tmrt'])
    predicted_value = model.predict(test_X)
    df = pd.DataFrame()
    df['code'] = bigdata[bigdata.date > 20201231]['code'].reset_index(drop=True)
    df['date'] = bigdata[bigdata.date > 20201231]['date'].reset_index(drop=True)
    df['predicted_target'] = predicted_target
    df['predicted_value'] = predicted_value
    df[['-1', '0', '1']] = predicted_prob
    df.to_csv(savepath + 'GDBTpredicted.csv', index=False)

def GDBTresidual():
    WLSreturn = pd.read_csv(savepath + 'WLS' + 'SZ.csv')
    bigdata = pd.read_csv(savepath + 'bigdata.csv')
    dataframe = bigdata.merge(WLSreturn.loc[:, ['date', 'code', 'residual']], how='left', on=['date', 'code'])
    dataframe = dataframe.dropna()
    training = dataframe[dataframe.date <= 20201230].drop(columns=['code'])
    test = dataframe[dataframe.date > 20201231].drop(columns=['code'])
    # training = tempdata[tempdata.date == 20170901]
    # test = tempdata[tempdata.date == 20170901]
    dates = training.date.unique()
    dff = pd.DataFrame()
    colname = ['BETA', 'MOMENTUM', 'SIZE', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY', 'SIZENL']
    for i in dates:
        data = dataframe[(dataframe.date == i)]
        X, y = data.loc[:, data.columns.isin(colname)].values, data.loc[:, 'residual'].values
        mean = np.mean(y)
        std_dev = np.std(y)
        Var_35 = norm.ppf(1 - 0.35, mean, std_dev)
        Var_75 = norm.ppf(1 - 0.75, mean, std_dev)
        # pdb.set_trace()
        tempp = np.where(data['residual'] > Var_35, 1,
                         np.where((data['residual'] > Var_75) & (data['residual'] <= Var_35), 0, -1))
        dff = dff.append(pd.DataFrame(tempp))
    # data = bigdata[(bigdata.date == 20170901)]
    # X, y = data.iloc[:, 2:-3].values, data.loc[:, 'tmrt'].values
    # mean = np.mean(y)
    # std_dev = np.std(y)
    # Var_35 = norm.ppf(1-0.35, mean, std_dev)
    # Var_75 = norm.ppf(1-0.75, mean, std_dev)
    # Var_50 = norm.ppf(0.5, mean, std_dev)
    # tempp = np.where(data['tmrt'] > Var_35, 1, np.where((data['tmrt'] > Var_75) & (data['tmrt'] <= Var_35), 0, -1))
    # #tempp = np.where(data['tmrt'] > Var_50, 1, -1)
    # dff = pd.DataFrame(tempp)
    X, y = training.loc[:, ~training.columns.isin(['date', 'residual'])].values, dff.values.flatten()
    test_X = test.loc[:, ~test.columns.isin(['date', 'residual'])].values
    # model = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=2000, num_leaves=2000, reg_alpha=0, reg_lambda=0,
    #                            max_depth=100, colsample_bytree=0.8, bagging_fraction=0.7, bagging_freq=1,
    #                            learning_rate=1, min_child_samples=500, random_state=2021, n_jobs=-1,
    #                            importance_type='gain')
    # model.fit(X, y)
    # predicted_target = model.predict(test_X)
    # predicted_prob = model.predict_proba(test_X)
    model = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=2000, reg_alpha=0, reg_lambda=0, max_depth=100,
                              n_estimators=100, colsample_bytree=0.8, bagging_fraction=0.7, subsample=0.8,
                              bagging_freq=1, learning_rate=0.05, min_child_samples=500, random_state=2021, n_jobs=-1,
                              importance_type='gain')
    model.fit(X, training.loc[:, 'residual'].values)
    predicted_value = model.predict(test_X)
    # pdb.set_trace()
    df = pd.DataFrame()
    df['date'] = dataframe[dataframe.date > 20201231]['date'].reset_index(drop=True)
    df['code'] = dataframe[dataframe.date > 20201231]['code'].reset_index(drop=True)
    #df['predicted_target'] = predicted_target
    df['predicted_residual'] = predicted_value
    df=df.merge(WLSreturn.loc[:, ['date', 'code', 'predict']], on = ['code', 'date'], how = 'left')
    df['predicted_WLS'] = df['predicted_residual'] + df['predict']
    #df[['-1', '0', '1']] = predicted_prob
    df.to_csv(savepath + 'GDBTresidualSZpredicted.csv', index=False)
    return df

def GDBTplt(returns, bigdata, savepath, method, methodreturn, methodcut):
    #'methodcut' = 'predicted_value' or '1' or 'probpred'
    #'method' = 'GDBT' or 'GDBTresidual'
    if method == 'GDBT':
        df = pd.read_csv(savepath + 'GDBTpredicted.csv').reset_index(drop=True)
    elif method == 'GDBTresidual':
        df = pd.read_csv(savepath + 'GDBTresidualSZpredicted.csv').reset_index(drop=True)
    if methodcut == 'predicted_value':
        def _corr(x):
            return pd.qcut(x['predicted_value'], q=10, labels=False)
    if methodcut=='predicted_WLS':
        def _corr(x):
            return pd.qcut(x['predicted_WLS'], q=10, labels=False)
    if methodcut == '1':
        def _corr(x):
            return pd.qcut(x['1'].rank(method='first'), q=10, labels=False)
    if methodcut == 'probpred':
        df['probpred'] = df['predicted_value'] * df['1']
        def _corr(x):
            return pd.qcut(x['probpred'].rank(method='first'), q=10, labels=False)
    #     return stats.spearmanr(x['predicted_value'],x[str(x['predicted_target'].values[0])])[0]

    temp = df.groupby(['date']).apply(lambda x: _corr(x)).reset_index(drop=False)
    temp = temp.rename(columns = {methodcut: 'qtile'})
    if methodreturn == 'SZ':
        returns = getexcessreturns(returns, 'SZ')
    elif methodreturn == 'SH':
        returns = getexcessreturns(returns, 'SH')
    else:
        returns = getreturns(returns, methodreturn)
    df=df.merge(temp.loc[:, ['level_1', 'qtile']], how = 'left', left_index=True, right_on='level_1').drop(columns = 'level_1')
    returns['tmrt'] = returns.groupby('code')['return'].shift(-1, axis=0)
    returns = returns.rename({'tradeday': 'date'}, axis=1)
    returns = returns.loc[:, ['code', 'date', 'tmrt', 'return']]
    bigdata = pd.merge(bigdata, returns, on=['code', 'date'], how='left')
    data = df.loc[:, ['code', 'date','qtile']].merge(bigdata.loc[:, ['code', 'date', 'tmrt']], on = ['code', 'date'], how = 'left')
    data = data.drop(columns = 'code')
    data = data.groupby(['date', 'qtile'])['tmrt'].mean().reset_index(drop=False)
    data = data.pivot_table(values = 'tmrt', index = data.date, columns = 'qtile').reset_index(drop=False)
    data.columns=['date', str(9), str(8), str(7), str(6), str(5), str(4), str(3), str(2), str(1), str(0)]
    data.to_csv(savepath + 'GDBT' + method + methodcut + methodreturn + 'decile.csv', index=False)
    deciplt(data, savepath, method, methodcut)

def GDBTaccuracy():
    df = pd.read_csv(savepath + 'GDBTpredicted.csv')
    WLSreturn = pd.read_csv(savepath + 'WLS' + 'SZ.csv')
    dataframe = df.loc[:, ['code', 'date', 'predicted_target']].merge(WLSreturn.loc[:, ['date', 'code', 'tmrt']], on=['code', 'date'], how='left')
    def func(x):
        if x > 0:
            return 1
        if x <= 0:
            return -1
    dataframe['actual_target'] = dataframe['tmrt'].apply(lambda x: func(x))
    def fun(x):
        return x[x == 1].count() / x.count()
    dt1 = dataframe[dataframe.predicted_target == 1].groupby('date')['actual_target'].apply(
        lambda x: fun(x)).reset_index(drop=False)
    dt1 = dt1.rename({'actual_target': '1'}, axis=1)
    def funcc(x):
        return x[x == -1].count() / x.count()
    dtn1 = dataframe[dataframe.predicted_target == -1].groupby('date')['actual_target'].apply(
        lambda x: fun(x)).reset_index(drop=False)
    dtn1 = dtn1.rename({'actual_target': '-1'}, axis=1)
    dt = dt1.merge(dtn1, on=['date'], how='outer')
    dt.to_csv(savepath + 'GDBTtargetaccuracy.csv', index=False)

def factor_return(datas, marketdata, returns, savepath, method, methodreturn):
    # if methodreturn == 'SZ':
    #     returns = getexcessreturns(returns, 'SZ')
    # elif methodreturn == 'SH':
    #     returns = getexcessreturns(returns, 'SH')
    # else:
    #     returns = getreturns(returns, methodreturn)
    returns['tmrt'] = returns.groupby('code')['return'].shift(-1, axis=0)
    returns = returns.rename({'tradeday': 'date'}, axis=1)
    marketdata = marketdata.rename({'tradeday': 'date'}, axis=1)
    returns = returns.loc[:, ['code', 'date', 'tmrt', 'return']]
    bigdata = pd.merge(datas, returns, on=['code', 'date'], how='left')
    bigdata = pd.merge(bigdata, marketdata, on=['code', 'date'], how='left')
    bigdata = bigdata[~bigdata['tmrt'].isnull()]
    bigdata = bigdata[~bigdata['circulated_market_value'].isnull()]
    bigdata = bigdata.dropna(axis=0).reset_index(drop=True)
    date = bigdata.loc[:, 'date'].unique()
    k = len(date)
    coltval = pd.DataFrame(bigdata.columns[2:12]).apply(lambda x: x + 'tval')
    colauto = pd.DataFrame(bigdata.columns[2:12]).apply(lambda x: x + 'auto')
    colname = np.append(colauto, bigdata.columns[2:-3].values)
    colname = np.append(colname, coltval.values)
    colname = np.append(colname, 'date')
    colname = np.append(colname, 'r2')
    df = pd.DataFrame(columns=colname)
    #df = pd.DataFrame(columns = ['first', 'second', 'intercept', 'date'])
    df.to_csv(savepath + method + methodreturn + 'factorreturn.csv', index=False)
    for i in range(k - 1):
        dff = []
        today = bigdata[bigdata.date == date[i]]
        if i < k - 22:
            ###today's factors & tomorrow's in dataframe form
            nextm = bigdata[bigdata.date == date[i + 22]]
            X, y = today.drop(columns=['tmrt', 'return', 'date']), nextm.drop(
                columns=['tmrt', 'return', 'date', 'circulated_market_value'])
            total = pd.merge(X, y, on='code', how='left')
            for j in bigdata.columns[2:12].values:
                meann = total[j + '_x'].mean()
                meant = total[j + '_y'].mean()
                coefficient = (total[j + '_x'] - meann) * (total[j + '_y'] - meant) * np.sqrt(
                    total['circulated_market_value'])
                vart = ((total[j + '_x'] - meann) ** 2) * np.sqrt(total['circulated_market_value'])
                vart1 = ((total[j + '_y'] - meant) ** 2) * np.sqrt(total['circulated_market_value'])
                auto = coefficient.sum() / np.sqrt(vart.sum()) / np.sqrt(vart1.sum())
                dff.append(auto)
        else:
            dff = np.append(dff, np.zeros(10) + np.nan).tolist()

        X, y = today.iloc[:, 2:-3].values, today.loc[:, 'tmrt'].values
        ### X: factors y: tomorrow return
        Xtmw, ytmw = bigdata[bigdata.date == date[i + 1]].iloc[:, 2:-3].values, bigdata[
                                                                                    bigdata.date == date[i + 1]].loc[:,
                                                                                'tmrt'].values
        ### tomottow's X, tomorrow's return

        if method == 'OLS':
            model = linear_model.LinearRegression()
            model.fit(X, y)
            f = model.coef_
            coefficient = f[1:] / 100
            intercept = model.intercept_ / 100
            yhat = model.predict(Xtmw)
        if method == 'LASSO':
            model = LassoLarsCV(cv=20).fit(X, y)
            model.fit(X, y)
            f = model.coef_
            coefficient = f[1:] / 100
            intercept = model.intercept_ / 100
            yhat = model.predict(Xtmw)
        if method == 'ENET':
            model = ElasticNetCV(cv=20, random_state=0)
            model.fit(X, y)
            f = model.coef_
            coefficient = f[1:] / 100
            intercept = model.intercept_/100
            yhat = model.predict(Xtmw)
        if method == 'PLS':
            model = PLSRegression(n_components=2)
            model.fit(X, y)
            f = model.coef_
            coefficient = f[1:] / 100
            intercept = model.intercept_/100
            yhat = model.predict(Xtmw)
        if method == 'PCR':
            # model = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
            #model = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
            # print(model1.explained_variance_ratio_)
            model1 = PCA(n_components=2)
            model1.fit(X)
            coefficient1 = model1.components_.dot(X.T)
            model = linear_model.LinearRegression()
            model.fit(coefficient1.T, y)
            f = model.coef_
            intercept = model.intercept_/100
            coefficient = f[1:]/ 100
        if method == 'GLM':
            model = linear_model.TweedieRegressor()
            model.fit(X, y)
            f = model.coef_
            coefficient = f[1:] / 100
            intercept = model.intercept_/100
            yhat = model.predict(Xtmw)
        if method == 'WLS':
            weight = np.sqrt(today['circulated_market_value'])
            weight = weight.values / sum(weight)
            weight = pd.DataFrame(np.diag(weight))
            Xtmw = sm.add_constant(Xtmw)
            X = sm.add_constant(X)
            industry_exp = today.iloc[:, 12:40].transpose()
            total_mkt = sum(today['circulated_market_value'])
            industry_weight = industry_exp.dot(today['circulated_market_value'])
            industry_weight = industry_weight / total_mkt
            industry_weight = -industry_weight.values[:-1] / industry_weight[-1]
            ss = X.shape[1]
            diag_R = np.diag(np.ones(ss))
            R = np.delete(diag_R, -1, axis=1)
            R[-1, -27:] = industry_weight
            W = R.dot(np.linalg.inv(R.T.dot(X.T).dot(weight).dot(X).dot(R))).dot(R.T).dot(X.T).dot(weight)
            W = pd.DataFrame(W)
            f = W.dot(y).to_frame().T
            coefficient = f.values[0][1:] / 100
            intercept = f.values[0][1] / 100
            yhat = np.array(f.dot(Xtmw.T))[0]
        tval = pd.melt(pd.DataFrame(gettval(X, y, f))).iloc[1:11, 1].values.tolist()
        dff.extend(coefficient.tolist())
        dff.extend(tval)
        dff.extend([intercept])
        dff.append(date[i + 1])
        corr_matrix = np.corrcoef(ytmw, yhat)
        corr = corr_matrix[0, 1]
        r2 = corr ** 2
        dff.append(r2)
        dff = pd.DataFrame(dff)
        dff = dff.transpose()
        #dff.columns = ['first', 'second', 'intercept', 'date']
        dff.columns = colname
        if ((date[i] // 10000) == (date[i + 1] // 10000)) & (i < k - 2):
            df = df.append(dff)
        elif i == k - 2:
            df = df.append(dff)
            df.to_csv(savepath + method + methodreturn + 'factorreturn.csv', header=False, mode='a', index=False)
            break
        else:
            df.to_csv(savepath + method + methodreturn + 'factorreturn.csv', header=False, mode='a', index=False)
            df = dff

## return each factors' t value
def gettval(X, Y, f):  ###f is factor return array
    err = Y - X.dot(f.T).flatten()
    se = (sum(np.array(err) ** 2) / (len(err) - X.shape[1]) / sum(np.square(X - mean(X)))) ** 0.5
    tvalue = f / se
    return tvalue

## return each style factors' yearly t value, %|t|>2, average return %,
## volatility, VIF and fsc
def tstat(dataframe, savepath, method):
    df = pd.DataFrame(dataframe.iloc[:, -12:-2].mean(), columns=['tavg']).reset_index()
    df['|t|>2'] = df['index'].apply(lambda x: dataframe.loc[dataframe[x] > 2, x].count() / dataframe.loc[:, x].count())
    df['avgrt %'] = df['index'].apply(lambda x: dataframe.loc[:, x.strip('tval')].mean() * 252)
    df['volatility'] = df['index'].apply(lambda x: dataframe.loc[:, x.strip('tval')].std() * np.sqrt(252))
    df['VIF'] = fctr2(dataframe)[:10].values
    df['FSC'] = df['index'].apply(
        lambda x: dataframe[~dataframe[x.strip('tval') + 'auto'].isnull()].loc[:, x.strip('tval') + 'auto'].mean())
    df = df.set_index('index')
    df.to_csv(savepath + method + 'fctrt.csv')

def fctr2(dataframe):  # dataframe = pd.read_csv(savepath + method + 'factorreturn.csv')
    df = dataframe.iloc[:, 10:48].copy()
    df['ic'] = 1
    name = df.columns
    df = np.matrix(df)
    VIF_list = [variance_inflation_factor(df, i) for i in range(df.shape[1])]
    VIF = pd.DataFrame({'feature': name, 'VIF': VIF_list})
    VIF = VIF[:-1].set_index('feature')
    return VIF

def deciturnover(savepath, method, methodreturn, price):
    df = pd.read_csv(savepath + method + methodreturn + '.csv')
    price = price.rename({'tradeday': 'date'}, axis=1)
    dataframe = df.merge(price, how = 'left', on = ['date', 'code'])
    dates = dataframe['date'].unique()
    dff = pd.DataFrame(columns=['tmrt', 'DecileRank', 'date', 'code', 'acc_last', 'acc_close', 'acc_open'])
    dff.to_csv(savepath + method + methodreturn + 'deciturnover.csv', index=False)
    for d in dates:
        df = dataframe.loc[dataframe.date == d].copy()
        df['DecileRank'] = pd.qcut(df['predict'].rank(method='first'), q=10, labels=False)
        datas = df.loc[:, ['tmrt', 'DecileRank', 'date', 'code', 'acc_last', 'acc_close', 'acc_open']]
        datas.to_csv(savepath + method + methodreturn + 'deciturnover.csv', mode='a', header=False, index=False)
    decile = pd.read_csv(savepath + method + methodreturn + 'deciturnover.csv')
    df = pd.DataFrame(columns=['code','date', 'return', 'DecileRank'])
    df.to_csv(savepath + method + methodreturn + 'decireturn.csv', index=False)
    rank = len(decile['DecileRank'].unique())
    k = len(dates)
    for i in range(rank):
        data = decile[decile.DecileRank == i]
        for j in range(k-1):
            data1 = data[data.date == dates[j]]
            data2 = data[data.date == dates[j+1]]
            dff= data2.merge(data1.loc[:, ['acc_last', 'code']], how='left', on = 'code')
            dff['acc_last_y'].fillna(dff.acc_open, inplace=True)
            dff['return'] = dff['acc_close']/dff['acc_last_y']
            dff = dff.loc[:, ['code','date', 'return', 'DecileRank']]
            df = df.append(dff)
        df.to_csv(savepath + method + methodreturn + 'decireturn.csv', mode='a', header=False, index=False)
        df = pd.DataFrame(columns=['code', 'date', 'return', 'DecileRank'])
    dataframe = pd.read_csv(savepath + method + methodreturn + 'decireturn.csv')
    dff = pd.DataFrame(columns=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 'date'])
    dff.to_csv(savepath + method + methodreturn + 'decile.csv', index=False)
    for d in rank:
        df = dataframe.loc[dataframe.date == d].copy()
        df['DecileRank'] = pd.qcut(df['predict'].rank(method='first'), q=10, labels=False)
        datas = df.loc[:, ['return', 'DecileRank']]
        Decmean = datas.groupby(['DecileRank']).mean()
        Decmean = pd.DataFrame(Decmean.transpose())
        Decmean['date'] = d
        Decmean.to_csv(savepath + method + methodreturn + 'decile.csv', mode='a', header=False, index=False)
    Decmean = df.loc[:, ['return', 'DecileRank']].groupby(['DecileRank']).mean()
    Decmean = pd.DataFrame(Decmean.transpose())
    Decmean['date'] = d
    Decmean.to_csv(savepath + method + methodreturn + 'decile.csv', mode='a', header=False, index=False)
    pdb.set_trace()

def decistat(savepath, method, methodreturn):
    dataframe = pd.read_csv(savepath + method + methodreturn + '.csv')
    dates = dataframe['date'].unique()
    dff = pd.DataFrame(columns=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 'date'])
    dff.to_csv(savepath + method + methodreturn + 'decile.csv', index=False)
    for d in dates:
        df = dataframe.loc[dataframe.date == d].copy()
        df['DecileRank'] = pd.qcut(df['predict'].rank(method='first'), q=10, labels=False)
        datas = df.loc[:, ['tmrt', 'DecileRank']]
        Decmean = datas.groupby(['DecileRank']).mean()
        Decmean = pd.DataFrame(Decmean.transpose())
        Decmean['date'] = d
        Decmean.to_csv(savepath + method + methodreturn + 'decile.csv', mode='a', header=False, index=False)
    decile = pd.read_csv(savepath + method + methodreturn + 'decile.csv')
    statistics = pd.DataFrame(decile.iloc[:, :-1].describe())
    # pdb.set_trace()
    statistics.drop(labels=['count', 'min', '25%', '50%', '75%', 'max'], axis=0, inplace=True)
    MD = getMD(decile).values
    statistics.loc['MD', :] = MD[0]
    ## there are 252 trading days in a year
    statistics.loc['mean', :] = statistics.loc['mean', :].apply(lambda x: x * 252)
    statistics.loc['std', :] = statistics.loc['std', :].apply(lambda x: x * np.sqrt(252))
    statistics.loc['SR', :] = statistics.loc['mean', :] / statistics.loc['std', :]
    statistics.to_csv(savepath + method + methodreturn + 'stat.csv')
    return statistics

def deciplt(df, savepath, method, methodreturn):  ###C:/Users/CCZQ/Desktop/decile.csv
    #dataframe = pd.read_csv(savepath + method + methodreturn + 'decile.csv')
    dataframe = df.copy()
    dataframe.loc[:, dataframe.columns != 'date'] = dataframe.loc[:,  dataframe.columns != 'date'].cumsum(axis=0).apply(lambda x: x / 100)
    dataframe.loc[:, 'date'] = dataframe.loc[:, 'date'].astype(str).apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111)

    for i in range(10):
        ax.plot(dataframe.loc[:, 'date'], dataframe.loc[:, str(i)], '--', label='%s' % i)
    # Major ticks every 6 months
    fmt_quart_year = mdates.MonthLocator(interval=3)
    ax.xaxis.set_major_locator(fmt_quart_year)

    # Minor ticks every month
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(fmt_month)

    # Text in the x axis will be displayed in 'YYYY-mm-dd' format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # pdb.set_trace()
    datemin = dataframe.loc[:, 'date'].min()
    datemax = dataframe.loc[:, 'date'].max()
    ax.set_xlim(datemin, datemax)

    # Format the coords message box, i.e. the numbers displayed as the cursor moves
    # across the axes within the interactive GUI
    # ax.format_xdata = mdates.DataFormatter('%Y-%m-%d')
    ax.grid(True)
    fig.suptitle(method + methodreturn + 'predicted decile performance', fontsize=20)
    plt.legend(loc=0)
    # plt.show()
    plt.savefig('%s/%s_%s.png' % (savepath, method, methodreturn))
    # plt.savefig('C:/Users/CCZQ/Desktop/cv.jpg')

def getreturns(returns, method):  # dataframe is downloaded from 'quote' and it is named returns
    if method == 'intraday return':
        returns['return'] = returns['close'] / returns['open'] - 1
        results = returns.loc[:, ['tradeday', 'code', 'return']]
    elif method == 'open return':
        returns['return'] = returns['open'] / returns['last'] - 1
        results = returns.loc[:, ['tradeday', 'code', 'return']]
    elif method == 'return':
        results = returns.loc[:, ['tradeday', 'code', 'return']]
    return results

def getexcessreturns(returns, method):
    benchmark = pd.read_csv(savepath + method + 'index.csv')
    returns = returns.loc[:, ['tradeday', 'code', 'return']]
    new = pd.merge(returns, benchmark, on=['tradeday'], how='left')
    new['return'] = new['return'] - new['SHrt']
    df = new.loc[:, ['tradeday', 'code', 'return']]
    return df

def getMD(dataframe):  ###dataframe is the decile data
    cumsum = dataframe.iloc[:, :-1].cumsum(axis=0).apply(lambda x: x / 100 + 1)
    cummax = cumsum.cummax()
    drawdown = (cummax - cumsum) / cummax
    df = pd.DataFrame()
    for i in range(10):
        df.loc[0, i] = drawdown.iloc[:, i].max()
    return df

def getindex(quote, method):  # weight = get_component_weight_series, index = 000300, quote = get_quote_series
    quote = quote.loc[:, ['tradeday', 'code', 'return']]
    if method == 'SH':
        weight = get_component_weight_series(20170901, 20210520, '000300')
    if method == 'SZ':
        weight = get_component_weight_series(20170901, 20210520, '000905')
    weight = weight.rename({'date': 'tradeday'}, axis=1)
    new = pd.merge(quote, weight, on=['code', 'tradeday'], how='left')
    new = new[~new['weight'].isnull()]
    SHrt = new[['return', 'weight']].prod(axis=1).groupby(new['tradeday']).sum()
    SHrt = pd.DataFrame(SHrt)
    SHrt.columns = ['SHrt']
    SHrt = SHrt.reset_index()
    SHrt.to_csv(savepath + method + 'index.csv', index=False)
    return SHrt

def freturnplt(dataframe, savepath, method, methodreturn):
    colname = ['BETA', 'MOMENTUM', 'SIZE', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY', 'SIZENL']
    # colname = ['first', 'second', 'intercept']
    dataframe.loc[:, dataframe.columns.isin(colname)] = dataframe.loc[:, dataframe.columns.isin(colname)].cumsum(axis=0)
    dataframe.dropna(inplace=True)
    dataframe.loc[:, 'date'] = dataframe.loc[:, 'date'].astype(int).astype(str).apply(
        lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111)
    for i in colname:
        ax.plot(dataframe.loc[:, 'date'], dataframe.loc[:, i], '--', label=i)
    # Major ticks every 6 months
    fmt_quart_year = mdates.MonthLocator(interval=3)
    ax.xaxis.set_major_locator(fmt_quart_year)

    # Minor ticks every month
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(fmt_month)

    # Text in the x axis will be displayed in 'YYYY-mm-dd' format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    datemin = dataframe.loc[0, 'date']
    datemax = dataframe.loc[len(dataframe) - 1, 'date']
    ax.set_xlim(datemin, datemax)

    # Format the coords message box, i.e. the numbers displayed as the cursor moves
    # across the axes within the interactive GUI
    # ax.format_xdata = mdates.DataFormatter('%Y-%m-%d')
    ax.grid(True)
    fig.suptitle(method + 'factor return', fontsize=18)
    plt.legend(loc=0)
    # plt.show()
    plt.savefig('%s/%s%s_%s.png' % (savepath, method, methodreturn, 'returnfactor'))

def r2plt(dataframe, savepath, method):
    dataframe.loc[:, 'r2'] = dataframe.loc[:, 'r2'].rolling(244).mean()
    dataframe.dropna(inplace=True)
    pdb.set_trace()
    dataframe.loc[:, 'date'] = dataframe.loc[:, 'date'].astype(int).astype(str).apply(
        lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111)
    ax.plot(dataframe.loc[:, 'date'], dataframe.loc[:, 'r2'], '--', label='r2')
    # Major ticks every 6 months
    fmt_quart_year = mdates.MonthLocator(interval=3)
    ax.xaxis.set_major_locator(fmt_quart_year)

    # Minor ticks every month
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(fmt_month)

    # Text in the x axis will be displayed in 'YYYY-mm-dd' format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    datemin = dataframe.loc[0, 'date']
    datemax = dataframe.loc[len(dataframe) - 1, 'date']
    ax.set_xlim(datemin, datemax)

    # Format the coords message box, i.e. the numbers displayed as the cursor moves
    # across the axes within the interactive GUI
    # ax.format_xdata = mdates.DataFormatter('%Y-%m-%d')
    ax.grid(True)
    fig.suptitle(method + 'r2cumsum', fontsize=18)
    plt.legend(loc=0)
    # plt.show()
    plt.savefig('%s/%s_%s.png' % (savepath, method, 'r2'))

def delipo(Java, bigdata):
    bigdata = bigdata.drop(['Unnamed: 0'], axis=1)
    Java = Java.rename({'wind_id_list': 'code'}, axis=1)
    bigdata = bigdata.rename(columns={'wind_id_list': 'code'})
    new = pd.merge(bigdata, Java, on='code', how='left')
    new[new.date <= new.ipo_date + 10000].loc[:, ['code', 'date']].to_csv(savepath + 'delipo.csv', index=False)
    new = new[new.date > new.ipo_date + 10000]
    new = new.drop(columns='ipo_date')
    # new.to_csv(savepath + 'delipo.csv', index = False)
    return new

def delST(bigdata, ST):
    codes = bigdata.code.unique()
    ST['cancel_date'] = ST['cancel_date'].fillna(20210520)
    ST = ST[ST.cancel_date > 20170901]
    ST.loc[:, ['cancel_date', 'execute_date']] = ST.loc[:, ['cancel_date', 'execute_date']].clip(20170901, 20210520)
    ST = ST[ST.code.isin(codes)].drop_duplicates(subset=None, keep='first', inplace=False)
    execute = ST.loc[:, ['code', 'execute_date']]
    execute['start'] = 1
    execute = execute.rename({'execute_date': 'date'}, axis=1)
    execute = execute.groupby(['code', 'date'])['start'].sum().reset_index()
    cancel = ST.loc[:, ['code', 'cancel_date']]
    cancel['end'] = -1
    cancel.groupby('code').sum()
    cancel = cancel.rename({'cancel_date': 'date'}, axis=1)
    cancel = cancel.groupby(['code', 'date'])['end'].sum().reset_index()
    new = bigdata.merge(execute, on=['code', 'date'], how='outer')
    neww = new.merge(cancel, on=['code', 'date'], how='outer')
    neww['end'] = neww['end'].fillna(0)
    neww['start'] = neww['start'].fillna(0)
    neww.sort_values(by=['code', 'date'], inplace=True)
    neww = neww.reset_index(drop=True)
    neww['end'] = neww['end'].cumsum(axis=0)
    neww['start'] = neww['start'].cumsum(axis=0)
    neww['total'] = neww['start'] + neww['end']
    neww[neww.total != 0].loc[:, ['code', 'date']].to_csv(savepath + 'delST.csv', index=False)
    neww = neww[neww.total == 0].reset_index(drop=True)
    bigdata = neww.drop(columns=['start', 'end', 'total'])
    # bigdata.to_csv(savepath + 'delST.csv', index = False)
    return bigdata

def handle_stoptrade(df, tradeday):
    # df = pd.read_csv(r'\\10.100.106.219\shared\个人文件夹\庾灿斌\实习\陈\stopdata.csv')
    # Data = windapi()
    # tradeday = Data.get_date_series(20170101,20210521,'trade')
    # df.groupby('date')

    tradeday['pos'] = tradeday.index
    df['stop'] = 1
    stopmatrix = df.pivot(index='date', columns='code', values='stop')
    stopmatrix = stopmatrix.reindex(index=tradeday['date'].values)
    stopmatrix.fillna(0, inplace=True)
    ref = stopmatrix - stopmatrix.shift(1)
    ref2 = stopmatrix * np.nan
    ref2[ref == 1] = 1
    ref2.fillna(0, inplace=True)
    ref2 = ref2.cumsum(axis=0)
    check = pd.melt(ref2.reset_index(), id_vars=['date'])
    check = check.merge(df, how='left', on=['date', 'code'])
    check['stop'].fillna(0, inplace=True)
    stoptradelist = check.groupby(['code', 'value']).agg({"stop": "sum", 'date': 'first'}).reset_index()
    stoptradelist = stoptradelist.query("value!=0")
    stoptradelist = stoptradelist.merge(tradeday, how='left', on='date')
    stoptradelist['endidx'] = stoptradelist['pos'] + stoptradelist['stop'] - 1
    stoptradelist['suspendidx'] = stoptradelist['endidx'] + stoptradelist['stop'].map(lambda x: 60 if x > 60 else x) + 1
    stoptradelist['endidx'] = stoptradelist['endidx'].clip(upper=len(tradeday) - 1)
    stoptradelist['suspendidx'] = stoptradelist['suspendidx'].clip(upper=len(tradeday) - 1)
    a = time.time()
    stoptradelist['enddate2'] = stoptradelist['endidx'].map(lambda x: tradeday.loc[x, 'date'])
    stoptradelist['suspenddate2'] = stoptradelist['suspendidx'].map(lambda x: tradeday.loc[x, 'date'])
    b = time.time()
    print(b - a)
    stoptradelist['enddate'] = stoptradelist['endidx'].map(lambda x: tradeday['date'].values[int(x)])
    stoptradelist['suspenddate'] = stoptradelist['suspendidx'].map(lambda x: tradeday['date'].values[int(x)])
    a = time.time()
    print(a - b)
    return stoptradelist

def delstop(bigdata, stoptb):
    codes = bigdata.code.unique()
    bigdata.sort_values(by=['code', 'date'], inplace=True)
    bigdata = bigdata.reset_index(drop=True)
    stoptb = stoptb[stoptb.suspenddate > 20170901].reset_index(drop=True)
    stoptb = stoptb[stoptb.code.isin(codes)]
    stoptb.loc[:, ['enddate', 'suspenddate']] = stoptb.loc[:, ['enddate', 'suspenddate']].clip(20170901, 20210520)
    stoptbb = stoptb.loc[:, ['code', 'enddate']]
    stoptbb = stoptbb.rename(columns={'enddate': 'date'})
    stoptbb['end'] = 1
    stoptbb = stoptbb.groupby(['code', 'date'])['end'].sum().reset_index()
    stoptn = stoptb.loc[:, ['code', 'suspenddate']]
    stoptn['sus'] = -1
    stoptn = stoptn.rename(columns={'suspenddate': 'date'})
    stoptn = stoptn.groupby(['code', 'date'])['sus'].sum().reset_index()
    neww = bigdata.merge(stoptbb, on=['code', 'date'], how='outer')
    neww = neww.merge(stoptn, on=['code', 'date'], how='outer')
    neww['end'] = neww['end'].fillna(0)
    neww['sus'] = neww['sus'].fillna(0)
    neww.sort_values(by=['code', 'date'], inplace=True)
    neww['end'] = neww['end'].cumsum(axis=0)
    neww['sus'] = neww['sus'].cumsum(axis=0)
    neww['cutdate'] = neww['end'] + neww['sus']
    neww[neww.cutdate != 0].loc[:, ['code', 'date']].to_csv(savepath + 'delstoptb.csv', index=False)
    neww = neww[neww['cutdate'] == 0].reset_index(drop=True)
    bigdata = neww.drop(columns=['end', 'sus', 'cutdate'])
    # bigdata.to_csv(savepath + 'bigdata.csv', index = False)
    return bigdata


if __name__ == "__main__":
    # returns = returns.rename(columns={'tradeday': 'date', 'code': 'wind_id_list'})
    # returns.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    # dataframe = pd.read_csv('C:/Users/CCZQ/Desktop/barra/riskdata_20170905.csv')
    method = 'GLM'
    methodreturn = ''
    methodcut = 'predicted_value'
    savepath = 'C:/Users/huili/Desktop/'

    market = pd.read_csv(savepath + 'chl_market_feature.csv').loc[:, ['date', 'code', 'circulated_market_value']]
    quote = pd.read_csv(savepath + 'chl_quote.csv').loc[:, ['date', 'code', 'return']]
    component = pd.read_csv(savepath + 'chl_component.csv')
    factors = pd.read_csv(savepath + 'chl_cne5.csv')
    # statistical_model(factors, quote, market, savepath, method, methodreturn)
    # decistat(savepath, method, methodreturn) ## return yearly performance for each group
    # dataframe = pd.read_csv(savepath + method + methodreturn + 'decile.csv')
    # deciplt(dataframe, savepath, method, methodreturn)
    # method = 'WLS'
    # statistical_model(factors, quote, market, savepath, method, methodreturn)
    # decistat(savepath, method, methodreturn)  ## return yearly performance for each group
    # dataframe = pd.read_csv(savepath + method + methodreturn + 'decile.csv')
    # deciplt(dataframe, savepath, method, methodreturn)
    # factor_return(factors, market, quote, savepath, method, methodreturn)
    dataframe = pd.read_csv(savepath + method + methodreturn + 'factorreturn.csv')
    # freturnplt(dataframe, savepath, method, methodreturn)
    # tstat(dataframe, savepath, method)
    r2plt(dataframe, savepath, method)
    pdb.set_trace()
    #GDBTaccuracy()
    ##method ='OLS', 'LASSO', 'ENET', 'PLS', 'PCR', 'GLM'. 'RF'

    # GDBTplt(returns, bigdata, savepath, method, methodreturn, methodcut)
    # methodcut = '1'
    # GDBTplt(returns, bigdata, savepath, method, methodreturn, methodcut)
    # methodcut = 'probpred'
    # GDBTplt(returns, bigdata, savepath, method, methodreturn, methodcut)
    # GDBTresidual()
    # method = 'GDBTresidual'
    # methodcut = 'predicted_WLS'
    # GDBTplt(returns, bigdata, savepath, method, methodreturn, methodcut)


    # #Get all files from Barra folder and concatenate them into one dataframe
    # files = glob.glob('C:/Users/CCZQ/Desktop/decile/*.csv')
    # # files
    # # ### Initialize empty dataframe
    # df = pd.DataFrame()
    # # ### Loop over list of files to append to empty dataframe
    # for f in files:
    #     print(f)
    #     data = pd.read_csv(f).loc[:, ['9', '0', 'date']]
    #     if len(df) == 0:
    #         df = data
    #     else:
    #         df = df.merge(data, how = 'outer', on = 'date')
    #     print(df)
    # # df
    # # df.to_csv(savepath + 'bigfactorreturn.csv', index = False)

    # # a = time.time()
    # # check1 = bigdata[bigdata['wind_id_list'] == '600378.SH']
    # # b = time.time()
    # # print(b - a)
    # # check2 = bigdata.query("wind_id_list=='600378.SH'")
    # # a = time.time()
    # # print(a - b)
    # bigdata = pd.read_csv('C:/Users/CCZQ/Desktop/bigdata.csv')
    # stoptb = pd.read_csv(savepath + 'stoptable.csv')
    # Java = pd.read_csv(savepath + 'ipo_date.csv')
    # ST = pd.read_csv(savepath + 'STlist.csv')
    # #
    # bigdata = delipo(Java, bigdata)
    # bigdata = delST(bigdata, ST)
    # bigdata = delstop(bigdata, stoptb)

    #
    #
    #
    # # start = time.time()
    # delipo(Java, bigdata)
    # # end = time.time()
    # # print(end-start)
    # #start1 = time.time()
    # method ='GLM'
    pdb.set_trace()
    deciturnover(savepath, method, methodreturn, price)
    # dataframe = pd.read_csv(savepath + method + methodreturn + 'decile.csv')
    # deciplt(dataframe, savepath, method, methodreturn)



    # dt = get_quote_series(20170901, 20210520)
    # # dt['tradeday', 'code', 'return'].to_csv(savepath + 'quote.csv', index = False)
    # dt.loc[:, ['tradeday', 'acc_last', 'acc_close', 'acc_open', 'code']].to_csv(savepath + 'price.csv', index = False)
    # price = pd.read_csv(savepath+'price.csv')
    # # quote = pd.read_csv(savepath + 'quote.csv')
    # # quote = quote.loc[:, ['tradeday', 'code', 'return']]
    # # quote.to_csv(savepath + 'quote.csv', index = False)
    # #quote = pd.read_csv(savepath + 'quote.csv')
    # #weight = get_component_weight_series(20170901, 20210520, '000300')
    # #weight.to_csv(savepath + 'weight.csv', index = False)
    # #weight = pd.read_csv(savepath + 'weight.csv')
    # # marketdata = get_market_feature_series(20170901, 20210520)
    # # marketdata.loc[:, ['tradeday', 'code', 'circulated_market_value']].to_csv(savepath + 'market.csv', index = False)

    #factor_return(bigdata, marketdata, returns, savepath, method, methodreturn)
    #

    #deciplt(dataframe, savepath, method, methodreturn)
    factor_return(bigdata, marketdata, returns, savepath, method, methodreturn)
    # # bigfactorreturn = pd.read_csv(savepath  + 'bigfactorreturn.csv')

    dataframe = pd.read_csv(savepath + method + methodreturn + 'factorreturn.csv')
    # tstat(dataframe, savepath, method)
    # r2plt(dataframe, savepath, method)
    # difference = pd.DataFrame()
    # # source = 'actual'
    freturnplt(dataframe, savepath, method, methodreturn)
    # # source = 'predicted'
    # freturnplt(dataframe, savepath, method, source = 'predicted')

    # df = pd.DataFrame(difference.describe())
    # df.loc['mean', :] = df.loc['mean', :] * df.loc['count', :]
    # df.loc['std', :] = df.loc['std', :] * np.sqrt(df.loc['count', :])
    # df.to_csv(savepath + 'factorreturndiff' + '.csv')
    pdb.set_trace()
