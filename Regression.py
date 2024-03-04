import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Linear Regression')
st.header('Data Generation')
np.random.seed(0)
N = st.select_slider(
    r'Select $\N$',
    options=[2,10,100,1000])

e = np.random.normal(size=N)
X = np.random.uniform(0,10,N)

st.header('Linear Regression')

beta_0 = st.select_slider(
    r'Select True $\beta_0$',
    options=np.linspace(0,10,101))

beta_1 = st.select_slider(
    r'Select True $\beta_1$',
    options=np.linspace(0,10,101))

epsilon = st.select_slider(
    r'Select True $\epsilon$',
    options=np.linspace(0,10,101))

import statsmodels.api as sm
y = beta_0 + beta_1 * X + epsilon * e
y_mean = np.mean(y)
y_center = y-y_mean
dfX = pd.DataFrame(X, columns=["x"])
dfX = sm.add_constant(dfX)
dfy = pd.DataFrame(y_center, columns=["y"])
df = pd.concat([dfX, dfy], axis=1)

option = st.selectbox(
    'Model Selection',
    ('Linear Regression', 'Ridge', 'Lasso','Elastic Net')
)
st.write('선택한 옵션:', option)
model = sm.OLS.from_formula("y ~ x", data = df)
if option == 'Linear Regression' :
    result = model.fit()
elif option == 'Ridge' :
    lambda_ = st.select_slider(
        r'Select $\lambda$',
        options=np.logspace(-3,5,8))
    result = model.fit_regularized(alpha=lambda_,L1_wt=0)
elif option == 'Lasso' :
    lambda_ = st.select_slider(
        r'Select $\lambda$',
        options=np.logspace(-3,5,8))
    result = model.fit_regularized(alpha=lambda_,L1_wt=1)
elif option == 'Elastic Net' :
    lambda_ = st.select_slider(
        r'Select $\lambda$',
        options=np.logspace(-3,5,8))
    L1_wt = st.select_slider(
        'Select $L1_wt$',
        options=np.linspace(0,1,10))
    result = model.fit_regularized(alpha=lambda_,L1_wt=L1_wt)

grid = np.linspace(min(X),max(X),N)
dfxx = pd.DataFrame(grid, columns=["x"])

fig, ax = plt.subplots()
sns.scatterplot(x=X,y=y,ax=ax,s=50, color ='black')
sns.lineplot(x=grid,y=result.predict(dfxx)+y_mean,ax=ax,color='red')
ax.set_title('Linear Regression')
st.pyplot(fig)