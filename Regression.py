import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

st.title('Linear Regression')
st.header('Data Generation')
st.latex(r'''Model : y_i = \beta_0 + \beta_1 x_i + \epsilon_i,\quad \epsilon_i \sim N(0,\sigma^2) \; and \; i=1,\cdots,N''') 

# 변수 선택
beta_0 = st.select_slider(
    r'Select True $\beta_0$',
    options=np.round(np.linspace(-3,3,61),1)
    ,value=0)

beta_1 = st.select_slider(
    r'Select True $\beta_1$',
    options=np.round(np.linspace(-3,3,61),1)
    ,value=1)

sigma = st.select_slider(
    r'Select True $\sigma$',
    options=np.round(np.linspace(0,4,41),1)
    ,value=1)

N = st.select_slider(
        r'Select $\N$',
        options=[5,10,50,100,500,1000]
        ,value=100)

# data 생성
if st.button('Data generate'):
    e = np.random.normal(size=N)
    if 'X' not in st.session_state:
        st.session_state['X'] = np.random.uniform(-5,5,N)
    if 'y' not in st.session_state:
        st.session_state['y'] = beta_0 + beta_1 * st.session_state['X'] + sigma * e
    st.session_state['X'] = np.random.uniform(-5,5,N)
    st.session_state['y'] = beta_0 + beta_1 * st.session_state['X'] + sigma * e
    fig, ax = plt.subplots()
    sns.scatterplot(x=st.session_state['X'],y=st.session_state['y'],ax=ax,s=50, marker='x',color = 'black')
    ax.set_title('Linear Regression')
    ax.set_xlim([-6,6])
    ax.set_ylim([-20,20])
    st.pyplot(fig)
else :  st.write('Click button!')

st.header('Regression')

from sklearn.linear_model import ElasticNet, LinearRegression

option = st.selectbox(
    'Model Selection',
    ('Linear Regression', 'Ridge', 'Lasso','Elastic Net')
)
if option == 'Linear Regression' :
    model = LinearRegression()
elif option == 'Ridge' :
    alpha_ = st.select_slider(
        r'Select $\alpha$',
        options=np.logspace(-3,5,8))
    model = ElasticNet(alpha = alpha_,l1_ratio=0)
elif option == 'Lasso' :
    alpha_ = st.select_slider(
        r'Select $\alpha$',
        options=np.logspace(-3,5,8))
    model = ElasticNet(alpha = alpha_,l1_ratio=1)
elif option == 'Elastic Net' :
    alpha_ = st.select_slider(
        r'Select $\alpha$',
        options=np.logspace(-3,5,8))
    L1_ratio = st.select_slider(
        'Select ' r'$L_1 \; ratio$',
        options=np.linspace(0,1,10))
    model = ElasticNet(alpha = alpha_,l1_ratio=L1_ratio)


X = st.session_state['X'].reshape(-1,1)
y = st.session_state['y'].reshape(-1,1)
model.fit(X,y)

grid = np.linspace(-6,6,num = N)
fig, ax = plt.subplots()
y_line = np.squeeze(model.predict(grid.reshape(-1,1)))
sns.scatterplot(x=st.session_state['X'],y=st.session_state['y'],ax=ax,s=50, marker='x',color = 'black')
y_mean = np.mean(st.session_state['y']) *np.ones(N)
line1, = ax.plot(grid,y_mean, color='black') # y mean
line2, = ax.plot(grid,y_line, color='red') # 회귀선
line3, = ax.plot(grid,beta_0 + beta_1 * grid, color='blue') # true line
legend_1 = ax.legend(handles=(line1,line2,line3),labels=(r'$\bar{y}$',option,'True line'),loc= 'upper right',fontsize = 'small')
ax.add_artist(legend_1)
beta_hat_0 = np.round(model.intercept_[0],5)
beta_hat_1 = np.round(model.coef_[0],5)
legend_2 = ax.legend(handles=(line1,line2,line3),
                     labels=(r'$\bar{y}$'+' = {}'.format(np.round(np.mean(st.session_state['y']),5)),
                             r'$\hat{\beta}_0$'+' = {}, '.format(beta_hat_0) + r'$\hat{\beta}_1$'+' = {}'.format(beta_hat_1),
                             r'$\beta_0$ = {}, '.format(beta_0) + r'$\beta_1$ = {}'.format(beta_1)),
                             loc = 'lower right',fontsize = 'small') # 추정량 비교
ax.set_title(option)
ax.set_xlim([-6,6])
ax.set_ylim([-20,20])
st.pyplot(fig)