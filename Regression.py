import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import ElasticNet, LinearRegression


st.title('Linear Regression')

st.latex(r'''Model : y_i = \beta_0 + \beta_1 x_i + \epsilon_i,\quad \epsilon_i \sim N(0,\sigma^2) \; and \; i=1,\cdots,N''') 

# 변수 선택
with st.sidebar :
    st.header('Data Generation')
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
    if 'count' not in st.session_state:
        st.session_state['count'] = 0
    if st.session_state['count'] == 0 :
        e = np.random.normal(size=N)
        if 'X' not in st.session_state:
            st.session_state['X'] = np.random.uniform(-5,5,N)
        if 'y' not in st.session_state:
            st.session_state['y'] = beta_0 + beta_1 * st.session_state['X'] + sigma * e
        st.session_state['X'] = np.random.uniform(-5,5,N)
        st.session_state['y'] = beta_0 + beta_1 * st.session_state['X'] + sigma * e
        fig, ax = plt.subplots()
        sns.scatterplot(x=st.session_state['X'],y=st.session_state['y'],ax=ax,s=50, marker='x',color = 'black')
        ax.set_title('Data')
        ax.set_xlim([-6,6])
        ax.set_ylim([-20,20])
        st.pyplot(fig)
        st.session_state['count'] += 1
        
    if st.button('Generate Data'):
        e = np.random.normal(size=N)
        if 'X' not in st.session_state:
            st.session_state['X'] = np.random.uniform(-5,5,N)
        if 'y' not in st.session_state:
            st.session_state['y'] = beta_0 + beta_1 * st.session_state['X'] + sigma * e
        st.session_state['X'] = np.random.uniform(-5,5,N)
        st.session_state['y'] = beta_0 + beta_1 * st.session_state['X'] + sigma * e
        fig, ax = plt.subplots()
        sns.scatterplot(x=st.session_state['X'],y=st.session_state['y'],ax=ax,s=50, marker='x',color = 'black')
        ax.set_title('Data')
        ax.set_xlim([-6,6])
        ax.set_ylim([-20,20])
        st.pyplot(fig)
        st.write('Done!')
    else : 
        pass

option = st.selectbox(
    'Model Selection',
    ('Linear Regression', 'Ridge', 'Lasso','Elastic Net')
)
if option == 'Linear Regression' :
    st.latex(r'''Loss(\beta_0,\beta_1) = \sum^{N}_{i=1}\left(y_i-(\beta_0 + \beta_1 x_{i})\right)^2''')
    st.latex(r'''\hat{\beta}_0,\hat{\beta}_1 = \argmin_{\beta_0, \beta_1} Loss(\beta_0,\beta_1)''')
elif option == 'Ridge' :
    st.latex(r'''Loss(\beta_0,\beta_1) = \sum^{N}_{i=1}\left(y_i-(\beta_0 + \beta_1 x_{i})\right)^2 + \alpha\beta_1^2   ''')
    st.latex(r'''\hat{\beta}_0,\hat{\beta}_1 = \argmin_{\beta_0, \beta_1} Loss(\beta_0,\beta_1)''')
elif option == 'Lasso' :
    st.latex(r'''Loss(\beta_0,\beta_1) = \sum^{N}_{i=1}\left(y_i-(\beta_0 + \beta_1 x_{i})\right)^2 + \alpha|\beta_1|   ''')
    st.latex(r'''\hat{\beta}_0,\hat{\beta}_1 = \argmin_{\beta_0, \beta_1} Loss(\beta_0,\beta_1)''')
elif option == 'Elastic Net' :
    st.latex(r'''Loss(\beta_0,\beta_1) = \sum^{N}_{i=1}\left(y_i-(\beta_0 + \beta_1 x_{i})\right)^2 + \alpha(L_1|\beta_1| + \frac{1-L_1}{2}\beta_1^2)   ''')
    st.latex(r'''\hat{\beta}_0,\hat{\beta}_1 = \argmin_{\beta_0, \beta_1} Loss(\beta_0,\beta_1)''')



if option == 'Linear Regression' :
    model = LinearRegression()
elif option == 'Ridge' :
    alpha_ = st.select_slider(
        r'Select $\alpha$',
        options=np.round(np.logspace(-3,5,8),5))
    model = ElasticNet(alpha = alpha_,l1_ratio=0)
elif option == 'Lasso' :
    alpha_ = st.select_slider(
        r'Select $\alpha$',
        options=np.round(np.logspace(-3,5,8),5))
    model = ElasticNet(alpha = alpha_,l1_ratio=1)
elif option == 'Elastic Net' :
    alpha_ = st.select_slider(
        r'Select $\alpha$',
        options=np.round(np.logspace(-3,5,8),5))
    L1_ratio = st.select_slider(
        'Select ' r'$L_1 \; ratio$',
        options=np.round(np.linspace(0,1,11),5),
        value= 0.5)
    model = ElasticNet(alpha = alpha_,l1_ratio=L1_ratio)


X = st.session_state['X'].reshape(-1,1)
y = st.session_state['y'].reshape(-1,1)
model.fit(X,y)
beta_hat_0 = np.round(model.intercept_[0],5)

if option == 'Linear Regression' :
    beta_hat_1 = np.round(model.coef_[0],5)[0]
else :
    beta_hat_1 = np.round(model.coef_[0],5)

grid = np.linspace(-6,6,num = N)
y_line = np.squeeze(model.predict(grid.reshape(-1,1)))
y_mean = np.mean(st.session_state['y']) *np.ones(N)


fig = go.Figure()

fig.add_trace(go.Scatter(x=st.session_state['X'],
                         y=st.session_state['y'],
                         mode = 'markers',
                         marker = dict(color = 'black', size = 5, symbol = 'x'),
                         name = 'Data'))
fig.add_trace(go.Scatter(x = grid,
                         y = y_mean,
                         mode = 'lines',
                         line = dict(color = 'black'),
                         name = 'y̅ = {}'.format(np.round(np.mean(st.session_state['y']),5))+ ' (Sample mean of y)'))
fig.add_trace(go.Scatter(x = grid,
                         y = y_line,
                         mode = 'lines',
                         line = dict(color = 'red'),
                         name = ' β̂₀ = {}, '.format(beta_hat_0) +'β̂₁ = {}'.format(beta_hat_1)+ ' ({})'.format(option))) 
fig.add_trace(go.Scatter(x = grid,
                         y = beta_0 + beta_1 * grid,
                         mode = 'lines',
                         line = dict(color = 'blue'),
                         name = 'β₀ = {}, '.format(beta_0) + 'β₁ = {}'.format(beta_1) + ' (True line)'))
st.plotly_chart(fig)


## matplotlib
# fig, ax = plt.subplots()

# sns.scatterplot(x=st.session_state['X'],y=st.session_state['y'],ax=ax,s=50, marker='x',color = 'black')
# line1, = ax.plot(grid,y_mean, color='black') # y mean
# line2, = ax.plot(grid,y_line, color='red') # 회귀선
# line3, = ax.plot(grid,beta_0 + beta_1 * grid, color='blue') # true line
# legend_1 = ax.legend(handles=(line1,line2,line3),labels=(r'$\bar{y}$',option,'True line'),loc= 'upper right',fontsize = 'small')
# ax.add_artist(legend_1)
# legend_2 = ax.legend(handles=(line1,line2,line3),
#                      labels=(r'$\bar{y}$'+' = {}'.format(np.round(np.mean(st.session_state['y']),5)),
#                              r'$\hat{\beta}_0$'+' = {}, '.format(beta_hat_0) + r'$\hat{\beta}_1$'+' = {}'.format(beta_hat_1),
#                              r'$\beta_0$ = {}, '.format(beta_0) + r'$\beta_1$ = {}'.format(beta_1)),
#                              loc = 'lower right',fontsize = 'small') # 추정량 비교

# ax.set_title(option)
# ax.set_xlim([-6,6])
# ax.set_ylim([-20,20])
# st.pyplot(fig)