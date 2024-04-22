import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def Generate_data(N,a,b,sigma):
    e = np.random.normal(size=N).reshape(-1,1)
    st.session_state['X'] = np.random.uniform(0,4*np.pi,N)
    st.session_state['y'] = np.squeeze(a*np.cos(st.session_state['X'].reshape(-1,1)) + b + sigma * e)
    

with st.sidebar :
    st.header('Data Generation(True model)')
    st.latex(r'''y_i = a\cos(x_{i}) + b + \epsilon_i,\quad \epsilon_i \sim N(0,\sigma^2) \; and \; i=1,\cdots,N''')
    N = st.select_slider(
            r'Select $\N$',
            options=[30,100,1000]
           ,value=30)
    a = st.select_slider(
        r'Select True $a$',
        options=np.round(np.linspace(-1,1,21),1)
        ,value=1)
    b = st.select_slider(
        r'Select True $b$',
        options=np.round(np.linspace(-1,1,21),1)
        ,value=0)
    sigma = st.select_slider(
        r'Select True $\sigma$',
        options=np.round(np.linspace(0,1,11),1)
        ,value=0.1)
    if st.button('Generate data'):
        Generate_data(N,a,b,sigma)
        st.write('Done!')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state['X'],
                                y=st.session_state['y'],
                                mode = 'markers',
                                marker = dict(color = 'black', size = 5, symbol = 'x'),
                                name = 'Data'))
        
        st.plotly_chart(fig, use_container_width=True)

option = st.selectbox(
    'Select Model',
    ('polynomial Regression', 'KNN regression'))

if option == 'polynomial Regression':
    st.title('Polynomial Regression')
    st.latex(r'''Model : y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_m x_i^m + \epsilon_i,\quad \epsilon_i \sim N(0,\sigma^2) \; and \; i=1,\cdots,N''')

    degree_1 = st.checkbox('m=1')
    degree_2 = st.checkbox('m=2')
    degree_3 = st.checkbox('m=3')
    degree_m_checkbox = st.checkbox('Custom degree m')
    degree_list = [degree_1,degree_2,degree_3]
    color_list = ['blue','red','green']
    if degree_m_checkbox:
        degree_m = st.number_input('Select m', min_value=1,max_value=20, value=4)
    grid = np.linspace(np.min(st.session_state['X']),np.max(st.session_state['X']),100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state['X'],
                            y=st.session_state['y'],
                            mode = 'markers',
                            marker = dict(color = 'black', size = 5, symbol = 'x'),
                            name = 'Data'))
    for i in range(3):
        if degree_list[i] :
            model_lr = make_pipeline(StandardScaler(),
                                    PolynomialFeatures(degree=i+1, include_bias=False),
                                    LinearRegression())
            model_lr.fit(st.session_state['X'].reshape(-1,1),st.session_state['y'])
            fig.add_trace(go.Scatter(x=grid,
                                    y=model_lr.predict(grid.reshape(-1,1)),
                                    mode = 'lines',
                                    line = dict(color = color_list[i]),
                                    name = 'm = {}'.format(i+1)))
    if degree_m_checkbox :
        model_lr = make_pipeline(StandardScaler(),
                                PolynomialFeatures(degree=degree_m, include_bias=False),
                                LinearRegression())
        model_lr.fit(st.session_state['X'].reshape(-1,1),st.session_state['y'])
        fig.add_trace(go.Scatter(x=grid,
                                y=model_lr.predict(grid.reshape(-1,1)),
                                mode = 'lines',
                                line = dict(color = 'brown'),
                                name = 'm = {}'.format(degree_m)))    


    st.plotly_chart(fig, use_container_width=True)
    st.divider()


    def MSE(X,y,degree,random_state = 0) :
        model_lr = make_pipeline(StandardScaler(),
                                PolynomialFeatures(degree=degree, include_bias=False),
                                LinearRegression())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
        model_lr.fit(X_train.reshape(-1,1),y_train)
        train_MSE = mean_squared_error(y_train,model_lr.predict(X_train.reshape(-1,1)))
        test_MSE = mean_squared_error(y_test,model_lr.predict(X_test.reshape(-1,1)))
        
        return train_MSE,test_MSE

    p=20
    MSE_matrix = np.zeros((p,2))
    for i in range(p):
        train_MSE, test_MSE = MSE(X=st.session_state['X'],y=st.session_state['y'],degree=i+1)
        MSE_matrix[i,0] = train_MSE
        MSE_matrix[i,1] = test_MSE


    fig = go.Figure()
    # train MSE
    fig.add_trace(go.Scatter(
        x=np.arange(1, p+1), 
        y=MSE_matrix[:,0], 
        mode='lines+markers', 
        marker=dict(symbol='x'),
        name='Train MSE' 
    ))
    # test MSE
    fig.add_trace(go.Scatter(
        x=np.arange(1, p+1), 
        y=MSE_matrix[:,1], 
        mode='lines+markers', 
        marker=dict(symbol='x'),
        name='Test MSE' 
    ))
    fig.update_layout(
        title='Training and Test MSE vs Polynomial Degree(Log Scale)',
        xaxis_title='Degree of Polynomial',
        yaxis_title='Mean Squared Error (log scale)',
        yaxis_type='log', 
        legend_title='MSE Type'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write('The best MSE is {:.5f} with m = {}'.format(np.min(MSE_matrix[:,1]),np.argmin(MSE_matrix[:,1])+1))
if option == 'KNN regression':
    st.title('KNN regression')
    k1 = st.checkbox('k=1')
    k2 = st.checkbox('k=2')
    k3 = st.checkbox('k=3')
    k_checkbox = st.checkbox('custom k')
    if k_checkbox:
        k = st.number_input('Select k', min_value=1,max_value=20, value=5)
    distance = st.checkbox('Distance weight?')
    k_list = [k1,k2,k3]
    color_list = ['blue','red','green']
    
    grid = np.linspace(np.min(st.session_state['X']),np.max(st.session_state['X']),100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state['X'],
                            y=st.session_state['y'],
                            mode = 'markers',
                            marker = dict(color = 'black', size = 5, symbol = 'x'),
                            name = 'Data'))
    for i in range(3):
        if k_list[i] :
            if distance :
                model_lr = KNeighborsRegressor(n_neighbors=i+1,weights='distance')
            else :
                model_lr = KNeighborsRegressor(n_neighbors=i+1)        
            model_lr.fit(st.session_state['X'].reshape(-1,1),st.session_state['y'])
            fig.add_trace(go.Scatter(x=grid,
                                    y=model_lr.predict(grid.reshape(-1,1)),
                                    mode = 'lines',
                                    line = dict(color = color_list[i]),
                                    name = 'k = {}'.format(i+1)))
    if k_checkbox :
        if distance :
            model_lr = KNeighborsRegressor(n_neighbors=k,weights='distance')
        else :
            model_lr = KNeighborsRegressor(n_neighbors=k)
        model_lr.fit(st.session_state['X'].reshape(-1,1),st.session_state['y'])
        fig.add_trace(go.Scatter(x=grid,
                                y=model_lr.predict(grid.reshape(-1,1)),
                                mode = 'lines',
                                line = dict(color = 'brown'),
                                name = 'k = {}'.format(k)))    

    st.plotly_chart(fig, use_container_width=True)
    st.divider()


    def MSE(X,y,k,random_state = 0) :
        if distance :
            model_lr = KNeighborsRegressor(n_neighbors=k,weights='distance')
        else :
            model_lr = KNeighborsRegressor(n_neighbors=k)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
        model_lr.fit(X_train.reshape(-1,1),y_train)
        train_MSE = mean_squared_error(y_train,model_lr.predict(X_train.reshape(-1,1)))
        test_MSE = mean_squared_error(y_test,model_lr.predict(X_test.reshape(-1,1)))
        return train_MSE,test_MSE

    k = st.number_input('Select maximum k', min_value=20,max_value=min([N-10,50]), value=20)
    MSE_matrix = np.zeros((k,2))
    for i in range(k):
        train_MSE, test_MSE = MSE(X=st.session_state['X'],y=st.session_state['y'],k=i+2)
        MSE_matrix[i,0] = train_MSE
        MSE_matrix[i,1] = test_MSE


    fig = go.Figure()
    # train MSE
    fig.add_trace(go.Scatter(
        x=np.arange(2, k+2), 
        y=MSE_matrix[:,0], 
        mode='lines+markers', 
        marker=dict(symbol='x'),
        name='Train MSE' 
    ))
    # test MSE
    fig.add_trace(go.Scatter(
        x=np.arange(2, k+2), 
        y=MSE_matrix[:,1], 
        mode='lines+markers', 
        marker=dict(symbol='x'),
        name='Test MSE' 
    ))
    fig.update_layout(
        title='Training and Test MSE vs k',
        xaxis_title='k',
        yaxis_title='Mean Squared Error',
        legend_title='MSE Type'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write('The best MSE is {:.5f} with k = {}'.format(np.min(MSE_matrix[:,1]),np.argmin(MSE_matrix[:,1])+2))

