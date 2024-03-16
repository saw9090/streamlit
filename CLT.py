import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm,chi2

st.title('Central Limit Theorem')
st.header('Central Limit Theorem(CLT)')
st.write('Let '+
        r'$X_1, X_2, \cdots, X_n$'+
        ' denote a random sample of '+
        r'$n$'+
        ' independent observations from a population with overall expected value (average) '+
        r'$\mu$'+
        ' and finite variance '+
        r'$\sigma^2$'+
        ' , and let '+
        r'$\bar{X}_n$'+
        'denote sample mean of that sample(which is itself a random variable). Then') 
st.latex(r'''\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0,1)
''')

st.header('Data Generation')

N = st.select_slider(
        r'Select $n$',
        options=[10,100,500,1000]
        ,value=100)
ns = st.select_slider(
        'Select the number of sample means ' r'$NS$',
        options=[10,100,500,1000]
        ,value=100)
option = st.selectbox(
    'Distribution Selection',
    ('Uniform Distribution', 'Normal Distribution', 'Chi-squared Distribution',)
)

if option == 'Uniform Distribution' :
    st.latex(r'''X \sim U(-a,a)''')
    a = st.select_slider(
        'Select '+r'$a$',
        options = range(1,6),
        value = 3)
    mat = np.random.uniform(-a,a,N*ns).reshape(N,ns)
    sample_mean = np.mean(mat,axis=0)
    grid1 = np.linspace(-a-1,a+1,1000)
    grid2 = np.linspace(min(sample_mean),max(sample_mean),1000)
    approx = norm.pdf(grid2,0,a/(3*N)**(1/2))

    fig, ax = plt.subplots(1,2, figsize=(8,3))
    ax[0].plot(grid1,1/a* np.ones(1000),color='black')
elif option == 'Normal Distribution' :
    st.latex(r'''X \sim N(\mu,\sigma^2)''')
    mu = st.select_slider(
        r'Select $\mu$',
        options = [-3,-2,-1,0,1,2,3],
        value = 0)
    sigma = st.select_slider(
        r'Select $\sigma$',
        options=np.round(np.linspace(0.1,4,40),1)
        ,value=1)
    mat = np.random.normal(mu,sigma,N*ns).reshape(N,ns)
    sample_mean = np.mean(mat,axis=0)
    grid1 = np.linspace(mu-3*sigma,mu+3*sigma,1000)
    grid2 = np.linspace(min(sample_mean),max(sample_mean),1000)
    approx = norm.pdf(grid2,mu,sigma/(N)**(1/2))

    fig, ax = plt.subplots(1,2, figsize=(8,3))
    ax[0].plot(grid1,norm.pdf(grid1,mu,sigma),color='black')
elif option == 'Chi-squared Distribution' :
    st.latex(r'''X \sim \chi^2(k)''')
    df = st.select_slider(
        'Select degree of freedom '+r'$k$',
        options=[1,2,3,4,5,6,7,8,9,10],
        value=1)
    mat = np.random.chisquare(df,N*ns).reshape(N,ns)
    sample_mean = np.mean(mat,axis=0)
    grid1 = np.linspace(0,20,1000)
    grid2 = np.linspace(df-3*(2*df/N)**(1/2),df+3*(2*df/N)**(1/2),1000)
    approx = norm.pdf(grid2,df,(2*df/N)**(1/2))

    fig, ax = plt.subplots(1,2, figsize=(8,3))
    ax[0].plot(grid1,chi2.pdf(grid1,df),color='black')

ax[0].set_title(option)
sns.histplot(sample_mean,stat='density',kde=True,ax=ax[1],color='blue',label='Sample Histogram of '+r'$\bar{X}$')
ax[1].plot(grid2,approx,color='black',label = 'Distribution by CLT')
ax[1].set_title('Approximate Distribution')
ax[1].legend(loc='upper right', bbox_to_anchor=(1.8,1.0))
st.pyplot(fig)

if st.button('Data Regeneration'):
    pass


