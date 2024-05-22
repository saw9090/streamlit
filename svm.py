import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

with st.sidebar :
    st.title('Data')
    data = np.array([
        [2, 8, 'blue'],  
        [3, 4, 'blue'],
        [1, 2, 'blue'],
        [7, 8, 'red'],
        [6, 5, 'red'],
        [8, 3, 'red']
    ])
    df = pd.DataFrame(data, columns=['x', 'y', 'label'])
    edited_df = st.data_editor(df, key='data_editor_key')
    x_coords = edited_df['x'].values
    y_coords = edited_df['y'].values
    labels = edited_df['label'].values

def plot_decision_boundary(X, y, model, title):
    """
    plot_decision_boundary

    Args:
        X (np.ndarray): 2D array of shape (n_samples, 2), where n_samples is the number of samples and each sample has 2 features.
        y (np.ndarray): 1D array of shape (n_samples,), where n_samples is the number of samples. Contains binary labels (0 or 1).
        model: A trained scikit-learn model with a predict method.
        title (str): Title for the plot.
    
    Returns:
        fig: Plotly figure object representing the decision boundary and data points.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    labels = np.array(['blue' if label == 0 else 'red' for label in y])
    symbols = ['circle' if label == 0 else 'square' for label in y]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers+text',
        marker=dict(color=labels, size=20, symbol=symbols),
        text=[str(i) for i in range(len(X))],
        textfont=dict(color='white', size=12),
        textposition='middle center'
    ))

    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.01),
        y=np.arange(y_min, y_max, 0.01),
        z=Z,
        showscale=False,
        colorscale=[[0, 'blue'], [1, 'red']],
        opacity=0.3,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[x_min, x_max], dtick=1),
        yaxis=dict(range=[y_min, y_max], dtick=1),
        showlegend=False,
        width=600,
        height=600
    )

    return fig
st.header('Support Vector Classifier')
option = st.selectbox(
    'Select Model',
    ('linear', 'rbf'))

C = st.select_slider(
        r'Select $C$',
        options=[0.1,1,10]
        ,value=1)

if option == 'rbf' :
    gamma = st.select_slider(
            r'Select $\gamma$',
            options=[0.1,1,10]
            ,value=1)

x_coords = edited_df['x'].astype(float).values
y_coords = edited_df['y'].astype(float).values
labels = edited_df['label'].values

label_dict = {'blue': 0, 'red': 1}
y = np.array([label_dict[label] for label in labels])

X = np.c_[x_coords, y_coords]


if option == 'linear' :
    model = svm.SVC(kernel=option,C=C)
elif option == 'rbf' :
    model = svm.SVC(kernel=option,gamma = gamma,C=C)
model.fit(X, y)
fig = plot_decision_boundary(X, y, model, 'SVM Decision Boundary')

st.plotly_chart(fig)

