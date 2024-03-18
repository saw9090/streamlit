import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots


st.title('Shrinkage Method')
st.header('Data Generation')
st.latex(r'''Model : y_i = \beta_1 x_{i1} + \beta_2 x_{i2} + \epsilon_i,\quad \epsilon_i \sim N(0,\sigma^2) \; and \; i=1,\cdots,N''')

N = st.select_slider(
        r'Select $\N$',
        options=[10,50,100,500,1000]
        ,value=500)

beta_1 = st.select_slider(
    r'Select True $\beta_1$',
    options=np.round(np.linspace(-1,1,21),1)
    ,value=0)

beta_2 = st.select_slider(
    r'Select True $\beta_2$',
    options=np.round(np.linspace(-1,1,21),1)
    ,value=0.5)

sigma = st.select_slider(
    r'Select True $\sigma$',
    options=np.round(np.linspace(0,5,51),1)
    ,value=0.3)


if st.button('Data generation'):
    e = np.random.normal(size=N)
    if 'X' not in st.session_state:
        st.session_state['X'] = np.random.uniform(-5,5,2*N).reshape(N,2)
    if 'y' not in st.session_state:
        st.session_state['y'] = np.squeeze(st.session_state['X'] @ np.array([beta_1,beta_2])) + sigma * np.random.normal(0,1,N)
    st.session_state['X'] = np.random.uniform(-5,5,2*N).reshape(N,2)
    st.session_state['y'] = np.squeeze(st.session_state['X'] @ np.array([beta_1,beta_2])) + sigma * np.random.normal(0,1,N)
    st.write('Done!')
else : st.write('Click button!')

beta_1_range = np.linspace(beta_1-1.2, beta_1+1.2, 481)
beta_2_range = np.linspace(beta_2-1.2, beta_2+1.2, 481)

# 손실 함수 값 계산을 위한 그리드 생성
B1, B2 = np.meshgrid(beta_1_range, beta_2_range)


# 손실 함수 정의
def Loss(beta_1,beta_2) :
    error = st.session_state['y'] - st.session_state['X'] @ np.array([beta_1,beta_2])
    return np.mean(error**2)

# 손실행렬 계산
if 'Loss' not in st.session_state:
    Loss_mat = np.zeros(B1.shape)
    for i in range(B1.shape[0]):
        for j in range(B1.shape[1]):
            Loss_mat[i, j] = Loss(B1[i,j],B2[i,j])
    st.session_state['Loss'] = Loss_mat


option = st.selectbox(
    'Model Selection',
    ('Ridge', 'Lasso')
)
if option == 'Ridge' :
    st.latex(r'''Loss(\beta_1,\beta_2) = \sum^{N}_{i=1}\left(y_i-(\beta_1 x_{i1} + \beta_2 x_{i2})\right)^2 \; ,subjects \; to \; \beta_1^2+\beta_2^2 < C^2''')
    st.latex(r'''\left(\Longleftrightarrow  Loss(\beta_1,\beta_2) = \sum^{N}_{i=1}\left(y_i-(\beta_1 x_{i1} + \beta_2 x_{i2})\right)^2 + \lambda(\beta_1^2+\beta_2^2) \right)   ''')
elif option == 'Lasso' :
    st.latex(r'''Loss(\beta_1,\beta_2) = \sum^{N}_{i=1}\left(y_i-(\beta_1 x_{i1} + \beta_2 x_{i2})\right)^2 \; ,subjects \; to \; |\beta_1|+|\beta_2| < C''')
    st.latex(r'''\left(\Longleftrightarrow  Loss(\beta_1,\beta_2) = \sum^{N}_{i=1}\left(y_i-(\beta_1 x_{i1} + \beta_2 x_{i2})\right)^2 + \lambda(|\beta_1|+|\beta_2|) \right)   ''')


C = st.select_slider(
    r'Select $C$',
    options=np.round(np.linspace(0.1,2,40),1),
    value=1)

# 페널티 설정
if option == 'Ridge' :
    index_ = (B1**2+B2**2 <= C**2)
elif option == 'Lasso' :
    index_ = (np.abs(B1)+np.abs(B2) <= C)

proj = np.zeros(B1.shape)
Loss_mat_copy = np.copy(st.session_state['Loss'])
proj[~index_] = Loss_mat_copy[~index_] = np.nan

# 조건 내의 최소값 찾기
min_index_basic = np.unravel_index(np.argmin(st.session_state['Loss']),Loss_mat_copy.shape)
beta_1_min_basic = B1[min_index_basic]
beta_2_min_basic = B2[min_index_basic]

min_index = np.unravel_index(np.nanargmin(Loss_mat_copy),Loss_mat_copy.shape)
beta_1_min = B1[min_index]
beta_2_min = B2[min_index]

fig = make_subplots(rows=1, cols=2,
                    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                    subplot_titles=("OLS", option))

# 첫 번째 subplot
beta_1_min_basic_truncated = int(beta_1_min_basic * 100) / 100
beta_2_min_basic_truncated = int(beta_2_min_basic * 100) / 100
legend_text_basic_1 = f'OLS estimator (β1={beta_1_min_basic:.2f}, β2={beta_2_min_basic:.2f})'
fig.add_trace(go.Surface(z=st.session_state['Loss'], x=beta_1_range, y=beta_2_range, showscale=False, colorscale='Cividis'), row=1, col=1)
# fig.add_trace(go.Surface(z=np.zeros(B1.shape), x=beta_1_range, y=beta_2_range, showscale=False),row=1,col=1)
fig.add_trace(go.Scatter3d(x=[beta_1_min_basic],
                           y=[beta_2_min_basic],
                           z=[st.session_state['Loss'][min_index_basic]],
                           mode='markers',
                           marker=dict(size=5, color='red'),
                           name=legend_text_basic_1),
              row=1, col=1)

# 두 번째 subplot
beta_1_min_truncated = int(beta_1_min * 100) / 100
beta_2_min_truncated = int(beta_2_min * 100) / 100
legend_text_basic_2 =  f'{option} estimator (β1={beta_1_min:.2f}, β2={beta_2_min:.2f})'
fig.add_trace(go.Surface(z=Loss_mat_copy, x=beta_1_range, y=beta_2_range, showscale=False, colorscale='Cividis'), row=1, col=2)
# fig.add_trace(go.Surface(z=proj, x=beta_1_range, y=beta_2_range, showscale=False),row=1,col=2)
fig.add_trace(go.Scatter3d(x=[beta_1_min],
                           y=[beta_2_min],
                           z=[Loss_mat_copy[min_index]],
                           mode='markers',
                           marker=dict(size=5, color='blue'),
                           name=legend_text_basic_2),
              row=1, col=2)

# 각 서브플롯의 z축 범위 조정
fig.update_layout(scene=dict(zaxis=dict(range=[np.nanmin(st.session_state['Loss']), np.nanmax(st.session_state['Loss'])])),
                  scene2=dict(zaxis=dict(range=[np.nanmin(Loss_mat_copy), np.nanmax(Loss_mat_copy)])))
fig.update_layout(
    scene=dict(
        xaxis_title='β₁',
        yaxis_title='β₂',
        zaxis_title='Loss'
    ),
    scene2=dict(
        xaxis_title='β₁',
        yaxis_title='β₂',
        zaxis_title='Loss'
    )
)

fig.update_layout(
    title_text='Estimator with Loss Function',
    title_font=dict(size=24) 
)
st.plotly_chart(fig)

# Ridge 경계선
def calculate_ridge_boundary(C, steps=100):
    theta = np.linspace(0, 2*np.pi, steps)
    x = C * np.cos(theta)
    y = C * np.sin(theta)
    return x, y

# Lasso 경계선
def calculate_lasso_boundary(C, steps=100):
    angles = np.linspace(0, 2 * np.pi, steps)
    x = C * np.cos(angles) / (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))
    y = C * np.sin(angles) / (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))
    return x, y

if option == 'Ridge':
    boundary_x, boundary_y = calculate_ridge_boundary(C)
elif option == 'Lasso':
    boundary_x, boundary_y = calculate_lasso_boundary(C)

fig = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]])

fig.add_trace(go.Contour(
    z=st.session_state['Loss'], x=beta_1_range, y=beta_2_range,
    colorscale='Cividis', contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=12, color='white'))
)) # Loss 등고선 그래프
fig.add_trace(go.Scatter(x=boundary_x, y=boundary_y, mode='lines', name=f'{option} Boundary', line=dict(color='red'))) # 페널티 경계선

fig.update_layout(
    xaxis_title='β₁',
    yaxis_title='β₂',
    title_text=f'Loss Function Contour with {option} Penalty Boundary',
    title_font=dict(size=24)
)

max_range = max(beta_1_range[-1] - beta_1_range[0], beta_2_range[-1] - beta_2_range[0])
x_center = (beta_1_range[-1] + beta_1_range[0]) / 2
y_center = (beta_2_range[-1] + beta_2_range[0]) / 2

fig.update_xaxes(range=[x_center - max_range / 2, x_center + max_range / 2])
fig.update_yaxes(range=[y_center - max_range / 2, y_center + max_range / 2])

st.plotly_chart(fig)
