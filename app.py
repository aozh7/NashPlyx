import json

import mantel
from dash import Dash, html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from io import StringIO
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.decomposition import PCA
import itertools
from more_itertools import unique_everseen

app = Dash(__name__, suppress_callback_exceptions=True)

np.random.seed(42)

tab_style = {
    'borderTop': '1px solid #696969',
    'borderBottom': '1px solid #696969',
    'backgroundColor': '#B0C4DE',
    'color': 'LightSlateGray',
    'font-size': '25px',
    'font-family': 'system-ui',
    'padding': '6px'
}

tab_selected_style = {
    'borderTop': '1px solid #696969',
    'borderBottom': '1px solid #696969',
    'backgroundColor': '#778899',
    'color': 'DarkSlateGray',
    'font-size': '25px',
    'font-family': 'system-ui',
    'fontWeight': 'bold',
    'padding': '6px'
}

sub_tab_style = {
    'borderTop': '1px solid #696969',
    'borderBottom': '1px solid #696969',
    'backgroundColor': '#D8BFD8',
    'color': 'LightSlateGray',
    'font-size': '20px',
    'font-family': 'system-ui',
    'padding': '40px',
    'height': '100px',
}

sub_tab_selected_style = {
    'borderTop': '1px solid #696969',
    'borderBottom': '1px solid #696969',
    'backgroundColor': '#DDA0DD',
    'color': 'DarkSlateGray',
    'font-size': '20px',
    'font-family': 'system-ui',
    'fontWeight': 'bold',
    'padding': '40px',
    'height': '100px',
}


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}


app.layout = html.Div(children=[
    html.Div([
        html.Label('Dataset', style={'font-weight': 'bold', 'font-family': 'system-ui',
                                     'color': 'DarkSlateGray', 'font-size': '30px', 'marginLeft': '30px'}),
        dcc.Dropdown(id='dataset-dropdown', options=['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA',
                    'GBM', 'HNSC', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'KICH', 'KIRC', 'KIRP', 'MESO', 'PAAD',
                    'PCPG', 'PRAD', 'READ', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM'], value='ACC',
                    searchable=False, style={'width': '40%', 'marginLeft': '20px', 'padding': '10px',
                                             'textAlign': 'center', 'color': 'DarkRed', 'font-size': '30px',
                                             'font-family': 'system-ui', 'height': '20px', 'marginBottom': '40px'}),
        html.Label('Genes', style={'font-weight': 'bold', 'font-family': 'system-ui', 'text-align': 'left',
                                   'color': 'DarkSlateGray', 'font-size': '20px', 'marginLeft': '20px',
                                   'padding': '10px'}),
        dcc.Dropdown(id='genes-list', multi=True, clearable=True,
                     style={'marginLeft': '20px', 'color': 'DarkSlateGray', 'padding': '10px', 'font-size': '20px',
                            'font-family': 'system-ui'})]),
    html.Div([
        dcc.Tabs(id='exp-meth-out', value='tab-0', children=[
            dcc.Tab(label='About', value='tab-0', children=[
                html.H1('NashPlyx', style={'font-family': 'system-ui', 'color': 'DarkSlateGray',
                                           'font-size': '30px', 'marginLeft': '200px', 'marginTop': '100px',
                                           'marginBottom': '50px'}),
                dcc.Markdown('''
                            NashPlyx is a tool for visualizing and comparison of transcriptomic and epitranscriptomic 
                            WGCNA results of TCGA datasets. Users can build transcriptional and epitranscriptional 
                            gene networks from the main "Transcriptome" and "Epitranscriptome" tabs, respectively. The 
                            two symmetric matrices generated from these tabs can be compared with Mantel's test from the 
                            main "Comparison" tab on the right. Tables generated from the "Data table" tabs can be downloaded
                            
                            This app was built with Plotly 
                            ''', style={'font-family': 'system-ui', 'font-size': '14px', 'marginLeft': '300px'})],
            style={'borderTop': '1px solid #696969', 'borderBottom': '1px solid #696969', 'backgroundColor': '#DarkRed',
                   'color': 'LightSlateGray', 'font-size': '25px', 'font-family': 'system-ui', 'padding': '6px',
                   'width': '250px', 'marginLeft': '200px'},
            selected_style={'borderTop': '1px solid #696969', 'borderBottom': '1px solid #696969', 'marginLeft': '220px',
                            'backgroundColor': 'Maroon', 'color': 'DarkSlateGray', 'font-size': '25px', 'font-family': 'system-ui',
                            'fontWeight': 'bold', 'padding': '6px', 'width': '220px'}),
            dcc.Tab(label='Transcriptome', value='tab-1', children=[
                dcc.Tabs(id='tr-sub-tabs', value='tr-sub-tab-1', children=[
                    dcc.Tab(label='Data table', value='tr-sub-tab-1', style=sub_tab_style, selected_style=sub_tab_selected_style,
                            children=[html.Div([
                                html.Button('Download WGCNA Data', id='exp-data-button',
                                            style={'font-family': 'system-ui', 'marginTop': '15px',
                                                   'text-align': 'left', 'color': 'DarkSlateGray', 'font-size': '16px',
                                                   'marginLeft': '30px', 'padding': '10px'}),
                                dcc.Download(id='exp-data-down'),
                            #dbc.Alert('No gene selected.', id='tr-sub-tab-1-alert', color='warning', dismissable=True)],
                            ]), html.Div(id='tr-data-tab')]),
                    dcc.Tab(label='Box plot', value='tr-sub-tab-2', style=sub_tab_style, selected_style=sub_tab_selected_style,
                            children=[html.Div(id='tr-box-tab')]),
                    dcc.Tab(label='Network', value='tr-sub-tab-3', style=sub_tab_style, selected_style=sub_tab_selected_style,
                            children=[html.Div(id='tr-netx-tab')]),
                    dcc.Tab(label='Heatmap', value='tr-sub-tab-4', style=sub_tab_style, selected_style=sub_tab_selected_style,
                            children=[html.Div(id='tr-heat-tab')]),
                    dcc.Tab(label='Dendrogram', value='tr-sub-tab-5', style=sub_tab_style, selected_style=sub_tab_selected_style,
                            children=[html.Div(id='tr-dend-tab')]),
                    dcc.Tab(label='PCA', value='tr-sub-tab-6', style=sub_tab_style, selected_style=sub_tab_selected_style,
                            children=[html.Div([
                                dcc.Slider(id='tr-pca-slide', min=1, value=2, step=1)]),
                                html.Div(id='tr-pca-tab')])
                ])], style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Epigenome', value='tab-2', children=[
                dcc.Tabs(id='epi-sub-tabs', value='epi-sub-tab-1', children=[
                    dcc.Tab(label='Data table', value='epi-sub-tab-1', style=sub_tab_style,selected_style=sub_tab_selected_style,
                            children=[html.Div([
                                html.Button('Download WGCNA Data', id='meth-data-button',
                                            style={'font-family': 'system-ui', 'marginTop': '15px', 'text-align': 'left',
                                                   'color': 'DarkSlateGray', 'font-size': '16px', 'marginLeft': '30px',
                                                   'padding': '10px'}),
                                dcc.Download(id='meth-data-down'),
                            ]), html.Div(id='epi-data-tab')]),
                    dcc.Tab(label='Box plot', value='epi-sub-tab-2', style=sub_tab_style, selected_style=sub_tab_selected_style,
                            children=[html.Div(id='epi-box-tab')]),
                    dcc.Tab(label='Network', value='epi-sub-tab-3', style=sub_tab_style, selected_style=sub_tab_selected_style,
                            children=[html.Div(id='epi-netx-tab')]),
                    dcc.Tab(label='Heatmap', value='epi-sub-tab-4', style=sub_tab_style, selected_style=sub_tab_selected_style,
                            children=[html.Div(id='epi-heat-tab')]),
                    dcc.Tab(label='Dendrogram', value='epi-sub-tab-5', style=sub_tab_style,
                            selected_style=sub_tab_selected_style, children=[html.Div(id='epi-dend-tab')]),
                    dcc.Tab(label='PCA', value='epi-sub-tab-6', style=sub_tab_style,
                            selected_style=sub_tab_selected_style,
                            children=[html.Div([
                                dcc.Slider(id='epi-pca-slide', min=1, value=2, step=1)]),
                                html.Div(id='epi-pca-tab')])
                ])], style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Comparison', value='tab-3',
                    children=[html.Div([
                        html.Button('Download Mantel test results', id='mant-data-button',
                        style={'font-family': 'system-ui', 'marginTop': '15px', 'text-align': 'left',
                               'color': 'DarkSlateGray', 'font-size': '16px', 'marginLeft': '30px', 'padding': '10px'})
                ]), html.Div(id='mant-tab')],
                    style={'borderTop': '1px solid #696969', 'borderBottom': '1px solid #696969',
                           'backgroundColor': 'MediumSlateBlue', 'color': 'LightSlateGray', 'font-size': '25px',
                           'font-family': 'system-ui', 'padding': '6px', 'width': '250px'},
                    selected_style={'borderTop': '1px solid #696969', 'borderBottom': '1px solid #696969',
                            'backgroundColor': 'SlateBlue', 'color': 'DarkSlateGray', 'font-size': '25px',
                            'font-family': 'system-ui', 'fontWeight': 'bold', 'padding': '6px', 'width': '400px'})
        ]),
        html.Div(id='exp-meth')]),
    dcc.Store('exp-data'), dcc.Store('exp-sim'),
    dcc.Store('meth-data'), dcc.Store('meth-sim')
], style={'backgroundColor': 'GhostWhite'})


@app.callback(
    Output('genes-list', 'options'),
    Input('dataset-dropdown', 'value'),
)
def update_genes_list(dataset):
    if dataset is None:
        raise PreventUpdate
    dataset_genes = np.load('/Volumes/SanDisk/Indices/' + dataset.lower() + '_exp_genes.npy', allow_pickle=True)
    return dataset_genes


@app.callback(
    Output('exp-data', 'data'),
    [Input('dataset-dropdown', 'value'),
     Input('genes-list', 'value')]
)
def update_exp(dataset, genes):
    if dataset is None:
        raise PreventUpdate
    if genes is None:
        raise PreventUpdate
    exp_genes = np.load('/Volumes/SanDisk/Indices/' + dataset.lower() + '_exp_genes.npy', allow_pickle=True)
    exp = pd.read_csv('/Volumes/SanDisk/' + dataset.lower() + '_tom_diss.tsv.gz', compression='gzip', usecols=genes)
    if len(genes) == 0:
        raise PreventUpdate
    exp = exp.set_index(exp_genes)
    exp = exp.loc[genes, genes]
    return exp.to_json()


@app.callback(
    Output('exp-data-down', 'data'),
    [Input('exp-data-button', 'n_clicks'),
     Input('dataset-dropdown', 'value'),
     Input('exp-data', 'data')]
)
def exp_down(n_clicks, dataset, exp):
    if n_clicks is None:
        raise PreventUpdate
    if dataset is None:
        raise PreventUpdate
    if exp is None:
        raise PreventUpdate
    return dcc.send_data_frame(pd.read_json(StringIO(exp)).to_csv, filename=dataset.lower() + '_exp_genes.csv')


@app.callback(
    Output('meth-data', 'data'),
    [Input('dataset-dropdown', 'value'),
     Input('genes-list', 'value')]
)
def update_meth(dataset, genes):
    if dataset is None:
        raise PreventUpdate
    if genes is None:
        raise PreventUpdate
    meth_genes = np.load('/Volumes/SanDisk/Indices/' + dataset.lower() + '_meth_genes.npy', allow_pickle=True)
    meth = pd.read_csv('/Volumes/SanDisk/' + dataset.lower() + '_meth_tom_diss.tsv.gz', compression='gzip', usecols=genes)
    if len(genes) == 0:
        raise PreventUpdate
    meth = meth.set_index(meth_genes)
    meth = meth.loc[genes, genes]
    return meth.to_json()


@app.callback(
    Output('meth-data-down', 'data'),
    [Input('meth-data-button', 'n_clicks'),
     Input('dataset-dropdown', 'value'),
     Input('meth-data', 'data')]
)
def meth_down(n_clicks, dataset, meth):
    if n_clicks is None:
        raise PreventUpdate
    if dataset is None:
        raise PreventUpdate
    if meth is None:
        raise PreventUpdate
    return dcc.send_data_frame(pd.read_json(StringIO(meth)).to_csv, filename=dataset.lower() + '_meth_genes.csv')


@app.callback(
    [Output('tr-pca-slide', 'max'),
     Output('epi-pca-slide', 'max')],
    Input('genes-list', 'value')
)
def update_slider(genes):
    if genes is None:
        raise PreventUpdate
    return [len(genes), len(genes)]

#
# @app.callback(
#     Output('alert_message', 'is_open'),
#     Input('genes-list', 'value')
# )
# def tr_tab1_alert(genes):
#     if len(genes) == 0:
#         return


@app.callback(
    Output('exp-sim', 'data'),
    Input('exp-data', 'data')
)
def render_exp(exp):
    if exp is None:
        raise PreventUpdate

    exp_genes = pd.read_json(StringIO(exp))
    exp_diss = df_to_plotly(exp_genes)
    exp_sim = (1 - np.array(exp_diss['z']))
    exp_sim = pd.DataFrame(exp_sim, columns=exp_genes.columns, index=exp_genes.index)
    return exp_sim.to_json()


@app.callback(
    Output('meth-sim', 'data'),
    Input('meth-data', 'data')
)
def render_meth(meth):
    if meth is None:
        raise PreventUpdate

    meth_genes = pd.read_json(StringIO(meth))
    meth_diss = df_to_plotly(meth_genes)
    meth_sim = (1 - np.array(meth_diss['z']))
    meth_sim = pd.DataFrame(meth_sim, columns=meth_genes.columns, index=meth_genes.index)
    return meth_sim.to_json()


@app.callback(
Output('mant-tab', 'children'),
    [Input('exp-meth-out', 'value'),
     Input('exp-data', 'data'),
     Input('meth-data', 'data')]
)
def render_mant(tab, exp, meth):
    if exp is None:
        raise PreventUpdate
    if meth is None:
        raise PreventUpdate

    if tab == 'tab-3':
        exp_diss = pd.read_json(StringIO(exp))
        meth_diss = pd.read_json(StringIO(meth))
        res = mantel.test(exp_diss, meth_diss)
        res_tab = [res[0], res[1], res[2]]
        fig = go.Figure(go.Table(header=dict(values=['r', 'p', 'z']),
                                 cells=dict(values=res_tab)))
        return html.Div([
            dcc.Graph(figure=fig)
        ])


@app.callback(
Output('mant-data-down', 'data'),
    [Input('mant-data-button', 'n_clicks'),
     Input('dataset-dropdown', 'value'),
     Input('exp-meth-out', 'value'),
     Input('exp-data', 'data'),
     Input('meth-data', 'data')]
)
def mant_down(n_clicks, dataset, tab, exp, meth):
    if n_clicks is None:
        raise PreventUpdate
    if dataset is None:
        raise PreventUpdate
    if exp is None:
        raise PreventUpdate
    if meth is None:
        raise PreventUpdate

    if tab == 'tab-3':
        exp_diss = pd.read_json(StringIO(exp))
        meth_diss = pd.read_json(StringIO(meth))
        res = mantel.test(exp_diss, meth_diss)
        res_tab = [res[0], res[1], res[2]]
        tab = go.Table(header=dict(values=['r', 'p', 'z']), cells=dict(values=res_tab))
        mant_tab = pd.DataFrame(tab)
        return mant_tab.to_json()


@app.callback(
    Output('tr-data-tab', 'children'),
    [Input('exp-meth-out', 'value'),
     Input('tr-sub-tabs', 'value'),
     Input('exp-data', 'data')]
)
def render_tabs(tab, tr_subtab, exp):
    if exp is None:
        raise PreventUpdate
    if tab == 'tab-1':
        if tr_subtab == 'tr-sub-tab-1':
            exp_diss = pd.read_json(StringIO(exp))
            exp_diss = df_to_plotly(exp_diss)
            fig = go.Figure(go.Table(columnwidth=75,
                                     header=dict(values=exp_diss['x']),
                                     cells=dict(values=exp_diss['z'])))
            return html.Div([
                dcc.Graph(figure=fig)
            ])
        else:
            pass
    else:
        pass


@app.callback(
    Output('epi-data-tab', 'children'),
    [Input('exp-meth-out', 'value'),
     Input('epi-sub-tabs', 'value'),
     Input('meth-data', 'data')]
)
def render_tabs(tab, epi_subtab, meth):
    if meth is None:
        raise PreventUpdate
    if tab == 'tab-2':
        if epi_subtab == 'epi-sub-tab-1':
            meth_diss = pd.read_json(StringIO(meth))
            meth_diss = df_to_plotly(meth_diss)
            fig = go.Figure(go.Table(columnwidth=75,
                                     header=dict(values=meth_diss['x']),
                                     cells=dict(values=meth_diss['z'])))
            return html.Div([
                dcc.Graph(figure=fig)
            ])
        else:
            pass
    else:
        pass


@app.callback(
    Output('tr-box-tab', 'children'),
    [Input('dataset-dropdown', 'value'),
     Input('exp-meth-out', 'value'),
     Input('tr-sub-tabs', 'value'),
     Input('exp-sim', 'data')]
)
def render_tabs(dataset, tab, tr_subtab, exp_sim):
    if exp_sim is None:
        raise PreventUpdate
    if tab == 'tab-1':
        if tr_subtab == 'tr-sub-tab-2':
            exp_sim = pd.read_json(StringIO(exp_sim))
            exp_mod = exp_sim.where(np.triu(np.ones(exp_sim.shape), k=1).astype(np.bool_))
            exp_unq = exp_mod.unstack().dropna()
            exp_unq = exp_unq[[z for z in exp_unq.index if z[0] is not z[1]]]
            exp_unq.index = exp_unq.index.map('_'.join)
            fig = px.box(exp_unq, orientation='h', points='all', hover_name=exp_unq.index, color=exp_unq.index)
            fig.update_layout(xaxis_title='WGCNA similarity', yaxis_title='Gene pair',
                              font=dict(size=12)).update_yaxes(showticklabels=False)

            config = {'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image', 'height': 800,
                                               'width': 1200, 'scale': 10}}
            return html.Div([
                dcc.Graph(figure=fig, config=config)
            ])
        else:
            pass
    else:
        pass


@app.callback(Output('epi-box-tab', 'children'),
                [Input('dataset-dropdown', 'value'),
                 Input('exp-meth-out', 'value'),
                 Input('epi-sub-tabs', 'value'),
                 Input('meth-sim', 'data')]
)
def render_tabs(dataset, tab, epi_subtab, meth_sim):
    if meth_sim is None:
        raise PreventUpdate
    if tab == 'tab-2':
        if epi_subtab == 'epi-sub-tab-2':
            meth_sim = pd.read_json(StringIO(meth_sim))
            meth_mod = meth_sim.where(np.triu(np.ones(meth_sim.shape), k=1).astype(np.bool_))
            meth_unq = meth_mod.unstack().dropna()
            meth_unq = meth_unq[[z for z in meth_unq.index if z[0] is not z[1]]]
            meth_unq.index = meth_unq.index.map('_'.join)
            fig = px.box(meth_unq, orientation='h', points='all', hover_name=meth_unq.index, color=meth_unq.index)
            fig.update_layout(xaxis_title='WGCNA similarity', yaxis_title='Gene pair',
                              font=dict(size=12)).update_yaxes(showticklabels=False)
            config = {'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image', 'height': 800,
                                               'width': 1200, 'scale': 10}}
            return html.Div([
                dcc.Graph(figure=fig, config=config)
            ])
        else:
            pass
    else:
        pass


@app.callback(
    Output('tr-netx-tab', 'children'),
    [Input('dataset-dropdown', 'value'),
     Input('exp-meth-out', 'value'),
     Input('tr-sub-tabs', 'value'),
     Input('exp-sim', 'data')]
)
def render_tabs(dataset, tab, tr_subtab, exp_sim):
    if exp_sim is None:
        raise PreventUpdate
    if tab == 'tab-1':
        if tr_subtab == 'tr-sub-tab-3':
            exp_sim = pd.read_json(StringIO(exp_sim))
            nxgraph = nx.from_pandas_adjacency(exp_sim, nx.MultiGraph)
            w = nx.get_edge_attributes(nxgraph, 'weight')
            p = nx.shell_layout(nxgraph)
            nxgraph_nodes = p.keys()
            nxgraph_edges = list(itertools.permutations(nxgraph_nodes, 2))
            nxgraph_edges = list(unique_everseen(nxgraph_edges, key=frozenset))

            for x in list(w.keys()):
                if w[x] == 0:
                    del w[x]

            edge_x = []
            edge_y = []
            for edge in nxgraph_edges:
                x0, y0 = p[edge[0]]
                x1, y1 = p[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

            edge_x = np.array(edge_x).reshape(len(nxgraph_edges), 3)
            edge_y = np.array(edge_y).reshape(len(nxgraph_edges), 3)

            def make_edge(n1, n2, width):
                return go.Scatter(x=n1, y=n2, line=dict(width=width, color='LightSlateGray'), mode='lines')

            edge_list = []
            for i in range(len(nxgraph_edges)):
                trace = make_edge(edge_x[i], edge_y[i], list(w.values())[i])
                edge_list.append(trace)

            node_x = []
            node_y = []
            for node in nxgraph_nodes:
                x, y = p[node]
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(nxgraph.nodes),
                                    textfont=dict(color='DarkSlateGray', size=16),
                                    marker=dict(color=list(range(len(nxgraph.nodes))), opacity=0.5,
                                                size=80, colorscale='Viridis'))
            node_trace.text = list(nxgraph.nodes)
            layout = go.Layout(showlegend=False, hovermode='closest',
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            fig = go.Figure(layout=layout)
            for trace in edge_list:
                fig.add_trace(trace)
            fig.add_trace(node_trace)
            config = {'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image', 'height': 1500,
                                               'width': 2000, 'scale': 5}}

            return html.Div([
                dcc.Graph(figure=fig, config=config)
            ])
        else:
            pass
    else:
        pass


@app.callback(
    Output('epi-netx-tab', 'children'),
    [Input('dataset-dropdown', 'value'),
     Input('exp-meth-out', 'value'),
     Input('epi-sub-tabs', 'value'),
     Input('meth-sim', 'data')]
)
def render_tabs(dataset, tab, epi_subtab, meth_sim):
    if meth_sim is None:
        raise PreventUpdate
    if tab == 'tab-2':
        if epi_subtab == 'epi-sub-tab-3':
            meth_sim = pd.read_json(StringIO(meth_sim))
            nxgraph = nx.from_pandas_adjacency(meth_sim, nx.MultiGraph)
            w = nx.get_edge_attributes(nxgraph, 'weight')
            p = nx.shell_layout(nxgraph)
            nxgraph_nodes = p.keys()
            nxgraph_edges = list(itertools.permutations(nxgraph_nodes, 2))
            nxgraph_edges = list(unique_everseen(nxgraph_edges, key=frozenset))

            for x in list(w.keys()):
                if w[x] == 0:
                    del w[x]

            edge_x = []
            edge_y = []
            for edge in nxgraph_edges:
                x0, y0 = p[edge[0]]
                x1, y1 = p[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

            edge_x = np.array(edge_x).reshape(len(nxgraph_edges), 3)
            edge_y = np.array(edge_y).reshape(len(nxgraph_edges), 3)

            def make_edge(n1, n2, width):
                return go.Scatter(x=n1, y=n2, line=dict(width=width, color='LightSlateGray'), mode='lines')

            edge_list = []
            for i in range(len(nxgraph_edges)):
                trace = make_edge(edge_x[i], edge_y[i], list(w.values())[i])
                edge_list.append(trace)

            node_x = []
            node_y = []
            for node in nxgraph_nodes:
                x, y = p[node]
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(nxgraph.nodes),
                                    textfont=dict(color='DarkSlateGray', size=16),
                                    marker=dict(color=list(range(len(nxgraph.nodes))), opacity=0.5,
                                                size=80, colorscale='Turbo'))
            node_trace.text = list(nxgraph.nodes)
            layout = go.Layout(showlegend=False, hovermode='closest',
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            fig = go.Figure(layout=layout)
            for trace in edge_list:
                fig.add_trace(trace)
            fig.add_trace(node_trace)
            config = {'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image', 'height': 1500,
                                               'width': 2000, 'scale': 5}}
            return html.Div([
                dcc.Graph(figure=fig, config=config)
            ])
        else:
            pass
    else:
        pass


@app.callback(
    Output('tr-heat-tab', 'children'),
    [Input('dataset-dropdown', 'value'),
     Input('exp-meth-out', 'value'),
     Input('tr-sub-tabs', 'value'),
     Input('exp-sim', 'data')]
)
def render_tabs(dataset, tab, tr_subtab, exp_sim):
    if exp_sim is None:
        raise PreventUpdate
    if tab == 'tab-1':
        exp_sim = pd.read_json(StringIO(exp_sim))
        exp_simm = exp_sim.mask(exp_sim == 0)
        if tr_subtab == 'tr-sub-tab-4':
            fig = go.Figure(go.Heatmap(df_to_plotly(exp_simm), colorscale='Viridis'))
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
            conf = {'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image', 'height': 1500,
                                             'width': 2000, 'scale': 10}}
            return html.Div([
                dcc.Graph(figure=fig, config=conf)
            ])
        else:
            pass
    else:
        pass


@app.callback(
    Output('epi-heat-tab', 'children'),
    [Input('dataset-dropdown', 'value'),
     Input('exp-meth-out', 'value'),
     Input('epi-sub-tabs', 'value'),
     Input('meth-sim', 'data')]
)
def render_tabs(dataset, tab, epi_subtab, meth_sim):
    if meth_sim is None:
        raise PreventUpdate
    if tab == 'tab-2':
        meth_sim = pd.read_json(StringIO(meth_sim))
        meth_simm = meth_sim.mask(meth_sim == 0)
        if epi_subtab == 'epi-sub-tab-4':
            fig = go.Figure(go.Heatmap(df_to_plotly(meth_simm), colorscale='Sunset'))
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
            conf = {'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image', 'height': 1000,
                                             'width': 1500, 'scale': 10}}
            return html.Div([
                dcc.Graph(figure=fig, config=conf)
            ])
        else:
            pass
    else:
        pass


@app.callback(
    Output('tr-dend-tab', 'children'),
    [Input('dataset-dropdown', 'value'),
     Input('exp-meth-out', 'value'),
     Input('tr-sub-tabs', 'value'),
     Input('exp-sim', 'data')]
)
def render_dend(dataset, tab, tr_subtab, exp_sim):
    if exp_sim is None:
        raise PreventUpdate
    if tab == 'tab-1':
        if tr_subtab == 'tr-sub-tab-5':
            exp_sim = pd.read_json(StringIO(exp_sim))
            if len(exp_sim) < 2:
                raise PreventUpdate
            fig = go.Figure(ff.create_dendrogram(exp_sim, orientation='left', labels=exp_sim.index))
            conf = {'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image', 'height': 1000,
                                             'width': 1500, 'scale': 10}}
            return html.Div([
                dcc.Graph(figure=fig, config=conf)
            ])
        else:
            pass
    else:
        pass


@app.callback(
    Output('epi-dend-tab', 'children'),
    [Input('dataset-dropdown', 'value'),
     Input('exp-meth-out', 'value'),
     Input('epi-sub-tabs', 'value'),
     Input('meth-sim', 'data')]
)
def render_dend(dataset, tab, epi_subtab, meth_sim):
    if meth_sim is None:
        raise PreventUpdate
    if tab == 'tab-2':
        meth_sim = pd.read_json(StringIO(meth_sim))
        if epi_subtab == 'epi-sub-tab-5':
            if len(meth_sim) < 2:
                raise PreventUpdate
            fig = go.Figure(ff.create_dendrogram(meth_sim, orientation='left', labels=meth_sim.index))
            conf = {'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image', 'height': 1500,
                                             'width': 2000, 'scale': 10}}
            return html.Div([
                dcc.Graph(figure=fig, config=conf)
            ])
        else:
            pass
    else:
        pass


@app.callback(
    Output('tr-pca-tab', 'children'),
    [Input('dataset-dropdown', 'value'),
     Input('exp-meth-out', 'value'),
     Input('tr-sub-tabs', 'value'),
     Input('exp-sim', 'data'),
     Input('tr-pca-slide', 'max'),
     Input('tr-pca-slide', 'value')]
)
def render_pca(dataset, tab, tr_subtab, exp_sim, ncomp_tr_max, ncomp_tr):
    if exp_sim is None:
        raise PreventUpdate
    if tab == 'tab-1':
        if tr_subtab == 'tr-sub-tab-6':
            exp_sim = pd.read_json(StringIO(exp_sim))
            if len(exp_sim) < 2:
                raise PreventUpdate
            if ncomp_tr == 1 and ncomp_tr_max == 1:
                raise PreventUpdate
            if ncomp_tr_max < ncomp_tr:
                raise PreventUpdate

            def pca_plot(n_comp):
                pca = PCA(n_comp)
                comps = pca.fit_transform(exp_sim)
                var = pca.explained_variance_ratio_.sum() * 100
                pca_gr = px.scatter_matrix(comps, dimensions=range(n_comp), labels=exp_sim.index,
                                           color=exp_sim.index, title=f'Total Explained Variance: {var:.2f}%')
                #pca_gr.update_layout(width=1000, height=800)
                return pca_gr

            conf = {'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image', 'height': 1500,
                                             'width': 2000, 'scale': 10}}
            return html.Div([
                dcc.Graph(figure=go.Figure(pca_plot(ncomp_tr)), config=conf)
            ])
        else:
            pass
    else:
        pass


@app.callback(
    Output('epi-pca-tab', 'children'),
    [Input('dataset-dropdown', 'value'),
     Input('exp-meth-out', 'value'),
     Input('epi-sub-tabs', 'value'),
     Input('meth-sim', 'data'),
     Input('epi-pca-slide', 'max'),
     Input('epi-pca-slide', 'value')]
)
def render_pca(dataset, tab, epi_subtab, meth_sim, ncomp_epi_max, ncomp_epi):
    if meth_sim is None:
        raise PreventUpdate
    if tab == 'tab-2':
        if epi_subtab == 'epi-sub-tab-6':
            meth_sim = pd.read_json(StringIO(meth_sim))
            if len(meth_sim) < 2:
                raise PreventUpdate
            if ncomp_epi == 1 and ncomp_epi_max == 1:
                raise PreventUpdate
            if ncomp_epi_max < ncomp_epi:
                raise PreventUpdate

            def pca_plot(n_comp):
                pca = PCA(n_comp)
                comps = pca.fit_transform(meth_sim)
                var = pca.explained_variance_ratio_.sum() * 100
                pca_gr = px.scatter_matrix(comps, dimensions=range(n_comp), labels=meth_sim.index,
                                           color=meth_sim.index, title=f'Total Explained Variance: {var:.2f}%')
                # pca_gr.update_traces(diagonal_visible=False)
                return pca_gr
            conf = {'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image', 'height': 1500,
                                             'width': 2000, 'scale': 10}}
            return html.Div([
                dcc.Graph(figure=go.Figure(pca_plot(ncomp_epi)), config=conf)
            ])
        else:
            pass
    else:
        pass


if __name__ == '__main__':
    app.run_server(debug=True)
