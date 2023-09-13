from dash import Dash, html, dcc, Input, Output
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import numpy as np
from more_itertools import unique_everseen

app = Dash(__name__)

data_drop_styles = {
    'color': 'DarkRed',
    'font-size': '25px',
    'font-family': 'system-ui'
}

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

app.layout = html.Div(children=[
    html.Div([
        html.Label('Dataset', style={'font-weight': 'bold', 'font-family': 'system-ui', 'text-align': 'left',
                                     'color': 'DarkSlateGray', 'font-size': '25px'}),
        dcc.Dropdown(id='dataset-dropdown', options=['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA',
                                                     'GBM', 'HNSC', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'KICH',
                                                     'KIRC', 'KIRP', 'MESO', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SKCM',
                                                     'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM'], value='ACC',
                     searchable=False)], style=data_drop_styles),
    html.Div([
        html.Label('Genes', style={'font-weight': 'bold',  'font-family': 'system-ui', 'text-align': 'left',
                                   'color': 'DarkSlateGray', 'font-size': '20px'}),
        dcc.Dropdown(id='genes-list', multi=True, clearable=True)], style={}),
    html.Div([
        dcc.Tabs(id='exp-meth-out', value='tab-0', children=[
            dcc.Tab(label='About', value='tab-0', children=[
                html.H1('NashPlyx'),
                dcc.Markdown('''
                            NashPlyx is a tool for network identification and analysis with TCGA datasets using WGCNA 
                            results. Users can build and compare gene networks with transcriptional and methylational
                            using the "Transcriptome" and "Epigenome" tabs.
                            ''')], style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Transcriptome', value='tab-1', children=[
                dcc.Tabs(id='tr-sub-tabs', value='tr-sub-tab-1', children=[
                    dcc.Tab(label='Data table', value='tr-sub-tab-1', #children=[
                        #html.Button('Download WGCNA Data', id='exp-data-button'),
                        #dcc.Download(id='exp-data-down')],
                            style=sub_tab_style, selected_style=sub_tab_selected_style),
                    dcc.Tab(label='Network', value='tr-sub-tab-2',
                            style=sub_tab_style, selected_style=sub_tab_selected_style),
                    dcc.Tab(label='Heatmap', value='tr-sub-tab-3', style=sub_tab_style, selected_style=sub_tab_selected_style),
                ])], style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Epigenome', value='tab-2', children=[
                dcc.Tabs(id='epi-sub-tabs', value='epi-sub-tab-1', children=[
                    dcc.Tab(label='Data table', value='epi-sub-tab-1', style=sub_tab_style, selected_style=sub_tab_selected_style),
                    dcc.Tab(label='Network', value='epi-sub-tab-2', style=sub_tab_style, selected_style=sub_tab_selected_style),
                    dcc.Tab(label='Heatmap', value='epi-sub-tab-3', style=sub_tab_style, selected_style=sub_tab_selected_style),
                ])], style=tab_style, selected_style=tab_selected_style)]),
    html.Div(id='exp-meth')]),
    dcc.Store('exp-data'),
    dcc.Store('meth-data')
])


@app.callback(
    Output('genes-list', 'options'),
    Input('dataset-dropdown', 'value'),
)
def update_genes_list(dataset):
    dataset_genes = np.load('/Volumes/SanDisk/Indices/' + dataset.lower() + '_exp_genes.npy', allow_pickle=True)
    return dataset_genes


@app.callback(
    Output('exp-data', 'data'),
    [Input('dataset-dropdown', 'value'), Input('genes-list', 'value')]
)
def update_exp(dataset, genes):
    exp_genes = np.load('/Volumes/SanDisk/Indices/' + dataset.lower() + '_exp_genes.npy', allow_pickle=True)
    exp_diss = pd.read_csv('/Volumes/SanDisk/' + dataset.lower() + '_tom_diss.tsv.gz',
                           compression='gzip', usecols=genes)
    exp_diss = exp_diss.set_index(exp_genes)
    exp_diss = exp_diss.loc[genes, :]
    return exp_diss.to_json()


@app.callback(
    Output('meth-data', 'data'),
    [Input('dataset-dropdown', 'value'), Input('genes-list', 'value')]
)
def update_meth(dataset, genes):
    meth_genes = np.load('/Volumes/SanDisk/Indices/' + dataset.lower() + '_meth_genes.npy', allow_pickle=True)
    meth_diss = pd.read_csv('/Volumes/SanDisk/' + dataset.lower() + '_meth_tom_diss.tsv.gz',
                            compression='gzip', usecols=genes)
    meth_diss = meth_diss.set_index(meth_genes)
    meth_diss = meth_diss.loc[genes, :]
    return meth_diss.to_json()


@app.callback(
    Output('exp-meth', 'children'),
    [Input('exp-meth-out', 'value'), Input('tr-sub-tabs', 'value'),
     Input('epi-sub-tabs', 'value'), Input('exp-data', 'data'),
     Input('meth-data', 'data')]
)
def render_tabs(tab, tr_subtab, epi_subtab, exp, meth):
    def df_to_plotly(df):
        return {'z': df.values.tolist(),
                'x': df.columns.tolist(),
                'y': df.index.tolist()}
    if tab == 'tab-1':
        exp_genes = pd.read_json(exp)
        exp_diss = df_to_plotly(exp_genes)
        exp_sim = (1 - np.array(exp_diss['z'])) * 10000
        exp_sim = pd.DataFrame(exp_sim, columns=exp_genes.columns, index=exp_genes.index)
        exp_simm = exp_sim.mask(exp_sim == 10000)
        if tr_subtab == 'tr-sub-tab-1':
            fig = go.Figure(go.Table(header=dict(values=exp_diss['x']),
                                     cells=dict(values=exp_diss['z'])))
            return html.Div([
                dcc.Graph(figure=fig)
            ])
        # elif tab == 'tr-sub-tab-2':
        #     return
        #
        #     #html.Div([
        #         #dcc.Graph(nx.draw_shell(nxg, with_labels=True, node_color='mediumpurple', node_size=3000,
        #                                 #width=w_, edge_color='gray'))
        #     #]))
        else:
            fig = go.Figure(go.Heatmap(df_to_plotly(exp_simm), colorscale='Viridis'))
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
            return html.Div([
                dcc.Graph(figure=fig)
            ])

    elif tab == 'tab-2':
        meth_genes = pd.read_json(meth)
        meth_diss = df_to_plotly(meth_genes)
        meth_sim = (1 - np.array(meth_diss['z'])) * 100000
        meth_sim = pd.DataFrame(meth_sim, columns=meth_genes.columns, index=meth_genes.index)
        meth_simm = meth_sim.mask(meth_sim == 100000)
        if epi_subtab == 'epi-sub-tab-1':
            fig = go.Figure(go.Table(header=dict(values=meth_diss['x']),
                                     cells=dict(values=meth_diss['z'])))
            return html.Div([
                dcc.Graph(figure=fig)
            ])
        # elif tab == 'epi-sub-tab-2':
        #     return

        else:
            fig = go.Figure(go.Heatmap(df_to_plotly(meth_simm), colorscale='Sunset'))
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
            return html.Div([
                dcc.Graph(figure=fig)
            ])
    else:
        pass


if __name__ == '__main__':
    app.run_server(debug=True)
