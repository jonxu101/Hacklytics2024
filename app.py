from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output
import dash_table

import collections
import bisect

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

data = pd.read_csv('./efficient_frontier.csv')
weights = pd.read_csv('./weights.csv')
n = len(data)
min_risk = data['Risk'][0]
max_risk = data['Risk'][n - 1]
incr = data['Risk'][0]

risk_return_df = pd.read_csv('./riskreturn.csv')

names = risk_return_df['Name'].tolist()
print(names)
print(weights.iloc[0].values)

trial_info_df = risk_return_df[['Website', 'OtherInfo']]

ind_map = collections.OrderedDict()
ind_map = {
    data['Risk'][i] : int(i) for i in range(n)
}

print(ind_map)
# Styles
component_box_style = {
    'border': '2px solid #DDD',  # Lighter border for the white box
    'backgroundColor': '#FFFFFF',  # White background for the box
    'padding': '20px',
    'boxSizing': 'border-box',
    'margin': '10px',
    'borderRadius': '5px',  # Rounded corners for the box
}

background_color = '#F0F0F0'  # Slight gray background

header_style = {
    'textAlign': 'center', 
    'color': '#333', 
    'fontWeight': 'bold'  # Make the header text bold
}

app.layout = html.Div([
    html.Div([  # Top Row with corrected vertical separator
        html.Div([  # Top Left Quadrant: Title and Slider
            html.H1("Clinical Trial Investing", style=header_style),
            dcc.Slider(
                min=min_risk,
                max=max_risk - 0.005,
                step=0.001,
                marks=None,
                value=data['Risk'][n//2],
                id='risk_slider',
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div([  # DataFrame display
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in trial_info_df.columns],
                    data=trial_info_df.to_dict('records'),
                    style_cell={'textAlign': 'left'},
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    },
                    style_as_list_view=True,
                )
            ], style={'marginTop': '20px'}),

        ], style={**component_box_style, 'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([  # Top Right Quadrant: Pie Chart
            dcc.Graph(id="graph"),
        ], style={**component_box_style, 'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'boxSizing': 'border-box', 'alignItems': 'stretch', 'backgroundColor': background_color, 'justifyContent': 'space-between'}),

    html.Div([
        dcc.Graph(id="plot"),
        html.Div(id='risk-output-container'),
    ], style={**component_box_style, 'width': '100%', 'boxSizing': 'border-box'}),
], style={'padding': '20px', 'boxSizing': 'border-box', 'fontFamily': 'Times New Roman', 'backgroundColor': background_color})




@app.callback(
    Output(component_id='graph', component_property='figure'),
    Input('risk_slider', 'value'))
def generate_chart(risk_slider):
    print("GENERATING GRAPH" + str(risk_slider))
    # ind = bisect.bisect_left(ind_map.keys(), risk_slider)
    risk_slider = round(risk_slider, 3)
    df = pd.DataFrame(zip(names, weights.iloc[ind_map[risk_slider]].values), columns = ['name', 'weight'])
    fig = px.pie(df, values='weight', names='name', hole=0.2)
    fig.update_layout(
        title_text='Portfolio Allocation Weights',
        font=dict(family="Times New Roman", size=20, color="RebeccaPurple")  # Update font here
    )
    return fig

@app.callback(
    Output('risk-output-container', 'children'),
    Input('risk_slider', 'value'))
def update_output(risk_slider):
    return 'You have selected risk="{}"'.format(risk_slider)

@app.callback(
    Output(component_id='plot', component_property='figure'),
    Input('risk_slider', 'value'))
def update_graph(risk_slider):
    # Create the scatter plot
    fig = px.scatter(data, x='Risk', y='Return')
    fig.add_scatter(x=[risk_slider], y=[data['Return'][ind_map[risk_slider]]], mode='markers', marker=dict(color='red', size=20), name='Selected Risk Level')
    
    fig.update_layout(
        title='Efficient Frontier (Risk Return Trade-off)',
        font=dict(family="Times New Roman", size=20, color="RebeccaPurple")  # Update font here
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)