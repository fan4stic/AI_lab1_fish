import dash
import dash_bootstrap_components as dbc
from dash import dcc
import plotly.express as px
import pandas as pd


# Column to display text
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
            ### Looking into Linear Regression

            This web application is a demonstration of the linear regression model (Ordinary Least Squares) implemented as part of the Fish Dimension Regression Analysis.
            """
        ),
        dcc.Link(dbc.Button('View Model', color='primary'), href='/predictions')
    ],
    md=4,
)

# Creating plotly graph
df = pd.read_csv('./assets/data/Fish.csv')
fig = px.scatter(df, x='Length3', y='Weight', labels={
                                                "Weight": "Fish Weight (grams)",
                                                "Length3": "Fish Cross Length (cm)"
                                            },)

# Column to display grpah
column2 = dbc.Col(
    [
        dcc.Graph(figure=fig),
    ]
)

# Layout for rendering to web app
layout = dbc.Row([column1, column2])