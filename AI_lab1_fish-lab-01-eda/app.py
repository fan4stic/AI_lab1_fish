import dash
import dash_bootstrap_components as dbc


external_stylesheets = [
    dbc.themes.BOOTSTRAP,                                       # Bootswatch theme
    'https://use.fontawesome.com/releases/v5.9.0/css/all.css',  # For social media icons
]

meta_tags=[
    {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, meta_tags=meta_tags)   # Initiat App
app.config.suppress_callback_exceptions = True
app.title = 'Fish Dimensions Regression Analysis'                 # Appears in browser title bar
server = app.server