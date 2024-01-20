import pandas as pd
import dash
from dash import dcc, html, callback
import plotly.express as px
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/covid', name="Covid-19 cases 📊")

# Load Dataset for line chart 
url_covid = 'https://storage.data.gov.my/healthcare/covid_cases.parquet'

cases_df = pd.read_parquet(url_covid)
cases_df['date'] = pd.to_datetime(cases_df['date'])  # Convert the 'date' column to datetime type

# Load Dataset for heatmap
url = 'https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_state.csv'

# Convert data into data frame
df = pd.read_csv(url)

# Convert the 'date' column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Extract month and year from the 'date' column
df['month_year'] = df['date'].dt.to_period('M')

# Convert the 'month_year' column to a string
df['month_year'] = df['month_year'].astype(str)

# Convert the 'month_year' column to a DatetimeIndex
df['month_year'] = pd.to_datetime(df['month_year'])

# Create a pivot table to separate the data monthly for every state
pivot_table = pd.pivot_table(df, values='cases_active', index=['state'], columns=['month_year'], aggfunc='sum', fill_value=0)

# Create a heatmap using Plotly Express
heatmap_fig = px.imshow(pivot_table.values,
                       labels=dict(x="Month", y="State", color="cases_cative"),
                       x=pivot_table.columns,
                       y=pivot_table.index,
                       color_continuous_scale="Agsunset",
                       title="Monthly Active Cases by State",
                       aspect="auto")

heatmap_fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(len(pivot_table.columns))),
                                     ticktext=pivot_table.columns.strftime('%b %Y')),
                          yaxis=dict(tickmode='array', tickvals=list(range(len(pivot_table.index))),
                                     ticktext=pivot_table.index),
                          yaxis_title="State",
                          xaxis_title="Month",
                          )

# Create widget (dropdown and slider for years)
columns = [{"label": col, "value": col} for col in ["Malaysia", "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang", "Perak", "Perlis", "Pulau Pinang", "Sabah", "Sarawak", "Selangor", "Terengganu", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"]]

dd = dcc.Dropdown(id="dist_column", options=columns, value="Malaysia", clearable=False)

slider_years = dcc.Slider(
    id='year_slider',
    min=df['date'].dt.year.min(),
    max=df['date'].dt.year.max(),
    value=df['date'].dt.year.max(),
    marks={year: str(year) for year in range(df['date'].dt.year.min(), df['date'].dt.year.max() + 1)},
    step=1
)

# Layout of Dash app
layout = html.Div(children=[
    html.Br(),
    html.H3("Overview of Daily Covid-19 cases in Malaysia", style={'textAlign': 'center'}),
    html.P("Data as of 13 January 2024", style={'color': 'grey', 'textAlign': 'right'}), 
    html.P("Select State:"),
    dd,
    dcc.Graph(id="line"),
    html.Div(id="prescriptive-analysis", style={'textAlign': 'center', 'padding': '20px'}),
    html.Br(),
    slider_years,
    dcc.Graph(id="heatmap", figure=heatmap_fig),
])

# Define callback
@callback(
    [Output("line", "figure"), Output("prescriptive-analysis", "children"), Output("heatmap", "figure")],
    [Input("dist_column", "value"), Input("year_slider", "value")]
)
def update_graph_and_analysis(dist_column, selected_year):
    dff = cases_df[(cases_df['state'] == dist_column)]

    # Create a line chart
    line_figure = px.line(dff, x="date", y="cases_new", color='state')
    line_figure.update_layout(yaxis={'title': 'Number of new cases (daily)'},
                              title=f"Covid-19 New Cases for {dist_column}")

    # Prescriptive Analysis
    trend = get_trend_label(dff['cases_new'])
    recommendation = get_recommendation(trend)

    # Update heatmap
    heatmap_fig = update_heatmap(selected_year)

    return line_figure, html.P(recommendation, style={'fontSize': '18px'}), heatmap_fig

def get_trend_label(series):
    # Calculate the average daily change in cases
    avg_change = series.diff().mean()

    # Classify the trend based on the average daily change
    if avg_change > 0:
        return "increasing"
    elif avg_change < 0:
        return "decreasing"
    else:
        return "stable"

def get_recommendation(trend):
    if trend == "increasing":
        return "The number of daily new cases is increasing. Please exercise caution and follow safety guidelines."
    elif trend == "decreasing":
        return "The number of daily new cases is decreasing. Continue to follow safety guidelines."
    else:
        return "The number of daily new cases is stable. Maintain vigilance and adhere to safety guidelines."
    
def update_heatmap(selected_year):
    filtered_df = df[df['date'].dt.year == selected_year]

     # Create a pivot table to separate the data monthly for all states based on selected year
    pivot_table_filtered = pd.pivot_table(filtered_df, values='cases_active', index=['state'],
                                          columns=['month_year'], aggfunc='sum', fill_value=0)
    
    # Update heatmap
    heatmap_fig = px.imshow(pivot_table_filtered.values,
                           labels=dict(x="Month", y="State", color="Active cases"),
                           x=pivot_table_filtered.columns,
                           y=pivot_table_filtered.index,
                           color_continuous_scale="Viridis",
                           title=f"Monthly Active Cases by State in Malaysia for {selected_year}",
                           aspect="auto")

    heatmap_fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(len(pivot_table_filtered.columns))),
                                         ticktext=pivot_table_filtered.columns.strftime('%b %Y')),
                              yaxis=dict(tickmode='array', tickvals=list(range(len(pivot_table_filtered.index))),
                                         ticktext=pivot_table_filtered.index),
                              yaxis_title="State",
                              xaxis_title="Month")

    return heatmap_fig