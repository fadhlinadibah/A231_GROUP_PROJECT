import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path='/', name="Home 🏠")

# Load Dataset for horizontal bar chart 
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

# Group by state and get total value local and import cases
df_grouped = df.groupby('state')[['cases_new', 'cases_import']].sum().reset_index()

# Calculate total confirmed cases for each state separately
df_grouped['total_cases'] = df_grouped['cases_new'] + df_grouped['cases_import']

# Function for horizontal bar chart
def create_horizontal_bar_chart(col_name="Malaysia", sorting_order="ascending"):
    if sorting_order == "ascending":
        df_grouped_sorted = df_grouped.sort_values(by='total_cases')
    else:
        df_grouped_sorted = df_grouped.sort_values(by='total_cases', ascending=False)

    fig = px.bar(data_frame=df_grouped_sorted, x='total_cases', y='state', orientation='h',
             color='state', color_discrete_map={'Johor':'#5F9EA0', 'Kedah':'#5F9EA0', 'Kelantan':'#5F9EA0', 'Melaka': '#5F9EA0', 'Negeri Sembilan':'#5F9EA0', 'Pahang':'#5F9EA0', 'Perak':'#5F9EA0', 
                                                'Perlis':'#5F9EA0', 'Pulau Pinang':'#5F9EA0', 'Sabah':'#5F9EA0', 'Sarawak':'#5F9EA0', 'Selangor':'#5F9EA0', 'Terengganu':'#5F9EA0', 
                                                'W.P. Kuala Lumpur':'#5F9EA0', 'W.P. Labuan':'#5F9EA0', 'W.P. Putrajaya':'#5F9EA0'})
    fig = fig.update_layout(xaxis={'title': 'Number of Confirmed cases'},
                            title=f"Covid-19 confirmed cases in Malaysia"
                            )
    return fig

# Load Dataset for creating card  
CONF_URL = 'https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_state.csv'
DEAD_URL = 'https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_malaysia.csv'
RECV_URL = 'https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_state.csv'

# Convert data into data frame
covid_conf_df = pd.read_csv(CONF_URL)
covid_dead_df = pd.read_csv(DEAD_URL)
covid_recv_df = pd.read_csv(RECV_URL)

# Calculate total value to display in card
def get_overall_total(df, column):
    return df[column].sum().sum()

# Calculate total for recovered cases
recv_overall_total = get_overall_total(covid_recv_df, 'cases_recovered')

# Calculate total for confirmed cases
columns_to_sum = ['cases_new', 'cases_import']
conf_overall_total = get_overall_total(covid_conf_df, columns_to_sum)

# Calculate total for dead cases
dead_overall_total = get_overall_total(covid_dead_df, 'deaths_new')

# Print the results
print('Overall Recovered:', recv_overall_total)
print('Overall Confirmed:', conf_overall_total)
print('Overall Dead:', dead_overall_total)

# Create cards to display total recovered cases, confirmed cases and deaths
def generate_card_content(card_header, *args):
    card_head_style = {'textAlign': 'center', 'fontSize': '150%', 'padding': '10px'}
    card_body_style = {'textAlign': 'center', 'fontSize': '200%', 'padding': '10px', 'width': '30%', 'display': 'inline-block', 'margin': 'auto'}

    cards = [
        html.Div(
            style={'border': '1px solid #ddd', 'borderRadius': '8px', 'margin': '10px', 'width': '30%', 'display': 'inline-block'},
            children=[
                html.H4(card_header, style=card_head_style),
                html.H2(f"{int(arg):,}", style=card_body_style),
            ]
        )
        for arg in args
    ]

    return cards

# Create widget for horizontal bar chart
sorting_radio = dcc.RadioItems(
    id='sorting_radio',
    options=[
        {'label': 'Ascending', 'value': 'ascending'},
        {'label': 'Descending', 'value': 'descending'}
    ],
    value='ascending',
    labelStyle={'display': 'inline-block', 'margin-right': '20px'}
)

# Layout of Dash App
layout = html.Div(children=[
    html.H2("Welcome to Covid-19 cases in Malaysia dashboard"),
    "This dashboard serves as a comprehensive resource for tracking the impact of the COVID-19 pandemic in Malaysia. ",
    "Stay informed with daily updates on confirmed cases, recoveries, and unfortunate losses. ",
    "By monitoring the evolving situation, we can contribute to the collective effort in managing and mitigating the effects of the virus.",
    html.Br(),
    html.P("Data as of 13 January 2024", style={'color': 'grey', 'textAlign': 'right'}),
    *generate_card_content("Recovered", recv_overall_total),
    *generate_card_content("Confirmed", conf_overall_total),
    *generate_card_content("Deaths", dead_overall_total),
    html.Br(),
    html.Br(),
    sorting_radio,
    dcc.Graph(id="bar_chart"),
])

# Define callback
@callback(Output("bar_chart", "figure"),
              [Input("sorting_radio", "value")])

def update_charts(sorting_order):
    # Create a horizontal bar chart to display total confirmed cases based on selected radio items
    bar_chart_figure = create_horizontal_bar_chart(sorting_order=sorting_order)

    return bar_chart_figure