import pandas as pd
import dash
from dash import dcc, html, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/deaths', name="Deaths 📈")

# Load Dataset
DEATH_AGE_URL = 'https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_age.csv'
death_age_df = pd.read_csv(DEATH_AGE_URL)

# Selecting only the desired columns
selected_columns = ['week', 'abs_0_4', 'abs_5_11', 'abs_12_17', 'abs_18_29', 'abs_30_39', 'abs_40_49', 'abs_50_59', 'abs_60_69', 'abs_70_79', 'abs_80_+']
death_age_selected_df = death_age_df[selected_columns]

# Extract numerical part from the 'week' column 
death_age_selected_df['week'] = death_age_selected_df['week'].astype(str).str.extract('(\d+)').astype(int)

# Group by 'week' and calculate the sum for each group
total_sum_df = death_age_selected_df.groupby('week').sum().reset_index()

# Arrange the 'week' column to start from week 1 to week 24
total_sum_df = total_sum_df[(total_sum_df['week'] >= 1) & (total_sum_df['week'] <= 24)]

# Calculate percentage of values with % sign 
percentage_df = total_sum_df.copy()
percentage_df[selected_columns[1:]] = (total_sum_df[selected_columns[1:]].div(total_sum_df[selected_columns[1:]].sum(axis=1), axis=0) * 100).round(2).astype(str) + '%'

# Change column names for better display
death_age_selected_df.columns = ['week', '0 - 4', '5 - 11', '12 - 17', '18 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79', '80 +']
total_sum_df.columns = ['week', '0 - 4', '5 - 11', '12 - 17', '18 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79', '80 +']
percentage_df.columns = ['week', '0 - 4', '5 - 11', '12 - 17', '18 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79', '80 +']

# Layout of the Dash app
layout = html.Div(children=[
    html.Br(),
    html.H3("Distribution of Deaths by Age Group", style={'textAlign': 'center'}),
    dcc.RadioItems(
        id='display-type',
        options=[
            {'label': 'Absolute', 'value': 'absolute'},
            {'label': 'Percentage of Deaths', 'value': 'percentage'}
        ],
        value='absolute',
        labelStyle={'display': 'in-line', 'margin-right': '20px'}
    ),
    html.Br(),
    dash_table.DataTable(
        id='death-table',
        page_size=25,
        style_cell={
            'border': '1px solid black',  # Add border to cells
            'color': 'black',  # Set text color to black
            'text-align': 'left',
            'whiteSpace': 'normal',  # Allow text to wrap to the next line
           
        },
        style_header={
            'background-color': 'darkslategray',
            'font-weight': 'bold',
            'color': 'white',
            'padding': '10px',
            'font-size': '18px',
        },
        style_data_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'center'
            } for c in death_age_selected_df.columns
        ]
    ),
])

@callback(
    Output('death-table', 'data'),
    [Input('display-type', 'value')]
)
def update_table(display_type):
    if display_type == 'absolute':
        return total_sum_df.to_dict('records')
    elif display_type == 'percentage':
        return percentage_df.to_dict('records')