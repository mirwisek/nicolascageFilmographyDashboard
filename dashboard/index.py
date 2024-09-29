##############################################
# Author: Mirwise
# Purpose: This script analyzes Nicolas Cage's filmography data and generates visualizations using Dash and Plotly. It performs various data transformations, such as removing duplicates, splitting genres, sorting movies, and calculating movie ratings. The script also creates treemaps, bar plots, network graphs, and word clouds to visualize different aspects of Nicolas Cage's movies, including genre distribution, movie ratings, directors he has worked with, and audience descriptions. The script uses pandas, plotly, networkx, and wordcloud libraries for data manipulation and visualization.
# Usage: python index.py
##############################################

import dash

from dash import html, dcc
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

from wordcloud import WordCloud
import base64
from io import BytesIO
from wordcloud import STOPWORDS


##############################################
# Loading data
##############################################

# Load data from CSV
df = pd.read_csv("../dataset/nicholas_data.csv")

##############################################
# Treemap for Genre vs Movie Count (movie name and rating)
##############################################

# Remove duplicates to get unique movies
df_unique = df.drop_duplicates(subset='Title')

# Convert 'Year' to numeric if it's not already
df_unique.loc[:, 'Year'] = pd.to_numeric(df_unique['Year'], errors='coerce')

# Split genres by ',' and explode the column to get one genre per row
df_genre = df_unique.assign(Genre=df_unique['Genre'].str.split(',')).explode('Genre')

# Sort movies by Rating within each genre and select top 3
df_genre_sorted = df_genre.sort_values(by=['Genre', 'Rating'], ascending=[True, False])

# For each genre, select the top 3 movies by Rating
df_top3_per_genre = df_genre_sorted.groupby('Genre').head(3)

# Create custom labels in the format "Title (Year) Rating"
def create_chip(title, year, rating):
    return f"{title} ({year}) ⭐ {rating}"

# Group by Genre and aggregate custom labels
df_top3_per_genre['Movie Chips'] = df_top3_per_genre.apply(
    lambda row: create_chip(row['Title'], int(row['Year']), row['Rating']), axis=1
)

# Group by Genre, aggregate the top 3 movies into a single string and keep the total count of movies in each genre
genre_count = df_genre.groupby('Genre').agg({
    'Title': 'count',  # Total number of movies in the genre
}).reset_index()

# Merge the top 3 movie chips into the same DataFrame
genre_count = genre_count.merge(
    df_top3_per_genre.groupby('Genre').agg({
        'Movie Chips': lambda x: "<br>".join(x)  # Combine top 3 movie chips into a single string
    }).reset_index(),
    on='Genre',
    how='left'
)

##############################################
# Bar chat plot for Popular and Flop movies
##############################################

# Drop rows with missing 'Year' values
df_unique = df_unique.dropna(subset=['Year'])

# Remove commas from 'Votes' and convert to integers
df_unique['Votes'] = df_unique['Votes'].astype(str).str.replace(',', '').astype(int)

# Normalize Rating to be on the same scale as Metascore (0-100)
df_unique['Rating_Normalized'] = df_unique['Rating'] * 10

# Sort data by year, newest to oldest
df_sorted = df_unique.sort_values(by='Year', ascending=False)

# Calculate median number of votes
median_votes = df_unique['Votes'].median()

# Define color and opacity for each movie based on popularity or flop status
def get_movie_status(row):
    if row['Rating'] > 7.0 and row['Metascore'] > 70 and row['Votes'] > median_votes:
        return {'color': 'rgba(144, 238, 144, 1)', 'bold': True}  # Light green for popular movie with full opacity
    elif row['Rating'] < 5.0 and row['Metascore'] < 40 and row['Votes'] < median_votes:
        return {'color': 'rgba(128, 128, 128, 1)', 'bold': True}  # Gray for flopped movie with full opacity
    else:
        return {'color': 'rgba(0, 0, 255, 0.3)', 'bold': False}  # Blue with lower opacity for normal movies

df_sorted['Bar_Status'] = df_sorted.apply(get_movie_status, axis=1)

##############################################
# Directors Network Graph
##############################################

# Network graph data - Nicolas Cage and directors
df_directors = df_unique.groupby('Director').size().reset_index(name='Count')
df_directors = df_directors[df_directors['Director'].notnull()]

# Create a network graph using networkx
G = nx.Graph()

# Add Nicolas Cage node
G.add_node("Nicolas Cage", size=50, color='red')

# Add director nodes and edges
for _, row in df_directors.iterrows():
    G.add_node(row['Director'], size=row['Count'] * 10, color='blue')  # Director node size depends on movie count
    G.add_edge("Nicolas Cage", row['Director'], weight=row['Count'])

# Create positions for each node
pos = nx.spring_layout(G)

# Extract node positions and attributes
node_x = []
node_y = []
node_text = []
node_size = []
node_color = []

for node, (x, y) in pos.items():
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    node_size.append(G.nodes[node]['size'])
    node_color.append(G.nodes[node]['color'])

# Extract edges and their weights
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)


##############################################
# Word Cloud of Reviews Text
##############################################

# Generate word cloud from review keywords
def generate_wordcloud(text):
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(['movie','film','will','make','go','someting', 'movies', 'say'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(text)
    return wordcloud

# Extract review text and combine into a single string
review_text = ' '.join(df_unique['Review'].dropna())

# Generate the word cloud image
wordcloud_img = generate_wordcloud(review_text)

# Save the word cloud to a bytes object
img_bytes = BytesIO()
wordcloud_img.to_image().save(img_bytes, format='PNG')
img_bytes.seek(0)

# Convert to base64 string for display in Dash
wordcloud_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

##############################################
# Main Application Layout
##############################################


# Initialize the app with Bootstrap support
app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])

# Define the custom color palette for the treemap
color_palette = ['#f9f0a7', '#f9bfa7', '#f9cfa7', '#f9e0a7', '#f9f0a7', '#f2f9a7', '#e1f9a7', '#d1f9a7']

# App layout
app.layout = html.Div(
    className="container-fluid mt-4",  # Bootstrap container
    children=[
        # Page Header
        html.Div(className="row", children=[
            html.Div(className="col", children=[
                html.H1("Nicolas Cage's Filmography", className="text-center mb-4")
            ])
        ]),

        # Treemap for Genre vs Movie Count with chips (movie name and rating)
        html.Div(className="row", children=[
            html.Div(className="col", children=[
                html.H2("Top 3 Nicolas Cage Movies by Genre", className="text-center mb-4"),
                dcc.Graph(
                    figure=go.Figure(
                        data=go.Treemap(
                            labels=genre_count['Genre'],
                            parents=[''] * len(genre_count),  # Treemap requires a parent-child structure, set all to empty for root
                            values=genre_count['Title'],  # Use the total number of movies in each genre for the treemap size
                            texttemplate='<span style="font-size:1.2em"><b>%{label}</b> (%{value})</span><br>' + genre_count['Movie Chips'].fillna(''),  # Add the total movie count using %{value}, and show top 3 movies
                            textinfo="label+text",  # Display both the label (genre) and the custom movie chips (text)
                            hoverinfo='label+value+text',  # Show genre, count, and movies on hover
                            marker=dict(colors=color_palette),  # Use the custom color palette
                            textfont=dict(size=14),  # Make the text larger for better visibility
                        ),
                        layout=go.Layout(
                            title="Nicolas' was actively involved in <b>Action</b> movies. Shown ratings are based on audience reviews.",
                            margin=dict(t=50, l=25, r=25, b=25),
                        )
                    )
                )
            ])
        ]),

        html.H2("Top 3 popular and flop movies of Nicholas' Career", className="text-center mb-4"),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    # Bar plot for Rating (Normalized to 100 scale) with hover template
                    go.Bar(
                        x=[f"<b>{title}</b>" if status['bold'] else title for title, status in zip(df_sorted['Title'], df_sorted['Bar_Status'])],
                        y=df_sorted['Rating_Normalized'],
                        name='Audience Rating',
                        marker=dict(color=[status['color'] for status in df_sorted['Bar_Status']]),
                        hovertemplate=df_sorted['Title'] + ' (' + df_sorted['Year'].astype(int).astype(str) + ')<extra></extra>',
                        textposition='auto',
                    ),
                    # Bar plot for Metascore with fixed orange color and hover template
                    go.Bar(
                        x=[f"<b>{title}</b>" if status['bold'] else title for title, status in zip(df_sorted['Title'], df_sorted['Bar_Status'])],
                        y=df_sorted['Metascore'],
                        name='Critic Metascore',
                        marker=dict(color=['rgba(255, 165, 0, 0.3)' if status['color'] == 'rgba(0, 0, 255, 0.3)' else status['color'] for status in df_sorted['Bar_Status']]),
                        hovertemplate = df_sorted['Title'] + ' (' + df_sorted['Year'].astype(int).astype(str) + ')<extra></extra>',
                        textposition='auto',
                    )
                ],
                layout=go.Layout(
                    title='<b>Face/Off, Adaptation, Leaving Las Vegas</b> were some of the popular movies of Nicolas Cage that had high audience engagement. <br>While, <b>Left Behind, The Wicker Man</b> and <b>Jiu Jitsu</b> were some of the flop movies.',
                    xaxis_title='Movies',
                    yaxis_title='Score (0-100)',
                    barmode='group',
                    xaxis=dict(tickangle=-45, automargin=True),
                    height=700
                )
            )
        ),

        # Network Graph of Directors
        html.H2("Directors Nicolas Cage has worked with", className="text-center mb-4"),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Scatter(
                        x=edge_x, y=edge_y,
                        mode='lines',
                        line=dict(width=1, color='grey'),
                        hoverinfo='none'
                    ),
                    go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        hoverinfo='text',
                        marker=dict(
                            size=node_size,
                            color=node_color,
                            line_width=2
                        )
                    )
                ],
                layout=go.Layout(
                    title="Did you know that <b>Jon Turteltaub</b> was Nicolas' faviorite director?",
                    showlegend=False,
                    margin=dict(t=50, l=25, r=25, b=25),
                    height=700
                )
            )
        ),

        # Word Cloud of Review Keywords
        html.H2("How audience describes Nicholas' movies as?", className="text-center mb-4"),
        html.Div(
            style={'text-align': 'center', 'margin-bottom': '30px'},  # Center-align the content inside this div
            children=[
                html.Img(
                    src=f'data:image/png;base64,{wordcloud_base64}',
                    style={'width': '40%', 'height': 'auto', 'display': 'block', 'margin': 'auto'}
                )
            ]
        )

    ]
)

# Run the app
if __name__ == '__main__':
    app.run_server(port=5000, debug=True)
