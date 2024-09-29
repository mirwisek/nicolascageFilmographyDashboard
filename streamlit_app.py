import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from io import BytesIO
import base64

##############################################
# Loading data
##############################################

# Load data from CSV
df = pd.read_csv('dataset/nicolas_data.csv')

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

# Custom color palette
color_palette = ['#f9f0a7', '#f9bfa7', '#f9cfa7', '#f9e0a7', '#f9f0a7', '#f2f9a7', '#e1f9a7', '#d1f9a7']


##############################################
# App Configuration and Layout
##############################################

# Set page configuration
st.set_page_config(
    page_title="Nicolas Cage - Filmography", 
    page_icon="🎬",  
    layout="wide", 
)
st.title("🎥 Nicolas Cage - Filmography Analysis")

# Display Treemap with Plotly
st.subheader("Top 3 Nicolas Cage Movies by Genre")
st.write("Nicolas Cage was actively involved in **Action** movies. Shown ratings are based on audience reviews.")

treemap = go.Figure(
    data=go.Treemap(
        labels=genre_count['Genre'],
        parents=[''] * len(genre_count),  # Treemap requires a parent-child structure, set all to empty for root
        values=genre_count['Title'],  # Use the total number of movies in each genre for the treemap size
        texttemplate='<span style="font-size:1.2em"><b>%{label}</b> (%{value})</span><br>' + genre_count['Movie Chips'].fillna(''),
        textinfo="label+text",  # Display both the label (genre) and the custom movie chips (text)
        hoverinfo='label+value+text',  # Show genre, count, and movies on hover
        marker=dict(colors=color_palette)  # Apply custom color palette
    )
)
st.plotly_chart(treemap, use_container_width=True)

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
        return {'color': 'rgba(144, 238, 144, 1)', 'bold': True}  # Light green for popular movie
    elif row['Rating'] < 5.0 and row['Metascore'] < 40 and row['Votes'] < median_votes:
        return {'color': 'rgba(128, 128, 128, 1)', 'bold': True}  # Gray for flopped movie
    else:
        return {'color': 'rgba(0, 0, 255, 0.3)', 'bold': False}  # Blue for normal movies

df_sorted['Bar_Status'] = df_sorted.apply(get_movie_status, axis=1)

st.subheader("Top 3 Popular and Flop Movies of Nicolas' Career")
st.write("**Face/Off, Adaptation, Leaving Las Vegas** were some of the popular movies of Nicolas Cage that had high audience engagement. While, **Left Behind, The Wicker Man** and **Jiu Jitsu** were some of the flop movies.")

bar_chart = go.Figure(
    data=[
        # Bar plot for Rating (Normalized to 100 scale)
        go.Bar(
            x=[f"<b>{title}</b>" if status['bold'] else title for title, status in zip(df_sorted['Title'], df_sorted['Bar_Status'])],
            y=df_sorted['Rating_Normalized'],
            name='Audience Rating',
            marker=dict(color=[status['color'] for status in df_sorted['Bar_Status']]),
            hovertemplate=df_sorted['Title'] + ' (' + df_sorted['Year'].astype(int).astype(str) + ')<extra></extra>',
        ),
        # Bar plot for Metascore with fixed color
        go.Bar(
            x=[f"<b>{title}</b>" if status['bold'] else title for title, status in zip(df_sorted['Title'], df_sorted['Bar_Status'])],
            y=df_sorted['Metascore'],
            name='Critic Metascore',
            marker=dict(color=['rgba(255, 165, 0, 0.3)' if status['color'] == 'rgba(0, 0, 255, 0.3)' else status['color'] for status in df_sorted['Bar_Status']]),
        )
    ],
    layout=go.Layout(
        xaxis_title='Movies',
        yaxis_title='Score (0-100)',
        barmode='group',
        height=600,
        xaxis=dict(tickangle=-45)  # Rotate x-axis labels to 45 degrees
    )
)
st.plotly_chart(bar_chart, use_container_width=True)

##############################################
# Directors Network Graph
##############################################

# Network graph data - Nicolas Cage and directors
df_directors = df_unique.groupby('Director').size().reset_index(name='Count')
df_directors = df_directors[df_directors['Director'].notnull()]

# Create a network graph using networkx
G = nx.Graph()

# Add Nicolas Cage node
G.add_node("Nicolas Cage", size=50, color='yellow')

# Add director nodes and edges
for _, row in df_directors.iterrows():
    G.add_node(row['Director'], size=row['Count'] * 10, color='orange')  # Director node size depends on movie count
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

# Extract edge positions (lines)
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

# Plot Network Graph
st.subheader("Directors Nicolas Cage Has Worked With")
st.write("Did you know that **Jon Turteltaub** was one of Nicolas Cage's favorite directors?")

network_chart = go.Figure(
    data=[
        # Add edges (lines)
        go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='rgba(227, 227, 227, 0.7)'),
            hoverinfo='none'
        ),
        # Add nodes (directors and Nicolas Cage)
        go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            hovertemplate=[f'{node} (Movies: {df_unique[df_unique["Director"] == node]["Title"].count()})' if node != 'Nicolas Cage' else node for node in node_text],
            marker=dict(size=node_size, color=node_color),
            showlegend=False
        )
    ],
    layout=go.Layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700
    )
)
st.plotly_chart(network_chart, use_container_width=True)

##############################################
# Word Cloud of Reviews Text
##############################################

# Generate word cloud from review keywords
def generate_wordcloud(text):
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(['movie', 'film', 'will', 'make', 'go', 'someting', 'movies', 'say', 'Nicolas', 'Cage'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(text)
    return wordcloud

# Extract review text and combine into a single string
review_text = ' '.join(df_unique['Review'].dropna())

# Generate the word cloud image
wordcloud_img = generate_wordcloud(review_text)

# Generate the word cloud image
wordcloud_img = generate_wordcloud(review_text)

# Save the word cloud to a BytesIO object to display it with st.image()
img_bytes = BytesIO()
wordcloud_img.to_image().save(img_bytes, format='PNG')
img_bytes.seek(0)


# Display Word Cloud with reduced width and center-aligned
st.subheader("How Audience Describes Nicolas' Movies?")
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{base64.b64encode(img_bytes.getvalue()).decode()}" width="600" />
    </div>
    """, 
    unsafe_allow_html=True
)