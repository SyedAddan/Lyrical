import os
import logging
import spacy
spacy.load('en_core_web_sm')
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling
import webbrowser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]


def visualize():
    """
        Visualizes the preprocessed data from /projectdir/data/processed
        and saves all the reports and figures generated in the process in /projectdir/reports.
    """
    
    logger = logging.getLogger(__name__)
    logger.info('visualizing preprocessed dataset')
    
    
    # paths
    processed_filepath = Path(os.path.join(PROJECT_DIR, "data", "processed", "lyrics_processed.csv"))
    figures_filepath = Path(os.path.join(PROJECT_DIR, "reports", "figures"))
    eda_report_filepath = Path(os.path.join(PROJECT_DIR, "reports", "eda_report"))
    
    
    # Load the dataset
    logger.info('loading preprocessed dataset')
    data = pd.read_csv(processed_filepath)
    
    
    
    
    
    # 1. Basic Information
    logger.info('1: Generating basic information report')
    profile = ydata_profiling.ProfileReport(data, title="Lyrics Dataset Profiling Report")
    profile.to_file(os.path.join(eda_report_filepath, "data_exploration_report.html"))
    _ = webbrowser.open(str(Path(os.path.join(eda_report_filepath, "data_exploration_report.html")).resolve()))
    
    
    
    
    
    # 2: In-depth analysis of each categorical column
    logger.info('2: Generating in-depth analysis of each categorical column')
    # Genre analysis
    logger.info('2.1: Genre Counts')
    genre_counts = data['genre'].value_counts()

    # Plotting genre counts
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
    plt.title('Genre Counts')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(figures_filepath, 'genre-counts.png'), bbox_inches='tight')



    # Artist analysis
    logger.info('2.2: Top 15 Artist Counts')
    artist_counts = data['artist'].value_counts().nlargest(15)

    # Plotting artist counts
    plt.figure(figsize=(12, 6))
    sns.barplot(x=artist_counts.values, y=artist_counts.index, palette='viridis')
    plt.title('Top 15 Artist Counts')
    plt.xlabel('Count')
    plt.ylabel('Artist')
    plt.savefig(os.path.join(figures_filepath, 'top-15-artists.png'), bbox_inches='tight')



    # Year analysis
    logger.info('2.3: Year Counts')
    year_counts = data['year'].value_counts().sort_index()

    # Plotting year counts
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', color='purple')
    plt.title('Year Counts')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(figures_filepath, 'year-counts.png'), bbox_inches='tight')
    
    
    
    
    
    # 3: Visualizations
    logger.info('3: Generating visualizations')
    # Genre distribution
    logger.info('3.1: Genre Distribution')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='genre', data=data)
    plt.title('Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(figures_filepath, 'genre_distribution.png'), bbox_inches='tight')



    # Lyrics length distribution
    logger.info('3.2: Lyrics Length Distribution')
    len_data = data.copy()
    len_data['lyrics_length'] = len_data['lyrics'].apply(lambda x: len(x))
    plt.figure(figsize=(12, 6))
    sns.histplot(data=len_data, x='lyrics_length', bins=30, kde=True)
    plt.title('Lyrics Length Distribution')
    plt.xlabel('Lyrics Length')
    plt.ylabel('Count')
    plt.savefig(os.path.join(figures_filepath, 'lyrics_length_distribution.png'), bbox_inches='tight')



    # Average lyrics length per genre
    logger.info('3.3: Average Lyrics Length per Genre')
    genre_avg_length = len_data.groupby('genre')['lyrics_length'].mean()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_avg_length.index, y=genre_avg_length.values)
    plt.title('Average Lyrics Length per Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average Length')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(figures_filepath, 'avg_lyrics_length_per_genre.png'), bbox_inches='tight')
    
    
    
    
    
    # 4: Wordclouds
    logger.info('4: Generating wordclouds')
    # Pop wordcloud
    logger.info('4.1: Pop Wordcloud')
    wc = WordCloud(background_color="white", width=1500, height=600, stopwords=spacy.lang.en.stop_words.STOP_WORDS)
    pop_wc = wc.generate(" ".join(data[data["genre"] == "pop"]["lyrics"])).to_image()
    pop_wc.save(os.path.join(figures_filepath, 'pop_wordcloud.png'))
    
    
    
    # Rap wordcloud
    logger.info('4.2: Rap Wordcloud')
    wc = WordCloud(background_color="white", width=1500, height=600, stopwords=spacy.lang.en.stop_words.STOP_WORDS)
    rap_wc = wc.generate(" ".join(data[data["genre"] == "rap"]["lyrics"])).to_image()
    rap_wc.save(os.path.join(figures_filepath, 'rap_wordcloud.png'))
    
    
    
    # Rock wordcloud
    logger.info('4.3: Rock Wordcloud')
    wc = WordCloud(background_color="white", width=1500, height=600, stopwords=spacy.lang.en.stop_words.STOP_WORDS)
    rock_wc = wc.generate(" ".join(data[data["genre"] == "rock"]["lyrics"])).to_image()
    rock_wc.save(os.path.join(figures_filepath, 'rock_wordcloud.png'))
    
    
    
    logger.info('visualizations generated successfully!')
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    visualize()