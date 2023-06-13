# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline
import gc

model_ckpt = "papluca/xlm-roberta-base-language-detection"
pipe = pipeline("text-classification", model=model_ckpt, device=-1)
PROJECT_DIR = Path(__file__).resolve().parents[2]

def identify_language(lyrics: str) -> str|np.nan:
    res = pipe([lyrics], truncation=True, max_length=128)
    return res[0]['label'] if res[0]['score'] > 0.5 else np.nan


def makeProcessed():
    """ 
        Preprocesses the raw data from /projectdir/data/raw
        and saves the preprocessed data in /projectdir/data/processed.
    """
    logger = logging.getLogger(__name__)
    logger.info('making preprocessed data set from raw data')
    
    input_filepath = Path(os.path.join(PROJECT_DIR, "data", "raw", "song_lyrics.csv"))
    processed_filepath = Path(os.path.join(PROJECT_DIR, "data", "processed", "lyrics_processed.csv"))
    
    logger.info('input_filepath: {}'.format(input_filepath))
    
    # this can take more than 12 hours to run
    with pd.read_csv(
        input_filepath,
        chunksize=5 * 10**3,
        usecols=["title", "artist", "year", "tag", "views", "lyrics"],
        dtype={"year": np.int16, "views": np.int32}
    ) as chunks:

        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx}...")

            # drop N.A. lyrics
            logger.info(f"dropping N.A. lyrics from chunk {idx}...")
            chunk = chunk.dropna(subset=["lyrics"])
            
            # drop romanizations
            logger.info(f"dropping romanizations from chunk {idx}...")
            chunk = chunk[chunk["artist"] != "Genius Romanizations"]
            chunk = chunk[~chunk["title"].str.contains(r"\(?romanized\)?", regex=True, na=False, case=False)]

            # remove invalid years
            logger.info(f"removing invalid years from chunk {idx}...")
            chunk = chunk[chunk["year"] < 2023]
            
            # remove duplicated entries
            logger.info(f"removing duplicated entries from chunk {idx}...")
            chunk = chunk.drop_duplicates(subset=["title", "artist", "year"])
            
            # remove special characters from lyrics
            logger.info(f"removing special characters from lyrics in chunk {idx}...")
            pattern = r"(?m)^\[.*?\]$"
            chunk["lyrics"] = chunk["lyrics"].str.replace(pattern, "", regex=True)
            
            # remove empty lines
            logger.info(f"removing empty lines from lyrics in chunk {idx}...")
            pattern = r"\n|\n\n"
            chunk["lyrics"] = chunk["lyrics"].str.replace(pattern, " ", regex=True)

            # drop lyrics that are too short or too long
            logger.info(f"dropping lyrics that are too short or too long from chunk {idx}...")
            chunk = chunk[chunk["lyrics"].str.len().between(10**2, 10**5)]

            # analyze language
            logger.info(f"analyzing language from chunk {idx}...")
            chunk["language"] = chunk["lyrics"].apply(identify_language)
            logger.info(f'{len(chunk[chunk["language"].isna()])} not identified lyrics using by the language detection model.')
            
            # drop non-english lyrics
            logger.info(f"dropping non-english lyrics from chunk {idx}...") 
            chunk = chunk[chunk["language"] == "en"][["artist", "tag", "lyrics"]]
            
            # save processed data
            logger.info(f"saving processed data from chunk {idx}...")
            chunk.to_csv(processed_filepath, mode="a", header=not os.path.exists(processed_filepath), index=False)
            
            logger.info(f"garbage collecting chunk {idx}...")
            del chunk
            gc.collect()
            
            logger.info(f"finished processing chunk {idx}...")
            logger.info("------------------------------------------")
            
    logger.info("finished semi-preprocssing from raw data and checkpointed in projectdir/data/processed")        
    
    logger.info("doing final preprocessing from processed data and saving in projectdir/data/processed")
    
    # Define the 100 most popular artists so that we can filter out the rest 
    best_100_artists = {'Led Zeppelin', 'LSD', 'Jon Hopkins', 'Dolly Parton', 'Billy Joel', 'Backstreet Boys', 'Muze Sikk', 'Kanye West', 'SALES', 'Ray Charles', 'Gucci Mane', 'Rihanna', 'Aerosmith', 'Juice WRLD', 'Soulja Boy', 'Snoop Dogg', 'Bon Jovi', 'twenty one pilots', 'Alabama', 'Kid Cudi', 'The Drums', 'Eminem', 'alt-J', 'AC/DC', 'Of Monsters and Men', 'Madonna', 'James Brown', 'Britney Spears', 'Johnny Cash', 'Guided by Voices', 'Bruce Springsteen', 'The White Stripes', 'The Rolling Stones', 'Tyler', 'Arctic Monkeys', 'Shania Twain', '50 Cent', 'Eagles', 'The Game', 'Maroon 5', 'Caravan Palace', 'Emily Dickinson', 'LMFAO', 'Rod Stewart', 'Imagine Dragons', 'june', 'Sia', 'Daft Punk', 'Kendrick Lamar', 'Matthew Mole', 'Shakira', 'Jack Stauber', 'U2', 'Katy Perry', 'Pink Floyd', 'Glee Cast', 'Taylor Swift', 'Radiohead', 'Frank Zappa', 'Whitney Houston', 'Bob Dylan', 'Abraham Lincoln', 'The Beatles', 'Stevie Wonder', 'Lil B', 'Marshmello', 'Queen', 'Mariah Carey', 'Metallica', 'JP Saxe', 'Elton John', 'Noah Kahan', 'Macklemore', 'Harry Styles', 'Prince', 'Frank Sinatra', 'Ed Sheeran', 'J. Cole', 'Burial', 'Michael Jackson', '2Pac', 'Ella Fitzgerald', 'The Weeknd', 'Joji', 'The Grateful Dead', 'Bruno Mars', 'Shawn Mendes', 'Miley Cyrus', 'Lil Wayne', 'Adele', 'Nirvana', 'Clean Bandit', 'Avril Lavigne', "Guns N' Roses", 'Drake', 'Coldplay', 'Black Eyed Peas', 'Van Morrison', 'AURORA', 'Elvis Presley'}
    
    # Read the semi-processed data
    logger.info("reading semi-processed data")
    semi_processed = pd.read_csv(processed_filepath, index_col=0)
    
    # Dropping the views column and renaming the tag and title columns
    logger.info("dropping views column and renaming tag and title columns")
    semi_processed = semi_processed.drop(columns=["views"]).rename({"tag":"genre", "title":"song"}, axis=1)
    
    # Choosing only the 100 most popular artists
    logger.info("choosing only the 100 most popular artists")
    semi_processed = semi_processed[semi_processed['artist'].isin(best_100_artists)]
    
    # Dropping the songs with year < 1950 and converting the year column to int16 for memory efficiency
    logger.info("dropping the songs with year < 1950 and converting the year column to int16 for memory efficiency")
    semi_processed = semi_processed[semi_processed['year'] > 1950]
    semi_processed = semi_processed.astype({"year":np.int16})

    # Dropping the songs with genre = 'misc'
    logger.info("dropping the songs with genre = 'misc'")
    all_processed = semi_processed[semi_processed['genre'] != 'misc']
    
    
    # Saving the processed data
    logger.info("saving the processed data")
    all_processed.to_csv(processed_filepath, index=False)
    
    
    logger.info('finished preprocessing from raw data and saved in projectdir/data/preprocessed')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    makeProcessed()
