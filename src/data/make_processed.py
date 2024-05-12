# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
# from transformers import pipeline
import gc

# model_ckpt = "papluca/xlm-roberta-base-language-detection"
# pipe = pipeline("text-classification", model=model_ckpt, device=-1)
PROJECT_DIR = Path(__file__).resolve().parents[2]

# def identify_language(lyrics: str) -> str|np.nan:
#     res = pipe([lyrics], truncation=True, max_length=128)
#     return res[0]['label'] if res[0]['score'] > 0.5 else np.nan


def makeProcessed():
    """ 
        Preprocesses the raw data from /projectdir/data/raw
        and saves the preprocessed data in /projectdir/data/processed.
    """
    logger = logging.getLogger(__name__)
    logger.info('making preprocessed data set from raw data')
    
    input_filepath = Path(os.path.join(PROJECT_DIR, "data", "raw", "song_lyrics.csv"))
    processed_filepath = Path(os.path.join(PROJECT_DIR, "data", "processed", "lyrics_processed.csv"))
    
    if not os.path.exists(input_filepath):
        logger.error(f'input_filepath: {input_filepath} does not exist! Please run make_raw.py first to download the raw data.')
        return
    
    logger.info('input_filepath: {}'.format(input_filepath))
    
    if os.path.exists(processed_filepath):
        logger.warning(f'processed_filepath: {processed_filepath} already exists! Please delete it to re-run the preprocessing.')
        return
    
    logger.info('----------------------------------------------------')
    
    logger.info('Staring Preprocessing!')
    logger.warning('Please make sure you have enough memory to run this script. If you are running this on a local machine, please make sure you have at least 16GB of RAM.')
    
    logger.info('----------------------------------------------------')
    
    with pd.read_csv(
        input_filepath,
        chunksize=5 * (10**3),
        usecols=["title", "artist", "year", "tag", "lyrics"],
        dtype={"year": np.int16}
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
            chunk = chunk[chunk['year'] > 1980]
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
            chunk = chunk[chunk["lyrics"].str.len().between(10**2, 10**4)]

            # Dropping the views column and renaming the tag and title columns
            logger.info(f"renaming tag and title columns to genre and song respectively from chunk {idx}...")
            chunk = chunk.rename({"tag":"genre", "title":"song"}, axis=1)

            # Dropping the songs with genre = 'misc'
            logger.info(f"dropping the songs with genre = 'misc' from chuck {idx}...")
            chunk = chunk[chunk['genre'] != 'misc']
            
            # # analyze language
            # logger.info(f"analyzing language from chunk {idx}...")
            # chunk["language"] = chunk["lyrics"].apply(identify_language)
            # logger.info(f'{len(chunk[chunk["language"].isna()])} not identified lyrics using by the language detection model.')
            
            # # drop non-english lyrics
            # logger.info(f"dropping non-english lyrics from chunk {idx}...") 
            # chunk = chunk[chunk["language"] == "en"][["artist", "tag", "lyrics"]]
            
            # save processed data
            logger.info(f"saving processed data from chunk {idx}...")
            chunk.to_csv(processed_filepath, mode="a", header=not os.path.exists(processed_filepath), index=False)
            
            logger.info(f"garbage collecting chunk {idx}...")
            del chunk
            gc.collect()
            
            logger.info(f"finished processing chunk {idx}...")
            logger.info("----------------------------------------------------")
            
    logger.info('finished preprocessing from raw data and saved in {processed_filepath}')
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    makeProcessed()
