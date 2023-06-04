# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
import gc
from pathlib import Path


def makeProcessed():
    """ 
        Builds Features from processed data in /projectdir/data/processed into
        cleaned data ready to be analyzed (saved in /projectdir/data/features).
    """
    logger = logging.getLogger(__name__)
    logger.info('building features from processed data')
    
    input_filepath = Path("../../data/processed")
    
    logger.info('input_filepath: {}'.format(input_filepath))
    
    # this can take more than 12 hours to run
    with pd.read_csv(
        "../data/raw/song_lyrics.csv",
        chunksize=5 * 10**3,
        usecols=["title", "artist", "year", "tag", "views", "lyrics"],
        dtype={"year": np.int16, "views": np.int32}
    ) as chunks:

        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx}")

            # drop N.A. lyrics
            chunk = chunk.dropna(subset=["lyrics"])
            
            # drop romanizations
            chunk = chunk[chunk["artist"] != "Genius Romanizations"]
            chunk = chunk[~chunk["title"].str.contains(r"\(?romanized\)?", regex=True, na=False, case=False)]

            # remove invalid years
            chunk = chunk[chunk["year"] < 2023]
            
            # remove duplicated entries
            chunk = chunk.drop_duplicates(subset=["title", "artist", "year"])
            
            # remove special characters from lyrics
            pattern = r"(?m)^\[.*?\]$"
            chunk["lyrics"] = chunk["lyrics"].str.replace(pattern, "", regex=True)
            
            # remove empty lines
            pattern = r"\n|\n\n"
            chunk["lyrics"] = chunk["lyrics"].str.replace(pattern, " ", regex=True)

            # drop lyrics that are too short or too long
            chunk = chunk[chunk["lyrics"].str.len().between(10**2, 10**5)]

            # analyze language
            chunk["language"] = chunk["lyrics"].apply(identify_language)
            print(f'{len(chunk[chunk["language_cld3"].isna()])} not identified lyrics using by the language detection model.')
            
            # drop non-english lyrics
            chuck = chunk[chunk["language"] == "en"][["artist", "tag", "lyrics"]]
            
            # save processed data
            chunk.to_csv("../data/processed/lyrics_processed.csv", mode="a", header=not os.path.exists("../data/processed/lyrics_processed.csv"), index=False)
            
            del chunk
            gc.collect()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    makeProcessed()
