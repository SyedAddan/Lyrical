# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
import subprocess
import zipfile

PROJECT_DIR = Path(__file__).resolve().parents[2]


def makeRaw():
    """ 
        Fetches the dataset used from the Kaggle dataset: 'nikhilnayak123/5-million-song-lyrics-dataset'
        and places it into (/projectdir/data/raw).
    """
    logger = logging.getLogger(__name__)
    logger.info('fetching raw data from kaggle')
    
    dataset_url = "nikhilnayak123/5-million-song-lyrics-dataset"
    destination_path = os.path.join(PROJECT_DIR, "data", "raw")
    if not(os.path.exists(destination_path)):
        os.makedirs(destination_path)
    destination_zip_path = os.path.join(destination_path, "5-million-song-lyrics-dataset.zip")
    destination_csv_path = os.path.join(destination_path, "song_lyrics.csv")
    
    # check if raw csv file already exists
    if not(os.path.exists(destination_csv_path)):
        # if it doesn't, check if the raw zip file exists
        if not(os.path.exists(destination_zip_path)):
            # if it doesn't, fetch it from kaggle
            command = f"kaggle datasets download -d {dataset_url}"
            
            # download the raw zip file
            try:
                subprocess.call(command, shell=True)
                logger.info(f'raw zip file fetched from {dataset_url}')
                logger.info(f"now placing raw zip file into {destination_path}")
                os.rename("5-million-song-lyrics-dataset.zip", destination_zip_path)
                logger.info(f'raw zip file placed into {destination_path}')
                
            except:
                logger.error(f'failed to fetch dataset from {dataset_url}')
                return
                
        else:
            # if it does, go on to unzip it
            logger.info(f"raw zip file already exists in {destination_path}")
            
        # unzip the raw zip file
        logger.info(f'unzipping raw zip file to {destination_path}')
        try:
            with zipfile.ZipFile(destination_zip_path, 'r') as zip_ref:
                zip_ref.extractall(destination_path)
            os.remove(destination_zip_path)
            os.rename(os.path.join(destination_path, "ds2.csv"), destination_csv_path)
            logger.info(f'raw zip file unzipped to {destination_path}')
        except:
            logger.error(f'failed to unzip raw zip file to {destination_path}')
            return
         
    else:
        # if it does, do nothing
        logger.info(f'raw csv file already exists in {destination_path}')
                
                

if __name__ == '__main__':
    # configure logging
    log_fmt = '%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # make the raw dataset
    makeRaw()