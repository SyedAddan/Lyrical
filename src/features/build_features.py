# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def buildFeatures():
    """ 
        Runs data processing scripts to turn raw data from (/projectdir/data/raw) into
        cleaned data ready to be analyzed (saved in /projectdir/data/processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making processed data set from raw data')
    
    input_filepath = Path("../../data/raw")
    
    logger.info('input_filepath: {}'.format(input_filepath))
    logger.info("made processed data")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    buildFeatures()
