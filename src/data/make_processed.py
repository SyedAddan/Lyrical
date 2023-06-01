# -*- coding: utf-8 -*-
import os
import logging
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
    logger.info("features built")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    makeProcessed()
