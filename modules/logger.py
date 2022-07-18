from logging import INFO, getLogger, StreamHandler, Formatter, FileHandler, DEBUG
import os

def get_logger(log_dir, log_name):
    logger = getLogger('root')
    logger.setLevel(INFO)

    if not logger.hasHandlers():
        sh = StreamHandler()
        sh.setLevel(DEBUG)

    fh = FileHandler(os.path.join(log_dir,log_name))
    fh.setLevel(DEBUG)

    if not logger.hasHandlers():
        formatter = Formatter('%(asctime)s [%(levelname)s] %(message)s')
        sh.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.addHandler(fh)

    return logger