import logging
from time import gmtime, strftime
import os.path


def initialize_logger(output_dir):
    logname = strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = \
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.FileHandler(os.path.join(output_dir, \
                                               "info_{}.log".format(logname)), "w", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # # create error file handler and set level to error
    # handler = logging.FileHandler(os.path.join(output_dir, "error.log"),"w", encoding=None, delay="true")
    # handler.setLevel(logging.ERROR)
    # formatter = logging.Formatter("%(levelname)s - %(message)s")
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    # # create debug file handler and set level to debug
    # handler = logging.FileHandler(os.path.join(output_dir, "all.log"),"w")
    # handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(levelname)s - %(message)s")
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
