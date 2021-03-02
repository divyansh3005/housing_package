import logging
import sys

from score_eval import pred_eval

LOG_FILENAME = "logfile.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
logging.info("main.....")
pred_eval() 
