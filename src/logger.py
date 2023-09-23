import logging

# Set up a specific logger with our desired output level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the threshold for this logger to DEBUG or above

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the log message handler to the logger
handler = logging.StreamHandler()  
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Write log results to a file
handler = logging.FileHandler('my_logs.txt', mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)