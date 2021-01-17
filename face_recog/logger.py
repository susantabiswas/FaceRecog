import logging
import sys

class LoggerFactory:
    def create_formatter(self, format_pattern:str):
        format_pattern = format_pattern or \
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        return logging.Formatter(format_pattern)

    def get_console_handler(self, formatter, level=logging.INFO, stream=sys.stdout):
        # create a stream handler, it can be for stdout or stderr
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        return console_handler
        
    def get_file_handler(self, formatter, level=logging.INFO, file_path:str='data/app.log'):
        file_handler = logging.FileHandler(filename=file_path) 
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        return file_handler

    def get_logger(self, logger_name, level=logging.DEBUG, format_pattern:str=None, file_path:str='data/app.log'):
        # Creates the default logger
        logger = logging.getLogger(logger_name)
        formatter = self.create_formatter(format_pattern=format_pattern)
        # Get the stream handlers, by default they are set at INFO level 
        console_handler = self.get_console_handler(formatter=formatter, level=logging.INFO)
        file_handler = self.get_file_handler(formatter=formatter, level=logging.INFO, file_path=file_path)
        # Add all the stream handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        # Set the logger level, this is different from stream handler level
        # Stream handler levels are further filters when data reaches them
        logger.setLevel(level)
        logger.propagate = False
        return logger

    def create_logger(self, logger_name, handlers, propagate_error:bool=False):
        logger = logging.getLogger(logger_name)
        # Add all the stream handlers
        for handler in handlers:
            logger.addHandler(handler)
        logger.propagate = propagate_error
        return logger

    