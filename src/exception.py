import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    This function captures the details of an error, including the file name,
    line number, and error message, and returns a formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    error_message = "Error occurred in Python script name [{0}] at line number [{1}] with error message [{2}]".format(
        file_name, line_number, str(error)
    )
    
    return error_message

class CustomException(Exception):
    """
    A custom exception class that extends the base Exception class and includes detailed error messages.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
