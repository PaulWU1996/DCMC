import numpy as np
import scipy.stats as stats

class State():
    """
        State: Describe the position of the target
    """
    def __init__(self, x = 0, y = 0, std_x = 0, std_y = 0):
        self.x = x
        self.y = y
        self.std_x = std_x
        self.std_y = std_y

def pos2vector(pos:State, range:State, vtype:str='2D'):
    if vtype == '2D':
        _, _, pdf = generate_2d_pdf(pos.x, pos.std_x, pos.y, pos.std_y, 0, range.x, 0, range.y, step=1)
    elif vtype == '1D':
        _, pdf = generate_1d_pdf(pos.x, pos.std_x, 0, range.x, step=1)
    else:
        raise ValueError('vtype must be 1D or 2D')
    pdf = pdf / np.max(pdf) # normalize the pdf and make the max value to 1
    return pdf

def generate_1d_pdf(mean, std, start, end, step):
    x = np.arange(start, end, step)
    y = stats.norm.pdf(x, mean, std)
    return x, y

def generate_2d_pdf(mean_x, std_x, mean_y, std_y, start_x, end_x, start_y, end_y, step):
    x, y = np.meshgrid(np.arange(start_x, end_x, step), np.arange(start_y, end_y, step))
    pos = np.dstack((x, y))
    rv = stats.multivariate_normal([mean_x, mean_y], [[std_x**2, 0], [0, std_y**2]])
    z = rv.pdf(pos)
    return x, y, z

class CustomException(Exception):
    def __init__(self, error_code, message,):
        super().__init__(error_code, message, )
        """
            Custom Exception for D-CMCM
            Error Code:
                0: Can be handled by the program
                1: Cannot be handled by the program but can be handled by the user
                -1: Cannot be handled by the program and cannot be handled by the user, the program will exit
        """
        self.error_code = error_code
