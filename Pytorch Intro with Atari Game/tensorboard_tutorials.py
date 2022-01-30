import math
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter()
    funcs = {'sin' : math.sin, 'cos' :math.cos, 'tan' : math.tan}

    for angle in range(-360, 360):
        angle_rad = angle * math.pi/180
        for name, fun in funcs.items ():
            val = fun(angle_rad)
            writer.add_scalar(name, val, angle)
        
    writer.close()

    # To make the tensorboardX display, use be script in CLI
    # tensorboard --logdir runs 
    # Run this script in same directory with runs folder
