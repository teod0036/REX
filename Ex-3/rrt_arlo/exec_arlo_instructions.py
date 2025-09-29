import move_arlo
from rdp import rdp
from arlo_path import path_to_arlo_instructions

def exec_instructions(arlo_instructions):
    for instruction in arlo_instructions:
        getattr(move_arlo, instruction[0])(instruction[1])

if __name__ == '__main__':
    path = [[0, 1.9], [-0.11842211,  1.75475859], [-0.24480625,  1.59975199], [-0.3711904 ,  1.44474538], [-0.54398795,  1.35659427], [-0.6791863 ,  1.20921243], [-0.61288822,  1.02052072], [-0.59856512,  0.82103425], [-0.60073514,  0.62104603], [-0.40089634,  0.61301759], [-0.26021383,  0.47086123], [-0.11412401,  0.3342678], [0.03176616, 0.19746116], [0, 0]]
    path.reverse()
    smooth_path = rdp(path, 0.05*2)
    instructions = path_to_arlo_instructions(smooth_path, [0, 1])
    exec_instructions(instructions)