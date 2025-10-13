import math
import numpy as np

def path_to_arlo_instructions(path, current_dir=[0, 1], current_dir_orthogonal=[-1, 0]):
    instructions = []
    current_point = np.asarray(path[0])
    for point in path[1:]:
        point = np.asarray(point)
        target_dir = point - current_point
        point_dist = np.linalg.norm(target_dir)
        target_dir /= point_dist
        dot_prod = np.clip(np.sum(target_dir * current_dir), -1, 1)
        cross_prod = np.sum(target_dir * current_dir_orthogonal) 
        rot_deg = (180/math.pi) * np.arccos(dot_prod) 

        withclock = (np.sign(cross_prod) < 0)
        instructions.append(["turn", (withclock, round(rot_deg, 2))]) 
        instructions.append(["forward", round(point_dist, 2)])

        current_dir = target_dir
        current_point = point

    return instructions
