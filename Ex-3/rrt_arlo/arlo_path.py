import math

def path_to_arlo_instructions(path, current_dir):
    instructions = []
    current_point = path[0] 
    for point in path[1:]:
        target_dir = [point[0] - current_point[0], point[1] - current_point[1]]
        dot_prod = current_dir[0] * target_dir[0] + current_dir[1] * target_dir[1]
        cross_prod = current_dir[0] * target_dir[1] + current_dir[1] * target_dir[0]
        pass
    return instructions