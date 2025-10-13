import math

def path_to_arlo_instructions(path, current_dir=[0, 1]):
    instructions = []
    current_point = path[0] 
    for point in path[1:]:
        target_dir = [point[0] - current_point[0], point[1] - current_point[1]]
        dot_prod = current_dir[0] * target_dir[0] + current_dir[1] * target_dir[1]
        cross_prod = current_dir[1] * target_dir[0] - current_dir[0] * target_dir[1]
        rot_deg = (180/math.pi) * math.atan2(cross_prod, dot_prod)
        rot_dir = (rot_deg < 0)
        instructions.append(["turn", (rot_dir, round(abs(rot_deg), 2))]) 

        point_dist = math.sqrt((current_point[0]-point[0])**2 + (current_point[1]-point[1])**2)
        instructions.append(["forward", round(point_dist, 2)])

        current_dir = target_dir
        current_point = point

    return instructions