from arlo_path import path_to_arlo_instructions
from rrt import RRT
from rdp import rdp

def plan_path(map, robot, start=[0, 0], goal=[0, 1.9], expand_dis=0.2, debug=False):
    rrt = RRT(
        start=start,
        goal=goal,
        robot_model=robot,
        map=map,
        expand_dis=expand_dis,
        path_resolution=map.resolution,
        )

    path = rdp(rrt.planning(animation=False), rrt.path_resolution*2)
    path.reverse()
    if debug:
        with open('path.txt', 'w') as f:
            f.write(str(path))

    return path_to_arlo_instructions(path)

if __name__ == "__main__":
    import grid_occ, robot_models
    from exec_arlo_instructions import exec_instructions


    path_res = 0.05
    map = grid_occ.GridOccupancyMap(low=(-1, 0), high=(1, 2), res=path_res)
    map.populate()

    robot = robot_models.PointMassModel(ctrl_range=[-path_res, path_res])

    instructions = plan_path(map=map, robot=robot, debug=True)
    exec_instructions(instructions)
