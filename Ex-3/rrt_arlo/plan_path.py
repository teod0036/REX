from arlo_path import path_to_arlo_instructions
from rrt import RRT
from rdp import rdp
import numpy as np

def plan_path(map, robot, start=np.array([0, 0], dtype=np.float32), goal=np.array([0, 1.9], dtype=np.float32), expand_dis=0.2, debug=False):
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
    import map.occupancy_grid_map as occupancy_grid_map
    import map.robot_models as robot_models
    from exec_arlo_instructions import exec_instructions


    path_res = 0.05
    map = occupancy_grid_map.OccupancyGridMap(low=np.array((-1, 0), dtype=np.float32), high=np.array((1, 2), dtype=np.float32), resolution=path_res)
    map.populate()

    robot = robot_models.PointMassModel(ctrl_range=[-path_res, path_res])

    instructions = plan_path(map=map, robot=robot, debug=True)
    
    exec_instructions(instructions)
