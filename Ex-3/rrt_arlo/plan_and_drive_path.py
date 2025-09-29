from arlo_path import path_to_arlo_instructions
from rrt import RRT
from rdp import rdp
from exec_arlo_instructions import exec_instructions

def plan_and_drive(map, robot):
    rrt = RRT(
        start=[0, 0],
        goal=[0, 1.9],
        robot_model=robot,
        map=map,
        expand_dis=0.2,
        path_resolution=map.resolution,
        )

    path = rdp(rrt.planning(animation=False), rrt.path_resolution*2)
    instructions = path_to_arlo_instructions(path)
    exec_instructions(instructions)

if __name__ == "__main__":
    import grid_occ, robot_models

    path_res = 0.05
    map = grid_occ.GridOccupancyMap(low=(-1, 0), high=(1, 2), res=path_res)
    map.populate()

    robot = robot_models.PointMassModel(ctrl_range=[-path_res, path_res])

    plan_and_drive(map, robot)