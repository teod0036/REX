from arlo_path import path_to_arlo_instructions
from rrt import RRT
from rdp import rdp
import numpy as np
import map_plot_markers as map_plot_markers

map_low = map_plot_markers.map_low
map_high = map_plot_markers.map_low
map_res = map_plot_markers.map_res

marker_radius = map_plot_markers.marker_radius_m



def plan_path(map, robot, start=np.array([0, 0], dtype=np.float32), goal=np.array([0, 1.9], dtype=np.float32), expand_dis=0.2, debug=False):
    rrt = RRT(
        start=start,
        goal=goal,
        robot_model=robot,
        map=map,
        expand_dis=expand_dis,
        path_resolution=map.resolution,
        )
    
    plan = rrt.planning(animation=False)
    if plan is None:
        print("no path found")
        return []
    
    path = rdp(plan, rrt.path_resolution*2)
    path.reverse()

    
    
    
    if debug:
        with open('path.txt', 'w') as f:
            f.write(str(path))

    return path_to_arlo_instructions(path)

if __name__ == "__main__":
    import map.occupancy_grid_map as occupancy_grid_map
    import map.robot_models as robot_models
    # from exec_arlo_instructions import exec_instructions


    map = occupancy_grid_map.OccupancyGridMap(low=map_low, high=map_high, resolution=map_res)
    map.populate()

    robot = robot_models.PointMassModel(ctrl_range=[-map_res, map_res])

    instructions = plan_path(map=map, robot=robot, debug=True)
    
    # exec_instructions(instructions)
