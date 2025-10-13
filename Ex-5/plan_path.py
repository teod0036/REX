from arlo_path import path_to_arlo_instructions
from rrt import RRT
from rdp import rdp
import numpy as np
import map_plot_markers

map_low = map_plot_markers.map_low
map_high = map_plot_markers.map_high
map_res = map_plot_markers.map_res

marker_radius = map_plot_markers.marker_radius_m

def plan_path(map, robot, current_dir=np.array([0,1]), current_dir_orthogonal=np.array([-1,0]), start=np.array([0.0, 0.0], dtype=np.float32), goal=np.array([0, 1.9], dtype=np.float32), expand_dis=0.2, debug=False):
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

    return path_to_arlo_instructions(path, current_dir=current_dir, current_dir_orthogonal=current_dir_orthogonal)
