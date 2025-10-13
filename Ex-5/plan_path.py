from arlo_path import path_to_arlo_instructions
from rrt import RRT
from rdp import rdp
import numpy as np

# def save_array(arr: np.ndarray, name : str):
#     datafile_name = f"{name}.npy"
#
#     with open(datafile_name, "wb") as f:
#         np.save(f, np.array(arr))
#         print(f"outputted array to {datafile_name}")
#
#
# def load_array(name: str) -> np.ndarray:
#     datafile_name = f"{name}.npy"
#
#     with open(datafile_name, "rb") as f:
#         return np.load(f)



def plan_path(map, robot,
              current_dir=np.array([0,1]),
              current_dir_orthogonal=np.array([-1,0]),
              start=np.array([0.0, 0.0], dtype=np.float32),
              goal=np.array([0, 1.9], dtype=np.float32),
              expand_dis=0.2,
              debug=False):
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

    # if debug:
    #     save_array(np.array(path), "path")

    return path_to_arlo_instructions(path, current_dir=current_dir, current_dir_orthogonal=current_dir_orthogonal)
