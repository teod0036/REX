import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer

from copy import deepcopy

from collections import defaultdict

# Flags
onRobot = False  # Whether or not we are running on the Arlo robot
showGUI = True  # Whether or not to open GUI windows
instruction_debug = False  # Whether you want to debug the isntrcution execution code, even if you don't have an arlo


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
    You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    # sys.path.append("../../../../Arlo/python")
    # robot.py is in this directory
    pass

if isRunningOnArlo():
    instruction_debug = False

try:
    print("selflocalize.py: assuming we are running on robot")
    import robot

    # onRobot = True
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    # onRobot = False


# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarks = {
    3: np.array((0.0, 0.0), dtype=np.float32),  # Coordinates for landmark 1
    11: np.array((300.0, 0.0), dtype=np.float32),  # Coordinates for landmark 2
}
landmarkIDs = list(landmarks.keys())
landmark_colors = [CRED, CGREEN]  # Colors used when drawing the landmarks


def jet(x):
    """Colour map for drawing particles. This function determines the colour of
    a particle from its weight."""

    r = (
        (x >= 3.0 / 8.0 and x < 5.0 / 8.0) * (4.0 * x - 3.0 / 2.0)
        + (x >= 5.0 / 8.0 and x < 7.0 / 8.0)
        + (x >= 7.0 / 8.0) * (-4.0 * x + 9.0 / 2.0)
    )
    g = (
        (x >= 1.0 / 8.0 and x < 3.0 / 8.0) * (4.0 * x - 1.0 / 2.0)
        + (x >= 3.0 / 8.0 and x < 5.0 / 8.0)
        + (x >= 5.0 / 8.0 and x < 7.0 / 8.0) * (-4.0 * x + 7.0 / 2.0)
    )
    b = (
        (x < 1.0 / 8.0) * (4.0 * x + 1.0 / 2.0)
        + (x >= 1.0 / 8.0 and x < 3.0 / 8.0)
        + (x >= 3.0 / 8.0 and x < 5.0 / 8.0) * (-4.0 * x + 5.0 / 2.0)
    )

    return (255.0 * float(r), 255.0 * float(g), 255.0 * float(b))


def draw_world(est_pose, particles, world, path=None):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE  # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x, y), 2, colour, 2)
        b = (
            int(particle.getX() + 15.0 * np.cos(particle.getTheta())) + offsetX,
            ymax
            - (int(particle.getY() + 15.0 * np.sin(particle.getTheta())) + offsetY),
        )
        cv2.line(world, (x, y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)
        fontScale = 1
        thickness = 2
        lineType = 2
        cv2.putText(
            world,
            f"{ID}",
            lm,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            (0, 0, 0),
            thickness,
            lineType,
        )
    
    if path is not None:
        for i in range(len(path)):
            if i >= (len(path)-1):
                break
            point1 = (int(path[i][0] * 100)+offsetX, int(ymax - (path[i][1] * 100+offsetY)))
            point2 = (int(path[i+1][0] * 100)+offsetX, int(ymax - (path[i+1][1] * 100+offsetY)))
            cv2.line(img=world, pt1=point1, pt2=point2, color=(0, 0, 255), thickness=2)        

    # Draw estimated robot pose
    a = (int(est_pose.getX()) + offsetX, ymax - (int(est_pose.getY()) + offsetY))
    b = (
        int(est_pose.getX() + 15.0 * np.cos(est_pose.getTheta())) + offsetX,
        ymax - (int(est_pose.getY() + 15.0 * np.sin(est_pose.getTheta())) + offsetY),
    )
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)


def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points.
        p = particle.Particle(
            600.0 * np.random.ranf() - 100.0,
            600.0 * np.random.ranf() - 250.0,
            np.mod(2.0 * np.pi * np.random.ranf(), 2.0 * np.pi),
            1.0 / num_particles,
        )
        particles.append(p)

    return particles


def initialize_camera():
    print("Opening and initializing camera")
    if isRunningOnArlo():
        cam = camera.Camera(0, robottype="arlo", useCaptureThread=False)
    else:
        cam = camera.Camera(0, robottype="macbookpro", useCaptureThread=False)
    return cam


def control_manually(action, velocity, angular_velocity):
    if action == ord("w"):  # Forward
        if velocity < 4.0:
            velocity += 4.0
    elif action == ord("s"):  # Backwards
        if velocity > -4.0:
            velocity -= 4.0
    elif action == ord("x"):  # Stop
        velocity = 0.0
        angular_velocity = 0.0
    elif action == ord("a"):  # Left
        if (angular_velocity < 0.2):  # clamp angular velocity so particles can't accelerate infinitely
            angular_velocity += 0.2
    elif action == ord("d"):  # Right
        if (angular_velocity > -0.2):  # clamp angular velocity so particles can't accelerate infinitely
            angular_velocity -= 0.2
    return velocity, angular_velocity


def RecalculatePath(goal, est_pose, instructions, path_coords=[]):
    # print statement for debugging reasons
    print("recalculating path")
    print()

    # Get the robots position in meters, since the path planning needs an input in meters
    pos_meter = np.array([est_pose.getX() / 100, est_pose.getY() / 100])

    # Get the robots direction and orthogonal directions as vectors since they are needed to calculate
    # the robot instructions to for following the path outputted by the RRT algorithm
    current_dir = np.array([np.cos(est_pose.getTheta()), np.sin(est_pose.getTheta())])
    current_dir_orthogonal = np.column_stack([-current_dir[1], current_dir[0]])

    # Get a list of instructions that the robot can execute
    # Instructions are created based on RRT algorithm output
    instructions = plan_path.plan_path(
        path_map,
        robot_model,
        current_dir=current_dir,
        current_dir_orthogonal=current_dir_orthogonal,
        start=pos_meter,
        goal=goal,
        path_coords=path_coords
    )  # type: ignore

    # remove instructions exceeding the maximum.
    if maxinstructions_per_execution is not None:
        instructions = instructions[:maxinstructions_per_execution]

    return instructions


def check_if_arrived(goal, est_pose, instructions, arrived):
    # Calculate how far the robot is from it's goal.
    # This value is used to check whether the robot has arrived or not.
    # The distance is in meters.
    dist_from_target = np.linalg.norm([goal[0] - (est_pose.getX() / 100), goal[1] - (est_pose.getY() / 100)])

    # Print statements for debugging reasons
    print(f"I am currently {dist_from_target} meters from the target position")
    print(f"Current target is: {goal}")
    print(f"Current posistion is: [{est_pose.getX()/100}, {est_pose.getY()/100}]")
    print(f"My instructions are {instructions}")
    print()
    # If the robot center is closer than 40 cm to it's target set the arrived flag to true.
    # If the arrived falg is already true, the robot has arrived at it's target.
    if dist_from_target <= 0.40:
        print("I am close to my target")
        print()
        if arrived:
            print("I have arrived")
            print(f"The target is at {goal}")
            print(f"I am at [{est_pose.getX()/100}, {est_pose.getY()/100}]")
            print()
            return True
        # Clear the instruction list to allow the robot to survey it's surroundings again
        # to make sure it is in the right place without driving away
        instructions[:] = []
        return True

    # If the arrived flag was set to true but the robot no longer fulfills the condition flip it to false
    # This usually happens when the robot recalculates it's position and realizes it is actually somewhere else
    elif arrived:
        print("I have realized i am not close to my target")
        print()
        return False


def turn_particles(instructions):
    # Unpack the direction and degrees rotated
    withclock, degrees = instructions[0][1]
    print(f"Rotating particles by {'+' if withclock else '-'}{degrees} degrees ")

    # Convert the degrees to radians
    radians = np.deg2rad(degrees)

    # If the robot rotated clockwise it means that the paritcles should rotate in the negative direction
    if withclock:
        radians = radians * -1

    # Set the angular velocity to the value obtained based on the direction and degrees rotated
    angular_velocity = radians

    # Set the angular uncertainty to the uncertainty used when rotating
    angular_uncertainty = angular_uncertainty_on_turn

    return angular_velocity, angular_uncertainty


def forward_particles(instructions): #This function doesn't do anything to the robot
    # Unpack the meters driven
    meters = instructions[0][1]

    # Convert meters to centimeters, since the particles move in units of centimeters
    centimeters = meters * 100

    # Set the velocity to the value obtained based on the meters dictated by the instruction
    velocity = centimeters
    print(f"Forwarding particles by {centimeters} cm")


    # Set the angular uncertainty to the uncertainty used when driving
    angular_uncertainty = angular_uncertainty_on_forward

    return velocity, angular_uncertainty


def generate_rotate_in_place(deg_per_rot):
    for i in range(360 // deg_per_rot):
        instructions.append(["turn", (False, deg_per_rot)])

# Main program #
if __name__ == "__main__":
    try:
        if showGUI:
            # Open windows
            WIN_RF1 = "Robot view"
            cv2.namedWindow(WIN_RF1)
            cv2.moveWindow(WIN_RF1, 50, 50)

            WIN_World = "World view"
            cv2.namedWindow(WIN_World)
            cv2.moveWindow(WIN_World, 500, 50)

        import plan_path
        import map.robot_models as robot_models
        import map.occupancy_grid_map as occupancy_grid_map

        # Initialize particles
        num_particles = 1000

        if instruction_debug:
            # smaller amount of particles to test pathfinding and the effect of instructions
            num_particles = 4

        particles = initialize_particles(num_particles)

        est_pose = particle.estimate_pose(particles)  # Initial random estimate

        # Driving parameters
        velocity = 0.0  # cm/instruction
        angular_velocity = 0.0  # radians/instruction
        velocity_uncertainty = 4  # cm/instruction

        # Representation of the uncertainty in drift to either side when moving forwards
        angular_uncertainty_on_forward = np.deg2rad(1.5)  # radians/instruction
        # Representation of the uncertainty of turning precision
        angular_uncertainty_on_turn = np.deg2rad(2)  # radians/instruction

        # Angular uncertainty is always equal to either angular_uncertainty_on_turn or angular_uncertainty_on_forward
        angular_uncertainty = angular_uncertainty_on_turn  # radians/instruction

        # More uncertainty parameters
        distance_measurement_uncertainty = 5.0 * 3  # cm
        angle_measurement_uncertainty = np.deg2rad(5)  # radians

        # Initialize the robot (XXX: You do this)
        if isRunningOnArlo():
            import exec_arlo_instructions as exec

            arlo = robot.Robot()

        # Create map used for pathfinding, map uses meters as it's unit
        map_res = 0.05
        path_map = occupancy_grid_map.OccupancyGridMap(
            low=np.array((-2, -5)), high=np.array((8, 5)), resolution=map_res
        )

        origins = []
        for origin in landmarks.values():
            origins.append((origin[0] / 100, origin[1] / 100))
        radius_squared = ((18 / 100) + (22.5 / 100)) ** 2
        path_map.plot_centroid(np.array(origins), np.array(radius_squared))

        # Create robot model for pathfinding
        path_res = map_res
        robot_model = robot_models.PointMassModel(ctrl_range=[-path_res, path_res])

        # Where the robot wants to go, position in meters
        goal = (landmarks[landmarkIDs[0]] + landmarks[landmarkIDs[1]]) / 2 / 100.0
        print(f"Target point: {goal}")

        # Allocate space for world map
        world = np.zeros((500, 500, 3), dtype=np.uint8)

        # Draw map
        draw_world(est_pose, particles, world)

        cam = initialize_camera()

        # Initialize the instruction list
        instructions = []

        # value to control how many degrees the robot rotates at a time when surveying its surroundings
        deg_per_rot = 30

        # Make the robot start by rotating around itself once
        generate_rotate_in_place(deg_per_rot)

        # The maximum amount of instructions the robot executs before surveying its surroundings.
        # This value should always be a multiple of 2, set value to None to remove cap
        maxinstructions_per_execution = 8

        # Initialize flag designating that the robot believes it has arrived
        arrived = False
        
        #used for drawing path
        path_coords=[]

        while True:
            if instruction_debug:
                time.sleep(0.2)

            # Move the robot according to user input (only for testing)
            action = cv2.waitKey(10)
            if action == ord("q"):  # Quit
                break

            if not isRunningOnArlo():
                velocity, angular_velocity = control_manually(action, velocity, angular_velocity)

            # Use motor controls to update particles
            # XXX: Make the robot drive
            # XXX: You do this

            # This code block mainly calculates a new path for the robot to take
            # Instructions having a length of 0 means the robot has run out of plan for where to go
            if len(instructions) == 0:
                instructions = RecalculatePath(goal, est_pose, instructions)
                if check_if_arrived(goal, est_pose, instructions, arrived):
                    arrived = True
                    print("Double checking if really arrived")
                    if check_if_arrived(goal, est_pose, instructions, arrived):
                        break
                    else:
                        arrived = False

                # Make the robot end every instruction sequence by rotating around itself once.
                generate_rotate_in_place(deg_per_rot)

            # This code block moves the robot and
            # updates the velocity and angular velocity used when updating the particles
            # This code block only runs if the robot has instructions
            # Instructions are empty at this point if the path planning algorithm didn't find a path
            if (isRunningOnArlo() or instruction_debug) and len(instructions) != 0:
                # reset the velocity and angular velocity to 0
                angular_velocity = 0
                velocity = 0

                # If the current instruction is a turn instruction update angular velocity accordingly
                if instructions[0][0] == "turn":
                    angular_velocity, angular_uncertainty = turn_particles(instructions)

                # If the current instruction is a forward instruction update velocity accordingly
                elif instructions[0][0] == "forward":
                    velocity, angular_uncertainty = forward_particles(instructions)

                # If the instruction is unknown print a message and do nothing
                else:
                    print("Unknown instruction, instructions have to be either turn or forward")

                # If run in debug mode simulate executing instructions by removing the first entry in the instruction list
                if instruction_debug:
                    del instructions[0]
                    if len(instructions) == 0:
                        velocity = 0
                        angular_velocity = 0

                # If not running in debug mode execute the next instruction.
                else:
                    exec.next(instructions)

            # predict particles after movement (prior):
            for p in particles:
                x_offset = np.cos(p.getTheta()) * velocity
                y_offset = np.sin(p.getTheta()) * velocity

                p = particle.move_particle(p, x_offset, y_offset, angular_velocity)

            # Add some noise
            particle.add_uncertainty(particles, velocity_uncertainty, angular_uncertainty)

            # Fetch next frame
            colour = cam.get_next_frame()

            # Detect objects
            objectIDs, dists, angles = cam.detect_aruco_objects(colour)
            if (
                not isinstance(objectIDs, type(None))
                and not isinstance(dists, type(None))
                and not isinstance(angles, type(None))
            ):
                # List detected objects
                objectDict = {}
                for i in range(len(objectIDs)):
                    print(f"{ objectIDs[i] = }, { dists[i] = }, { angles[i] = }")

                    # XXX: Do something for each detected object - remember, the same ID may appear several times
                    if objectIDs[i] not in objectDict:
                        objectDict[objectIDs[i]] = (dists[i], angles[i])
                    elif dists[i] < objectDict[objectIDs[i]][0]:
                        objectDict[objectIDs[i]] = (dists[i], angles[i])

                # Compute particle weights
                # XXX: You do this

                # put positions and weights into homogenous numpy arrays for vectorized operations
                positions = np.array([(p.getX(), p.getY()) for p in particles], dtype=np.float32)
                orientations = np.array(
                    [(np.cos(p.getTheta()), np.sin(p.getTheta())) for p in particles],
                    dtype=np.float32,
                )
                orientations_orthogonal = np.column_stack([orientations[:, 1], -orientations[:, 0]])  # 90° rotated
                weights = np.array([p.getWeight() for p in particles], dtype=np.float32)

                # scale the weights for each observation (multiply by likelihood)
                for objID, (objDist, objAngle) in objectDict.items():
                    if objID not in landmarkIDs:
                        continue

                    # vector from particle to landmark
                    v = landmarks[objID][np.newaxis, :] - positions
                    distances = np.linalg.norm(v, axis=1)

                    # accumulate likelihood for each object for each measurement
                    distance_pdf = (1 / (distance_measurement_uncertainty * np.sqrt(2 * np.pi))) * np.exp(
                        -0.5 * ((objDist - distances) / distance_measurement_uncertainty) ** 2
                    )

                    # angles from particle direction to landmark direction
                    v /= distances[:, np.newaxis]
                    dot = np.clip(np.sum(v * orientations, axis=1), -1.0, 1.0)
                    cross = np.sum(v * orientations_orthogonal, axis=1)
                    angles = np.sign(cross) * np.arccos(dot)

                    angle_pdf = (1 / (angle_measurement_uncertainty * np.sqrt(2 * np.pi))) * np.exp(
                        -0.5 * ((objAngle - angles) / angle_measurement_uncertainty) ** 2
                    )

                    weights *= distance_pdf * angle_pdf

                # normalise weights (compute the posterior)
                weights += 1e-12  # avoid problems with zeroes
                weights /= np.sum(weights)

                # Resampling
                # XXX: You do this
                # resample particles to avoid degenerate particles
                num_effective_particles = 1 / np.sum(np.square(weights))
                if num_effective_particles < num_particles / 2: # if less than half of the particles contribute meaningfully
                    cumulative_sum = np.cumsum(weights)
                    cumulative_sum[-1] = 1.0  # fix issues with zeroes

                    # use uniform distribution once, and do systematic resampling
                    offset = np.random.rand()
                    positions = (np.arange(num_particles) + offset) / num_particles
                    indices = np.searchsorted(cumulative_sum, positions, side='right').astype(int)

                    # deep copy particless to avoid reference errors
                    particles = [deepcopy(particles[i]) for i in indices]

                    # too many degenerate particles - reset weights to uniform distribution
                    for p in particles:
                        p.setWeight(1.0 / num_particles)
                else:
                    # set weights for visualization
                    for i, p in enumerate(particles):
                        p.setWeight(weights[i])

                # Draw detected objects
                cam.draw_aruco_objects(colour)
            else:
                # No observation - reset weights to uniform distribution
                for p in particles:
                    p.setWeight(1.0 / num_particles)

            est_pose = particle.estimate_pose(particles)  # The estimate of the robots current pose

            if showGUI:
                # Draw map
                draw_world(est_pose, particles, world, path_coords)

                # Show frame
                cv2.imshow(WIN_RF1, colour)

                # Show world
                cv2.imshow(WIN_World, world)

    finally:
        # Make sure to clean up even if an exception occurred

        # Close all windows
        cv2.destroyAllWindows()

        # Clean-up capture thread
        cam.terminateCaptureThread()
