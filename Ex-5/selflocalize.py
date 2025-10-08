import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer

from copy import deepcopy

# from copy import deepcopy


# Flags
onRobot = False  # Whether or not we are running on the Arlo robot
showGUI = True  # Whether or not to open GUI windows
instruction_debug = False #whether you want to debug the isntrcution execution code, even if you don't have an arlo

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot

if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    #sys.path.append("../../../../Arlo/python")
    #robot.py is in this directory
    pass

if isRunningOnArlo():
    instruction_debug = False
    showGUI = False

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
    7: np.array((300.0, 0.0), dtype=np.float32)  # Coordinates for landmark 2
}
landmarkIDs = list(landmarks.keys())
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks





def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    x = max(0, min(1, x)) 

    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX, 
                                     ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)
        fontScale = 1
        thickness = 2
        lineType = 2
        cv2.putText(world, f"{ID}", lm, cv2.FONT_HERSHEY_SIMPLEX, fontScale, landmark_colors[i], thickness, lineType)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX, 
         ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)



def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles


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
            num_particles = 2

        particles = initialize_particles(num_particles)

        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        # Driving parameters
        velocity = 0.0 #cm/instruction
        angular_velocity = 0.0 #radians/instruction
        velocity_uncertainty = 4 #cm/instruction
        angular_uncertainty = 0.01 #radians/instruction
        angular_uncertainty_on_forward = 0.103 #radians/instruction
        angular_uncertainty_on_turn = 0.01 #radians/instruction

        #More uncertainty parameters
        distance_measurement_uncertainty = 30.0  # cm
        angle_measurement_uncertainty = 11.5 * (2 * np.pi / 360) # radians


        # Initialize the robot (XXX: You do this)
        if isRunningOnArlo():
            import exec_arlo_instructions as exec
            arlo = robot.Robot()

        #create map used for pathfinding
        map_res = 0.05
        path_map = occupancy_grid_map.OccupancyGridMap(low=np.array((-1, -2.5)), high=np.array((4, 2.5)), resolution=map_res) 
        origins = []
        for origin in landmarks.values():
            origins.append((origin[0]/100, origin[1]/100))
        radius_squared = ((18 / 100) + (22.5 / 100))**2
        path_map.plot_centroid(np.array(origins), np.array(radius_squared))
        
        #create robot model for pathfinding
        path_res = map_res
        robot_model = robot_models.PointMassModel(ctrl_range=[-path_res, path_res])

        #Where the robot wants to go, position in meters
        goal = (landmarks[landmarkIDs[0]] + landmarks[landmarkIDs[1]]) / 2 / 100.
        print(f"Target point: {goal}")

        # Allocate space for world map
        world = np.zeros((500,500,3), dtype=np.uint8)

        # Draw map
        draw_world(est_pose, particles, world)

        print("Opening and initializing camera")
        if isRunningOnArlo():
            #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
            cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
        else:
            #cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
            cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=False)

        instructions = []
        for i in range(12):
            instructions.append(["turn", (True, 30)])
        maxinstructions_per_execution = 4
        arrived = False
        search = False

        while True:
            if instruction_debug:
                time.sleep(0.2)

            # Move the robot according to user input (only for testing)
            action = cv2.waitKey(10)
            if action == ord('q'): # Quit
                break
        
            if not isRunningOnArlo():
                if action == ord('w'): # Forward
                    if velocity < 4.0:
                        velocity += 4.0
                elif action == ord('s'): # Backwards
                    if velocity > -4.0:
                        velocity -= 4.0
                elif action == ord('x'): # Stop
                    velocity = 0.0
                    angular_velocity = 0.0
                elif action == ord('a'): # Left
                    if angular_velocity < 0.2:
                        angular_velocity += 0.2
                elif action == ord('d'): # Right
                    if angular_velocity > -0.2:
                        angular_velocity -= 0.2



            
            # Use motor controls to update particles
            # XXX: Make the robot drive
            # XXX: You do this
            if len(instructions) == 0:
                if search:
                    search = True
                    pos_meter = np.array([est_pose.getX() / 100, est_pose.getY() / 100])
                    current_dir = [np.cos(est_pose.getTheta()), np.sin(est_pose.getTheta())]
                    instructions = plan_path.plan_path(path_map, robot_model, current_dir=current_dir, start=pos_meter, goal=goal) #type: ignore
                    if len(instructions) == 0:
                        print(f"Current target is: {goal}")
                        print(f"Current posistion is: [{est_pose.getX()/100}, {est_pose.getX()/100}]")
                    if maxinstructions_per_execution is not None:
                        instructions = instructions[:maxinstructions_per_execution]
                else:
                    search = False
                    instructions = []
                    for i in range(12):
                        instructions.append(["turn", (True, 30)])
                #The distance is in meters
                dist_from_target = np.linalg.norm([goal[0]-(est_pose.getX()/100), goal[1]-(est_pose.getY()/100)])
                if len(instructions) == 2 and dist_from_target < 0.20:
                    if arrived:
                        print("I have arrived")
                        print(f"The target is at {goal}")
                        print(f"I am at [{est_pose.getX()/100}, {est_pose.getY()/100}]")
                        break
                    arrived = True
                    instructions = []
                    for i in range(12):
                        instructions.append(["turn", (True, 30)])
                elif arrived:
                    arrived = False

            if (isRunningOnArlo() or instruction_debug) and len(instructions) != 0:
                angular_velocity = 0
                velocity = 0
                if instructions[0][0] == "turn":
                    withclock, degrees = instructions[0][1]
                    radians = np.radians(degrees)
                    if withclock:
                        radians = radians * -1
                    angular_velocity = radians
                    angular_uncertainty = angular_uncertainty_on_turn
                elif instructions[0][0] == "forward":
                    meters = instructions[0][1]
                    #instructions have their argument in meters, so they have to be converted to centimeters
                    centimeters = meters * 100
                    velocity = centimeters
                    angular_uncertainty = angular_uncertainty_on_forward
                else:
                    print("Unknown instruction, instructions have to be either turn or forward")
                if instruction_debug:
                    del instructions[0]
                    if len(instructions) == 0:
                        velocity = 0
                        angular_velocity = 0
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
            if not isinstance(objectIDs, type(None)) and not isinstance(dists, type(None)) and not isinstance(angles, type(None)):

                # List detected objects
                # for i in range(len(objectIDs)):
                #     print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])

                # XXX: Do something for each detected object - remember, the same ID may appear several times
                objectDict = {objectIDs[i] : (dists[i], angles[i]) for i in range(len(objectIDs))}

                # Compute particle weights
                # XXX: You do this

                # put positions and weights into homogenous numpy arrays for vectorized operations
                positions = np.array([(p.getX(), p.getY()) for p in particles], dtype=np.float32)
                orientations = np.array([(np.cos(p.getTheta()), np.sin(p.getTheta())) for p in particles], dtype=np.float32)
                orientations_orthogonal = orientations[:, [1,0]]
                orientations_orthogonal[:, 0] *= -1
                weights = np.array([p.getWeight() for p in particles], dtype=np.float32)

                # scale the weights for each observation (multiply by likelihood)
                for (objID, (objDist, objAngle)) in objectDict.items():
                    if objID in landmarkIDs:
                        # compute distance and angles to presumed location of landmarks
                        v = landmarks[objID][np.newaxis, :] - positions
                        distances = np.linalg.norm(v, axis=1)
                        v /= distances[:, np.newaxis]
                        angles = np.multiply(np.sign(np.sum(v * orientations_orthogonal, axis=1)),
                                             np.arccos(np.sum(v * orientations, axis=1)))

                        # create normal distributions centered around measurements
                        distance_pdf = ((1 / (distance_measurement_uncertainty * np.sqrt(2 * np.pi))) *
                                        np.exp(-0.5 * ((objDist - distances) / distance_measurement_uncertainty) ** 2))
                        angle_pdf = ((1 / (angle_measurement_uncertainty * np.sqrt(2 * np.pi))) *
                                        np.exp(-0.5 * ((objAngle - angles) / angle_measurement_uncertainty) ** 2))
                        
                        # compute the weights for this particular landmark
                        weights_l = distance_pdf * angle_pdf

                        # compute the culminative weights for all landmarks
                        weights *= weights_l

                # normalise weights (compute the posterior)
                weights += 0.0000001 # avoid problems with zeroes 
                weights /= np.sum(weights) 

                # Resampling
                # XXX: You do this

                # resample particles to avoid degenerate particles
                cumulative_sum = np.cumsum(weights)
                cumulative_sum[-1] = 1. # avoid problems with zeroes
                indicies = np.searchsorted(cumulative_sum, np.random.uniform(size=num_particles))
                particles = [deepcopy(particles[index]) for index in indicies] # copy by value

                # reset weights to uniform distribution (for the next cycle of weighting and resampling)
                for p in particles:
                    p.setWeight(1.0 / num_particles)

                # Draw detected objects
                cam.draw_aruco_objects(colour)
            else:
                # No observation - reset weights to uniform distribution
                for p in particles:
                    p.setWeight(1.0/num_particles)

        
            est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

            if showGUI:
                # Draw map
                draw_world(est_pose, particles, world)
        
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
