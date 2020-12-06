'''
    This is a simple python scripts that enables you to control the simulator using a ps4 controller

    Credits to  https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/
                https://carla.readthedocs.io/en/latest/
                https://gist.github.com/claymcleod/028386b860b75e4f5472
'''


import glob
import os
import sys
import threading
import random
import time
import numpy as np
import cv2
import pygame
IM_WIDTH = 640
IM_HEIGHT = 480

try:
    # TODO: Make a smart way to change the path
    sys.path.append(glob.glob('D:\\Users\\cjrs2\\Downloads\\CARLA 0 9 10 1\\CARLA_0.9.10.1\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla




class CamaraManager:
    def __init__(self):
        self.frame = None

    def update_frame(self, frame):
        self.frame = frame

    def render(self, display_surface):
        image = self.frame[:,:,::-1]
        image = cv2.resize(image, (IM_WIDTH, IM_HEIGHT))
        image = pygame.surfarray.make_surface(image.swapaxes(0,1))
         # Show frames
        display_surface.fill((255,255,255))
        display_surface.blit(image, (0, 0)) 
        # Draws the surface object to the screen.   
        pygame.display.update() 
        

def process_img(data, display_surface, camaraManager, show_image=True):
    i = np.array(data.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    frame = i3

    if show_image:
        camaraManager.update_frame(frame)
        camaraManager.render(display_surface)


IM_WIDTH = 640
IM_HEIGHT = 480

def setup_pygame():
        pygame.init()
        # assigning values to X and Y variable 
        X = IM_WIDTH
        Y = IM_HEIGHT
        
        # create the display surface object 
        # of specific dimension..e(X, Y). 
        display_surface = pygame.display.set_mode((X, Y )) 
                
        
        # set the pygame window name 
        pygame.display.set_caption('Carla') 

        return display_surface

class SimulatorManager:
    def __init__(self):
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.actor_list = None
    
    def begin_simulation(self):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)
        self.actor_list = []

        self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()

    # TODO: car might be its own class
    def spawn_mycar(self): 
        bp = self.blueprint_library.filter('cybertruck')[0]

        spawn_point = random.choice(self.world.get_map().get_spawn_points())

        vehicle = self.world.spawn_actor(bp, spawn_point)
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        # vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

        self.actor_list.append(vehicle)

        # Add camera
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors
        # get the blueprint for this sensor
        blueprint = self.blueprint_library.find('sensor.camera.rgb')
        # change the dimensions of the image
        blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
        blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
        blueprint.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle
        spawn_point = carla.Transform(carla.Location(x=2.5, z=1.7))

        # spawn the sensor and attach to vehicle.
        sensor = self.world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

        # add sensor to list of actors
        self.actor_list.append(sensor)

        # Returns camara
        return vehicle, sensor

    def clear_actors(self):
        for actor in self.actor_list:
            actor.destroy()





def main():
    controls = {'throttle':0, 'brake': 0, 'steering':0, 'reverse': 0}

    image = CamaraManager()
    simulatorManager = SimulatorManager()

    try:
        
        # Set up pygame
        display_surface = setup_pygame()

        # Initial carla configuration
        simulatorManager.begin_simulation()

        # Spawns a car 
        vehicle, camera = simulatorManager.spawn_mycar()

        # Starts the sensor of the camera and process it
        camera.listen(lambda data: process_img(data, display_surface, image))

        # Sets the PS4 controller
        controller(controls, vehicle)
        print('Exit')

    finally:
        print('destroying actors')
        simulatorManager.clear_actors()
        print('done.')


def controller(controls, vehicle):
    ps4 = PS4Controller()
    ps4.init()
    ps4.listen(vehicle, controls)


'''
    Adapted from https://gist.github.com/claymcleod/028386b860b75e4f5472
    This is the class than handles the controller inputs and changes
    the values for the car.
'''
class PS4Controller(object):
    """Class representing the PS4 controller. Pretty straightforward functionality."""

    controller = None
    axis_data = None
    button_data = None
    hat_data = None

    def init(self):
        """Initialize the joystick components"""
        
        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

    def listen(self, vehicle, controls):
        """Listen for events to happen"""
        
        if not self.axis_data:
            self.axis_data = {}

        if not self.button_data:
            self.button_data = {}
            for i in range(self.controller.get_numbuttons()):
                self.button_data[i] = False

        if not self.hat_data:
            self.hat_data = {}
            for i in range(self.controller.get_numhats()):
                self.hat_data[i] = (0, 0)
    
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    self.axis_data[event.axis] = round(event.value,2)
                elif event.type == pygame.JOYBUTTONDOWN:
                    self.button_data[event.button] = True
                elif event.type == pygame.JOYBUTTONUP:
                    self.button_data[event.button] = False
                elif event.type == pygame.JOYHATMOTION:
                    self.hat_data[event.hat] = event.value

                # TODO: Toggle reverse
                if (self.button_data[2] == True):
                    print('Triangle, reverse') 
                # Reads R2, to increase throttle 
                if (5 in self.axis_data):
                    if self.axis_data[5] < 0:
                        controls['throttle'] = 0
                    else:
                        controls['throttle'] = self.axis_data[5]
                # Reads L2, to break
                if (4 in self.axis_data):
                    if self.axis_data[4] == -1:
                        controls['brake'] = 0
                    else:
                        controls['brake'] += self.axis_data[4]
                # TODO: implement recording using share button
                if (self.button_data[8] == True):
                    print('Share start recording')

                # Check for the left analag, in the x axis. Steering
                if (0 in self.axis_data and (self.axis_data[0] > 0.0 or self.axis_data[0] < 0.0)):
                    if self.axis_data[0] > 0:
                        if (controls['steering'] < 1.0 and  self.axis_data[0] > 0.20):
                            controls['steering'] = self.axis_data[0]

                    elif self.axis_data[0] < 0:
                        if (controls['steering'] > -1.0 and  self.axis_data[0] < -0.20):
                            controls['steering'] = self.axis_data[0]
                    else:
                        controls['steering'] = 0
                
                # Prints the commands sent to the vehicle
                print(controls)
                # Apply the controls
                vehicle.apply_control(carla.VehicleControl(throttle=controls['throttle'], steer=controls['steering'], brake=controls['brake']))
                # Clears the dict 
                controls = {'throttle':0, 'brake': 0, 'steering':0, 'reverse': 0}

if __name__ == '__main__':

    main()
