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
import csv
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn

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
        self.recording = False

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
        

    def process_img(self, data, display_surface, camaraManager, show_image=True):
        i = np.array(data.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        frame = i3

        if show_image:
            camaraManager.update_frame(frame)
            camaraManager.render(display_surface)

    def get_frame(self):
        return self.frame

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
        camera.listen(lambda data: image.process_img(data, display_surface, image))

        # Controls the car using a neural network
        control(vehicle, image)

    finally:
        done = True

        print('destroying actors')
        simulatorManager.clear_actors()
        print('done.')


'''Load the model weights'''
def create_model(device):
    pretrained_weights = torch.load('C:\\Users\\cjrs2\\OneDrive\\Escritorio\\Ml\\ImitationLearningCarla\\driving_10.weights')
    model = models.resnet50(pretrained=False)
    # Steering, Throttle, Brake, reverse
    model.fc = nn.Linear(2048, 3)
    model.load_state_dict(pretrained_weights )
    model.to(device)

    return model

''' Loop that makes inference and send commands to carla'''
def control(vehicle, image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(device)
    
    while True:
        # Listen to pygame event otherwise queue gets fill up and freezes
        pygame.event.get()
        # Gets the image, and converts it from H X W X C to C X H X W
        current_image = image.get_frame()
        current_image = current_image.transpose((2, 0, 1))
        current_image = torch.from_numpy(current_image)
        # Adds 'batch_size' dimension
        current_image = current_image[None, :, :, :,]
        # Prediction, returns [[throttle, steering, brake]]
        prediction = model(current_image.float().to(device))
        # Move to cpu and convert from tensor to np
        prediction = prediction.cpu().detach().numpy()
        # The first dimension is always the one
        prediction = prediction[0]
        # Build the control base on the prediction
        controls = {'throttle':prediction[0], 'steering':prediction[1], 'brake': prediction[2]}
        # Temporaly we only care about hard breaking
        if controls['brake'] > 0.4 or controls['brake'] < 0:
                controls['brake'] = 0
        print(controls)
        # Apply the control
        vehicle.apply_control(carla.VehicleControl(throttle=float(controls['throttle']), steer=float(controls['steering']), brake=float(controls['brake'])))

if __name__ == '__main__':

    main()
