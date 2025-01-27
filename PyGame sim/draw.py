import pygame
import numpy as np
import pygame_widgets
from pygame_widgets.button import Button
import csv

# Initialize Pygame and set up the screen/fonts
pygame.init()
info = pygame.display.Info()
width = info.current_w-20
height = info.current_h-80
screen = pygame.display.set_mode((width, height))
font = pygame.font.SysFont("Arial", 18)
pygame.display.set_caption('DC motor simulator')
clock = pygame.time.Clock()

# if save button is pressed, save the data to CSV
save_data = False
button = Button(
    screen, width-250, 30, 220, 80, text='Click to save to CSV',
    fontSize=24, margin=20,
    inactiveColour=(255, 255, 255),
    pressedColour=(155, 255, 155), radius=20,
    onClick=lambda: globals().update(save_data=True)
)
# Log button functionality
def save_data_log(data):
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        fieldnames = ["Time", "X", "Y", "Target X", "Target Y", "Heading", "Speed", 
                      "Left Voltage", "Right Voltage","min dist","l"]
        # Write the headers
        writer.writerow(fieldnames)

        # Convert np.arrays to lists to write them as strings in CSV
        rows = zip(
            data['time'], data['x'], data['y'],
            [val[0] for val in data['target']],  # target_x
            [val[1] for val in data['target']],  # target_y
            data['heading'], data['speed'],
            data['left voltage'], data['right voltage'],
            data['Closest object'],data["l"]
        )

        # Write each row of data
        for row in rows:
            writer.writerow(row)

# Define colors
WHITE = (255, 200, 100)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Convert coordinates for Pygame screen (from world coordinates) 
def to_screen_coords(x, y):
    scale = height / 50
    return int(width / 2 + x * scale), int(height / 2 - y * scale)

def to_world_coords(x, y):
    scale = height / 50
    return (x - (width / 2))/scale, (-y + (height / 2)) / scale

# Function to draw the robot
def draw_robot(x, y, phi):
    l = 0.4
    c = np.cos(phi)
    s = np.sin(phi)
    vertices = np.array([
        [x + l * 4 * c, y + l * 4 * s],
        [x - l * s, y + l * c],
        [x + l * s, y - l * c],
        [x + l * 4 * c, y + l * 4 * s]
    ])
    pygame.draw.polygon(screen, GREEN, [to_screen_coords(v[0], v[1]) for v in vertices], 0)

# Main draw function
def draw(data, pgons, target, FPS, running, events, tr):
    global save_data  
    # save data if the save data button has been pressed
    if save_data is True:
        save_data_log(data)
        save_data = False

    x = data['x'][-1]
    y = data['y'][-1]
    heading = data['heading'][-1]

    screen.fill((0,0,40))  # Clear screen

    pygame.draw.lines(screen, RED, False, [to_screen_coords(tr[i][0], tr[i][1]) for i in range(len(tr))], 1)

    # Draw path traveled
    if (len(data['x'])>1):
        pygame.draw.lines(screen, GREEN, False, [to_screen_coords(data['x'][i], data['y'][i]) for i in range(len(data['x']))], 1)

    # Draw obstacles
    for pgon in pgons:
        pygame.draw.polygon(screen, WHITE, [to_screen_coords(*point) for point in pgon.xy])

    # Draw the target point
    pygame.draw.circle(screen, RED, to_screen_coords(target[0], target[1]), 5)  

    # Draw the robot
    draw_robot(x, y, heading)

    # Write the texts
    exit_text = font.render('Space: run/stop', True, (255, 255, 255))
    mouse_text = font.render('Mouse click: select goal', True, (255, 255, 255))
    disturbance_text = font.render('Number keys: sim speed', True, (255, 255, 255))
    screen.blit(exit_text, (10, 70))
    screen.blit(mouse_text, (10, 90))
    screen.blit(disturbance_text, (10, 110))

    if (running):
        running_text = font.render('SIM RUNNING', True, (50, 255, 50))
        # run_time_text = font.render(f'Run time: {round(data['time'][-1])}', True, (255, 255, 255))
        frame_rate_text = font.render(f'Frame rate: {round(clock.get_fps())}', True, (255, 255, 255))
        screen.blit(running_text, (10, 10))
        # screen.blit(run_time_text, (10, 30))
        screen.blit(frame_rate_text, (10, 50))
    else:
        running_text = font.render('SIM STOPPED', True, (255, 50, 50))
        screen.blit(running_text, (10, 10))

    pygame_widgets.update(events)

    # Update display
    pygame.display.flip()
    clock.tick(FPS)  # Adjust to control the animation speed
