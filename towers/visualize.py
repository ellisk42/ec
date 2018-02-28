import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import sys
from time import time


# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

# --- pygame setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Simple pygame example')
clock = pygame.time.Clock()

# --- pybox2d world setup ---
from tower_common import *

# Create the world
world = TowerWorld()

plan = eval(sys.argv[1]) #[(10,False),(11,False),(12,False),(11,True)]
perturbation = float(sys.argv[2])

originalPlan = plan

def my_draw_polygon(polygon, body, fixture):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, body.userData["color"], vertices)
polygonShape.draw = my_draw_polygon

height, successes = TowerWorld().sampleStability(plan, perturbation, N = 100)
mass = sum(w*h for _,w,h in plan)
print "This tower has height %f, mass %f, and succeeds %d/100 of the time with a perturbation of %f"%(height, mass, int(successes*100),perturbation)

# --- main game loop ---

state = "BUILDING"
timer = None
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False

    screen.fill((0, 0, 0, 0))
    # Draw the world
    for body in [world.ground_body] + world.blocks:
        for fixture in body.fixtures:
            fixture.shape.draw(body, fixture)

    # Make Box2D simulate the physics of our world for one step.
    world.step(TIME_STEP)

    if state == "BUILDING":
        if world.unmoving():
            if plan != []:
                world.placeBlock(*plan[0])
                plan = plan[1:]
                timer = time() + 2
            elif time() > timer:
                state = "DESTROYING"
    elif state == "DESTROYING":
        world.impartImpulses(perturbation)
        state = "SPECTATING"
        timer = time() + 2
    elif state == "SPECTATING":
        if world.unmoving() and time() > timer:
            plan = originalPlan
            world.clearWorld()
            state = "BUILDING"

    # Flip the screen and try to keep at the target FPS
    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()
