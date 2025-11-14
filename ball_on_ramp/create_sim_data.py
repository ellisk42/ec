import pymunk
import pymunk.pygame_util
import pygame
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import math
import yaml
import os
from datetime import datetime
import argparse
import pickle

def save_video(frames: list, path: str, fps=60):
    """
    frames: list of np.darray objects
    path: output mp4 filename
    """
    imageio.mimsave(path, frames, fps=fps, codec="libx264")
    print(f"Save file at {path}")

def add_box_on_ramp(space: pymunk.Space, s1: pymunk.Segment, 
        s2: pymunk.Segment, t: float, size=(50, 50), 
        friction=0.9, elasticity=0.05):
    """ Add box on a ramp
    """
    s1_a, s1_b = s1.a, s1.b
    s2_a, s2_b = s2.a, s2.b
    x = s1_a[0] + t * (s2_b[0] - s1_a[0])
    if x <= s1_b[0]:
        angle = math.atan2(s1_b[1] - s1_a[1], s1_b[0] - s1_a[0])
        y = s1_a[1] + t * (s1_b[1] - s1_a[1])
    else:
        angle = 0 # at the ground
        y = 0 + size[1] / 2 # box size y-offset

    box_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    box_body.position = x, y
    box_body.angle = angle

    box_shape = pymunk.Poly.create_box(box_body, size)
    box_shape.friction = friction
    box_shape.elasticity = elasticity

    space.add(box_body, box_shape)
    return (box_body, box_shape)

def render(space: pymunk.Space,
           size=(640, 480),
           xlim=(0, 640),
           ylim=(-10, 480)) -> np.ndarray:
    """
    Render the current Pymunk space to an RGB numpy array.
    Works on a headless machine because it uses matplotlib's Agg backend.
    """
    import matplotlib
    matplotlib.use("Agg")  # force non-interactive backend
    import matplotlib.pyplot as plt
    from pymunk.matplotlib_util import DrawOptions

    # Create off-screen figure
    dpi = 100
    fig_w = size[0] / dpi
    fig_h = size[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    # Optional: invert y so it matches typical screen coordinates
    #ax.invert_yaxis()
    ax.axis("off")
    plt.subplots_adjust(
        left=0, right=1, bottom=0, top=1,
        wspace=0, hspace=0
    )
    fig.patch.set_alpha(0.0)      # Remove fig background (optional)
    ax.margins(0)

    options = DrawOptions(ax)
    space.debug_draw(options)

    # Draw and grab the pixel buffer
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape((h, w, 3))

    plt.close(fig)
    return img

def create_env(config_filepath, args):
    # Load config
    with open(config_filepath, "r") as file:
        config = yaml.safe_load(file)

    space = pymunk.Space()
    space.gravity = 0, -981 # y-direction
    r_segment = pymunk.Segment(space.static_body, 
            config["ramp_segment"]["a"], 
            config["ramp_segment"]["b"], 
            config["ramp_segment"]["radius"]
            )

    # Set up segment
    r_segment.elasticity = config["ramp_segment"]["elasticity"]
    r_segment.friction = config["ramp_segment"]["friction"]

    h_segment = pymunk.Segment(space.static_body, 
        config["horizontal_segment"]["a"], 
        config["horizontal_segment"]["b"], 
        config["horizontal_segment"]["radius"]
        )

    # Set up segment
    h_segment.elasticity = config["horizontal_segment"]["elasticity"]
    h_segment.friction = config["horizontal_segment"]["friction"]

    # Set up circle
    body = pymunk.Body(mass=config["ball"]["mass"], 
                       moment=config["ball"]["moment"])
    body.position = config["ball"]["position"]
    circle = pymunk.Circle(body, radius=config["ball"]["radius"])
    circle.elasticity = config["ball"]["elasticity"]
    circle.friction = config["ball"]["friction"]
    space.add(body, circle, r_segment, h_segment)
    box_body, box_shape = add_box_on_ramp(space, r_segment, h_segment, args.box_position)
    return space, body, box_body

def simulate_env(space, body, args):
    dt = args.dt
    simulation_time = args.simulation_time
    num_steps = int(simulation_time / dt)
    frames = []
    ball_position = []
    print("Starting Pymunk simulation")

    for i in range(num_steps):
        space.step(dt)
        frames.append(render(space))
        ball_position.append((body.position[0], body.position[0]))
    return ball_position, frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for ball on ramp')
    parser.add_argument("--box_position", help="The x ratio position of the box obstacle", type=float, default=0.5)
    parser.add_argument("--simulation_time", help="Simulation time for pymunk (in second)", type=float, default=2.0)
    parser.add_argument("--config_filepath", help="Path to config file", type=str, default="config.yaml")
    parser.add_argument("--dt", help="dt for simulation", type=float, default=0.02)
    parser.add_argument("--save_folder", help="Save folder for video and results", type=str, default="results")
    parser.add_argument("--save_video", help="Whether to save video", type=bool, default=True)

    args = parser.parse_args()
    space, body, box_body = create_env(args.config_filepath, args)
    ball_positions, frames = simulate_env(space, body, args)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    save_folder = args.save_folder
    if args.save_video:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        save_filename = os.path.join(save_folder, f"ball_on_ramp_{timestamp}.mp4")
        save_video(frames, save_filename, fps=int(1/args.dt))
    
    res = {
        "ball_position": ball_positions,
        "box_position": (box_body.position[0], box_body.position[1])
    }
    res_filename = os.path.join(save_folder, f"ball_on_ramp_{timestamp}.pkl")
    with open(res_filename, "wb") as file:
        pickle.dump(res, file)
    print(f"Save result to {res_filename}")