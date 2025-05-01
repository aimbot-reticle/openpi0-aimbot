import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
import tqdm
import tyro

faulthandler.enable()


class RobotEnv:
    # Jayjun implementation
    def __init__(self, use_reticle: bool = False):
        pass

    def get_observation(self):
        #     {
        #     "left_shoulder_image": left_shoulder_image,
        #     "right_shoulder_image": right_shoulder_image,
        #     "wrist_image": wrist_image,
        #     "state": joint_states, # joint position, gripper position
        # }
        
        ##   Depth images Processing
        # # convert depth to gray rgb image 
        # left_shoulder_depth_image = np.repeat(left_shoulder_depth_image[:, :, np.newaxis], 3, axis=2)
        # right_shoulder_depth_image = np.repeat(right_shoulder_depth_image[:, :, np.newaxis], 3, axis=2)
        # wrist_depth_image = np.repeat(wrist_depth_image[:, :, np.newaxis], 3, axis=2)
        
        # left_shoulder_depth_image = (left_shoulder_depth_image - np.min(left_shoulder_depth_image)) / (min(np.max(left_shoulder_depth_image),3) - np.min(left_shoulder_depth_image))
        # right_shoulder_depth_image = (right_shoulder_depth_image - np.min(right_shoulder_depth_image)) / (min(np.max(right_shoulder_depth_image),3) - np.min(right_shoulder_depth_image))
        # wrist_depth_image = (wrist_depth_image - np.min(wrist_depth_image)) / (min(np.max(wrist_depth_image),3) - np.min(wrist_depth_image))
        
        # left_shoulder_depth_image = (left_shoulder_depth_image * 255).astype(np.uint8)
        # right_shoulder_depth_image = (right_shoulder_depth_image * 255).astype(np.uint8)
        # wrist_depth_image = (wrist_depth_image * 255).astype(np.uint8)
        pass
    
    def step(self, action):
        # action is (8,), 7 joint position + 1 gripper position
        pass
    


CONTROL_FREQUENCY = 15

@dataclasses.dataclass
class Args:
    # Rollout parameters
    max_timesteps: int = 400
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Initialize the Panda environment. Using joint postion action space and gripper position action space.
    env = RobotEnv()
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = env.get_observation()

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/left_shoulder_image": image_tools.resize_with_pad(
                            curr_obs["left_shoulder_image"], 224, 224
                        ),  
                        "observation/right_shoulder_image": image_tools.resize_with_pad(
                            curr_obs["right_shoulder_image"], 224, 224
                        ),
                        "observation/wrist_image": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        
                        # # Depth images Input
                        # "observation/left_shoulder_depth": image_tools.resize_with_pad(
                        #     curr_obs["left_shoulder_depth"], 224, 224
                        # ),
                        # "observation/right_shoulder_depth": image_tools.resize_with_pad(
                        #     curr_obs["right_shoulder_depth"], 224, 224
                        # ),
                        # "observation/wrist_depth": image_tools.resize_with_pad(
                        #     curr_obs["wrist_depth"], 224, 224
                        # ),
                        
                        "observation/state": curr_obs["state"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint position actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1


                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / CONTROL_FREQUENCY:
                    time.sleep(1 / CONTROL_FREQUENCY - elapsed_time)
            
            except KeyboardInterrupt:
                break


        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
            },
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")



if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
