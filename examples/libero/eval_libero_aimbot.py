import collections
from copy import deepcopy
import dataclasses
import json
import logging
import math
import os
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from crosshair.reticle_builder import ReticleBuilder
from crosshair.config import CONFIG_DICT
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map

from scipy.spatial.transform import Rotation as R

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

def is_open(gripper_qpos):
    if abs(gripper_qpos[0]) > 0.035 and abs(gripper_qpos[1]) > 0.035:
        return True
    return False


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "localhost"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    ) # can be multiple task suites separated by comma
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_path: str = "runs/evaluation"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    
    model_name: str = "pi0_fast_libero"             # Model name
    task_start_id: int = 0                          # Start task ID
    task_end_id: int = 10                            # End task ID
    
    save_video_num: int = 100  # Number of videos to save per task
    
    use_reticle: bool = True  # Use reticle in the environment
    reticle_config_key: str = "large_crosshair_dynamic_default_color"  # Reticle configuration key    


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    print("Input Task Suite:", args.task_suite_name.split(","))
    if args.task_suite_name == "all":
        task_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    else:
        task_suites = args.task_suite_name.split(",")
        
    print(f"connected to {args.host}:{args.port}")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    
    for task_suite_name in tqdm.tqdm(task_suites):

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        logging.info(f"Task suite: {task_suite_name}")

        pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

        save_dir = os.path.join(args.save_path, task_suite_name, args.model_name)
        os.makedirs(save_dir, exist_ok=True)
        save_dir_video = os.path.join(save_dir, "video")
        os.makedirs(save_dir_video, exist_ok=True)
        
        
        if task_suite_name == "libero_spatial":
            max_steps = 250  # longest training demo has 193 steps
        elif task_suite_name == "libero_object":
            max_steps = 300  # longest training demo has 254 steps
        elif task_suite_name == "libero_goal":
            max_steps = 350  # longest training demo has 270 steps
        elif task_suite_name == "libero_10":
            max_steps = 560  # longest training demo has 505 steps
        elif task_suite_name == "libero_90":
            max_steps = 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task suite: {task_suite_name}")

            
        print(f"Task suite: {task_suite_name} has total task number: {num_tasks_in_suite}, run on task from {args.task_start_id} to {args.task_end_id}")
        
        if args.use_reticle:
            print(f"Using reticle with configuration key: {args.reticle_config_key}")
            config = CONFIG_DICT[args.reticle_config_key]
            shooting_line_config = config["shooting_line"]
            scope_reticle_config = config["scope_reticle"]
            
            MAX_EE_TABLE_DIST = 0.4
            FIXCAM_TOLERANCE = 18
            WSTCAM_TOLERANCE = 12
                    
            if hasattr(scope_reticle_config, "line_length_cfg"):
                scope_reticle_config.line_length_cfg.maxdist = MAX_EE_TABLE_DIST
            
            if hasattr(scope_reticle_config, "circle_radius_cfg"):
                scope_reticle_config.circle_radius_cfg.maxdist = MAX_EE_TABLE_DIST
            
            reticle_builder = ReticleBuilder(
                shooting_line_config=shooting_line_config,
                scope_reticle_config=scope_reticle_config,
            )


        
        # Start evaluation
        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(args.task_start_id, args.task_end_id)):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, use_depth=args.use_reticle)

            results = {"task_id": task_id, "task_description":task_description, "data": []} 

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description}")

                # Reset environment
                env.reset()
                action_plan = collections.deque()
                success = False

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []

                logging.info(f"Starting episode {task_episodes+1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue
                        
                        if args.use_reticle:        
                            gripper_pos = deepcopy(obs["robot0_eef_pos"])
                            gripper_quat = deepcopy(obs["robot0_eef_quat"])
                                
                            front_depth = np.flipud(obs["agentview_depth"]).squeeze()
                            front_depth_real = get_real_depth_map(env.sim, front_depth)
                            
                            agentview_rgb = reticle_builder.render_on_fix_camera(
                                camera_rgb=np.flipud(obs["agentview_image"]).astype(np.uint8),
                                camera_depth=front_depth_real,
                                camera_extrinsics=np.linalg.inv(get_camera_extrinsic_matrix(env.sim, "agentview")),
                                camera_intrinsics=get_camera_intrinsic_matrix(env.sim, "agentview", LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION),
                                gripper_pos=gripper_pos,
                                gripper_quat=gripper_quat,
                                gripper_open=is_open(obs["robot0_gripper_qpos"]),
                                image_height=LIBERO_ENV_RESOLUTION,
                                image_width=LIBERO_ENV_RESOLUTION,
                                tolerance=FIXCAM_TOLERANCE,
                            )
                            
                            wrist_depth = np.flipud(obs["robot0_eye_in_hand_depth"]).squeeze()
                            wrist_depth_real = get_real_depth_map(env.sim, wrist_depth)
                            
                            
                            robot0_eye_in_hand_rgb = reticle_builder.render_on_wst_camera(
                                wrist_camera_rgb=np.flipud(obs["robot0_eye_in_hand_image"]).astype(np.uint8),
                                wrist_camera_depth=wrist_depth_real,
                                wrist_camera_extrinsics=np.linalg.inv(get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand")),
                                wrist_camera_intrinsics= get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION),
                                gripper_pos=gripper_pos,
                                gripper_quat=gripper_quat,
                                gripper_open=is_open(obs["robot0_gripper_qpos"]),
                                image_height=LIBERO_ENV_RESOLUTION,
                                image_width=LIBERO_ENV_RESOLUTION,
                                tolerance=WSTCAM_TOLERANCE,
                            )
                            
                            img = np.ascontiguousarray(agentview_rgb)
                            wrist_img = np.ascontiguousarray(robot0_eye_in_hand_rgb)
                            
                        else:
                            # Get preprocessed image
                            # IMPORTANT: rotate 180 degrees to match train preprocessing
                            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )

                        # Save preprocessed image for replay video
                        if args.use_reticle:
                            replay_images.append(np.concatenate((img, wrist_img), axis=1))
                        else:
                            replay_images.append(np.concatenate((img[:, ::-1], wrist_img[:, ::-1]), axis=1))

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(task_description),
                            }

                            # Query model to get action
                            action_chunk = client.infer(element)["actions"]
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            success = True
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.error(f"Caught exception: {e}")
                        break

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                
                if episode_idx < args.save_video_num:
                    imageio.mimwrite(
                        os.path.join(save_dir, "video", f"task{task_id}-seed{args.seed}-{task_segment}_ep{episode_idx}_{suffix}.mp4"),
                        [np.asarray(x) for x in replay_images],
                        fps=30,
                    )
                    
                results["data"].append({"episode": episode_idx, "success": success})

                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
            json_name = f"task{task_id}-seed{args.seed}-{processed_task_description}.json"
            with open(os.path.join(save_dir, json_name), "w") as f:
                json.dump(results, f, indent=2)
                
            # Log final results
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
        logging.info(f"Total episodes: {total_episodes}")
    env.close()


def _get_libero_env(task, resolution, seed, use_depth=False):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    if use_depth: env_args["camera_depths"] = True
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
