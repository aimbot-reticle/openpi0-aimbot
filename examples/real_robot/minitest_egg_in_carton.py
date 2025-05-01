import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import PIL

np.set_printoptions(precision=5, suppress=True)




left_shoulder_image = np.array(PIL.Image.open("examples/real_robot/reticle_samples_egg/left_shoulder_image_236.png"))
right_shoulder_image = np.array(PIL.Image.open("examples/real_robot/reticle_samples_egg/right_shoulder_image_236.png"))
wrist_image = np.array(PIL.Image.open("examples/real_robot/reticle_samples_egg/wrist_image_236.png"))
instruction = "put the egg inside the egg carton"

state = [-0.00725937,  0.13311845, -0.04766899, -2.08065104,  0.02973011,  2.02222753,  1.13670385,  0.88123362]
state = np.array(state, dtype=np.float32)

request_data = {
    "observation/left_shoulder_image": image_tools.resize_with_pad(left_shoulder_image, 224, 224),  
    "observation/right_shoulder_image": image_tools.resize_with_pad(right_shoulder_image, 224, 224),
    "observation/wrist_image": image_tools.resize_with_pad(wrist_image, 224, 224),
    "observation/state": state,
    "prompt": instruction,
}

policy_client = websocket_client_policy.WebsocketClientPolicy(
    host="localhost",
    port=8001,
)
tstart = time.time()
pred_action_chunk = policy_client.infer(request_data)["actions"]
tend = time.time()
print(f"Time taken: {tend - tstart} seconds")
print(pred_action_chunk[0]) # expected [-0.01511198  0.11611952 -0.05221765 -2.14998979  0.08095009  2.1068709 1.12064988  1.        ]
print(pred_action_chunk.shape)