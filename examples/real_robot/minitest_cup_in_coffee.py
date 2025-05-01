import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import PIL

np.set_printoptions(precision=5, suppress=True)



left_shoulder_image = np.array(PIL.Image.open("examples/real_robot/reticle_samples_coffee/left_shoulder_image_224.png"))
right_shoulder_image = np.array(PIL.Image.open("examples/real_robot/reticle_samples_coffee/right_shoulder_image_224.png"))
wrist_image = np.array(PIL.Image.open("examples/real_robot/reticle_samples_coffee/wrist_image_224.png"))
instruction = "put the cup on the coffee machine"

state = [-6.36629835e-02, -3.31839137e-02, -1.23551235e-01, -2.74614859e+00, 2.19897553e-01, 2.61483908e+00, -1.66412270e+00, 2.32637301e-03]
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
print(pred_action_chunk[0]) # expected [-0.07001272 -0.02224026 -0.12195967 -2.75122962  0.25098871  2.64748625 -1.74296515  1.        ]
print(pred_action_chunk.shape)