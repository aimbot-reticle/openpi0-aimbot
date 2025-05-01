import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import PIL

np.set_printoptions(precision=5, suppress=True)




left_shoulder_image = np.array(PIL.Image.open("examples/real_robot/reticle_samples_drawer/left_shoulder_image_77.png"))
right_shoulder_image = np.array(PIL.Image.open("examples/real_robot/reticle_samples_drawer/right_shoulder_image_77.png"))
wrist_image = np.array(PIL.Image.open("examples/real_robot/reticle_samples_drawer/wrist_image_77.png"))
instruction = "put the tennis ball inside the drawer"

state = [-0.37561423,  0.72214317, -0.1908751,  -1.69940007,  1.05602407,  1.32654977, -0.27157781,  0.99659354]
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
print(pred_action_chunk[0]) #  expect [-0.40962731  0.75692577 -0.19266555 -1.62441164  1.0104553   1.25835662 -0.3051943   1.        ]
# print(pred_action_chunk.shape)