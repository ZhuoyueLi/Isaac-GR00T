import cv2

def map_task(key) -> str:
    """
    Map the task key to the task name

    Parameters:
    - key: task key
    """

    tasks = {
        'banana_from_right_stove_to_oven_tray': 'Move banana from right stove to oven tray',
        'pot_from_sink_to_left_stove': 'Move pot from sink to left stove',
        'open_microwave': 'Open the microwave',
        'pot_from_left_to_right_stove': 'Move pot from left to right stove',
        'banana_from_tray_to_right_stove': 'Move banana from tray to right stove',
        'pull_oven_tray': 'Pull the oven tray',
        'banana_from_right_stove_to_sink': 'Move banana from right stove to sink',
        'close_oven': 'Close the oven',
        'push_toaster_lever': 'Push down the toaster lever',
        'pot_from_right_to_left_stove': 'Move pot from right to left stove',
        'pot_from_right_stove_to_sink': 'Move pot from right stove to sink',
        'open_ice': 'Open the ice box',
        'open_oven': 'Open the oven',
        'push_oven_tray': 'Push the oven tray',
        'close_microwave': 'Close the microwave',
        'banana_from_sink_to_right_stove': 'Move banana from sink to right stove',
        'pot_from_left_stove_to_sink': 'Move pot from left stove to sink',
        'close_ice': 'Close the ice box',
        'pickup_toast_and_put_to_sink': 'Pick up toast and put it in the sink',
        'pot_from_sink_to_right_stove': 'Move pot from sink to right stove',
    }

    if key in tasks:
        return tasks[key]
    else:
        return key

def resize_and_crop(
        cam_img_array, position, des_width: int = 500, des_height: int = 500
    ):
    """
    Resizes and crops an image to a square shape, centered or aligned to the left or right side.

    Arguments:
    - image_array: a NumPy array representing the image to be resized and cropped.
    - position: a string specifying the position of the crop within the image.
                Valid values are 'left', 'right', 'center', and 'center_right'.
    - size: a tuple specifying the desired size of the output image after resizing. The default value is (500, 500).

    Returns:
    - image_resized: a NumPy array representing the resized and cropped image.

    The method first calculates the size of the largest square that fits inside the original image,
    then crops the image to a square centered or aligned to the specified position.
    Finally, the cropped image is resized to the specified output size using the OpenCV library.

    Note that this method modifies the input image array in-place,
    so make a copy of the original image if you need to keep it intact.
    """


    # cam_image = np.load(cam_img_path)

    size = (des_width, des_height)
    # O Get the height and width of the image
    height, width, _ = cam_img_array.shape
    # Calculate the size of the square
    square_size = min(width, height)
    # Calculate the left, top, right, and bottom coordinates of the square
    if position == "left":
        # Crop the left side of the image
        left = 0
        top = 0
        right = square_size
        bottom = square_size
    elif position == "right":
        # Crop the right side of the image
        left = width - square_size
        top = 0
        right = width
        bottom = square_size
    elif position == "center":
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size 
        bottom = top + square_size 
    elif position == "top_center_new_lab":
        left = 550
        top = 0
        right = 1700
        bottom = top + square_size 
    elif position == "front_center_new_lab":
        left = 0
        top = 0
        right = 1700
        bottom = top + square_size
    elif position == "center_left":
        left = (width - square_size) // 2 - 30
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
    elif position == "center_far_left":
        left = (width - square_size) // 2 - 100
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
    elif position == "center_right":
        left = (width - square_size) // 2 + 100
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
    else:
        raise ValueError("Invalid position. Use 'left' or 'right' or 'center'.")

    # Crop the image to create a square
    image_cropped = cam_img_array[top:bottom, left:right]

    # Resize the image to 500x500
    image_resized = cv2.resize(image_cropped, size)

    # Downscale image to 250x250
    # image_resized = cv2.resize(image_resized, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)

    return image_resized

def kitchen_dataset_obs_transforms(obs, size=(224,224)):
        """These transforms are embedded within the dataset, so EVERY policy,
        regardless of their own transforms needs to undergo these beforehand"""

        primary_image = obs["top_cam"]  # [H, W, 3]
        secondary_image = obs["side_cam"]  # [H, W, 3]
        primary_image = resize_and_crop(primary_image, "top_center_new_lab", size[0], size[1])
        secondary_image = resize_and_crop(secondary_image, "front_center_new_lab", size[0], size[1])

        obs['primary_camera'] = primary_image
        obs['secondary_camera'] = secondary_image

        return obs

def kitchen_dataset_obs_transforms_apr25(obs, size=(256,256)):
        """These transforms are embedded within the dataset, so EVERY policy,
        regardless of their own transforms needs to undergo these beforehand"""

        side_top = 0
        side_left = 500
        side_width = 1373
        side_height = 1080
        side_bottom = side_top + side_height
        side_right = side_left + side_width

        top_left = 530
        top_top = 112
        top_width = 1072
        top_height = 968
        top_bottom = top_top + top_height
        top_right = top_left + top_width

        primary_image = obs["top_cam"]  # [H, W, 3]
        secondary_image = obs["side_cam"]  # [H, W, 3]

        # primary_image = primary_image[top_top:top_bottom, top_left:top_right]
        # secondary_image = secondary_image[side_top:side_bottom, side_left:side_right]
        primary_image = cv2.resize(primary_image, (size[0], size[1]))
        secondary_image = cv2.resize(secondary_image, (size[0], size[1]))
        # obs['primary_camera'] = primary_image
        # obs['secondary_camera'] = secondary_image

        print(obs.keys())
        # save and show the img
        # cv2.imwrite("/home/irl-admin/LZY/Isaac-GR00T/real_kitchen_evaluation/images/first/first.jpg", primary_image)
        # cv2.imwrite("/home/irl-admin/LZY/Isaac-GR00T/real_kitchen_evaluation/images/second/second.jpg", secondary_image)

        gr00t_obs = {
            "video.ego_view": primary_image[None, :],
            "video.second_view": secondary_image[None, :],
            "state.gripper_state": obs["gripper_width"][None, :],
            "state.joint_pos": obs["joint_pos"][None, :],

            "annotation.human.action.task_description": ["pick up the green pepper from the table and put it in a bowl"],
            # "annotation.human.action.task_description": ["pick up the green pepper and put it in the hand"],
            # "annotation.human.action.task_description": ["pick up the eraser from bowl and put it in the hand"],
            # "annotation.human.action.task_description": ["pick up the cups and stack them together"],
            # "annotation.human.action.task_description": ["pick up the eraser and then remove black marks"],
            # "annotation.human.action.task_description": ["stack the blue cup, then the yellow cup,then the green cup at the specific location"],


            # "annotation.human.action.task_description": ["pick up the eraser and put it on the hand"],
            # "annotation.human.action.task_description": ["pick up the eraser from the table and put it in a bowl"],
            # "annotation.human.action.task_description": ["pick up the green pepper and put it on the hand"],

            # "annotation.human.action.task_description": ["pick up the eraser from the table and put it in a bowl, pick up the eraser and then remove black marks"],
            
            # "annotation.human.action.task_description": ["pick up the eraser and then remove black marks,then pick up the eraser and put it on the hand"],
            # "annotation.human.action.task_description": ["at first pick up the green pepper from the table and put it in a bowl,then pick up the eraser from the table and put it in a bowl"],
            # "annotation.human.action.task_description": ["pick up the eraser and then remove black marks,then put it in a bowl"],



            
        }
        shape = gr00t_obs["video.ego_view"].shape
        img_type =  gr00t_obs["video.ego_view"].dtype
        print(f"shape of image:{shape}\n type of image:{img_type}")
        return gr00t_obs

if __name__ == "__main__":
    t = [
        "banana_from_right_stove_to_oven_tray",
        "banana_from_right_stove_to_sink",
        "banana_from_sink_to_right_stove",
        "banana_from_tray_to_right_stove",
        "close_ice",
        "close_microwave",
        "close_oven",
        "open_ice",
        "open_microwave",
        "open_oven",
        "pickup_toast_and_put_to_sink",
        "pot_from_left_stove_to_sink",
        "pot_from_left_to_right_stove",
        "pot_from_right_stove_to_sink",
        "pot_from_right_to_left_stove",
        "pot_from_sink_to_left_stove",
        "pot_from_sink_to_right_stove",
        "pull_oven_tray",
        "push_oven_tray",
        "push_toaster_lever",
    ]

    tt = ([map_task(i) for i in t])

    for i in tt:
        print(i)