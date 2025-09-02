from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler
import cv2 
import numpy as np

# Initialize handler
handler = DigitHandler()
digits = DigitHandler.list_digits()
print("Available DIGIT devices:", digits)
# # List all connected DIGIT sensors
# serials = handler.get_serials()
# print("Available DIGIT serial numbers:", serials)

# if len(serials) == 0:
#     raise Exception("No DIGIT sensor found. Check USB connection.")

# Connect to first found DIGIT

# digit = Digit('D20157')
digit = Digit('D20188')
#digit = Digit('D20049')
# digit = Digit('D20095')
digit.connect()

digit.set_intensity(Digit.LIGHTING_MAX)
qvga_res = Digit.STREAMS["QVGA"]
digit.set_resolution(qvga_res)
fps_15 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
digit.set_fps(fps_15)
print(digit.info())

print("Connected to DIGIT")

while True:
    # Get a frame from the sensor
    frame = digit.get_frame()
    if frame is not None:
        # Display the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("DIGIT Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("No frame received")

# digit.show_view()

# saved_frames = []

# num_frames = 1000

# for i in range(100):
#     _ = digit.get_frame()

# ref_frame = digit.get_frame()
# while True:
#     # saved_frames.append(frame)  # <-- this was missing!
#     if ref_frame is not None:
#         diff = digit.get_diff(ref_frame)
#         cv2.imshow("Difference", diff)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

# saved_frames = np.array(saved_frames)  # (num_frames, H, W, 3)

# print(f"Saved {len(saved_frames)} frames with shape {saved_frames.shape}")

# np.save('digit_frames.npy', saved_frames)
# print("Saved to digit_frames.npy")

