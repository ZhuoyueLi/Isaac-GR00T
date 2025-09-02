## 1B model 100k

- 'pot_from_right_stove_to_sink'           | 1
- 'pot_from_sink_to_right_stove'           | 0
- 'pot_from_left_stove_to_sink'            | 1
- 'pot_from_sink_to_left_stove'            | 0

- 'pot_from_left_to_right_stove'           | 1
- 'pot_from_right_to_left_stove'           | 0

- 'banana_from_right_stove_to_sink'        | 1
- 'banana_from_sink_to_right_stove'        | 0
- 'open_microwave'                         | 1
- 'close_microwave'                        | 1
- 'open_oven'                              | 1
- 'close_oven'                             | 1
- 'open_ice'                               | 0
- 'close_ice'                              | 0

- 'push_toaster_lever'                     | 1
- 'pickup_toast_and_put_to_sink'           | 1

- 'pull_oven_tray'                         | 0
- 'push_oven_tray'                         | 0
- 'banana_from_right_stove_to_oven_tray'   | 1
- 'banana_from_tray_to_right_stove'        | 0

# zero shot finetuned, single image

- 'pot_from_right_stove_to_sink'           | 1
- 'pot_from_sink_to_right_stove'           | 0
- 'pot_from_left_stove_to_sink'            | 1
- 'pot_from_sink_to_left_stove'            | 0

- 'pot_from_left_to_right_stove'           | 0
- 'pot_from_right_to_left_stove'           | 0

- 'banana_from_right_stove_to_sink'        | 1
- 'banana_from_sink_to_right_stove'        | 0
- 'open_microwave'                         | 1
- 'close_microwave'                        | 1
- 'open_oven'                              | 1
- 'close_oven'                             | 1
- 'open_ice'                               | 1
- 'close_ice'                              | 1

- 'push_toaster_lever'                     | 1
- 'pickup_toast_and_put_to_sink'           | 0

- 'pull_oven_tray'                         | 1
- 'push_oven_tray'                         | 1
- 'banana_from_right_stove_to_oven_tray'   | 0
- 'banana_from_tray_to_right_stove'        | 0

# zero shot finetuned, two images

- 'pot_from_right_stove_to_sink'           | 1
- 'pot_from_sink_to_right_stove'           | 0
- 'pot_from_left_stove_to_sink'            | 0
- 'pot_from_sink_to_left_stove'            | 0

- 'pot_from_left_to_right_stove'           | 0
- 'pot_from_right_to_left_stove'           | 0

- 'banana_from_right_stove_to_sink'        | 0
- 'banana_from_sink_to_right_stove'        | 0
- 'open_microwave'                         | 0
- 'close_microwave'                        | 0
- 'open_oven'                              | 0
- 'close_oven'                             | 0
- 'open_ice'                               | 0
- 'close_ice'                              | 0

- 'push_toaster_lever'                     | 0
- 'pickup_toast_and_put_to_sink'           | 0

- 'pull_oven_tray'                         | 0
- 'push_oven_tray'                         | 0
- 'banana_from_right_stove_to_oven_tray'   | 0
- 'banana_from_tray_to_right_stove'        | 0


## segmented

# finetuned ver of zero shpt

- agent: 
-- finetuned ver of zero shpt
train_run_dir: /media/irl-admin/93a784d0-a1be-419e-99bd-9b2cd9df02dc/flower_data/models/horeka_trains/08-01-35
checkpoint: "checkpoint_60000"
single_image: false

ensemble_strategy: null
pred_action_horizon: 20
multistep: 15

- notes:
- - pot_from_sink_to_right_stove: mostly failed because it went into a loop just before dropping the pot off / or couldn't get the pot out of the sink wall
- - pot_from_sink_to_left_stove: couldn't really pick the pot up at all for some reason
- - pot_from_right_to_left_stove: tries to drop into the sink
- - push_toaster_lever: pushes ~1cm left of the lever, so misses the actual lever. probably happens because we use p99-p01 normalization because the toaster is right at the edge of the workspace, other times it doesn't push hard enough, one time it ripped the toaster
- - pickup_toast_and_put_to_sink: sometimes rips the toaster
- - push_oven_tray: sometimes crashes into the kitchen

Task Success Rates:
banana_from_right_stove_to_oven_tray: 40.00% (2 success, 3 failure)
pot_from_sink_to_left_stove: 0.00% (0 success, 5 failure)
open_microwave: 100.00% (5 success, 0 failure)
pot_from_left_to_right_stove: 80.00% (4 success, 1 failure)
banana_from_tray_to_right_stove: 0.00% (0 success, 5 failure)
pull_oven_tray: 100.00% (5 success, 0 failure)
banana_from_right_stove_to_sink: 100.00% (5 success, 0 failure)
close_oven: 80.00% (4 success, 1 failure)
push_toaster_lever: 0.00% (0 success, 5 failure)
pot_from_right_to_left_stove: 40.00% (2 success, 3 failure)
pot_from_right_stove_to_sink: 80.00% (4 success, 1 failure)
open_ice: 100.00% (5 success, 0 failure)
open_oven: 60.00% (3 success, 2 failure)
push_oven_tray: 20.00% (1 success, 4 failure)
close_microwave: 100.00% (5 success, 0 failure)
banana_from_sink_to_right_stove: 60.00% (3 success, 2 failure)
pot_from_left_stove_to_sink: 100.00% (5 success, 0 failure)
close_ice: 100.00% (5 success, 0 failure)
pickup_toast_and_put_to_sink: 40.00% (2 success, 3 failure)
pot_from_sink_to_right_stove: 20.00% (1 success, 4 failure)