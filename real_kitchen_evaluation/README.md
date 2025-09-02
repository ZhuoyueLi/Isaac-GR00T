## Common evaluation on the real kitchen environment

Models and environments are connected via their respective clients.

Currently we have a a client for the (joint-space-)`real-robot` environments from the latest real-robot repository, and the legacy (joint-space) RealRobotEnvironment from `evaluating_real_world` repository.

For clients we have Octo, Crossformer, OpenVLA, FlowerVLA and MoDE.

### Usage

Start the respective servers of the models and the environments, then run `kitchen_eval.py`. You can configure the tasks and the connection information in `config/kitchen_eval.yaml`.