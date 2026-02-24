# Configuration

All runtime parameters are managed with [Hydra](https://hydra.cc/) and stored under `config/`.

| File | Purpose |
|------|---------|
| `config/collect.yaml` | Data collection settings (cameras, GELLO, NUC, episode length) |
| `config/eval.yaml` | Policy evaluation settings (observation cameras, policy server) |
| `config/postprocess.yaml` | Postprocessing and FoundationStereo settings |
| `config/task/pusht.yaml` | Push-T task (2-DOF, XY only) |
| `config/task/pickandplace_mug.yaml` | Pick-and-place task (6-DOF + gripper) |
| `config/policy/default.yaml` | Default policy parameters (action horizon, observation keys) |

Override any parameter from the command line using Hydra syntax:

```bash
python -m client.collect task=pickandplace_mug max_episode_timesteps=500
```
