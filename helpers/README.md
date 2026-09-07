# helpers

Turns mocap into reference clips a policy can actually track.

A clip has to obey physics or the imitation reward is unreachable: airborne, the centre
of mass must follow a parabola under gravity and angular momentum must stay constant.
Hand animation breaks this routinely. These tools detect it and correct it, changing
only root translation and flight timing — joint angles, style and limb timing survive.

```bash
python use.py                    # list clips with a physics report
python use.py backflip           # make one live for the sim and training
python checkclip.py ../animations/backflip.csv
python build.py dm spinkick      # retarget a DeepMimic clip from clips/
python build.py all              # rebuild the standard set
```

| file | role |
|---|---|
| `dm_common.py` | skeleton definition, quaternion and FK helpers |
| `dm_native.py` | reads DeepMimic clip files |
| `retarget.py` | DeepMimic humanoid3d → mma joint rotations |
| `ballistic.py` | flight retiming and parabolic COM correction |
| `angular.py` | angular momentum conservation during flight |
| `checkclip.py` | the gate: implied gravity, drift, joint speed |
| `writecsv.py` | writes the 150-column CSV and the training npz |
| `build.py` | ties the above together |
| `use.py` | selects the live clip |
| `bake_npz.py` | CSV → npz only, when the clip needs no correction |

`clips/` holds the raw DeepMimic source files.
