# helpers

Turns mocap into clips a policy can actually track.

The catch is that a clip has to obey physics or the reward is impossible to earn — in
the air the body has to fall like a thrown object and can't change its spin. Hand
animation breaks this all the time. These tools spot it and fix it by nudging only
where the body travels and how long it hangs, so the pose and style stay as animated.

```bash
python use.py                    # list clips, with a note on which are physical
python use.py backflip           # make one live
python checkclip.py ../animations/backflip.csv
python build.py dm spinkick      # retarget a DeepMimic clip from clips/
python build.py all              # rebuild everything
```

`use.py` and `checkclip.py` are the two you'll actually use day to day. The rest
(`retarget`, `ballistic`, `angular`, `writecsv`, `dm_common`, `dm_native`) are the
pipeline underneath, and `clips/` holds the raw DeepMimic files.
