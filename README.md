Snake Zero
===

## Summary
This is the classic Snake game, but I have made some rule adjustments and tried training an AI model with `Reinforcement Learning`.
This version of Snake allows for 2-player gameplay, adding a certain level of strategy and competitiveness.
I specifically used `MLX` for training, which also serves as practice for the new framework.


## Start Game
You can modify p1 or p2 to walk automatically by using `p1_auto` or `p2_auto` in games/main.py currently.
```
python games/main.py
```

## Basic Rules 

Rules 1.	Eating a red candy earns 5 points.

<img width="627" alt="Snake Play Sample 1" src="https://github.com/yeyuting0307/snake-zero/assets/35023161/77ca9c8a-860b-44ad-afe3-7787dc25912f">

Rules 2. Eating the opponent’s body earns 5 points and decreases the opponent’s score by 5 points.

<img width="748" alt="Snake Play Sample 2" src="https://github.com/yeyuting0307/snake-zero/assets/35023161/8786f017-fa39-4f4e-b15b-116dba21640d">

Rules 3. Eating your own body causes your body color to lighten. After accumulating three such instances, you lose the game.

<img width="751" alt="Snake Play Sample 3" src="https://github.com/yeyuting0307/snake-zero/assets/35023161/f46c71de-1856-49c9-97bc-85c3469740d8">

Rules 4. In the last 10 seconds, the number of newly spawned red candies increases to 10.

<img width="747" alt="Snake Play Sample 4" src="https://github.com/yeyuting0307/snake-zero/assets/35023161/3ce7e32a-95b1-401f-80c1-c371b717f93e">

Rules 5. The first player to reach 100 points wins.

<img width="748" alt="Snake Play Sample 5" src="https://github.com/yeyuting0307/snake-zero/assets/35023161/7fc8b3e0-e2e0-48fb-870f-dcd3660c2e93">


## TODO
- [x] pygame for snake
- [x] example model of MLX
- [x] episode collector
- [ ] train code example
- [ ] integrate ai model with game play
  
