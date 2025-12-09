# NFL Player Trajectory Prediction

Systematic baseline development for the [2026 NFL Big Data Bowl](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) trajectory prediction challenge.

## Problem Statement

Predict where NFL players will move during passing plays. Given tracking data up to the moment a quarterback releases the ball, forecast player positions while the ball is in the air (~2 seconds, 20 frames at 10 Hz).

**Challenge:** Players don't move in straight lines. Receivers adjust routes toward catch points while defenders react to offensive movements. Accurate prediction requires capturing both physics (momentum) and intent (player objectives).

## Approach

### 1. Feature Engineering

**18 features across 5 categories:**

- **Position & Motion:** Current x/y coordinates, speed, acceleration, velocity components
- **Temporal Changes:** 5-frame velocity changes capturing acceleration patterns
- **Physics Projections:** Constant-velocity baseline predictions
- **Spatial Context:** Distance to ball landing location, ball coordinates
- **Player Roles:** Receiver vs defender indicators

### 2. Model Architecture

- **Algorithm:** LightGBM with separate regressors for x and y coordinates
- **Validation:** 5-fold cross-validation grouped by play_id (prevents data leakage)
- **Training:** 562,000 prediction samples across 14,000 plays

### 3. Results

**Overall Performance:** 1.33 RMSE (Root Mean Squared Error in yards)

**Error Analysis:**

| Time Horizon | RMSE | Sample Size |
|--------------|------|-------------|
| 0-0.5s | 0.44 yards | 230k |
| 0.5-1.0s | 0.76 yards | 191k |
| 1.0-1.5s | 1.63 yards | 83k |
| 1.5-2.0s | 2.63 yards | 36k |
| 2.0+s | 3.85 yards | 21k |

**Player Role Breakdown:**
- Receivers: 0.94 RMSE
- Defenders: 1.51 RMSE (60% harder to predict)

## Key Findings

### What Worked
1. **Physics baselines** provide strong starting point for short predictions
2. **Temporal features** (velocity changes) capture acceleration patterns effectively
3. **Spatial context** (distance/angle to ball) improves trajectory adjustments
4. **Systematic validation** with grouped k-fold prevents data leakage

### Architectural Limitations

**Why single-frame LightGBM caps at ~1.3 RMSE:**

1. **No trajectory context** - Model sees only the last frame, ignoring 20+ frames of movement history
   - Cannot detect if player is accelerating, decelerating, or cutting
   - Misses patterns like "receiver runs straight then cuts toward ball"

2. **Independent predictions** - Each time step predicted separately
   - Doesn't ensure smooth, physically plausible trajectories
   - Can't learn temporal dependencies (position at t=2 depends on t=1)

3. **Error compounds with time** - Physics projections assume constant velocity
   - Breaks down as players adjust trajectories (2.0+s: 3.85 RMSE)
   - Particularly problematic for defenders reacting to receivers

### What's Needed for Competitive Performance

Top solutions (0.5-0.6 RMSE) use **sequence-to-sequence models**:
- Input: Full trajectory sequence (all 20+ frames)
- Architecture: LSTM/GRU/Transformers
- Output: Future trajectory sequence with temporal consistency
- Learns: Acceleration curves, cutting behavior, pursuit angles

## Project Structure
```
├── analysis.ipynb       # Complete analysis with visualizations
├── README.md           # This file
└── requirements.txt    # Dependencies
```

## Key Learnings

1. **Research first, execute second** - Understanding past winning approaches saves weeks of iteration
2. **Choose appropriate architecture** - Trajectory prediction fundamentally requires sequence models
3. **Error analysis drives improvement** - Systematic breakdown reveals where model fails
4. **Domain knowledge matters** - NFL-specific patterns (routes, coverage) inform better features

## Next Steps

**To improve this baseline:**
- Separate models for receivers vs defenders (fundamentally different behaviors)
- Cap predictions at 2.5s or use distance-based weighting
- Ensemble multiple models with different feature sets

**To reach competitive performance:**
- Implement sequence model (GRU/LSTM) using full trajectory history
- Add player interaction features (defender-receiver pairings, zone coverage)
- Incorporate NFL domain knowledge (route trees, coverage schemes)

## Technical Notes

- **Data:** 562k training samples from 14,000 plays (2023-2024 NFL seasons)
- **Evaluation:** RMSE between predicted and actual (x, y) coordinates
- **Hardware:** Trained locally on 2015 MacBook Pro (CPU only)

## Acknowledgments

Competition data provided by the NFL and AWS as part of the 2026 Big Data Bowl.

---

**Author:** Andre DeGregorio  
**Competition:** [NFL Big Data Bowl 2026](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)
