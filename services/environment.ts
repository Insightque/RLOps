
import { SimState } from '../types';

export interface EnvStepResult {
  nextState: SimState;
  reward: number;
  done: boolean;
}

export class PointMass1DEnv {
  private state: SimState;

  constructor() {
    this.state = this.reset();
  }

  reset(): SimState {
    this.state = {
      position: (Math.random() - 0.5) * 1.8,
      velocity: 0,
      target: (Math.random() - 0.5) * 1.8
    };
    return this.state;
  }

  getStateVector(s: SimState): number[] {
    // Relative distance is normalized to encourage better learning
    return [s.position / 2.5, s.velocity / 1.0, (s.target - s.position) / 2.5];
  }

  step(action: number): EnvStepResult {
    // Physics Logic
    const newVelocity = this.state.velocity * 0.92 + action * 0.1;
    const newPosition = this.state.position + newVelocity;
    
    // Reward Logic
    const dist = Math.abs(this.state.target - newPosition);
    let reward = Math.max(-2.0, -dist); // Distance penalty
    
    if (dist < 0.15) reward += 0.3; // Proximity bonus
    if (dist < 0.05) reward += 1.5; // Success bonus

    // Termination
    const done = dist < 0.05 || Math.abs(newPosition) > 2.8;

    this.state = {
      position: newPosition,
      velocity: newVelocity,
      target: this.state.target
    };

    return { nextState: this.state, reward, done };
  }
}
