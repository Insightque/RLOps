
import * as tf from '@tensorflow/tfjs';
import { SimState, TrainingMetric } from '../types';

/**
 * Stabilized DDPG implementation for RLOps.
 * Features: Gradient Clipping, Reduced Learning Rates, and Enhanced Q-Monitoring.
 */
export class DDPGSimulator {
  private actor: tf.LayersModel;
  private critic: tf.LayersModel;
  private targetActor: tf.LayersModel;
  private targetCritic: tf.LayersModel;
  
  private actorOptimizer: tf.Optimizer;
  private criticOptimizer: tf.Optimizer;
  
  private replayBuffer: { s: number[], a: number[], r: number, ns: number[], d: boolean }[] = [];
  private readonly bufferSize = 10000;
  private readonly batchSize = 64;
  private readonly gamma = 0.99;
  private readonly tau = 0.005;
  private readonly warmupSteps = 250; // Increased warmup
  
  private epsilon = 1.0;
  private readonly epsilonDecay = 0.997;
  private readonly epsilonMin = 0.05;
  
  private stepCount: number = 0;

  constructor() {
    const stateDim = 3; 
    const actionDim = 1;

    this.actor = this.createActor(stateDim, actionDim);
    this.targetActor = this.createActor(stateDim, actionDim);
    this.targetActor.setWeights(this.actor.getWeights());
    // Reduced Learning Rates for stability
    this.actorOptimizer = tf.train.adam(0.0001); 

    this.critic = this.createCritic(stateDim, actionDim);
    this.targetCritic = this.createCritic(stateDim, actionDim);
    this.targetCritic.setWeights(this.critic.getWeights());
    this.criticOptimizer = tf.train.adam(0.0002);
  }

  private createActor(stateDim: number, actionDim: number): tf.LayersModel {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [stateDim], kernelInitializer: 'heNormal' }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: 'heNormal' }));
    model.add(tf.layers.dense({ 
      units: actionDim, 
      activation: 'tanh', 
      kernelInitializer: tf.initializers.randomUniform({ minval: -0.003, maxval: 0.003 }) 
    }));
    return model;
  }

  private createCritic(stateDim: number, actionDim: number): tf.LayersModel {
    const stateInput = tf.input({ shape: [stateDim] });
    const actionInput = tf.input({ shape: [actionDim] });
    
    const sBranch = tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: 'heNormal' }).apply(stateInput);
    const combined = tf.layers.concatenate().apply([sBranch as tf.SymbolicTensor, actionInput]);
    const d1 = tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: 'heNormal' }).apply(combined);
    const output = tf.layers.dense({ units: 1 }).apply(d1) as tf.SymbolicTensor;
    
    return tf.model({ inputs: [stateInput, actionInput], outputs: output });
  }

  reset(): SimState {
    return {
      position: (Math.random() - 0.5) * 1.8,
      velocity: 0,
      target: (Math.random() - 0.5) * 1.8
    };
  }

  private getStateVector(s: SimState): number[] {
    return [s.position, s.velocity, (s.target - s.position)];
  }

  async trainStep(currentState: SimState): Promise<{ nextState: SimState, metric: TrainingMetric, noiseLevel: number }> {
    this.stepCount++;
    const sVec = this.getStateVector(currentState);

    // 1. Action selection
    const action = tf.tidy(() => {
      let act: number;
      const sTensor = tf.tensor2d([sVec]);
      const pred = this.actor.predict(sTensor) as tf.Tensor;
      const rawAct = pred.dataSync()[0];

      if (this.stepCount < this.warmupSteps) {
        act = (Math.random() - 0.5) * 2;
      } else {
        const noise = (Math.random() - 0.5) * 2 * this.epsilon;
        act = Math.max(-1, Math.min(1, rawAct + noise));
      }
      return act;
    });

    // 2. Physics & Reward
    const newVelocity = currentState.velocity * 0.92 + action * 0.1;
    const newPosition = currentState.position + newVelocity;
    
    const dist = Math.abs(currentState.target - newPosition);
    // Rewards are negative distance. Capped to prevent extreme gradients.
    let reward = Math.max(-2.0, -dist); 
    
    if (dist < 0.1) reward += 0.5; // Encouragement
    if (dist < 0.05) reward += 1.0; // Goal reached bonus

    const done = dist < 0.05 || Math.abs(newPosition) > 2.8;
    
    const nextState: SimState = {
      position: newPosition,
      velocity: newVelocity,
      target: currentState.target
    };
    const nsVec = this.getStateVector(nextState);

    // 3. Replay Storage
    this.replayBuffer.push({ s: sVec, a: [action], r: reward, ns: nsVec, d: done });
    if (this.replayBuffer.length > this.bufferSize) this.replayBuffer.shift();

    let cLoss = 0;
    let aLoss = 0;
    let avgQ = 0;

    // 4. Learning Phase with Gradient Clipping
    if (this.stepCount > this.warmupSteps && this.replayBuffer.length > this.batchSize) {
      const batch = [];
      for (let i = 0; i < this.batchSize; i++) {
        batch.push(this.replayBuffer[Math.floor(Math.random() * this.replayBuffer.length)]);
      }

      const states = tf.tensor2d(batch.map(b => b.s));
      const actions = tf.tensor2d(batch.map(b => b.a));
      const rewards = tf.tensor2d(batch.map(b => [b.r]));
      const nextStates = tf.tensor2d(batch.map(b => b.ns));
      const terminals = tf.tensor2d(batch.map(b => [b.d ? 1 : 0]));

      // --- CRITIC UPDATE WITH GRADIENT CLIPPING ---
      const criticGrads = tf.variableGrads(() => {
        const nextActions = this.targetActor.predict(nextStates) as tf.Tensor;
        const nextQ = this.targetCritic.predict([nextStates, nextActions]) as tf.Tensor;
        const targetQ = rewards.add(nextQ.mul(this.gamma).mul(tf.scalar(1).sub(terminals)));
        const currentQ = this.critic.predict([states, actions]) as tf.Tensor;
        
        // Monitoring Avg Q
        avgQ = currentQ.mean().dataSync()[0];
        
        return tf.losses.meanSquaredError(targetQ, currentQ) as tf.Scalar;
      });

      cLoss = criticGrads.value.dataSync()[0];
      
      // Manual Clipping
      const clippedCriticGrads: {[key: string]: tf.Tensor} = {};
      for (const key in criticGrads.grads) {
        clippedCriticGrads[key] = tf.clipByValue(criticGrads.grads[key], -1, 1);
      }
      this.criticOptimizer.applyGradients(clippedCriticGrads);

      // --- ACTOR UPDATE WITH GRADIENT CLIPPING ---
      const actorGrads = tf.variableGrads(() => {
        const predActions = this.actor.predict(states) as tf.Tensor;
        const qValues = this.critic.predict([states, predActions]) as tf.Tensor;
        return tf.mean(qValues).mul(-1) as tf.Scalar;
      });

      aLoss = actorGrads.value.dataSync()[0];

      const clippedActorGrads: {[key: string]: tf.Tensor} = {};
      for (const key in actorGrads.grads) {
        clippedActorGrads[key] = tf.clipByValue(actorGrads.grads[key], -1, 1);
      }
      this.actorOptimizer.applyGradients(clippedActorGrads);

      this.updateTargetNetworks();
      
      if (this.epsilon > this.epsilonMin) {
        this.epsilon *= this.epsilonDecay;
      }

      // Cleanup
      tf.dispose([states, actions, rewards, nextStates, terminals, criticGrads.value, actorGrads.value]);
      tf.dispose(Object.values(criticGrads.grads));
      tf.dispose(Object.values(actorGrads.grads));
    }

    return {
      nextState,
      metric: {
        step: this.stepCount,
        reward: reward,
        criticLoss: cLoss,
        actorLoss: aLoss,
        qValue: avgQ // Logged the average Q value from batch
      },
      noiseLevel: this.epsilon
    };
  }

  private updateTargetNetworks() {
    tf.tidy(() => {
      const update = (net: tf.LayersModel, targetNet: tf.LayersModel) => {
        const weights = net.getWeights();
        const targetWeights = targetNet.getWeights();
        const newWeights = weights.map((w, i) => 
          w.mul(this.tau).add(targetWeights[i].mul(1 - this.tau))
        );
        targetNet.setWeights(newWeights);
      };
      update(this.actor, this.targetActor);
      update(this.critic, this.targetCritic);
    });
  }
}
