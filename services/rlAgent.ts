
import * as tf from '@tensorflow/tfjs';
import { Experience } from '../types';

export class DDPGAgent {
  private actor: tf.LayersModel;
  private critic: tf.LayersModel;
  private targetActor: tf.LayersModel;
  private targetCritic: tf.LayersModel;
  
  private actorOptimizer: tf.Optimizer;
  private criticOptimizer: tf.Optimizer;
  
  private replayBuffer: Experience[] = [];
  private readonly bufferSize = 10000;
  private readonly batchSize = 64;
  private readonly gamma = 0.99;
  private readonly tau = 0.005;
  
  public epsilon = 1.0;
  private readonly epsilonDecay = 0.9975;
  private readonly epsilonMin = 0.05;

  constructor(private stateDim: number, private actionDim: number) {
    this.actor = this.createActor();
    this.targetActor = this.createActor();
    this.targetActor.setWeights(this.actor.getWeights());
    this.actorOptimizer = tf.train.adam(0.0001); 

    this.critic = this.createCritic();
    this.targetCritic = this.createCritic();
    this.targetCritic.setWeights(this.critic.getWeights());
    this.criticOptimizer = tf.train.adam(0.0002);
  }

  private createActor(): tf.LayersModel {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [this.stateDim], kernelInitializer: 'heNormal' }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: 'heNormal' }));
    model.add(tf.layers.dense({ 
      units: this.actionDim, 
      activation: 'tanh', 
      kernelInitializer: tf.initializers.randomUniform({ minval: -0.003, maxval: 0.003 }) 
    }));
    return model;
  }

  private createCritic(): tf.LayersModel {
    const stateInput = tf.input({ shape: [this.stateDim] });
    const actionInput = tf.input({ shape: [this.actionDim] });
    
    const sBranch = tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: 'heNormal' }).apply(stateInput);
    const combined = tf.layers.concatenate().apply([sBranch as tf.SymbolicTensor, actionInput]);
    const d1 = tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: 'heNormal' }).apply(combined);
    const output = tf.layers.dense({ units: 1 }).apply(d1) as tf.SymbolicTensor;
    
    return tf.model({ inputs: [stateInput, actionInput], outputs: output });
  }

  getAction(stateVector: number[], useNoise: boolean = true): { action: number, qValue: number } {
    return tf.tidy(() => {
      const sTensor = tf.tensor2d([stateVector]);
      const pred = this.actor.predict(sTensor) as tf.Tensor;
      const rawAct = pred.dataSync()[0];
      
      const qTensor = this.critic.predict([sTensor, pred]) as tf.Tensor;
      const qVal = qTensor.dataSync()[0];

      let finalAct = rawAct;
      if (useNoise) {
        const noise = (Math.random() - 0.5) * 2 * this.epsilon;
        finalAct = Math.max(-1, Math.min(1, rawAct + noise));
      }
      return { action: finalAct, qValue: qVal };
    });
  }

  storeExperience(exp: Experience) {
    this.replayBuffer.push(exp);
    if (this.replayBuffer.length > this.bufferSize) this.replayBuffer.shift();
  }

  async train(): Promise<{ actorLoss: number, criticLoss: number, avgQ: number }> {
    if (this.replayBuffer.length < this.batchSize) {
      return { actorLoss: 0, criticLoss: 0, avgQ: 0 };
    }

    const batch = [];
    for (let i = 0; i < this.batchSize; i++) {
      batch.push(this.replayBuffer[Math.floor(Math.random() * this.replayBuffer.length)]);
    }

    const states = tf.tensor2d(batch.map(b => b.s));
    const actions = tf.tensor2d(batch.map(b => b.a));
    const rewards = tf.tensor2d(batch.map(b => [b.r]));
    const nextStates = tf.tensor2d(batch.map(b => b.ns));
    const terminals = tf.tensor2d(batch.map(b => [b.d ? 1 : 0]));

    let avgQ = 0;
    
    // Update Critic
    const criticGrads = tf.variableGrads(() => {
      const nextActions = this.targetActor.predict(nextStates) as tf.Tensor;
      const nextQ = this.targetCritic.predict([nextStates, nextActions]) as tf.Tensor;
      const targetQ = rewards.add(nextQ.mul(this.gamma).mul(tf.scalar(1).sub(terminals)));
      const currentQ = this.critic.predict([states, actions]) as tf.Tensor;
      avgQ = currentQ.mean().dataSync()[0];
      return tf.losses.meanSquaredError(targetQ, currentQ) as tf.Scalar;
    });

    const clippedCriticGrads: {[key: string]: tf.Tensor} = {};
    for (const key in criticGrads.grads) {
      clippedCriticGrads[key] = tf.clipByValue(criticGrads.grads[key], -1, 1);
    }
    this.criticOptimizer.applyGradients(clippedCriticGrads);

    // Update Actor
    const actorGrads = tf.variableGrads(() => {
      const predActions = this.actor.predict(states) as tf.Tensor;
      const qValues = this.critic.predict([states, predActions]) as tf.Tensor;
      return tf.mean(qValues).mul(-1) as tf.Scalar;
    });

    const clippedActorGrads: {[key: string]: tf.Tensor} = {};
    for (const key in actorGrads.grads) {
      clippedActorGrads[key] = tf.clipByValue(actorGrads.grads[key], -1, 1);
    }
    this.actorOptimizer.applyGradients(clippedActorGrads);

    this.updateTargetNetworks();
    
    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay;
    }

    const cLoss = criticGrads.value.dataSync()[0];
    const aLoss = actorGrads.value.dataSync()[0];

    tf.dispose([states, actions, rewards, nextStates, terminals, criticGrads.value, actorGrads.value]);
    tf.dispose(Object.values(criticGrads.grads));
    tf.dispose(Object.values(actorGrads.grads));

    return { actorLoss: aLoss, criticLoss: cLoss, avgQ };
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
