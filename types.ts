
export interface TrainingMetric {
  step: number;
  reward: number;
  criticLoss: number;
  actorLoss: number;
  qValue: number;
}

export interface SimState {
  position: number;
  velocity: number;
  target: number;
}

export interface TrainingLog {
  timestamp: string;
  type: 'INFO' | 'WARN' | 'ERROR';
  message: string;
}

export interface PipelineStatus {
  stage: 'IDLE' | 'DATA_INGESTION' | 'TRAINING' | 'EVALUATION' | 'DEPLOYMENT';
  progress: number;
  active: boolean;
}

export interface Experience {
  s: number[];
  a: number[];
  r: number;
  ns: number[];
  d: boolean;
}
