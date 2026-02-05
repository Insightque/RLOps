
import React, { useState, useEffect, useRef } from 'react';
import { SimState, TrainingMetric, TrainingLog, PipelineStatus } from './types';
import { PointMass1DEnv } from './services/environment';
import { DDPGAgent } from './services/rlAgent';
import { analyzeTrainingPerformance } from './services/geminiService';
import SimulatorCanvas from './components/SimulatorCanvas';
import MetricsCharts from './components/MetricsCharts';

const App: React.FC = () => {
  const [pipeline, setPipeline] = useState<PipelineStatus>({
    stage: 'IDLE',
    progress: 0,
    active: false
  });
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [simState, setSimState] = useState<SimState>({ position: 0, velocity: 0, target: 0 });
  const [noiseLevel, setNoiseLevel] = useState(1.0);
  const [currentQ, setCurrentQ] = useState(0);
  const [logs, setLogs] = useState<TrainingLog[]>([]);
  const [aiAnalysis, setAiAnalysis] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const agentRef = useRef<DDPGAgent | null>(null);
  const envRef = useRef<PointMass1DEnv | null>(null);
  const simStateRef = useRef<SimState>({ position: 0, velocity: 0, target: 0 });
  const stepCountRef = useRef(0);
  const isTrainingRef = useRef(false);
  const loopTimeoutRef = useRef<number | null>(null);
  
  const warmupSteps = 300;

  const initSystem = () => {
    envRef.current = new PointMass1DEnv();
    agentRef.current = new DDPGAgent(3, 1);
    const initialState = envRef.current.reset();
    simStateRef.current = initialState;
    setSimState(initialState);
    setMetrics([]);
    setNoiseLevel(1.0);
    setCurrentQ(0);
    stepCountRef.current = 0;
  };

  useEffect(() => {
    initSystem();
    return () => stopTraining();
  }, []);

  const addLog = (message: string, type: 'INFO' | 'WARN' | 'ERROR' = 'INFO') => {
    setLogs(prev => [{
      timestamp: new Date().toLocaleTimeString(),
      type,
      message
    }, ...prev.slice(0, 49)]);
  };

  const trainingLoop = async () => {
    if (!isTrainingRef.current || !agentRef.current || !envRef.current) return;

    const agent = agentRef.current;
    const env = envRef.current;

    stepCountRef.current++;
    const sVec = env.getStateVector(simStateRef.current);

    // 1. Act
    const { action, qValue } = agent.getAction(sVec, true);
    
    // 2. Step Environment
    const { nextState, reward, done } = env.step(action);
    const nsVec = env.getStateVector(nextState);

    // 3. Store Experience
    agent.storeExperience({ s: sVec, a: [action], r: reward, ns: nsVec, d: done });

    // 4. Train (Async)
    let actorLoss = 0, criticLoss = 0, avgQ = qValue;
    if (stepCountRef.current > warmupSteps) {
      const result = await agent.train();
      actorLoss = result.actorLoss;
      criticLoss = result.criticLoss;
      avgQ = result.avgQ;
    }

    // Update Refs & Local State
    simStateRef.current = nextState;
    
    // UI Updates
    setSimState(nextState);
    setNoiseLevel(agent.epsilon);
    setCurrentQ(avgQ);
    
    const newMetric: TrainingMetric = {
      step: stepCountRef.current,
      reward,
      criticLoss,
      actorLoss,
      qValue: avgQ
    };

    setMetrics(prev => [...prev.slice(-200), newMetric]);
    setPipeline(prev => ({ ...prev, progress: Math.min(99, prev.progress + 0.002) }));

    if (stepCountRef.current === warmupSteps) {
      addLog("Warmup Finished. Neural updates active.", "INFO");
    }

    if (done) {
      simStateRef.current = env.reset();
      setSimState(simStateRef.current);
    }

    // Schedule next step only after this one is fully processed
    if (isTrainingRef.current) {
      loopTimeoutRef.current = window.setTimeout(trainingLoop, 16);
    }
  };

  const startTraining = () => {
    if (isTrainingRef.current) return;
    addLog("Pipeline modular run started.", "INFO");
    setPipeline(prev => ({ ...prev, stage: 'TRAINING', active: true }));
    isTrainingRef.current = true;
    trainingLoop();
  };

  const stopTraining = () => {
    isTrainingRef.current = false;
    if (loopTimeoutRef.current) {
      clearTimeout(loopTimeoutRef.current);
      loopTimeoutRef.current = null;
    }
    setPipeline(prev => ({ ...prev, stage: 'IDLE', active: false }));
    addLog("Pipeline execution paused.", "WARN");
  };

  const handleReset = () => {
    stopTraining();
    initSystem();
    addLog("Full system reset performed.", "INFO");
  };

  const handleAiAnalysis = async () => {
    if (metrics.length < 20) return;
    setIsAnalyzing(true);
    const analysis = await analyzeTrainingPerformance(metrics);
    setAiAnalysis(analysis);
    setIsAnalyzing(false);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <header className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-emerald-400">
            RLOps Modular Engine
          </h1>
          <p className="text-slate-400 mt-1 uppercase text-[10px] font-bold tracking-widest">Stable Decoupled Architecture</p>
        </div>
        <div className="flex gap-3">
          <button onClick={handleReset} className="px-4 py-2 border border-slate-700 hover:bg-slate-800 rounded-lg text-slate-300 font-bold transition-all">Reset All</button>
          {pipeline.active ? (
            <button onClick={stopTraining} className="px-6 py-2 bg-rose-600 hover:bg-rose-500 rounded-lg font-bold shadow-lg">Stop</button>
          ) : (
            <button onClick={startTraining} className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg font-bold shadow-lg">Run Pipeline</button>
          )}
        </div>
      </header>

      <div className="bg-slate-800/80 backdrop-blur rounded-2xl p-6 border border-slate-700 mb-8">
        <div className="flex justify-between items-center mb-4">
          <div className="flex gap-8">
            <div className="flex flex-col">
              <span className="text-slate-500 text-[9px] font-black uppercase tracking-widest">Pipeline State</span>
              <span className={`text-sm font-bold ${pipeline.active ? 'text-emerald-400' : 'text-slate-400'}`}>
                {pipeline.active ? (stepCountRef.current < warmupSteps ? 'DATA COLLECTION' : 'GRADIENT DESCENT') : 'READY'}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-slate-500 text-[9px] font-black uppercase tracking-widest">Epsilon</span>
              <span className="text-indigo-400 font-mono text-sm">{(noiseLevel * 100).toFixed(1)}%</span>
            </div>
            <div className="flex flex-col">
              <span className="text-slate-500 text-[9px] font-black uppercase tracking-widest">Target Q</span>
              <span className="text-emerald-400 font-mono text-sm">{currentQ.toFixed(3)}</span>
            </div>
          </div>
          <span className="text-slate-500 text-xs font-mono">{Math.floor(pipeline.progress)}% COMPLETED</span>
        </div>
        <div className="w-full bg-slate-900 rounded-full h-1 border border-slate-700">
          <div className="h-full bg-indigo-500 transition-all duration-1000" style={{ width: `${pipeline.progress}%` }} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 flex flex-col gap-6">
          <SimulatorCanvas state={simState} />
          <div className="bg-slate-800/50 rounded-2xl border border-slate-700 h-[380px] overflow-hidden flex flex-col">
            <div className="p-4 border-b border-slate-700 bg-slate-800/80 font-bold text-slate-400 text-xs uppercase tracking-widest">Pipeline Events</div>
            <div className="p-4 overflow-y-auto flex-1 font-mono text-[10px] space-y-2">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-2 border-l border-slate-700 pl-3">
                  <span className="text-slate-600">{log.timestamp}</span>
                  <span className={log.type === 'ERROR' ? 'text-rose-500' : log.type === 'WARN' ? 'text-amber-500' : 'text-indigo-400'}>{log.type}</span>
                  <span className="text-slate-300">{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="lg:col-span-2">
          <MetricsCharts data={metrics} />
          
          <div className="mt-8 bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl relative overflow-hidden group">
            <div className="flex justify-between items-center mb-6 relative z-10">
              <h3 className="text-xl font-bold text-slate-200 flex items-center gap-2">
                <div className="p-1 bg-indigo-500/20 rounded">
                  <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                </div>
                RLOps Advisor
              </h3>
              <button onClick={handleAiAnalysis} disabled={isAnalyzing || metrics.length < 20} className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-bold shadow-lg shadow-indigo-900/40 disabled:opacity-20 transition-all active:scale-95">
                {isAnalyzing ? 'Analyzing...' : 'Analyze Health'}
              </button>
            </div>
            <div className="bg-slate-900/60 rounded-xl p-6 border border-slate-700/50 min-h-[140px] relative z-10">
              {aiAnalysis ? (
                <div className="text-slate-300 text-sm leading-relaxed prose prose-invert max-w-none prose-sm whitespace-pre-wrap">{aiAnalysis}</div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-slate-500 text-sm gap-2 italic">
                  Run training and click analyze for AI insights.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <footer className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4 pb-8">
        {[
          { label: 'Step Reward', value: metrics.length > 0 ? metrics[metrics.length - 1].reward.toFixed(3) : '0.000', color: 'text-emerald-400' },
          { label: 'Critic Loss', value: metrics.length > 0 ? metrics[metrics.length - 1].criticLoss.toFixed(5) : '0.00000', color: 'text-rose-400' },
          { label: 'Actor Loss', value: metrics.length > 0 ? metrics[metrics.length - 1].actorLoss.toFixed(5) : '0.00000', color: 'text-indigo-400' },
          { label: 'Total Steps', value: stepCountRef.current, color: 'text-slate-100' },
        ].map((stat, i) => (
          <div key={i} className="bg-slate-800/30 p-4 rounded-xl border border-slate-700/50">
            <p className="text-slate-500 text-[9px] font-black uppercase tracking-widest mb-1">{stat.label}</p>
            <p className={`text-2xl font-mono ${stat.color}`}>{stat.value}</p>
          </div>
        ))}
      </footer>
    </div>
  );
};

export default App;
