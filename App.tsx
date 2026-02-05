
import React, { useState, useEffect, useRef } from 'react';
import { SimState, TrainingMetric, TrainingLog, PipelineStatus } from './types';
import { DDPGSimulator } from './services/rlEngine';
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

  const simulatorRef = useRef<DDPGSimulator | null>(null);
  const trainingIntervalRef = useRef<number | null>(null);
  const simStateRef = useRef<SimState>({ position: 0, velocity: 0, target: 0 });

  const initSimulator = () => {
    simulatorRef.current = new DDPGSimulator();
    simStateRef.current = simulatorRef.current.reset();
    setSimState(simStateRef.current);
    setMetrics([]);
    setNoiseLevel(1.0);
    setCurrentQ(0);
  };

  useEffect(() => {
    initSimulator();
  }, []);

  const addLog = (message: string, type: 'INFO' | 'WARN' | 'ERROR' = 'INFO') => {
    setLogs(prev => [{
      timestamp: new Date().toLocaleTimeString(),
      type,
      message
    }, ...prev.slice(0, 49)]);
  };

  const startTraining = () => {
    if (pipeline.active || !simulatorRef.current) return;
    
    addLog("Initializing Stabilized RLOps Pipeline...", "INFO");
    setPipeline({ stage: 'TRAINING', progress: 5, active: true });
    
    setTimeout(() => {
      addLog("Grad-Clip enabled. LR adjusted (Critic: 0.0002).", "INFO");
      addLog("Starting DDPG Warmup...", "INFO");

      trainingIntervalRef.current = window.setInterval(async () => {
        if (!simulatorRef.current) return;

        const { nextState, metric, noiseLevel: currentNoise } = await simulatorRef.current.trainStep(simStateRef.current);
        simStateRef.current = nextState;
        setSimState(nextState);
        setNoiseLevel(currentNoise);
        setCurrentQ(metric.qValue);
        
        setMetrics(prev => [...prev.slice(-200), metric]);
        
        setPipeline(prev => ({ 
          ...prev, 
          progress: Math.min(99, prev.progress + 0.003) 
        }));

        if (metric.step === 250) {
          addLog("Warmup complete. Updates starting.", "INFO");
        }

        const dist = Math.abs(nextState.position - nextState.target);
        const outOfBounds = Math.abs(nextState.position) > 2.8;
        
        if (dist < 0.05 || outOfBounds) {
          const newState = simulatorRef.current.reset();
          simStateRef.current = newState;
          setSimState(newState);
        }
      }, 50);
    }, 1000);
  };

  const stopTraining = () => {
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current);
      trainingIntervalRef.current = null;
    }
    setPipeline({ stage: 'IDLE', progress: 0, active: false });
    addLog("Training sequence halted.", "WARN");
  };

  const handleReset = () => {
    stopTraining();
    initSimulator();
    addLog("Neural weights and buffer purged. Ready for fresh run.", "INFO");
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
            RLOps Deep Engine
          </h1>
          <p className="text-slate-400 mt-1">Real-time DDPG Neural Policy Optimization</p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={handleReset} 
            className="px-4 py-2 border border-slate-700 hover:bg-slate-800 rounded-lg text-slate-300 font-bold transition-all"
          >
            Reset Engine
          </button>
          {pipeline.active ? (
            <button onClick={stopTraining} className="px-6 py-2 bg-rose-600 hover:bg-rose-500 rounded-lg font-bold shadow-lg transition-all">Stop</button>
          ) : (
            <button onClick={startTraining} className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg font-bold shadow-lg transition-all">Start Pipeline</button>
          )}
        </div>
      </header>

      <div className="bg-slate-800/90 backdrop-blur rounded-2xl p-6 border border-slate-700 mb-8 shadow-2xl">
        <div className="flex justify-between items-center mb-4">
          <div className="flex flex-wrap gap-6 items-center">
            <span className={`px-3 py-1 rounded-full text-[10px] font-black tracking-widest ${pipeline.active ? 'bg-emerald-500/20 text-emerald-400 animate-pulse' : 'bg-slate-700 text-slate-400'}`}>
              {pipeline.active ? (metrics.length < 250 ? 'WARMUP PHASE' : 'ACTIVE TRAINING') : 'SYSTEM IDLE'}
            </span>
            <div className="flex flex-col">
              <span className="text-slate-500 text-[10px] uppercase font-bold tracking-tighter">Exploration (Epsilon)</span>
              <span className="text-indigo-400 font-mono text-sm">{(noiseLevel * 100).toFixed(1)}%</span>
            </div>
            <div className="flex flex-col">
              <span className="text-slate-500 text-[10px] uppercase font-bold tracking-tighter">Batch Q-Avg</span>
              <span className={`font-mono text-sm ${currentQ < 0 ? 'text-rose-400' : 'text-emerald-400'}`}>{currentQ.toFixed(3)}</span>
            </div>
          </div>
          <span className="hidden md:inline text-slate-500 text-xs font-mono">{Math.floor(pipeline.progress)}% PIPELINE SYNC</span>
        </div>
        <div className="w-full bg-slate-900 rounded-full h-1.5 border border-slate-700 overflow-hidden">
          <div className="h-full bg-gradient-to-r from-indigo-500 to-emerald-500 transition-all duration-700" style={{ width: `${pipeline.progress}%` }} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 flex flex-col gap-6">
          <SimulatorCanvas state={simState} />
          <div className="bg-slate-800/50 rounded-2xl border border-slate-700 h-[380px] overflow-hidden flex flex-col shadow-lg">
            <div className="p-4 border-b border-slate-700 bg-slate-800/80 font-bold text-slate-400 text-xs uppercase tracking-widest">Telemetry Logs</div>
            <div className="p-4 overflow-y-auto flex-1 font-mono text-[10px] space-y-2 scrollbar-thin scrollbar-thumb-slate-700">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-2 border-l border-slate-700 pl-3">
                  <span className="text-slate-600 tabular-nums">{log.timestamp}</span>
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
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-emerald-500/5 pointer-events-none" />
            <div className="flex justify-between items-center mb-6 relative z-10">
              <h3 className="text-xl font-bold text-slate-200 flex items-center gap-2">
                <div className="p-1 bg-indigo-500/20 rounded">
                  <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                </div>
                Gemini Training Advisor
              </h3>
              <button onClick={handleAiAnalysis} disabled={isAnalyzing || metrics.length < 20} className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-bold shadow-lg shadow-indigo-900/40 disabled:opacity-20 transition-all active:scale-95">
                {isAnalyzing ? 'Processing Gradients...' : 'Generate Analysis'}
              </button>
            </div>
            <div className="bg-slate-900/60 rounded-xl p-6 border border-slate-700/50 min-h-[140px] relative z-10">
              {aiAnalysis ? (
                <div className="text-slate-300 text-sm leading-relaxed prose prose-invert max-w-none prose-sm">{aiAnalysis}</div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-slate-500 text-sm gap-2">
                  <div className="w-8 h-8 border-2 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin" />
                  <p className="italic">Collecting neural metrics for deep analysis...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <footer className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4 pb-8">
        {[
          { label: 'Step Reward', value: metrics.length > 0 ? metrics[metrics.length - 1].reward.toFixed(3) : '0.000', color: 'text-emerald-400' },
          { label: 'Batch Q-Avg', value: metrics.length > 0 ? metrics[metrics.length - 1].qValue.toFixed(4) : '0.0000', color: 'text-indigo-400' },
          { label: 'Critic Loss', value: metrics.length > 0 ? metrics[metrics.length - 1].criticLoss.toFixed(4) : '0.0000', color: 'text-rose-400' },
          { label: 'Buffer Size', value: metrics.length > 0 ? metrics[metrics.length - 1].step : '0', color: 'text-slate-100' },
        ].map((stat, i) => (
          <div key={i} className="bg-slate-800/30 p-4 rounded-xl border border-slate-700/50 shadow-inner">
            <p className="text-slate-500 text-[9px] font-black uppercase tracking-widest mb-1">{stat.label}</p>
            <p className={`text-2xl font-mono ${stat.color}`}>{stat.value}</p>
          </div>
        ))}
      </footer>
    </div>
  );
};

export default App;
