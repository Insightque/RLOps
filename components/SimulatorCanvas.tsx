
import React, { useEffect, useRef } from 'react';
import { SimState } from '../types';

interface Props {
  state: SimState;
}

const SimulatorCanvas: React.FC<Props> = ({ state }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = 100;

    // Draw Track
    ctx.strokeStyle = '#334155';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(canvas.width, centerY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Target (Green)
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(centerX + state.target * scale, centerY, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText("TARGET", centerX + state.target * scale - 20, centerY - 15);

    // Draw Agent (Blue)
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(centerX + state.position * scale, centerY, 12, 0, Math.PI * 2);
    ctx.fill();
    
    // Simple Shadow/Aura
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX + state.position * scale, centerY, 16, 0, Math.PI * 2);
    ctx.stroke();

  }, [state]);

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 shadow-inner">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider">Live Simulator</h3>
        <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded">Continuous Control (1D)</span>
      </div>
      <canvas 
        ref={canvasRef} 
        width={400} 
        height={150} 
        className="w-full h-auto bg-slate-900 rounded-lg border border-slate-700"
      />
    </div>
  );
};

export default SimulatorCanvas;
