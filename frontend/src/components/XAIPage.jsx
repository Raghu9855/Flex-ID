import React, { useState } from 'react';
import {
    BrainCircuit,
    Play,
    Activity,
    Image as ImageIcon,
    ShieldCheck,
    ShieldAlert,
    Cpu,
    Server,
    Info
} from 'lucide-react';

const IMG_BASE = 'http://localhost:5000/results';

export default function XAIPage() {
    const [imageKey, setImageKey] = useState(Date.now());
    const [selectedModel, setSelectedModel] = useState('fedavg'); // 'fedavg' | 'fedprox'
    const [selectedScenario, setSelectedScenario] = useState('no_attack'); // 'no_attack' | 'under_attack'

    // Refresh image cache
    const refreshImage = () => {
        setImageKey(Date.now());
    };

    // Determine the path based on selection
    const getWeightsPath = () => {
        if (selectedModel === 'fedavg') {
            return selectedScenario === 'no_attack'
                ? 'results/fedavgeachround/round-10-weights.pkl'
                : 'results/fedunderattack/round-10-weights.pkl';
        } else {
            return selectedScenario === 'no_attack'
                ? 'results/fedproxeachround/round-10-weights.pkl'
                : 'results/fedproxunderattack/round-10-weights.pkl';
        }
    };

    // Unique output name based on selection
    const getOutputName = () => `shap_summary_${selectedModel}_${selectedScenario}.png`;

    const commandToRun = `python explain_model.py --weights ${getWeightsPath()} --output results/${getOutputName()}`;

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Header */}
            <div className="flex flex-col gap-2 pb-6 border-b border-white/5">
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                    <BrainCircuit className="w-8 h-8 text-purple-400" />
                    Explainable AI (XAI)
                </h1>
                <p className="text-slate-400 max-w-2xl">
                    Generate Feature Importance explanations for any of your trained models.
                    Compare how different models (FedAvg vs FedProx) rank features under different conditions.
                </p>
            </div>

            {/* Configuration & Command Area */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Visual Selectors */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Model Selection */}
                    <div className="bg-slate-900/50 border border-white/10 rounded-xl p-6">
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Server className="w-4 h-4" /> 1. Select Model Strategy
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                            <button
                                onClick={() => setSelectedModel('fedavg')}
                                className={`p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${selectedModel === 'fedavg'
                                    ? 'border-blue-500 bg-blue-500/10 text-white shadow-[0_0_20px_rgba(59,130,246,0.2)]'
                                    : 'border-white/5 bg-black/20 text-slate-500 hover:border-white/10 hover:bg-white/5'
                                    }`}
                            >
                                <span className="font-bold">FedAvg</span>
                                <span className="text-xs opacity-70">Standard Averaging</span>
                            </button>
                            <button
                                onClick={() => setSelectedModel('fedprox')}
                                className={`p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${selectedModel === 'fedprox'
                                    ? 'border-pink-500 bg-pink-500/10 text-white shadow-[0_0_20px_rgba(236,72,153,0.2)]'
                                    : 'border-white/5 bg-black/20 text-slate-500 hover:border-white/10 hover:bg-white/5'
                                    }`}
                            >
                                <span className="font-bold">FedProx</span>
                                <span className="text-xs opacity-70">Robust Aggregation</span>
                            </button>
                        </div>
                    </div>

                    {/* Scenario Selection */}
                    <div className="bg-slate-900/50 border border-white/10 rounded-xl p-6">
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Cpu className="w-4 h-4" /> 2. Select Scenario
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                            <button
                                onClick={() => setSelectedScenario('no_attack')}
                                className={`p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${selectedScenario === 'no_attack'
                                    ? 'border-emerald-500 bg-emerald-500/10 text-white shadow-[0_0_20px_rgba(16,185,129,0.2)]'
                                    : 'border-white/5 bg-black/20 text-slate-500 hover:border-white/10 hover:bg-white/5'
                                    }`}
                            >
                                <ShieldCheck className={selectedScenario === 'no_attack' ? "text-emerald-400" : ""} />
                                <span className="font-bold">Normal Operation</span>
                            </button>
                            <button
                                onClick={() => setSelectedScenario('under_attack')}
                                className={`p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${selectedScenario === 'under_attack'
                                    ? 'border-red-500 bg-red-500/10 text-white shadow-[0_0_20px_rgba(239,68,68,0.2)]'
                                    : 'border-white/5 bg-black/20 text-slate-500 hover:border-white/10 hover:bg-white/5'
                                    }`}
                            >
                                <ShieldAlert className={selectedScenario === 'under_attack' ? "text-red-400" : ""} />
                                <span className="font-bold">Under Attack</span>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Command Panel */}
                <div className="bg-slate-950 border border-white/10 rounded-xl p-6 flex flex-col justify-between h-full bg-gradient-to-b from-slate-900 to-black">
                    <div>
                        <h3 className="font-bold text-white flex items-center gap-2 mb-4">
                            <Activity className="w-5 h-5 text-purple-400" />
                            Generate & View
                        </h3>
                        <p className="text-sm text-slate-400 mb-4">
                            Run this command in your terminal to generate the plot for the selected configuration.
                        </p>
                        <div className="bg-black p-4 rounded-lg border border-slate-800 font-mono text-xs text-green-400 break-all select-all selection:bg-green-900">
                            {commandToRun}
                        </div>
                    </div>

                    <button
                        onClick={refreshImage}
                        className="mt-6 w-full flex items-center justify-center gap-2 px-6 py-4 bg-slate-800 hover:bg-purple-600 hover:text-white text-slate-200 rounded-xl transition-all font-bold group border border-white/5"
                    >
                        <Play className="w-5 h-5 group-hover:fill-current" />
                        Refresh Plot Result
                    </button>
                </div>
            </div>

            {/* Results Display */}
            <div className="bg-slate-900 border border-white/10 rounded-xl overflow-hidden min-h-[500px] flex flex-col">
                <div className="p-4 border-b border-white/5 bg-black/20 flex justify-between items-center">
                    <h3 className="font-bold text-white flex items-center gap-2">
                        <ImageIcon className="w-5 h-5 text-slate-400" />
                        Feature Importance Summary
                    </h3>
                    <div className="text-xs text-slate-500 font-mono">
                        Showing: {selectedModel.toUpperCase()} - {selectedScenario === 'no_attack' ? 'NORMAL' : 'ATTACK'}
                    </div>
                </div>

                <div className="flex-1 p-8 flex items-center justify-center bg-black/40">
                    <div className="relative group w-full max-w-4xl flex justify-center">
                        <img
                            src={`${IMG_BASE}/${getOutputName()}?t=${imageKey}`}
                            alt="Waiting for generated SHAP plot..."
                            className="max-w-full h-auto rounded shadow-2xl border border-white/5"
                            onError={(e) => {
                                e.target.style.display = 'none';
                                e.target.parentElement.nextElementSibling?.classList.remove('hidden'); // Show fallback
                            }}
                            onLoad={(e) => {
                                e.target.style.display = 'block';
                                e.target.parentElement.nextElementSibling?.classList.add('hidden'); // Hide fallback
                            }}
                        />
                        {/* Fallback Message (hidden when image loads) */}
                        <div className="hidden flex flex-col items-center justify-center text-slate-500 p-12 text-center">
                            <Info className="w-12 h-12 mb-4 opacity-50" />
                            <p className="text-lg font-medium mb-1">No Plot Available</p>
                            <p className="text-sm max-w-md">
                                Run the command above, then click <strong>Refresh Plot Result</strong>.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Help/Explanation */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-slate-800/50 p-6 rounded-xl border border-white/5">
                    <h4 className="font-bold text-white mb-2">How to read this plot?</h4>
                    <ul className="list-disc list-inside text-sm text-slate-400 space-y-2">
                        <li><strong>Top is #1</strong>: The feature at the very top is the <em>most important</em> factor the AI looks at.</li>
                        <li><strong>Longer is Stronger</strong>: The longer the bar, the more influence that feature has on the final decision.</li>
                    </ul>
                </div>
                <div className="bg-slate-800/50 p-6 rounded-xl border border-white/5">
                    <h4 className="font-bold text-white mb-2">Simple Interpretation</h4>
                    <p className="text-sm text-slate-400 leading-relaxed">
                        Imagine the AI is a detective. Use this chart to see its "clues".
                        <br /><br />
                        If <strong>Destination Bytes</strong> is the longest bar, it means simple <em>file transfer size</em> is the main clue the AI uses to catch attacks.
                    </p>
                </div>
            </div>
        </div>
    );
}
