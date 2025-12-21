import React, { useState, useEffect } from 'react';
import {
    Server,
    Database,
    BrainCircuit,
    Shield,
    Zap,
    Activity,
    Terminal,
    Cpu,
    Globe,
    X,
    Radio,
    Play
} from 'lucide-react';

export default function ProjectWorkflow() {
    const [started, setStarted] = useState(false);
    const [step, setStep] = useState(0);
    const [logs, setLogs] = useState(["> SYSTEM_STATUS: OFFLINE", "> WAITING FOR INIT..."]);

    const steps = [
        {
            title: "PROTOCOL: PARTITION",
            desc: "SEGMENTING DATASET. Features are split and distributed to clients. No single client holds all data.",
            duration: 4000,
            log: "> LOADING DATASET [CIC-IDS2018]...\n> SPLITTING: 3 SHARDS CREATED\n> ROUTING TO SECURE ENCLAVES..."
        },
        {
            title: "PROTOCOL: DISTRIBUTE",
            desc: "Initializing Global Model. Architecture broadcast to clients. Data remains encrypted at source.",
            duration: 4000,
            log: "> ESTABLISHING SECURE CHANNEL...\n> SYNCING MODEL ARCHITECTURE...\n> DATA LOCALITY: ENFORCED"
        },
        {
            title: "PROTOCOL: TRAIN_LOCAL",
            desc: "Clients executing local gradient descent. Private data accessed securely on-device.",
            duration: 4500,
            log: "> CLIENT_1: TRAINING BATCH [================>]\n> CLIENT_2: OPTIMIZING WEIGHTS...\n> CLIENT_3: CALCULATING GRADIENTS...\n> PRIVACY SHIELD: ACTIVE"
        },
        {
            title: "PROTOCOL: UPLOAD_WEIGHTS",
            desc: "Encrypting and transmitting learned model parameters. No raw telemetry extracted.",
            duration: 4000,
            log: "> ENCRYPTING PAYLOAD...\n> UPLOADING TENSORS...\n> BANDWIDTH OPTIMIZATION: ON"
        },
        {
            title: "PROTOCOL: AGGREGATE",
            desc: "Fusion Engine active. Merging local intelligence into Global Master Model.",
            duration: 3500,
            log: "> EXECUTING FEDAVG()...\n> MERGING PARAMETERS...\n> OUTLIER REJECTION: ACTIVE"
        },
        {
            title: "PROTOCOL: BROADCAST",
            desc: "Deploying upgraded neural weights to entire network grid. Defense matrix updated.",
            duration: 3500,
            log: "> PUSHING UPDATE v1.0.4...\n> UPDATING DEFENSE MATRIX...\n> NETWORK SYNCHRONIZED"
        },
        {
            title: "PROTOCOL: XAI_ANALYSIS",
            desc: "DECIPHERING BLACK BOX. SHAP Scan initiated to isolate intrusion signatures.",
            duration: 6000,
            log: "> INITIATING KERNEL EXPLAINER...\n> SCANNING FEATURE IMPORTANCE...\n> ANOMALY DETECTED: [Dst Bytes]"
        }
    ];

    useEffect(() => {
        if (!started) return;

        let currentTimer;
        const runStep = (index) => {
            const nextIndex = index >= steps.length - 1 ? 0 : index + 1;
            setStep(index);
            // Update pseudo-terminal logs
            setLogs(prev => [...prev.slice(-4), steps[index].log]);

            currentTimer = setTimeout(() => {
                runStep(nextIndex);
            }, steps[index].duration);
        };
        runStep(0);
        return () => clearTimeout(currentTimer);
    }, [started]);

    return (
        <div className="w-full relative rounded-2xl overflow-hidden font-mono border border-cyan-500/20 bg-black shadow-2xl h-[600px] group">

            {/* --- OFFLINE OVERLAY --- */}
            {!started && (
                <div className="absolute inset-0 z-50 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center p-8 text-center animate-in fade-in duration-500">
                    <div className="w-24 h-24 mb-6 rounded-full border-4 border-slate-700 flex items-center justify-center relative">
                        <div className="absolute inset-0 border-4 border-t-cyan-500 rounded-full animate-spin duration-3s"></div>
                        <Activity className="w-10 h-10 text-slate-500" />
                    </div>
                    <h2 className="text-3xl font-bold text-white mb-2 tracking-widest">SYSTEM OFFLINE</h2>
                    <p className="text-slate-400 mb-8 max-w-md">Secure Federated Learning Simulation Environment. Authorization Required to Initialize.</p>
                    <button
                        onClick={() => setStarted(true)}
                        className="group relative px-8 py-4 bg-cyan-900/30 border border-cyan-500 text-cyan-400 font-bold tracking-widest hover:bg-cyan-500 hover:text-black transition-all overflow-hidden"
                    >
                        <span className="relative z-10 flex items-center gap-2">
                            <Play className="w-5 h-5 fill-current" /> INITIALIZE_SYSTEM
                        </span>
                        <div className="absolute inset-0 bg-cyan-500 translate-y-full group-hover:translate-y-0 transition-transform duration-300 z-0"></div>
                    </button>
                </div>
            )}

            {/* CRT Scanline & Vignette Effect */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[size:100%_2px,3px_100%] pointer-events-none z-0" />
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_50%,rgba(0,0,0,0.6))] pointer-events-none z-0" />

            {/* Background Grid */}
            <div className="absolute inset-0 opacity-20 bg-[linear-gradient(to_right,#083344_1px,transparent_1px),linear-gradient(to_bottom,#083344_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_60%_at_50%_50%,#000_70%,transparent_100%)] z-0" />

            <div className="w-full h-full p-4 relative z-10 flex flex-col md:flex-row gap-6">

                {/* LEFT PANEL: VISUALIZATION STAGE */}
                <div className="flex-1 border border-cyan-500/30 bg-black/60 relative p-6 clip-corner flex flex-col justify-between overflow-hidden group">
                    {/* Decorative Corner Lines */}
                    <div className="absolute top-0 left-0 w-8 h-8 border-l-2 border-t-2 border-cyan-500" />
                    <div className="absolute top-0 right-0 w-8 h-8 border-r-2 border-t-2 border-cyan-500" />
                    <div className="absolute bottom-0 left-0 w-8 h-8 border-l-2 border-b-2 border-cyan-500" />
                    <div className="absolute bottom-0 right-0 w-8 h-8 border-r-2 border-b-2 border-cyan-500" />

                    {/* Header Overlay */}
                    <div className="absolute top-4 left-6 flex items-center gap-2 text-cyan-400 text-xs tracking-widest">
                        <Radio className="w-4 h-4 animate-pulse" />
                        <span>LIVE_FEED :: {started ? `STEP ${step + 1}/${steps.length}` : 'STANDBY'}</span>
                    </div>

                    {/* MAIN VISUAL CENTER */}
                    <div className="flex-1 relative flex flex-col items-center justify-center">

                        {/* SERVER NODE */}
                        <div className={`
                            relative z-20 transition-all duration-700
                            ${(step === 4 || step === 6) ? 'scale-125' : 'scale-100'}
                        `}>
                            {/* Hexagon Wrapper */}
                            <div className={`
                                w-24 h-24 flex items-center justify-center bg-slate-900 border-2 clip-hex transition-all duration-500
                                ${step === 6 ? 'border-pink-500 shadow-[0_0_40px_rgba(236,72,153,0.6)]' :
                                    step === 4 ? 'border-cyan-400 shadow-[0_0_40px_rgba(34,211,238,0.5)]' : 'border-slate-700'}
                            `}>
                                {step === 6 ? (
                                    <BrainCircuit className="w-10 h-10 text-pink-500 animate-pulse" />
                                ) : (
                                    <Globe className={`w-10 h-10 ${step === 4 ? 'text-cyan-400' : 'text-slate-600'}`} />
                                )}
                            </div>
                            <div className="mt-2 text-center">
                                <span className={`text-[10px] tracking-[0.2em] font-bold ${step === 6 ? 'text-pink-500' : 'text-cyan-500'}`}>FEDERATED_SERVER</span>
                            </div>

                            {/* PARTITION ANIMATION (Center Source) */}
                            {step === 0 && (
                                <div className="absolute inset-0 flex items-center justify-center z-50">
                                    <Database className="w-12 h-12 text-white animate-ping" />
                                </div>
                            )}

                            {/* SCAN EFFECT for XAI */}
                            {step === 6 && (
                                <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[140%] h-[140%] border border-pink-500/30 rounded-full animate-[spin_4s_linear_infinite]" />
                            )}
                        </div>

                        {/* CONNECTION PIPES */}
                        <div className="absolute inset-0 z-0">
                            {[1, 2, 3, 4].map((i) => (
                                <div key={i}
                                    className="absolute top-1/2 left-1/2 w-full h-[2px] bg-slate-800 origin-left"
                                    style={{
                                        transform: `translate(0, 0) rotate(${165 - (i * 30)}deg) translateY(50px)`,
                                        width: '160px'
                                    }}
                                >
                                    {/* Data Partition Packets (Outward) */}
                                    {step === 0 && (
                                        <div className="absolute top-1/2 -translate-y-1/2 w-4 h-2 bg-white shadow-[0_0_15px_#fff] rounded animate-flow-out" />
                                    )}

                                    {/* Distribute/Broadcast (Outward) */}
                                    {(step === 1 || step === 5) && (
                                        <div className="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-cyan-400 shadow-[0_0_10px_#22d3ee] rounded-full animate-flow-out" />
                                    )}

                                    {/* Upload (Inward) */}
                                    {step === 3 && (
                                        <div className="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-emerald-400 shadow-[0_0_10px_#34d399] rounded-full animate-flow-in" />
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* CLIENT NODES FOOTER */}
                    <div className="flex justify-around items-end pt-12 pb-4">
                        {[1, 2, 3, 4].map((id) => (
                            <div key={id} className={`flex flex-col items-center group/node transition-all duration-300 ${step === 0 ? 'scale-110' : 'scale-100'}`}>
                                <div className={`
                                    w-16 h-16 flex items-center justify-center bg-black border clip-hex transition-all duration-300
                                    ${step === 2 ? 'border-emerald-500 bg-emerald-900/10 shadow-[0_0_20px_rgba(16,185,129,0.3)]' : 'border-slate-800'}
                                    ${step === 0 ? 'border-white/50 bg-white/10' : ''}
                                `}>
                                    <Cpu className={`w-6 h-6 ${step === 2 ? 'text-emerald-400' : 'text-slate-700'}`} />
                                </div>
                                <div className="mt-2 flex items-center gap-1">
                                    <div className={`w-1.5 h-1.5 rounded-full ${step === 2 ? 'bg-emerald-500 animate-ping' : 'bg-slate-700'}`} />
                                    <span className="text-[10px] text-slate-500 font-bold">CLIENT_0{id}</span>
                                </div>

                                {/* Training Status Tag */}
                                {step === 2 && (
                                    <div className="absolute -translate-y-12 px-2 py-0.5 bg-emerald-500/20 border border-emerald-500/50 text-emerald-400 text-[9px] font-bold tracking-widest animate-pulse backdrop-blur-sm">
                                        TRAINING
                                    </div>
                                )}

                                {/* Partition Received Tag */}
                                {step === 0 && (
                                    <div className="absolute -translate-y-12 px-2 py-0.5 bg-white/20 border border-white/50 text-white text-[9px] font-bold tracking-widest animate-bounce">
                                        +DATA SHARD
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

                </div>

                {/* RIGHT PANEL: INFO & LOGS */}
                <div className="w-full md:w-80 flex flex-col gap-4">

                    {/* INFO CARD */}
                    <div className="bg-slate-900/80 border border-cyan-500/30 p-6 clip-corner relative overflow-hidden">
                        <div className="absolute inset-0 bg-cyan-500/5 z-0" />
                        <h3 className="relative z-10 text-2xl font-bold italic text-white mb-2 tracking-tighter">
                            {steps[step].title}
                        </h3>
                        <div className="w-12 h-1 bg-cyan-500 mb-4 relative z-10" />
                        <p className="relative z-10 text-sm text-cyan-100/70 leading-relaxed font-sans">
                            {steps[step].desc}
                        </p>

                        {/* XAI Stats Hologram */}
                        {step === 6 && (
                            <div className="mt-4 p-3 bg-black/50 border border-pink-500/30 rounded text-xs text-pink-300 animate-in slide-in-from-right-4 fade-in duration-300">
                                <div className="flex justify-between mb-1">
                                    <span>Dst Bytes</span>
                                    <span className="font-bold">94%</span>
                                </div>
                                <div className="w-full h-1 bg-pink-900/50">
                                    <div className="h-full bg-pink-500 w-[94%]" />
                                </div>
                                <div className="flex justify-between mt-2 mb-1">
                                    <span>Duration</span>
                                    <span className="font-bold">82%</span>
                                </div>
                                <div className="w-full h-1 bg-pink-900/50">
                                    <div className="h-full bg-pink-500 w-[82%]" />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* TERMINAL LOGS */}
                    <div className="flex-1 bg-black border border-slate-700 clip-corner p-4 font-mono text-[10px] text-green-400 overflow-hidden relative">
                        <div className="absolute top-0 left-0 right-0 h-6 bg-slate-800 flex items-center px-2 text-slate-400 text-[9px] tracking-widest border-b border-slate-700">
                            <Terminal className="w-3 h-3 mr-2" /> SYSTEM_LOGS.log
                        </div>
                        <div className="mt-6 flex flex-col gap-2 opacity-80">
                            {logs.map((log, i) => (
                                <div key={i} className="whitespace-pre-wrap animate-in fade-in slide-in-from-left-2 duration-300">
                                    {log}
                                </div>
                            ))}
                            <div className="w-2 h-4 bg-green-500 animate-pulse mt-2" />
                        </div>
                    </div>

                </div>
            </div>

            {/* CUSTOM STYLES for Tech look */}
            <style jsx>{`
                .clip-corner {
                    clip-path: polygon(
                        0 0, 
                        100% 0, 
                        100% calc(100% - 20px), 
                        calc(100% - 20px) 100%, 
                        0 100%
                    );
                }
                .clip-hex {
                    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
                }
                @keyframes flow-out {
                    0% { left: 50%; width: 0; opacity: 1; }
                    100% { left: 100%; width: 0; opacity: 0; }
                }
                @keyframes flow-in {
                    0% { left: 100%; width: 0; opacity: 0; }
                    100% { left: 50%; width: 0; opacity: 1; } 
                }
                /* Hacky way to reuse flow anims with rotate */
                .animate-flow-out {
                    animation: flowNode 1.5s infinite linear;
                }
                .animate-flow-in {
                    animation: flowNodeReverse 1.5s infinite linear;
                }
                @keyframes flowNode {
                    0% { transform: translateX(0); opacity: 1; }
                    100% { transform: translateX(140px); opacity: 0; }
                }
                @keyframes flowNodeReverse {
                    0% { transform: translateX(140px); opacity: 0; }
                    100% { transform: translateX(0); opacity: 1; }
                }
            `}</style>
        </div>
    );
}
