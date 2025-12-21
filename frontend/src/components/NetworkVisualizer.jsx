import React, { useEffect, useState } from 'react';
import { Server, Users, Activity, Zap, AlertTriangle, XCircle, Info, Play, CheckCircle2 } from 'lucide-react';
import { io } from 'socket.io-client';

const SOCKET_URL = 'http://localhost:5000';

export default function NetworkVisualizer() {
    const [activeNodes, setActiveNodes] = useState(new Set());
    const [roundPopup, setRoundPopup] = useState(null);
    const [errorPopup, setErrorPopup] = useState(null);
    const [waitingPopup, setWaitingPopup] = useState(false);
    const [connectedPopup, setConnectedPopup] = useState(false); // New: Clients Connected
    const [latestRound, setLatestRound] = useState(0);
    const [isRoundRunning, setIsRoundRunning] = useState(false);

    useEffect(() => {
        const socket = io(SOCKET_URL);

        const handleLog = (data) => {
            // Extract ID from log "[Client_0] ..."
            const match = data.match(/^\[(.*?)\]/);
            if (match) {
                const nodeId = match[1];
                triggerActivity(nodeId);
            }

            // Detect Round START
            if (data.includes('starting...')) {
                const startMatch = data.match(/Round (\d+) starting/);
                if (startMatch) {
                    const rnd = parseInt(startMatch[1]);
                    setLatestRound(rnd);
                    setIsRoundRunning(true);
                    setLatestRound(rnd);
                    setIsRoundRunning(true);
                    setWaitingPopup(false);
                    setConnectedPopup(false);
                }
            }

            // Detect Round Completion
            if (data.includes('weights saved to')) {
                const roundMatch = data.match(/Round (\d+) weights saved/);
                if (roundMatch) {
                    const rnd = parseInt(roundMatch[1]);
                    setRoundPopup(`Round ${rnd}`);
                    setLatestRound(rnd);
                    setIsRoundRunning(false);
                    setTimeout(() => setRoundPopup(null), 4000);
                }
            }

            // Detect Server Waiting for Clients
            if (data.includes('Requesting initial parameters from one random client')) {
                setWaitingPopup(true);
                setConnectedPopup(false);
            }

            // Detect Clients Connected / Initial Parameters Received
            // Logic: Server receives initial params => Clients are online
            if (data.includes('Received initial parameters from one random client') ||
                data.includes('strategy sampled')) {
                setWaitingPopup(false);
                setConnectedPopup(true);

                // Show for 3 seconds then auto-dismiss (or keep until round start?)
                // User requested "popup like server connected... and then live animation"
                // Lets keep it brief or until round start. Auto-dismiss is cleaner for flow.
                setTimeout(() => setConnectedPopup(false), 4000);

                // Trigger Pulse on ALL nodes to show "Connection Established"
                ['Client_0', 'Client_1', 'Client_2', 'Client_3', 'FL_Server'].forEach(id => {
                    triggerActivity(id);
                });
            }

            // Detect Errors
            const lowerData = data.toLowerCase();

            // 1. Critical process failures
            if (data.includes('exited with code') && !data.includes('exited with code 0')) {
                setErrorPopup("Process Crashed");
                return;
            }

            // 2. Scan content for specific error signatures
            // We strip the "[Client_X] ERROR:" prefix added by server.js
            const cleanMessage = lowerData.replace(/^\[.*?\]\s*(error:)?\s*/, '');

            const isNoisyLog =
                cleanMessage.includes('info') ||
                cleanMessage.includes('warning') ||
                cleanMessage.includes('deprecated') ||
                cleanMessage.includes('futurewarning') ||
                cleanMessage.includes('tensorflow') ||
                cleanMessage.includes('onednn');

            if (!isNoisyLog) {
                if (cleanMessage.includes('traceback') ||
                    cleanMessage.includes('exception') ||
                    (cleanMessage.includes('error') && !cleanMessage.includes('standard error'))) {

                    setErrorPopup("Process Error Detected");
                }
            }
        };

        socket.on('log', handleLog);

        return () => socket.disconnect();
    }, []);

    const triggerActivity = (nodeId) => {
        setActiveNodes(prev => {
            const next = new Set(prev);
            next.add(nodeId);
            return next;
        });

        setTimeout(() => {
            setActiveNodes(prev => {
                const next = new Set(prev);
                next.delete(nodeId);
                return next;
            });
        }, 500);
    };

    const getNodeColor = (id) => {
        if (activeNodes.has(id)) return 'text-emerald-400 drop-shadow-[0_0_15px_rgba(52,211,153,0.8)] scale-110';
        return 'text-slate-600';
    };

    const getBorderColor = (id) => {
        if (activeNodes.has(id)) return 'border-emerald-500/50 bg-emerald-500/10 shadow-[0_0_30px_rgba(52,211,153,0.2)]';
        return 'border-slate-700 bg-slate-800/50';
    };

    const clients = [0, 1, 2, 3];
    const radius = 140;

    return (
        <div className="relative h-[450px] w-full bg-slate-950 rounded-xl border border-slate-800 overflow-hidden flex items-center justify-center shadow-inner group">
            {/* Background Grid */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(30,41,59,0.3)_1px,transparent_1px),linear-gradient(90deg,rgba(30,41,59,0.3)_1px,transparent_1px)] bg-[size:40px_40px] opacity-20"></div>

            {/* Connections Layer (SVG) */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <svg className="w-[600px] h-[450px] overflow-visible">
                    <defs>
                        <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#0f172a" stopOpacity="0" />
                            <stop offset="50%" stopColor="#34d399" stopOpacity="1" />
                            <stop offset="100%" stopColor="#0f172a" stopOpacity="0" />
                        </linearGradient>
                        <filter id="neonGlow">
                            <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
                            <feMerge>
                                <feMergeNode in="coloredBlur" />
                                <feMergeNode in="SourceGraphic" />
                            </feMerge>
                        </filter>
                    </defs>
                    {clients.map((id, index) => {
                        const angle = (index / clients.length) * 2 * Math.PI;
                        const x = Math.cos(angle) * radius + 300;
                        const y = Math.sin(angle) * radius + 225;
                        const isActive = activeNodes.has(`Client_${id}`) || activeNodes.has('FL_Server');

                        return (
                            <g key={`line-${id}`}>
                                {/* Base Line */}
                                <line
                                    x1="300" y1="225"
                                    x2={x} y2={y}
                                    stroke={isActive ? '#34d399' : '#1e293b'}
                                    strokeWidth={isActive ? 2 : 1}
                                    strokeOpacity={isActive ? 0.8 : 0.3}
                                    strokeDasharray={isActive ? "5,5" : "0"}
                                    className="transition-all duration-300"
                                >
                                    {isActive && (
                                        <animate
                                            attributeName="stroke-dashoffset"
                                            from="100"
                                            to="0"
                                            dur="1s"
                                            repeatCount="indefinite"
                                        />
                                    )}
                                </line>

                                {/* Moving Particle Packet */}
                                {isActive && (
                                    <circle r="4" fill="#10b981" filter="url(#neonGlow)">
                                        <animateMotion
                                            dur="0.6s"
                                            repeatCount="indefinite"
                                            keyPoints="0;1"
                                            keyTimes="0;1"
                                            calcMode="linear"
                                            path={`M 300 225 L ${x} ${y}`}
                                        />
                                    </circle>
                                )}
                            </g>
                        );
                    })}
                </svg>
            </div>

            {/* Central Server Node */}
            <div className={`absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-20 transition-all duration-500 ${activeNodes.has('FL_Server') ? 'scale-110' : 'scale-100'}`}>
                {/* Outer Ring */}
                <div className={`absolute -inset-4 rounded-full border border-dashed transition-all duration-1000 ${activeNodes.has('FL_Server') ? 'border-emerald-500/30 animate-[spin_10s_linear_infinite]' : 'border-slate-700/30'
                    }`}></div>

                <div className={`relative p-6 rounded-full border-2 bg-slate-900/80 backdrop-blur-md shadow-2xl transition-all duration-300 ${getBorderColor('FL_Server')}`}>
                    {activeNodes.has('FL_Server') && (
                        <div className="absolute inset-0 rounded-full bg-emerald-500/20 animate-ping"></div>
                    )}
                    <Server className={`w-12 h-12 transition-all duration-300 ${getNodeColor('FL_Server')}`} />
                </div>

                <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 whitespace-nowrap">
                    <div className="px-3 py-1 bg-black/60 rounded-full border border-slate-700/50 backdrop-blur text-[10px] font-bold text-emerald-400 tracking-wider shadow-lg">
                        FL SERVER
                    </div>
                </div>
            </div>

            {/* Client Nodes */}
            {clients.map((id, index) => {
                const angle = (index / clients.length) * 2 * Math.PI;
                const x = Math.cos(angle) * radius;
                const y = Math.sin(angle) * radius;
                const nodeId = `Client_${id}`;

                return (

                    <div
                        key={nodeId}
                        className={`absolute left-1/2 top-1/2 rounded-xl border-2 transition-all duration-500 z-10 ${getBorderColor(nodeId)}`}
                        style={{
                            transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`,
                        }}
                    >
                        <div className={`p-3 bg-slate-900/90 backdrop-blur-sm rounded-xl relative overflow-hidden group-hover:scale-105 transition-transform duration-300`}>
                            {/* Inner Scanline/Highlight */}
                            {activeNodes.has(nodeId) && (
                                <div className="absolute inset-0 bg-gradient-to-tr from-emerald-500/0 via-emerald-500/10 to-emerald-500/0 animate-pulse"></div>
                            )}

                            <ActivePulse active={activeNodes.has(nodeId)} />
                            <Users className={`w-6 h-6 transition-all duration-300 relative z-10 ${getNodeColor(nodeId)}`} />
                        </div>

                        <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 whitespace-nowrap opacity-70 group-hover:opacity-100 transition-opacity">
                            <span className="text-[9px] font-mono text-slate-400 bg-slate-900/80 px-2 py-0.5 rounded border border-slate-800">CLIENT {id}</span>
                        </div>
                    </div>
                );

            })}

            {/* Round Completion Popup */}
            {roundPopup && (
                <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 z-50 flex items-center justify-center pointer-events-none">
                    <div className="relative">
                        <div className="absolute -inset-4 bg-red-500/50 blur-xl animate-pulse rounded-full"></div>
                        <div className="relative bg-gradient-to-r from-red-600 to-orange-600 text-white px-10 py-6 rounded-2xl shadow-[0_0_60px_rgba(220,38,38,0.6)] border-2 border-white/20 animate-bounce-in transform scale-110">
                            <div className="text-4xl font-black flex items-center gap-4 uppercase tracking-tighter drop-shadow-lg">
                                <Zap className="w-10 h-10 fill-yellow-300 text-yellow-300 animate-pulse" />
                                {roundPopup}
                                <span className="bg-white/20 px-2 rounded text-red-100">DONE</span>
                            </div>
                            <div className="text-center text-sm font-bold text-red-100 mt-2 tracking-widest uppercase">
                                Global Aggregation Complete
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Error Popup */}
            {errorPopup && (
                <div className="absolute inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
                    <div className="bg-slate-900 border-2 border-red-500 rounded-2xl p-8 shadow-[0_0_50px_rgba(239,68,68,0.4)] max-w-md text-center relative pointer-events-auto">
                        <div className="mx-auto w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mb-4">
                            <AlertTriangle className="w-8 h-8 text-red-500" />
                        </div>
                        <h3 className="text-2xl font-bold text-white mb-2">Training Interrupted</h3>
                        <p className="text-red-300 mb-6 font-mono text-sm bg-red-950/50 p-3 rounded border border-red-900/50">
                            {errorPopup}
                        </p>
                        <p className="text-slate-400 text-sm mb-6">
                            An error occurred in the network process. Please check the terminal logs for details and restart the server/clients.
                        </p>
                        <button
                            onClick={() => setErrorPopup(null)}
                            className="bg-red-600 hover:bg-red-500 text-white font-bold py-2 px-6 rounded-lg transition-colors flex items-center gap-2 mx-auto"
                        >
                            <XCircle className="w-4 h-4" />
                            Dismiss & Restart
                        </button>
                    </div>
                </div>
            )}

            {/* Waiting for Checks (Start Clients) Popup */}
            {waitingPopup && !errorPopup && !connectedPopup && (
                <div className="absolute inset-0 z-[55] flex items-center justify-center bg-black/40 backdrop-blur-[2px] animate-in fade-in duration-200">
                    <div className="bg-slate-900 border-2 border-blue-500 rounded-2xl p-6 shadow-[0_0_50px_rgba(59,130,246,0.4)] max-w-sm text-center relative pointer-events-auto transform scale-100 hover:scale-105 transition-transform">
                        <div className="mx-auto w-14 h-14 bg-blue-500/20 rounded-full flex items-center justify-center mb-4 animate-bounce">
                            <Play className="w-7 h-7 text-blue-400 fill-blue-400" />
                        </div>
                        <h3 className="text-xl font-bold text-white mb-2">Server Ready!</h3>
                        <p className="text-blue-200 text-sm mb-4">
                            The server is waiting for clients to connect.
                        </p>
                        <div className="bg-blue-900/40 border border-blue-500/30 rounded p-3 mb-4">
                            <p className="text-xs font-mono text-blue-300 font-bold uppercase tracking-wider">Action Required</p>
                            <p className="text-white font-bold mt-1">PLEASE START YOUR CLIENTS</p>
                        </div>
                        <button
                            onClick={() => setWaitingPopup(false)}
                            className="text-slate-500 hover:text-white text-xs underline"
                        >
                            Dismiss
                        </button>
                    </div>
                </div>
            )}

            {/* Clients Connected Popup */}
            {connectedPopup && !errorPopup && (
                <div className="absolute inset-0 z-[55] flex items-center justify-center pointer-events-none">
                    <div className="relative">
                        <div className="absolute -inset-4 bg-emerald-500/50 blur-xl animate-pulse rounded-full"></div>
                        <div className="relative bg-gradient-to-r from-emerald-600 to-green-600 text-white px-8 py-5 rounded-2xl shadow-[0_0_60px_rgba(16,185,129,0.6)] border-2 border-white/20 animate-bounce-in transform scale-105">
                            <div className="text-2xl font-black flex items-center gap-3 uppercase tracking-tight drop-shadow-lg">
                                <CheckCircle2 className="w-8 h-8 text-white animate-bounce" />
                                Clients Connected
                            </div>
                            <div className="text-center text-xs font-bold text-emerald-100 mt-1 tracking-widest uppercase">
                                Network Established • Training Starting
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Top Right: Status */}
            <div className="absolute top-4 right-4 flex flex-col items-end gap-2">
                <div className="flex items-center gap-2 text-xs text-slate-500 bg-slate-900/50 px-3 py-1.5 rounded-full border border-slate-800">
                    <Activity className="w-4 h-4 text-emerald-500 animate-pulse" />
                    <span>Live Network Activity</span>
                </div>

                {/* Round Counter / Status */}
                {latestRound > 0 && (
                    <div className={`flex items-center gap-2 text-xs font-bold px-3 py-1.5 rounded-full border shadow-[0_0_10px_rgba(0,0,0,0.2)] transition-colors duration-500
                        ${isRoundRunning
                            ? 'text-blue-200 bg-blue-900/40 border-blue-500/30'
                            : 'text-red-200 bg-red-900/40 border-red-500/30'
                        }
                    `}>
                        <span className={`w-2 h-2 rounded-full animate-ping ${isRoundRunning ? 'bg-blue-400' : 'bg-red-500'}`}></span>
                        {isRoundRunning ? `Round ${latestRound} Running...` : `Round ${latestRound} Completed`}
                    </div>
                )}
            </div>
        </div>
    );
}

// Helper for Pulse Effect
function ActivePulse({ active }) {
    if (!active) return null;
    return (
        <span className="absolute -inset-2 rounded-xl bg-emerald-500/20 blur-md animate-pulse"></span>
    );
}
