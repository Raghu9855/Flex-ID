import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Play, Square, Terminal, ShieldAlert, BadgeInfo, FileImage, Activity, FileJson, BarChart, CheckCircle, Download, RefreshCw, History } from 'lucide-react';
import NetworkVisualizer from './NetworkVisualizer';

const API_URL = 'http://localhost:5000/api';
const IMG_BASE_URL = 'http://localhost:5000/results';

export default function TrainingControl() {
    // Manual Commands State
    const [serverCmd, setServerCmd] = useState('python ml/4_server.py --strategy fedavg --rounds 10');

    const [clientCmds, setClientCmds] = useState([
        { id: 'Client_0', cmd: 'python ml/client.py --cid 0 --batch_size 1024' },
        { id: 'Client_1', cmd: 'python ml/client.py --cid 1 --batch_size 1024' },
        { id: 'Client_2', cmd: 'python ml/client.py --cid 2 --batch_size 1024' },
        { id: 'Client_3', cmd: 'python ml/client.py --cid 3 --batch_size 1024' }
    ]);

    const [status, setStatus] = useState('Idle');

    // Checkpoints State
    const [checkpointMode, setCheckpointMode] = useState('no_attack');

    // Normal Checkpoints
    const [fedAvgCheckpoints, setFedAvgCheckpoints] = useState([]);
    const [fedProxCheckpoints, setFedProxCheckpoints] = useState([]);

    // Under Attack Checkpoints
    const [fedAvgAttackCheckpoints, setFedAvgAttackCheckpoints] = useState([]);

    const [fedProxAttackCheckpoints, setFedProxAttackCheckpoints] = useState([]);
    const [historyFiles, setHistoryFiles] = useState([]);
    const [metricsSummary, setMetricsSummary] = useState(null);

    const updateClientCmd = (idx, value) => {
        const newCmds = [...clientCmds];
        newCmds[idx].cmd = value;
        setClientCmds(newCmds);
    };

    const runCommand = async (id, commandLine) => {
        try {
            const parts = commandLine.trim().split(/\s+/);
            const cmd = parts[0];
            const args = parts.slice(1);

            await axios.post(`${API_URL}/run-command`, {
                id,
                command: cmd,
                args
            });
            console.log(`Started ${id}`);
            setStatus(`Running ${id}`);
        } catch (e) {
            console.error(`Failed to start ${id}`, e);
            alert(`Failed to start ${id}`);
            setStatus(`Failed to start ${id}`);
        }
    };

    const stopTraining = async () => {
        try {
            await axios.post(`${API_URL}/stop-all`);
            setStatus('Stopped');
        } catch (e) {
            console.error(e);
            setStatus('Error stopping processes');
        }
    };

    // Poll for checkpoints
    useEffect(() => {
        const fetchArtifacts = async () => {
            const fetchFolder = async (path, setter) => {
                try {
                    const res = await axios.get(`${API_URL}/files`, { params: { path } });
                    if (res.data && res.data.children) {
                        const weights = res.data.children
                            .filter(f => f.name.includes('weights.pkl'))
                            .sort((a, b) => {
                                const getRound = n => parseInt(n.match(/round-(\d+)/)?.[1] || 0);
                                return getRound(a.name) - getRound(b.name);
                            });
                        setter(weights);
                    }
                } catch (e) {
                    setter([]); // Clear on error or not found
                }
            };

            // Normal
            fetchFolder('results/fedavgeachround', setFedAvgCheckpoints);
            fetchFolder('results/fedproxeachround', setFedProxCheckpoints);

            // Under Attack
            fetchFolder('results/fedunderattack', setFedAvgAttackCheckpoints);
            fetchFolder('results/fedunderattack', setFedAvgAttackCheckpoints);
            fetchFolder('results/fedproxunderattack', setFedProxAttackCheckpoints);

            // History Files
            try {
                const hRes = await axios.get(`${API_URL}/files`, { params: { path: 'results' } });
                if (hRes.data && hRes.data.children) {
                    const hFiles = hRes.data.children
                        .filter(f => f.type === 'file' && f.name.includes('history.pkl'))
                        .map(f => f.name);
                    setHistoryFiles(hFiles);
                }
            } catch (e) {
                // ignore
            }

            // Metrics Summary
            try {
                const mRes = await axios.get(`${IMG_BASE_URL}/metrics_summary.json?t=${Date.now()}`);
                setMetricsSummary(mRes.data);
            } catch (e) {
                // ignore
            }
        };

        const interval = setInterval(fetchArtifacts, 5000);
        fetchArtifacts(); // Initial load
        return () => clearInterval(interval);
    }, []);

    // Helper Component for Checkpoints List
    const CheckpointList = ({ title, files, folder }) => (
        <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700/50">
            <h3 className="text-sm font-bold text-slate-400 mb-3 uppercase tracking-wider flex items-center gap-2">
                <FileJson className="w-4 h-4" /> {title}
            </h3>
            {files.length === 0 ? (
                <div className="text-center py-6 text-slate-600 border border-dashed border-slate-700/50 rounded-lg text-sm">
                    No checkpoints found.
                </div>
            ) : (
                <div className="flex gap-3 overflow-x-auto pb-2 custom-scrollbar">
                    {files.map((file, idx) => (
                        <div key={idx} className="min-w-[160px] bg-slate-800 p-3 rounded-lg border border-white/5 hover:border-indigo-500/50 transition-colors group relative">
                            <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                <a href={`http://localhost:5000/results/${folder}/${file.name}`} download className="p-1.5 bg-indigo-600 rounded-md text-white hover:bg-indigo-500 block">
                                    <Download className="w-3 h-3" />
                                </a>
                            </div>
                            <div className="w-8 h-8 bg-indigo-500/10 rounded-full flex items-center justify-center mb-2">
                                <FileJson className="w-4 h-4 text-indigo-400" />
                            </div>
                            <div className="font-mono text-xs text-indigo-300 mb-1">Round {file.name.match(/round-(\d+)/)?.[1]}</div>
                            <div className="text-[10px] text-slate-500 truncate" title={file.name}>{file.name}</div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );

    const HistoryFileItem = ({ name, label, exists }) => (
        <div className={`flex items-center justify-between p-4 rounded-lg border transition-all ${exists ? 'bg-slate-800 border-slate-600 hover:border-amber-500/50' : 'bg-slate-900/50 border-slate-800 opacity-60'}`}>
            <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${exists ? 'bg-amber-500/10 text-amber-500' : 'bg-slate-800 text-slate-600'}`}>
                    <FileJson className="w-5 h-5" />
                </div>
                <div>
                    <h4 className={`text-sm font-bold ${exists ? 'text-slate-200' : 'text-slate-500'}`}>{label}</h4>
                    <p className="text-[10px] text-slate-500 font-mono">{name}</p>
                </div>
            </div>
            {exists && (
                <a
                    href={`http://localhost:5000/results/${name}`}
                    download
                    className="p-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg transition-colors shadow-lg"
                    title="Download History"
                >
                    <Download className="w-4 h-4" />
                </a>
            )}
            {!exists && (
                <span className="text-[10px] text-slate-600 uppercase font-bold tracking-wider px-2 py-1 bg-slate-900 rounded">
                    Not Found
                </span>
            )}
        </div>
    );

    return (
        <div className="space-y-8">
            {/* Top Row: Visualizer & Controls */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 items-stretch">
                {/* 1. Network Process Visualizer */}
                <div className="flex flex-col space-y-4 h-full">
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        <Activity className="w-6 h-6 text-emerald-400" />
                        Live Network Process
                    </h2>
                    <div className="flex-1">
                        <NetworkVisualizer />
                    </div>
                </div>

                {/* 2. Manual Controls */}
                <div className="flex flex-col space-y-4">
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        <Terminal className="w-5 h-5 text-indigo-400" />
                        Manual Execution Control
                    </h2>

                    <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg flex flex-col h-[450px] overflow-hidden">
                        <p className="text-slate-500 text-[10px] mb-4 flex items-start gap-2 bg-slate-700/50 p-2 rounded border border-slate-600/30 shrink-0">
                            <BadgeInfo className="w-3 h-3 text-blue-400 shrink-0 mt-0.5" />
                            Order: Start Server → Start Clients
                        </p>

                        {/* Server Section */}
                        <div className="mb-4 p-3 bg-slate-900/50 rounded-lg border border-emerald-500/20 shrink-0">
                            <h3 className="text-xs font-bold text-emerald-400 mb-2 flex items-center gap-2 uppercase">
                                1. Server
                                <span className="text-[10px] font-normal text-slate-500 lowercase">(port 8080)</span>
                            </h3>
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={serverCmd}
                                    onChange={(e) => setServerCmd(e.target.value)}
                                    className="flex-1 font-mono text-xs bg-black/40 border border-slate-600 text-green-300 p-2 rounded focus:border-emerald-500 focus:outline-none"
                                />
                                <button
                                    onClick={() => runCommand('FL_Server', serverCmd)}
                                    className="px-4 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded flex items-center gap-1.5 transition-colors text-xs"
                                >
                                    <Play className="w-3 h-3" /> Run
                                </button>
                            </div>
                            <div className="mt-2 flex gap-2">
                                <button
                                    onClick={() => !serverCmd.includes('--attack') && setServerCmd(serverCmd + ' --attack')}
                                    className="text-[10px] bg-red-900/30 text-red-400 border border-red-900/50 px-2 py-1 rounded hover:bg-red-900/50 transition-colors"
                                >
                                    + Add Attack Flag
                                </button>
                            </div>
                        </div>

                        {/* Clients Section */}
                        <div className="mb-4 flex-1 overflow-auto custom-scrollbar">
                            <h3 className="text-xs font-bold text-blue-400 mb-2 uppercase">2. Clients</h3>
                            <div className="space-y-2">
                                {clientCmds.map((client, idx) => (
                                    <div key={client.id} className="flex items-center gap-2">
                                        <span className="font-mono text-slate-500 text-[10px] w-12 uppercase">{client.id}</span>
                                        <input
                                            type="text"
                                            value={client.cmd}
                                            onChange={(e) => updateClientCmd(idx, e.target.value)}
                                            className="flex-1 font-mono text-xs bg-black/40 border border-slate-600 text-blue-300 p-1.5 rounded focus:border-blue-500 focus:outline-none"
                                        />
                                        <button
                                            onClick={() => runCommand(client.id, client.cmd)}
                                            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded flex items-center gap-1 text-[10px] transition-colors"
                                        >
                                            <Play className="w-2.5 h-2.5" /> Start
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Controls */}
                        <div className="pt-3 border-t border-slate-700 mt-auto shrink-0">
                            <button
                                onClick={stopTraining}
                                className="w-full py-2.5 bg-red-600 hover:bg-red-500 text-white font-bold rounded-lg shadow-lg hover:shadow-red-500/20 transition-all flex items-center justify-center gap-2 text-xs"
                            >
                                <Square className="w-4 h-4 fill-current" /> STOP ALL PROCESSES
                            </button>
                            <div className="text-center text-slate-500 mt-2 text-[10px]">
                                Status: <span className="font-bold text-white uppercase">{status}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* 3. Client Data Analysis (Before/After SMOTE) */}
            <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
                <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                    <FileImage className="w-5 h-5 text-amber-400" />
                    Client Data Analysis (Balancing)
                </h2>

                <div className="space-y-12">
                    {[0, 1, 2, 3].map(i => (
                        <div key={i} className="border-b border-slate-700 pb-8 last:border-0 last:pb-0">
                            <h3 className="text-lg font-bold text-slate-300 mb-4 flex items-center gap-2">
                                <BadgeInfo className="w-4 h-4 text-slate-500" />
                                Client {i}
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* Before */}
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between text-xs text-red-300 uppercase font-bold tracking-wider mb-2">
                                        <span>Before SMOTE</span>
                                        <span className="bg-red-900/50 px-2 py-0.5 rounded">Imbalanced</span>
                                    </div>
                                    <div className="relative group bg-slate-900 rounded-lg overflow-hidden border border-red-900/30 hover:border-red-500/50 transition-colors min-h-[200px] flex items-center justify-center">
                                        <ImageWithRetry
                                            src={`${IMG_BASE_URL}/client_${i}_dist_before_smote.png`}
                                            alt={`Client ${i} Before`}
                                        />
                                    </div>
                                </div>

                                {/* After */}
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between text-xs text-emerald-300 uppercase font-bold tracking-wider mb-2">
                                        <span>After Balancing</span>
                                        <span className="bg-emerald-900/50 px-2 py-0.5 rounded">Balanced</span>
                                    </div>
                                    <div className="relative group bg-slate-900 rounded-lg overflow-hidden border border-emerald-900/30 hover:border-emerald-500/50 transition-colors min-h-[200px] flex items-center justify-center">
                                        <ImageWithRetry
                                            src={`${IMG_BASE_URL}/client_${i}_dist_after_hybrid.png`}
                                            alt={`Client ${i} After`}
                                            fallbackColor="border-emerald-900/30"
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* 4. Global Model Checkpoints */}
            <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        <FileJson className="w-5 h-5 text-indigo-400" />
                        Global Model Checkpoints
                    </h2>

                    {/* Tabs */}
                    <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-700">
                        <button
                            onClick={() => setCheckpointMode('no_attack')}
                            className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all ${checkpointMode === 'no_attack'
                                ? 'bg-indigo-600 text-white shadow-lg'
                                : 'text-slate-400 hover:text-white hover:bg-white/5'
                                }`}
                        >
                            NO ATTACK
                        </button>
                        <button
                            onClick={() => setCheckpointMode('under_attack')}
                            className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all ${checkpointMode === 'under_attack'
                                ? 'bg-red-600 text-white shadow-lg'
                                : 'text-slate-400 hover:text-white hover:bg-white/5'
                                }`}
                        >
                            UNDER ATTACK
                        </button>
                    </div>
                </div>

                {checkpointMode === 'no_attack' ? (
                    <div className="space-y-4 animate-fade-in-up">
                        <CheckpointList
                            title="FedAvg Strategy"
                            files={fedAvgCheckpoints}
                            folder="fedavgeachround"
                        />
                        <CheckpointList
                            title="FedProx Strategy"
                            files={fedProxCheckpoints}
                            folder="fedproxeachround"
                        />
                    </div>
                ) : (
                    <div className="space-y-4 animate-fade-in-up">
                        <CheckpointList
                            title="FedAvg (Under Attack)"
                            files={fedAvgAttackCheckpoints}
                            folder="fedunderattack"
                        />
                        <CheckpointList
                            title="FedProx (Under Attack)"
                            files={fedProxAttackCheckpoints}
                            folder="fedproxunderattack"
                        />
                    </div>
                )}
            </div>

            {/* NEW: Training History Archives */}
            <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg mt-8">
                <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                    <History className="w-5 h-5 text-amber-500" />
                    Training History Archives
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {checkpointMode === 'no_attack' ? (
                        <>
                            <HistoryFileItem
                                name="fedavg_history.pkl"
                                label="FedAvg History"
                                exists={historyFiles.includes('fedavg_history.pkl')}
                            />
                            <HistoryFileItem
                                name="fedprox_history.pkl"
                                label="FedProx History"
                                exists={historyFiles.includes('fedprox_history.pkl')}
                            />
                        </>
                    ) : (
                        <>
                            <HistoryFileItem
                                name="fedavg_underattack_history.pkl"
                                label="FedAvg Attack History"
                                exists={historyFiles.includes('fedavg_underattack_history.pkl')}
                            />
                            <HistoryFileItem
                                name="fedprox_underattack_history.pkl"
                                label="FedProx Attack History"
                                exists={historyFiles.includes('fedprox_underattack_history.pkl')}
                            />
                        </>
                    )}
                </div>
            </div>

            {/* 5. Final Global Results */}
            <div className="bg-gradient-to-br from-slate-900 to-slate-800 p-1 rounded-xl shadow-2xl mt-8">
                <div className="bg-slate-900/90 backdrop-blur-xl p-8 rounded-lg border border-white/10">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                            <BarChart className="w-8 h-8 text-emerald-400" />
                            Final Global Results
                            <span className="text-sm font-normal text-slate-500 bg-slate-800 px-3 py-1 rounded-full border border-white/5 ml-auto">
                                Training Summary
                            </span>
                        </h2>

                        {/* Tabs (Synced with Checkpoints) */}
                        <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-700">
                            <button
                                onClick={() => setCheckpointMode('no_attack')}
                                className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all ${checkpointMode === 'no_attack'
                                    ? 'bg-emerald-600 text-white shadow-lg'
                                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                NO ATTACK
                            </button>
                            <button
                                onClick={() => setCheckpointMode('under_attack')}
                                className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all ${checkpointMode === 'under_attack'
                                    ? 'bg-red-600 text-white shadow-lg'
                                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                UNDER ATTACK
                            </button>
                        </div>
                    </div>

                    <div className="bg-black/40 rounded-xl overflow-hidden border border-white/10 p-4 relative group animate-fade-in-up">
                        <ImageWithRetry
                            src={`${IMG_BASE_URL}/${checkpointMode === 'no_attack' ? 'comparison_metrics.png' : 'comparison_metrics_underattack.png'}`}
                            alt={checkpointMode === 'no_attack' ? "Normal Results" : "Under Attack Results"}
                        />
                        <div className="absolute inset-x-0 bottom-0 p-4 bg-black/80 backdrop-blur opacity-0 group-hover:opacity-100 transition-opacity flex justify-center pointer-events-none group-hover:pointer-events-auto">
                            <a
                                href={`${IMG_BASE_URL}/${checkpointMode === 'no_attack' ? 'comparison_metrics.png' : 'comparison_metrics_underattack.png'}`}
                                download
                                className="bg-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-bold shadow-lg hover:bg-emerald-500 transition-colors"
                            >
                                Download Graph
                            </a>
                        </div>
                    </div>
                </div>
            </div>

            {/* Metrics Text Summary */}
            {metricsSummary && (
                <div className="mt-4 bg-slate-900/80 p-4 rounded-lg border border-slate-700/50 backdrop-blur-sm">
                    <div className="flex justify-between items-center mb-4 border-b border-slate-700/50 pb-2">
                        <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                            <Activity className="w-3 h-3 text-emerald-400" /> Performance Metrics
                        </h4>
                        <div className="flex gap-8 text-[10px] font-bold uppercase tracking-widest">
                            <span className="text-blue-400 border-b-2 border-blue-500/50 pb-0.5">FedAvg</span>
                            <span className="text-pink-400 border-b-2 border-pink-500/50 pb-0.5">FedProx</span>
                        </div>
                    </div>
                    <div className="space-y-3 font-mono text-xs">
                        <div className="flex justify-between items-center">
                            <span className="text-slate-400">Final Accuracy</span>
                            <div className="flex gap-8 w-32 justify-end">
                                <span className="text-blue-200">{metricsSummary[checkpointMode === 'no_attack' ? 'fedavg' : 'fedavg_attack']?.last_accuracy || '-'}</span>
                                <span className="text-pink-200">{metricsSummary[checkpointMode === 'no_attack' ? 'fedprox' : 'fedprox_attack']?.last_accuracy || '-'}</span>
                            </div>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-slate-400">Final F1 Score</span>
                            <div className="flex gap-8 w-32 justify-end">
                                <span className="text-blue-200">{metricsSummary[checkpointMode === 'no_attack' ? 'fedavg' : 'fedavg_attack']?.last_f1 || '-'}</span>
                                <span className="text-pink-200">{metricsSummary[checkpointMode === 'no_attack' ? 'fedprox' : 'fedprox_attack']?.last_f1 || '-'}</span>
                            </div>
                        </div>
                        <div className="flex justify-between items-center border-t border-slate-800/50 pt-2 mt-2">
                            <span className="text-slate-500">Rounds Completed</span>
                            <div className="flex gap-8 w-32 justify-end">
                                <span className="text-slate-400">{metricsSummary[checkpointMode === 'no_attack' ? 'fedavg' : 'fedavg_attack']?.rounds || '-'}</span>
                                <span className="text-slate-400">{metricsSummary[checkpointMode === 'no_attack' ? 'fedprox' : 'fedprox_attack']?.rounds || '-'}</span>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>

    );
}

function ImageWithRetry({ src, alt, fallbackColor = 'border-slate-700' }) {
    const [retryCount, setRetryCount] = useState(0);
    const [hasError, setHasError] = useState(false);

    useEffect(() => {
        if (hasError) {
            const timer = setTimeout(() => {
                setHasError(false); // Reset error to try again
                setRetryCount(c => c + 1); // Trigger re-render with new URL
            }, 3000); // Retry every 3s
            return () => clearTimeout(timer);
        }
    }, [hasError]);

    if (hasError) {
        return (
            <div className={`w-full h-full flex flex-col items-center justify-center p-8 text-center animate-pulse bg-slate-900 border ${fallbackColor} rounded-lg`}>
                <RefreshCw className="w-8 h-8 text-slate-600 mb-2 animate-spin" />
                <span className="text-xs text-slate-500">Generating chart...</span>
            </div>
        );
    }

    return (
        <img
            src={`${src}?retry=${retryCount}`}
            alt={alt}
            className="w-full h-auto object-contain hover:scale-105 transition-transform duration-500"
            onError={() => setHasError(true)}
        />
    );
}
