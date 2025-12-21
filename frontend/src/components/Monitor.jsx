import React, { useEffect, useState, useRef } from 'react';
import { io } from 'socket.io-client';
import { Terminal, Activity, Server, Users, Layers, Filter, Clock, Hash, Trash2 } from 'lucide-react';

const SOCKET_URL = 'http://localhost:5000';

const CHANNELS = [
    { id: 'all', label: 'All Events', icon: Layers },
    { id: 'FL_Server', label: 'FL Server', icon: Server },
    { id: 'Client_0', label: 'Client 0', icon: Users },
    { id: 'Client_1', label: 'Client 1', icon: Users },
    { id: 'Client_2', label: 'Client 2', icon: Users },
    { id: 'Client_3', label: 'Client 3', icon: Users },
];

export default function Monitor() {
    const [logs, setLogs] = useState({
        all: [],
        FL_Server: [],
        Client_0: [],
        Client_1: [],
        Client_2: [],
        Client_3: [],
        System: []
    });
    const [activeTab, setActiveTab] = useState('all');
    const [autoScroll, setAutoScroll] = useState(true);
    const logsEndRef = useRef(null);

    useEffect(() => {
        const socket = io(SOCKET_URL);

        const addLog = (channel, msg) => {
            setLogs(prev => {
                const now = new Date();
                const timeStr = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

                // Construct Log Object
                const logEntry = {
                    id: Date.now() + Math.random(),
                    time: timeStr,
                    msg: typeof msg === 'string' ? msg : JSON.stringify(msg),
                    source: channel || 'System'
                };

                const newLogs = { ...prev };
                // Keep last 1000 logs in 'all'
                newLogs.all = [...(newLogs.all || []), logEntry].slice(-1000);

                // Add to specific channel
                if (channel && newLogs[channel] !== undefined) {
                    newLogs[channel] = [...(newLogs[channel] || []), logEntry].slice(-200);
                }
                return newLogs;
            });
        };

        socket.on('connect', () => addLog('System', 'Connected to real-time event stream.'));
        socket.on('disconnect', () => addLog('System', 'Disconnected from event stream.'));
        socket.on('log', (data) => {
            if (!data) return;
            // Parse "[ID] Msg" pattern
            const match = typeof data === 'string' ? data.match(/^\[(.*?)\] (.*)/) : null;
            if (match) {
                addLog(match[1], match[2]);
            } else {
                addLog('System', data.toString());
            }
        });

        return () => socket.disconnect();
    }, []);

    // Auto-scroll logic
    useEffect(() => {
        if (autoScroll && logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs, activeTab, autoScroll]);

    const activeLogs = logs[activeTab] || [];

    // Helper to style log messages
    const renderLogMessage = (text) => {
        let className = "text-slate-300";
        if (/error/i.test(text)) className = "text-red-400 font-semibold";
        else if (/warning/i.test(text)) className = "text-amber-400";
        else if (/Round \d+/.test(text)) className = "text-indigo-400 font-bold";
        else if (/accuracy|loss/.test(text)) className = "text-emerald-400";

        return <span className={className}>{text}</span>;
    };

    const clearLogs = () => {
        if (confirm('Clear all logs for this view?')) {
            setLogs(prev => ({ ...prev, [activeTab]: [] }));
        }
    };

    return (
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-xl overflow-hidden flex flex-col h-[600px] transition-colors">
            {/* Top Bar */}
            <div className="bg-slate-50 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-indigo-500/10 rounded-lg">
                        <Terminal className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                    </div>
                    <div>
                        <h3 className="font-bold text-slate-800 dark:text-slate-100 text-sm">System Monitor</h3>
                        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                            </span>
                            Real-time
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setAutoScroll(!autoScroll)}
                        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${autoScroll ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300' : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400'}`}
                    >
                        Auto-scroll: {autoScroll ? 'ON' : 'OFF'}
                    </button>
                    <button
                        onClick={clearLogs}
                        className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md transition-colors"
                        title="Clear Logs"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex px-4 pt-2 gap-2 overflow-x-auto bg-slate-50 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 no-scrollbar">
                {CHANNELS.map(tab => {
                    const Icon = tab.icon;
                    const isActive = activeTab === tab.id;
                    return (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`
                                flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-t-lg transition-all border-t border-x border-b-0
                                ${isActive
                                    ? 'bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 border-slate-200 dark:border-slate-800 translate-y-[1px]'
                                    : 'bg-transparent text-slate-500 dark:text-slate-500 border-transparent hover:text-slate-700 dark:hover:text-slate-300'}
                            `}
                        >
                            <Icon className="w-3.5 h-3.5" />
                            {tab.label}
                            {activeTab !== tab.id && logs[tab.id]?.length > 0 && (
                                <span className="ml-1 w-1.5 h-1.5 bg-indigo-500 rounded-full"></span>
                            )}
                        </button>
                    );
                })}
            </div>

            {/* Log Viewer Area */}
            <div className="flex-1 overflow-y-auto p-0 bg-white dark:bg-slate-900 scroll-smooth">
                {activeLogs.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-slate-400 dark:text-slate-600">
                        <Activity className="w-12 h-12 mb-3 opacity-20" />
                        <p className="text-sm">No activity recorded for this channel.</p>
                    </div>
                ) : (
                    <table className="w-full text-left text-xs font-mono border-collapse">
                        <thead className="sticky top-0 bg-slate-50 dark:bg-slate-950 shadow-sm z-10 text-slate-500 dark:text-slate-400">
                            <tr>
                                <th className="py-2 px-4 w-24 font-semibold border-b border-slate-200 dark:border-slate-800"><Clock className="w-3 h-3 inline mr-1" />Time</th>
                                <th className="py-2 px-4 w-32 font-semibold border-b border-slate-200 dark:border-slate-800"><Hash className="w-3 h-3 inline mr-1" />Source</th>
                                <th className="py-2 px-4 font-semibold border-b border-slate-200 dark:border-slate-800">Message</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 dark:divide-slate-800/50">
                            {activeLogs.map((log) => (
                                <tr key={log.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors group">
                                    <td className="py-1.5 px-4 text-slate-400 whitespace-nowrap align-top">{log.time}</td>
                                    <td className="py-1.5 px-4 whitespace-nowrap align-top">
                                        <span className={`
                                            px-1.5 py-0.5 rounded text-[10px] font-bold uppercase
                                            ${log.source === 'FL_Server' ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300' :
                                                log.source.includes('Client') ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300' :
                                                    'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'}
                                        `}>
                                            {log.source}
                                        </span>
                                    </td>
                                    <td className="py-1.5 px-4 text-slate-600 dark:text-slate-300 break-all align-top leading-relaxed">
                                        {renderLogMessage(log.msg)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
                <div ref={logsEndRef} />
            </div>
        </div>
    );
}
