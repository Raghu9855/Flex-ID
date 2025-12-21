import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Database, FileDigit, CheckCircle, AlertCircle, X, Download, HardDrive, BarChart2, PieChart } from 'lucide-react';
import {
    Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';

const API_URL = 'http://localhost:5000/api';

// Custom Tooltip for Recharts
const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-slate-900/90 border border-white/10 p-3 rounded-lg shadow-xl backdrop-blur-md">
                <p className="text-slate-300 text-xs font-semibold mb-1">{label}</p>
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-400" />
                    <span className="text-emerald-400 font-mono font-bold">
                        {payload[0].value} samples
                    </span>
                </div>
            </div>
        );
    }
    return null;
};

export default function DataPrep() {
    const [status, setStatus] = useState('');
    const [error, setError] = useState('');
    const [report, setReport] = useState(null);
    const [parsedData, setParsedData] = useState([]);
    const [showReport, setShowReport] = useState(false);
    const [chartType, setChartType] = useState('radar'); // 'radar' or 'bar'

    const fetchReport = async () => {
        try {
            const res = await axios.get('http://localhost:5000/results/partition_report.txt');
            setReport(res.data);
            parseReportData(res.data);
            setShowReport(true);
        } catch (e) {
            console.error("Report not found", e);
            setError("Report not found. Ensure partitioning has been run.");
        }
    };

    const parseReportData = (text) => {
        const clients = [];
        const blocks = text.split('------------------------------');

        blocks.forEach(block => {
            if (!block.trim()) return;
            const lines = block.trim().split('\n');

            // Find the line that starts with "Client" (case-insensitive)
            // This handles the first block which includes the file header
            const clientLineIndex = lines.findIndex(line =>
                line.trim().toLowerCase().startsWith('client')
            );

            if (clientLineIndex === -1) return;

            // Extract client name from that line
            const clientName = lines[clientLineIndex].replace(':', '').trim();

            const clientData = {
                name: clientName,
                id: clientName.split(' ')[1] || '0',
                stats: { total: 0, train: 0, test: 0 },
                distribution: []
            };

            // Process lines starting from the client line
            lines.slice(clientLineIndex + 1).forEach(line => {
                if (line.includes('Total Samples')) clientData.stats.total = line.split(':')[1].trim();
                if (line.includes('Train Set')) clientData.stats.train = line.split(':')[1].trim();
                if (line.includes('Test Set')) clientData.stats.test = line.split(':')[1].trim();

                if (line.includes('Label Distribution')) {
                    const distStr = line.split('Label Distribution:')[1];
                    if (distStr) {
                        distStr.split(',').forEach(item => {
                            const [label, count] = item.split(':').map(s => s.trim());
                            if (label && count) {
                                clientData.distribution.push({
                                    subject: label,
                                    A: parseInt(count, 10),
                                    fullMark: 100 // Scale reference
                                });
                            }
                        });
                    }
                }
            });

            if (clientData.distribution.length > 0) {
                clients.push(clientData);
            }
        });

        setParsedData(clients);
    };

    useEffect(() => {
        fetchReport();
    }, []);

    const COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'];

    return (
        <div className="p-6 bg-slate-800 rounded-xl shadow-lg border border-slate-700">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <Database className="w-6 h-6 text-blue-400" />
                Data Partitioning Analysis
            </h2>

            {/* Status Messages */}
            {status && (
                <div className="mb-6 p-4 bg-emerald-900/20 border border-emerald-500/30 rounded-lg flex items-center gap-3 animate-in fade-in slide-in-from-top-2">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    <span className="text-emerald-200">{status}</span>
                </div>
            )}

            {error && (
                <div className="mb-6 p-4 bg-red-900/20 border border-red-500/30 rounded-lg flex items-center gap-3 animate-in fade-in slide-in-from-top-2">
                    <AlertCircle className="w-5 h-5 text-red-400" />
                    <span className="text-red-200">{error}</span>
                </div>
            )}

            {showReport ? (
                <div className="space-y-8 animate-in fade-in duration-500">

                    {/* Visualizations Section */}
                    <div className="grid grid-cols-12 gap-6">
                        {/* Summary Stats Column */}
                        <div className="col-span-12 xl:col-span-3 space-y-4">
                            <div className="flex items-center justify-between xl:justify-start gap-4 mb-2">
                                <h3 className="text-lg font-bold text-slate-300">Client Overview</h3>
                                <div className="flex bg-slate-900/50 p-1 rounded-lg border border-white/5">
                                    <button
                                        onClick={() => setChartType('radar')}
                                        className={`p-1.5 rounded transition-all ${chartType === 'radar' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}
                                        title="Radar View"
                                    >
                                        <PieChart className="w-4 h-4" />
                                    </button>
                                    <button
                                        onClick={() => setChartType('bar')}
                                        className={`p-1.5 rounded transition-all ${chartType === 'bar' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}
                                        title="Bar View"
                                    >
                                        <BarChart2 className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-4 max-h-[600px] overflow-y-auto custom-scrollbar pr-2">
                                {parsedData.map((client, idx) => (
                                    <div key={idx} className="bg-slate-900/40 p-4 rounded-xl border border-white/5 hover:border-blue-500/30 transition-colors">
                                        <div className="flex justify-between items-center mb-3">
                                            <span className="font-bold text-slate-200">{client.name}</span>
                                            <span className="text-xs bg-slate-800 text-slate-400 px-2 py-1 rounded font-mono">ID: {client.id}</span>
                                        </div>
                                        <div className="grid grid-cols-3 gap-2 text-center">
                                            <div className="p-2 bg-slate-800/50 rounded">
                                                <div className="text-[10px] text-slate-500 uppercase">Total</div>
                                                <div className="font-mono text-sm text-white">{client.stats.total}</div>
                                            </div>
                                            <div className="p-2 bg-slate-800/50 rounded">
                                                <div className="text-[10px] text-emerald-500/80 uppercase">Train</div>
                                                <div className="font-mono text-sm text-emerald-400">{client.stats.train}</div>
                                            </div>
                                            <div className="p-2 bg-slate-800/50 rounded">
                                                <div className="text-[10px] text-amber-500/80 uppercase">Test</div>
                                                <div className="font-mono text-sm text-amber-400">{client.stats.test}</div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Charts Grid */}
                        <div className="col-span-12 xl:col-span-9">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {parsedData.map((client, idx) => (
                                    <div key={idx} className="bg-slate-900/30 rounded-xl border border-white/5 p-4 relative group">
                                        <h4 className="absolute top-4 left-4 z-10 text-sm font-bold text-slate-400 flex items-center gap-2">
                                            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                                            {client.name} Distribution
                                        </h4>
                                        <div className="h-[250px] w-full mt-4">
                                            <ResponsiveContainer width="100%" height="100%">
                                                {chartType === 'radar' ? (
                                                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={client.distribution}>
                                                        <PolarGrid stroke="#334155" />
                                                        <PolarAngleAxis
                                                            dataKey="subject"
                                                            tick={{ fill: '#94a3b8', fontSize: 10 }}
                                                        />
                                                        <PolarRadiusAxis
                                                            angle={30}
                                                            domain={[0, 'auto']}
                                                            tick={{ fill: '#475569', fontSize: 10 }}
                                                            axisLine={false}
                                                        />
                                                        <Radar
                                                            name={client.name}
                                                            dataKey="A"
                                                            stroke="#3b82f6"
                                                            strokeWidth={2}
                                                            fill="#3b82f6"
                                                            fillOpacity={0.3}
                                                        />
                                                        <Tooltip content={<CustomTooltip />} />
                                                    </RadarChart>
                                                ) : (
                                                    <BarChart data={client.distribution} layout="vertical" margin={{ left: 40, right: 20 }}>
                                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                                                        <XAxis type="number" stroke="#94a3b8" fontSize={10} />
                                                        <YAxis
                                                            dataKey="subject"
                                                            type="category"
                                                            stroke="#94a3b8"
                                                            fontSize={10}
                                                            width={80}
                                                        />
                                                        <Tooltip content={<CustomTooltip />} />
                                                        <Bar dataKey="A" radius={[0, 4, 4, 0]} barSize={20}>
                                                            {client.distribution.map((entry, index) => (
                                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                                            ))}
                                                        </Bar>
                                                    </BarChart>
                                                )}
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Download Section */}
                    <div className="mt-8 pt-8 border-t border-slate-700">
                        <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                            <HardDrive className="w-6 h-6 text-emerald-400" />
                            Generated Client Partitions
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            {parsedData.map((client, i) => (
                                <div key={i} className="bg-gradient-to-br from-slate-800 to-slate-900 p-4 rounded-xl border border-white/5 hover:border-emerald-500/50 transition-all group relative overflow-hidden">
                                    <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-2xl -mr-12 -mt-12 group-hover:bg-emerald-500/10 transition-colors"></div>
                                    <div className="relative z-10">
                                        <div className="flex items-center justify-between mb-4">
                                            <div className="bg-emerald-500/20 p-2 rounded-lg">
                                                <FileDigit className="w-6 h-6 text-emerald-400" />
                                            </div>
                                            <span className="text-xs font-mono text-slate-500">.pkl</span>
                                        </div>
                                        <h4 className="text-lg font-bold text-white mb-1">Partition #{client.id}</h4>
                                        <p className="text-xs text-slate-400 mb-4">{client.name} Isolated Data</p>
                                        <a
                                            href={`http://localhost:5000/images/client_partition_${client.id}.pkl`}
                                            download={`client_partition_${client.id}.pkl`}
                                            className="flex items-center justify-center gap-2 w-full py-2 bg-white/5 hover:bg-emerald-500 text-slate-300 hover:text-white rounded-lg text-sm font-medium transition-all group-hover:translate-y-0"
                                        >
                                            <Download className="w-4 h-4" /> Download
                                        </a>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            ) : (
                <div className="text-center py-20">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                    <p className="text-slate-400">Loading partition analysis...</p>
                </div>
            )}
        </div>
    );
}
