import React from 'react';
import { Shield, Server, Users, BrainCircuit, Activity, Lock, Database, Code, Cpu, Lightbulb, PlayCircle } from 'lucide-react';
import ProjectWorkflow from './ProjectWorkflow';

export default function Home() {
    return (
        <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500">

            {/* Hero Section */}
            <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-blue-900/50 to-purple-900/50 border border-white/10 p-8 md:p-12">
                <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/20 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none"></div>
                <div className="relative z-10">
                    <h1 className="text-4xl md:text-5xl font-bold text-white mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                        Secure Federated Learning
                    </h1>
                    <h2 className="text-xl md:text-2xl text-blue-200 mb-6">
                        Intrusion Detection System (Flex-ID)
                    </h2>
                    <p className="text-slate-300 max-w-2xl text-lg leading-relaxed mb-8">
                        A robust, privacy-preserving framework for detecting network intrusions using Federated Learning.
                        This project leverages the CSE-CIC-IDS2018 dataset to simulate realistic cyber-attacks and defends against them using distributed Deep Neural Networks.
                    </p>

                    <div className="flex flex-wrap gap-4 items-center">
                        <div className="flex items-center gap-2 bg-white/10 px-4 py-2 rounded-lg border border-white/5">
                            <Shield className="w-5 h-5 text-emerald-400" />
                            <span className="font-medium text-slate-200">Robust Defense</span>
                        </div>
                        <div className="flex items-center gap-2 bg-white/10 px-4 py-2 rounded-lg border border-white/5">
                            <Lock className="w-5 h-5 text-amber-400" />
                            <span className="font-medium text-slate-200">Privacy First</span>
                        </div>
                        <div className="flex items-center gap-2 bg-white/10 px-4 py-2 rounded-lg border border-white/5">
                            <BrainCircuit className="w-5 h-5 text-purple-400" />
                            <span className="font-medium text-slate-200">XAI Enabled</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* LIVE SIMULATION SECTION */}
            <div>
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                    <Activity className="w-6 h-6 text-cyan-400" />
                    Live System Simulation
                </h3>
                <ProjectWorkflow />
            </div>

            {/* Project Overview Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-slate-900/50 backdrop-blur-sm p-8 rounded-xl border border-white/10">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                        <Server className="w-6 h-6 text-blue-400" />
                        Project Overview
                    </h3>
                    <div className="space-y-4 text-slate-300 leading-relaxed">
                        <p>
                            Traditional centralized intrusion detection systems require raw data to be sent to a central server, raising privacy concerns and high bandwidth costs.
                            <strong> Flex-ID</strong> solves this by training models locally on client devices (e.g., hospitals, IoT gateways) and only sharing model updates (weights) with the global server.
                        </p>
                        <p>
                            We employ two key strategies:
                        </p>
                        <ul className="list-disc pl-5 space-y-2 text-slate-400">
                            <li>
                                <strong className="text-white">FedAvg (Federated Averaging):</strong> The standard algorithm for aggregating local model updates.
                            </li>
                            <li>
                                <strong className="text-white">FedProx:</strong> An advanced algorithm designed to handle data heterogeneity (Non-IID data) and stragglers, ensuring robust convergence even when clients behave differently.
                            </li>
                        </ul>

                        <div className="mt-8 pt-6 border-t border-white/5">
                            <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <Code className="w-5 h-5 text-emerald-400" />
                                Technologies
                            </h4>

                            <div className="space-y-4">
                                {/* Machine Learning */}
                                <div className="bg-slate-800/40 rounded-lg p-3 border border-white/5">
                                    <div className="text-xs font-bold text-blue-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                                        <BrainCircuit className="w-3 h-3" /> Machine Learning
                                    </div>
                                    <div className="flex flex-wrap gap-2">
                                        {['TensorFlow', 'Flower (flwr)', 'Scikit-learn', 'Imbalanced-learn', 'Pandas', 'NumPy'].map(tech => (
                                            <span key={tech} className="px-2 py-1 bg-blue-500/10 text-blue-200 text-xs rounded border border-blue-500/20">
                                                {tech}
                                            </span>
                                        ))}
                                    </div>
                                </div>

                                {/* Web Stack */}
                                <div className="bg-slate-800/40 rounded-lg p-3 border border-white/5">
                                    <div className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                                        <Server className="w-3 h-3" /> Web Stack
                                    </div>
                                    <div className="flex flex-wrap gap-2">
                                        {['React + Vite', 'TailwindCSS', 'Node.js', 'Express', 'Socket.IO', 'Python'].map(tech => (
                                            <span key={tech} className="px-2 py-1 bg-purple-500/10 text-purple-200 text-xs rounded border border-purple-500/20">
                                                {tech}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Model Architecture Section */}
                <div className="bg-slate-900/50 backdrop-blur-sm p-8 rounded-xl border border-white/10">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                        <Cpu className="w-6 h-6 text-purple-400" />
                        Model Architecture
                    </h3>
                    <p className="text-slate-300 mb-6">
                        Each client trains a Deep Neural Network (DNN) built with <strong>TensorFlow/Keras</strong>, optimized for tabular network traffic data.
                    </p>

                    <div className="space-y-3">
                        <div className="flex items-center p-3 bg-slate-800/50 rounded-lg border border-white/5">
                            <div className="w-8 h-8 rounded bg-blue-500/20 flex items-center justify-center mr-4">
                                <span className="text-blue-400 font-mono text-xs">IN</span>
                            </div>
                            <div>
                                <div className="text-white font-medium">Input Layer</div>
                                <div className="text-xs text-slate-400">Matches processed feature dimension</div>
                            </div>
                        </div>

                        {[256, 128, 64].map((units, i) => (
                            <div key={i} className="flex items-center p-3 bg-slate-800/50 rounded-lg border border-white/5 relative">
                                <div className="absolute left-7 top-0 bottom-full w-px bg-slate-700 -z-10 h-4 -mt-2"></div>
                                <div className="w-8 h-8 rounded bg-purple-500/20 flex items-center justify-center mr-4">
                                    <BrainCircuit className="w-4 h-4 text-purple-400" />
                                </div>
                                <div>
                                    <div className="text-white font-medium">Dense Block ({units} Units)</div>
                                    <div className="text-xs text-slate-400 flex gap-2">
                                        <span className="bg-slate-700 px-1 rounded">ReLU</span>
                                        <span className="bg-slate-700 px-1 rounded">Batch Norm</span>
                                        <span className="bg-slate-700 px-1 rounded">Dropout (0.3)</span>
                                        <span className="bg-slate-700 px-1 rounded">L2 Reg</span>
                                    </div>
                                </div>
                            </div>
                        ))}

                        <div className="flex items-center p-3 bg-slate-800/50 rounded-lg border border-white/5">
                            <div className="w-8 h-8 rounded bg-emerald-500/20 flex items-center justify-center mr-4">
                                <span className="text-emerald-400 font-mono text-xs">OUT</span>
                            </div>
                            <div>
                                <div className="text-white font-medium">Output Layer</div>
                                <div className="text-xs text-slate-400">Softmax Activation (Multi-class classification)</div>
                            </div>
                        </div>
                    </div>

                    <div className="mt-8 pt-6 border-t border-white/5">
                        <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Lightbulb className="w-5 h-5 text-amber-400" />
                            Why this Architecture?
                        </h4>
                        <ul className="space-y-3 text-sm text-slate-300">
                            <li className="flex gap-3">
                                <span className="block w-1.5 h-1.5 mt-2 rounded-full bg-blue-400 shrink-0"></span>
                                <div>
                                    <strong className="text-slate-200">Complex Pattern Recognition:</strong>
                                    <span className="block text-slate-400">Network attacks are sophisticated. A deep architecture (3 hidden layers) allows the model to capture non-linear relationships and intricate traffic patterns that simpler models miss.</span>
                                </div>
                            </li>
                            <li className="flex gap-3">
                                <span className="block w-1.5 h-1.5 mt-2 rounded-full bg-purple-400 shrink-0"></span>
                                <div>
                                    <strong className="text-slate-200">Federated Compatibility:</strong>
                                    <span className="block text-slate-400">Deep Neural Networks are gradient-based, making them perfectly suited for <strong>FedAvg</strong> and <strong>FedProx</strong>, which rely on averaging weight matrices across clients.</span>
                                </div>
                            </li>
                            <li className="flex gap-3">
                                <span className="block w-1.5 h-1.5 mt-2 rounded-full bg-emerald-400 shrink-0"></span>
                                <div>
                                    <strong className="text-slate-200">Robust Generalization:</strong>
                                    <span className="block text-slate-400">We incorporate <strong>L2 Regularization</strong> and <strong>Dropout (0.3)</strong> to prevent the model from memorizing local client data (overfitting), ensuring it performs well on unseen global threats.</span>
                                </div>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>

        </div>
    );
}
