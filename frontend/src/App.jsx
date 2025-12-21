import React, { useState } from 'react';
import DataPrep from './components/DataPrep';
import TrainingControl from './components/TrainingControl';
import Monitor from './components/Monitor';
import ResultsView from './components/ResultsView';
import XAIPage from './components/XAIPage';
import Home from './components/Home';
import { LayoutDashboard, Database, Activity, BarChart2, Github, Terminal, Home as HomeIcon, Menu, ChevronLeft, BrainCircuit } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const NavItem = ({ id, label, icon: Icon }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 font-medium whitespace-nowrap overflow-hidden ${activeTab === id
        ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-[0_0_15px_rgba(37,99,235,0.2)]'
        : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
        }`}
      title={!sidebarOpen ? label : ''}
    >
      <Icon className={`w-5 h-5 flex-shrink-0 ${activeTab === id ? 'text-blue-400 animate-pulse' : ''}`} />
      <span className={`transition-opacity duration-300 ${sidebarOpen ? 'opacity-100' : 'opacity-0 w-0'}`}>
        {label}
      </span>
    </button>
  );

  return (
    <div className="flex h-screen bg-[#0a0c10] text-slate-200 overflow-hidden font-sans selection:bg-blue-500/30">
      {/* Background Gradients */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-0 w-[500px] h-[500px] bg-blue-600/5 rounded-full blur-[100px]" />
        <div className="absolute bottom-0 right-0 w-[500px] h-[500px] bg-purple-600/5 rounded-full blur-[100px]" />
      </div>

      {/* Sidebar - Collapsible */}
      <aside
        className={`${sidebarOpen ? 'w-64' : 'w-20'} bg-slate-900/50 backdrop-blur-xl border-r border-white/5 flex flex-col z-20 flex-shrink-0 transition-all duration-300 ease-in-out`}
      >
        <div className="p-6 border-b border-white/5 flex items-center justify-between">
          <div className={`flex items-center gap-3 overflow-hidden transition-all duration-300 ${sidebarOpen ? 'opacity-100' : 'opacity-0 w-0'}`}>
            <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-2.5 rounded-xl shadow-lg shadow-blue-500/20 flex-shrink-0">
              <LayoutDashboard className="w-6 h-6 text-white" />
            </div>
            <div className="whitespace-nowrap">
              <h1 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                FL-IDS
              </h1>
            </div>
          </div>

          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-white/10 rounded-lg text-slate-400 transition-colors mx-auto"
          >
            {sidebarOpen ? <ChevronLeft className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        <nav className="flex-1 p-4 space-y-2 overflow-y-auto custom-scrollbar overflow-x-hidden">
          <div className={`text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 px-4 transition-opacity duration-300 ${sidebarOpen ? 'opacity-100' : 'opacity-0'}`}>
            Menu
          </div>
          <NavItem id="home" label="Overview" icon={HomeIcon} />
          <NavItem id="dataprep" label="Data & Partitions" icon={Database} />
          <NavItem id="training" label="Training Control" icon={Activity} />
          <NavItem id="results" label="Results & Analysis" icon={BarChart2} />
          <NavItem id="xai" label="Explainable AI" icon={BrainCircuit} />

          <div className="my-6 border-t border-white/5"></div>
          <NavItem id="monitor" label="System Monitor" icon={Terminal} />
        </nav>

        <div className="p-4 border-t border-white/5">
          <div className={`bg-slate-950/50 rounded-xl p-4 border border-white/5 transition-all duration-300 ${sidebarOpen ? '' : 'p-2 flex justify-center'}`}>
            {sidebarOpen ? (
              <>
                <div className="flex items-center gap-2 text-xs text-emerald-400 font-mono mb-2">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  Active
                </div>
                <p className="text-[10px] text-slate-500">Port: 5000</p>
              </>
            ) : (
              <div className="w-3 h-3 bg-emerald-500 rounded-full shadow-[0_0_10px_#10b981]" title="System Active"></div>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col relative z-10 overflow-hidden w-full">
        {/* Top Header */}
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-8 bg-slate-900/30 backdrop-blur-sm flex-shrink-0">
          <div className="flex items-center gap-4">
            {!sidebarOpen && (
              <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-2 rounded-lg shadow-lg shadow-blue-500/20 mr-2">
                <LayoutDashboard className="w-5 h-5 text-white" />
              </div>
            )}
            <h2 className="text-white font-medium text-lg">
              {activeTab === 'home' && 'Project Overview'}
              {activeTab === 'dataprep' && 'Data Preparation & Partitioning'}
              {activeTab === 'training' && 'Federated Training Control'}
              {activeTab === 'results' && 'Results Visualization'}
              {activeTab === 'xai' && 'Explainable AI Model'}
              {activeTab === 'monitor' && 'System Logs & Monitoring'}
            </h2>
          </div>
          <div className="flex items-center gap-4">
            <a href="https://github.com" target="_blank" rel="noreferrer" className="text-slate-400 hover:text-white transition-colors">
              <Github className="w-5 h-5" />
            </a>
          </div>
        </header>

        {/* Scrollable Page Content */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 custom-scrollbar">
          <div className="max-w-7xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
            {activeTab === 'home' && <Home />}
            {activeTab === 'dataprep' && <DataPrep />}
            {activeTab === 'training' && <TrainingControl />}
            {activeTab === 'results' && <ResultsView />}
            {activeTab === 'xai' && <XAIPage />}
            {activeTab === 'monitor' && (
              <div className="h-[calc(100vh-140px)] rounded-xl overflow-hidden border border-white/10 shadow-2xl">
                <Monitor />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
