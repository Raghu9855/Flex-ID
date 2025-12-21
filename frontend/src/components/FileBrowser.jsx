import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Folder, FileText, Image as ImageIcon, ChevronRight, ChevronDown, Download, RefreshCw } from 'lucide-react';

const API_URL = 'http://localhost:5000/api';
const STATIC_URL = 'http://localhost:5000';

export default function FileBrowser({ initialPath = 'results', title = 'Project Artifacts', allowedPattern = null }) {
    const [structure, setStructure] = useState({});
    const [expanded, setExpanded] = useState({ [initialPath]: true });
    const [loading, setLoading] = useState(false);

    // Mock structure for now, ideally backend provides a directory listing endpoint.
    // Since we don't have a specific "list dir" endpoint for the whole project exposed safely,
    // we might need to add one or just hardcode the known structure for the demo if user accepts.
    // BUT the user asked for "folder like", so listing real files is better.
    // I will assume I can fetch a list from a new endpoint I'll create or just simulating for known folders.
    // Actually, I should create an endpoint in server.js to list 'results' content.

    // For this step, I will create the UI component and then I will update server.js to serve the file list.

    const toggleFolder = (path) => {
        setExpanded(prev => ({ ...prev, [path]: !prev[path] }));
    };

    // Placeholder data until we connect API
    const [files, setFiles] = useState([
        { name: 'results', type: 'folder', children: [] }
    ]);

    const fetchFiles = async () => {
        setLoading(true);
        try {
            const res = await axios.get(`${API_URL}/files/${encodeURIComponent(initialPath)}`);
            setFiles([res.data]);
        } catch (e) {
            console.error("Failed to fetch files", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchFiles();
    }, []);

    const FileItem = ({ item, path }) => {
        const fullPath = path ? `${path}/${item.name}` : item.name;
        const isFolder = item.type === 'directory';
        const isExpanded = expanded[fullPath];

        if (isFolder) {
            // Filter children if pattern is provided
            const children = item.children || [];
            const visibleChildren = allowedPattern
                ? children.filter(child => allowedPattern.test(child.name))
                : children;

            return (
                <div className="pl-4">
                    <div
                        className="flex items-center gap-2 py-2 px-3 hover:bg-white/5 rounded-lg cursor-pointer text-slate-300 hover:text-white transition-colors select-none"
                        onClick={() => toggleFolder(fullPath)}
                    >
                        {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                        <Folder className={`w-4 h-4 ${isExpanded ? 'text-amber-400' : 'text-slate-500'}`} />
                        <span className="font-medium text-sm">{item.name}</span>
                    </div>
                    {isExpanded && (
                        <div className="border-l border-white/10 ml-3">
                            {visibleChildren.length > 0 ? (
                                visibleChildren.map((child, i) => (
                                    <FileItem key={i} item={child} path={fullPath} />
                                ))
                            ) : (
                                <div className="py-2 px-3 text-xs text-slate-500 italic">No matching files</div>
                            )}
                        </div>
                    )}
                </div>
            );
        }

        return (
            <div className="pl-9 py-2 px-3 flex items-center justify-between hover:bg-white/5 rounded-lg group text-slate-400 hover:text-blue-300 transition-colors">
                <div className="flex items-center gap-2">
                    {item.name.endsWith('.png') ? <ImageIcon className="w-4 h-4 text-purple-400" /> : <FileText className="w-4 h-4" />}
                    <span className="text-sm">{item.name}</span>
                </div>
                <a
                    href={`${STATIC_URL}/${fullPath}`}
                    target="_blank"
                    rel="noreferrer"
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded"
                    title="Open/Download"
                >
                    <Download className="w-4 h-4" />
                </a>
            </div>
        );
    };

    return (
        <div className="bg-slate-900/50 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
            <div className="p-4 border-b border-white/10 flex justify-between items-center bg-white/5">
                <h3 className="text-white font-bold flex items-center gap-2">
                    <Folder className="w-5 h-5 text-amber-400" />
                    {title}
                </h3>
                <button onClick={fetchFiles} className="p-2 hover:bg-white/10 rounded-full text-slate-400 hover:text-white transition-all">
                    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                </button>
            </div>
            <div className="p-2 max-h-[400px] overflow-y-auto custom-scrollbar">
                {files.map((f, i) => (
                    <FileItem key={i} item={f} />
                ))}
            </div>
        </div>
    );
}
