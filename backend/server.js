const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const multer = require('multer');

// Configure Multer to save uploaded file as 'combined_ids2018_raw.csv'
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const dataDir = path.join(PROJECT_ROOT, 'data');
        if (!fs.existsSync(dataDir)) {
            fs.mkdirSync(dataDir);
        }
        cb(null, dataDir);
    },
    filename: function (req, file, cb) {
        // Force filename to match what 1_process_data.py expects
        cb(null, 'combined_ids2018_raw.csv');
    }
});
const upload = multer({ storage: storage });

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*", // Allow all for dev
        methods: ["GET", "POST"]
    }
});

const PORT = 5000;
const PROJECT_ROOT = path.join(__dirname, '..'); // Assuming backend is inside 7SemProject/backend

// Store running processes to manage lifecycles
// key: type (e.g., 'server', 'client-0'), value: process object
const runningProcesses = {};

// Helper to spawn process and stream logs
function spawnPythonProcess(command, args, processId, socket) {
    if (runningProcesses[processId]) {
        console.log(`Process ${processId} already running. Killing it first.`);
        killProcess(processId);
    }

    console.log(`Starting ${processId}: ${command} ${args.join(' ')}`);

    // Check if python or python3
    // On Windows it's often 'python', on Linux 'python3'. 
    // We'll assume 'python' for Windows environment based on user metadata.
    const pythonCmd = 'python';

    // Force unbuffered output for real-time logs
    let finalArgs = args;
    if (command === 'python' || command === 'python3') {
        if (!args.includes('-u')) {
            finalArgs = ['-u', ...args];
        }
    }

    const proc = spawn(command, finalArgs, {
        cwd: PROJECT_ROOT,
        shell: true
    });

    runningProcesses[processId] = proc;

    proc.stdout.on('data', (data) => {
        const msg = data.toString();
        // socket.emit('log', { id: processId, data: msg });
        io.emit('log', `[${processId}] ${msg}`); // Broadcast to all for now
        // console.log(`[${processId}] ${msg}`);
    });

    proc.stderr.on('data', (data) => {
        const msg = data.toString();
        // Keep the ID info clean so frontend can route it to the correct tab
        io.emit('log', `[${processId}] ERROR: ${msg}`);
        console.error(`[${processId} ERROR] ${msg}`);
    });

    proc.on('close', (code) => {
        console.log(`[${processId}] exited with code ${code}`);
        io.emit('log', `[${processId}] Process exited with code ${code}`);
        delete runningProcesses[processId];

        // Auto-generate plots when training finishes successfully
        if (processId === 'FL_Server' && code === 0) {
            io.emit('log', `[System] Training complete. Generating visualization plots...`);
            // We can spawn it directly here.
            // Note: We use a slight delay or just run it. Using spawnPythonProcess again is fine.
            // We give it a unique ID so it doesn't conflict (though FL_Server is already deleted above)
            spawnPythonProcess('python', ['ml/plot_history.py'], 'Plotter', io);
        }
    });
}

function killProcess(processId) {
    const proc = runningProcesses[processId];
    if (proc) {
        console.log(`Killing ${processId}...`);
        // On Windows with shell:true, normal .kill() might not kill the subprocess tree.
        // Using taskkill for robustness on Windows
        const pid = proc.pid;
        spawn("taskkill", ["/pid", pid, '/f', '/t']);
        delete runningProcesses[processId];
    }
}

function killAll() {
    Object.keys(runningProcesses).forEach(id => killProcess(id));
}

// --- Endpoints ---

// 0. Upload Dataset
app.post('/api/upload-dataset', upload.single('dataset'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: "No file uploaded" });
    }
    console.log(`Dataset uploaded: ${req.file.path}`);
    io.emit('log', `[Upload] New dataset uploaded: ${req.file.originalname}`);
    res.json({ message: "Dataset uploaded successfully" });
});

// 1. Run Preprocessing
app.post('/api/run-preprocessing', (req, res) => {
    spawnPythonProcess('python', ['ml/1_process_data.py'], 'preprocessing', io);
    res.json({ message: "Preprocessing started" });
});

// 2. Run Partitioning
app.post('/api/run-partitioning', (req, res) => {
    spawnPythonProcess('python', ['ml/2_create_partitions.py'], 'partitioning', io);
    res.json({ message: "Partitioning started" });
});

// 3. Start Training
// 3. Run Generic Command
app.post('/api/run-command', (req, res) => {
    const { command, args, id } = req.body;

    if (!command || !id) {
        return res.status(400).json({ message: "Missing command or id" });
    }

    // Default to 'python' if command is python
    // but we allow full flexibility. 
    // For safety in this specific app context, we might prepend 'python' if not present, 
    // but the user asked for full manual typing "python ...". 
    // So we assume the user sends the split command. 
    // Actually, spawn requires command and args array.
    // We'll let the frontend parse the string "python script.py --arg" into cmd="python", args=["script.py", "--arg"]

    spawnPythonProcess(command, args || [], id, io);
    res.json({ message: `Started process ${id}` });
});

// 4. Stop Specific Process
app.post('/api/stop-process', (req, res) => {
    const { id } = req.body;
    if (id && runningProcesses[id]) {
        killProcess(id);
        res.json({ message: `Stopped process ${id}` });
    } else {
        res.status(404).json({ message: "Process not found or already stopped" });
    }
});

// 5. Stop All (Panic Button)
app.post('/api/stop-all', (req, res) => {
    killAll();
    res.json({ message: "All processes stopped" });
});

// 5. XAI
app.post('/api/xai', (req, res) => {
    const { round, weights } = req.body; // Optional args
    const args = ['ml/explain_model.py'];
    if (round) {
        args.push('--round');
        args.push(String(round));
    }
    spawnPythonProcess('python', args, 'XAI_Explainer', io);
    res.json({ message: "XAI started" });
});

// 5.5 Run Comparison
app.post('/api/run-comparison', (req, res) => {
    const { mode } = req.body;

    // Determine paths based on mode
    // Paths are relative to PROJECT_ROOT
    let fedavgPath, fedproxPath;

    // We assume round 10 as the final model for now, or we could find the latest.
    // Given the previous user edit, they hardcoded round-10.
    if (mode === 'under_attack') {
        fedavgPath = 'results/fedunderattack/round-10-weights.pkl';
        fedproxPath = 'results/fedproxunderattack/round-10-weights.pkl';
    } else {
        // Default / No Attack
        fedavgPath = 'results/fedavgeachround/round-10-weights.pkl';
        fedproxPath = 'results/fedproxeachround/round-10-weights.pkl';
    }

    console.log(`[Comparison] Running for mode: ${mode}`);
    console.log(`FA: ${fedavgPath}, FP: ${fedproxPath}`);

    const args = [
        'ml/compare_results.py',
        '--fedavg', fedavgPath,
        '--fedprox', fedproxPath,
        '--mode', mode || 'manual'
    ];

    // We use child_process.exec or spawn, but we need to capture stdout for the JSON
    // spawnPythonProcess is for long running tasks with socket emission.
    // Here we want a request-response cycle (though it might take time).
    // Let's use a new spawn and buffer output.

    const pythonCmd = 'python'; // or python3
    const proc = spawn(pythonCmd, args, { cwd: PROJECT_ROOT });

    let stdoutData = '';
    let stderrData = '';

    proc.stdout.on('data', (data) => {
        stdoutData += data.toString();
        // Also stream to logs if needed
        io.emit('log', `[Comparison] ${data.toString()}`);
    });

    proc.stderr.on('data', (data) => {
        stderrData += data.toString();
        console.error(`[Comparison ERR] ${data.toString()}`);
    });

    proc.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({ message: "Comparison script failed", error: stderrData });
        }

        // Parse JSON
        try {
            const startMarker = '__JSON_START__';
            const endMarker = '__JSON_END__';

            const startIndex = stdoutData.indexOf(startMarker);
            const endIndex = stdoutData.indexOf(endMarker);

            if (startIndex !== -1 && endIndex !== -1) {
                const jsonStr = stdoutData.substring(startIndex + startMarker.length, endIndex).trim();
                const result = JSON.parse(jsonStr);
                res.json(result);
            } else {
                throw new Error("Could not find JSON output markers");
            }
        } catch (e) {
            console.error("JSON Parse Error:", e);
            res.status(500).json({ message: "Failed to parse comparison results", raw: stdoutData });
        }
    });
});

// 6. List Files (for File Browser)
// 6. List Files (for File Browser)
app.get('/api/files', (req, res) => {
    const folderName = req.query.path || 'results';
    // Security: Only allow specific folders checking relative path
    const targetPath = path.join(PROJECT_ROOT, folderName);

    // Basic check to ensure we are within project
    if (!targetPath.startsWith(PROJECT_ROOT)) {
        return res.status(403).json({ message: "Access denied" });
    }

    if (!fs.existsSync(targetPath)) {
        return res.json({ name: folderName, type: 'directory', children: [] });
    }

    const getDirRecursive = (dir) => {
        const stats = fs.statSync(dir);
        const name = path.basename(dir);

        if (!stats.isDirectory()) {
            return { name, type: 'file', size: stats.size };
        }

        const children = fs.readdirSync(dir).map(child => {
            return getDirRecursive(path.join(dir, child));
        });

        return { name, type: 'directory', children };
    };

    try {
        const tree = getDirRecursive(targetPath);
        res.json(tree);
    } catch (e) {
        res.status(500).json({ message: e.message });
    }
});

// 6. Results (Serve Static)
// We expose the 'results' folder and the root (for images usually saved in root? check script)
app.use('/results', express.static(path.join(PROJECT_ROOT, 'results')));
app.use('/images', express.static(PROJECT_ROOT)); // Some images might be in root

app.get('/api/check-status', (req, res) => {
    res.json({ running: Object.keys(runningProcesses) });
});

server.listen(PORT, () => {
    console.log(`Backend running on http://localhost:${PORT}`);
});

// Cleanup on exit
process.on('SIGINT', () => {
    console.log("Shutting down...");
    killAll();
    process.exit();
});

process.on('SIGTERM', () => {
    console.log("Shutting down...");
    killAll();
    process.exit();
});
