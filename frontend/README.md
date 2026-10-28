# FLEX-ID — Apple-Inspired Frontend Dashboard

This directory contains the user interface for **FLEX-ID**, redesigned as a minimalist, high-precision, Apple-inspired cybersecurity product.

## Technology Stack
* **HTML5**: Semantic, accessible markup.
* **CSS3**: Custom design tokens, tabular numerals, light/dark themes (`prefers-color-scheme` + manual toggle), and responsive layout.
* **Vanilla JavaScript (ES6+)**: Zero framework dependencies, zero build steps, native `fetch` client communicating with FastAPI REST endpoints.

## Structure
```
frontend/
├── index.html        # Semantic HTML5 layout for all 5 core views
├── css/
│   └── style.css     # Apple design system, custom properties, typography & components
└── js/
    ├── api.js        # Native fetch API client for FastAPI endpoints
    └── app.js        # State, tab routing, SVG charts, preset loaders, and event handlers
```

## Running
The frontend is served directly by the FastAPI backend:
```powershell
.\run.ps1 api
```
Access the application at `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/ui`.
Interactive OpenAPI documentation remains available at `http://127.0.0.1:8000/docs`.
