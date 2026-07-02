/**
 * Main application script
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize viewer
    const viewer = new FaceViewer('viewer-3d');
    
    // Initialize exporter
    const exporter = new DocExporter();
    
    // File input handler
    const fileInput = document.getElementById('file-input');
    const btnLoad = document.getElementById('btn-load');
    const btnReset = document.getElementById('btn-reset');
    
    btnLoad.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            viewer.loadModel(url);
        }
    });
    
    btnReset.addEventListener('click', () => {
        viewer.resetCamera();
    });
    
    // PDF download
    const btnPDF = document.getElementById('btn-pdf');
    btnPDF.addEventListener('click', () => {
        exporter.generatePDF();
    });
    
    // All MD download
    const btnAllMD = document.getElementById('btn-all-md');
    btnAllMD.addEventListener('click', () => {
        exporter.downloadAllMD();
    });
    
    // Individual MD downloads
    document.querySelectorAll('.btn-download').forEach(btn => {
        if (btn.href && btn.href.endsWith('.md')) {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const url = btn.href;
                const filename = url.split('/').pop();
                exporter.downloadSingleMD(url, filename);
            });
        }
    });
    
    // Smooth scroll for navigation
    document.querySelectorAll('.nav a').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Demo: Generate sample face geometry
    window.generateDemoFace = () => {
        const vertices = generateFaceVertices();
        const faces = generateFaceFaces(vertices.length / 3);
        viewer.loadFromGeometry(vertices, faces);
    };
    
    function generateFaceVertices() {
        const vertices = [];
        const segments = 30;
        
        for (let i = 0; i < segments; i++) {
            for (let j = 0; j < segments; j++) {
                const phi = (i / segments) * Math.PI;
                const theta = (j / segments) * 2 * Math.PI;
                
                // Basic sphere with face-like deformation
                let r = 0.5;
                
                // Nose deformation
                if (phi > 0.3 && phi < 0.7 && theta > 0.8 && theta < 1.2) {
                    r += 0.15 * Math.sin((phi - 0.3) * Math.PI / 0.4);
                }
                
                // Eye sockets
                if (phi > 0.4 && phi < 0.6) {
                    if (theta > 0.3 && theta < 0.6) {
                        r -= 0.05;
                    }
                    if (theta > 1.8 && theta < 2.1) {
                        r -= 0.05;
                    }
                }
                
                const x = r * Math.sin(phi) * Math.cos(theta);
                const y = r * Math.cos(phi);
                const z = r * Math.sin(phi) * Math.sin(theta);
                
                vertices.push([x, y, z]);
            }
        }
        
        return vertices;
    }
    
    function generateFaceFaces(vertexCount) {
        const faces = [];
        const segments = 30;
        
        for (let i = 0; i < segments - 1; i++) {
            for (let j = 0; j < segments - 1; j++) {
                const idx = i * segments + j;
                
                faces.push([idx, idx + segments, idx + 1]);
                faces.push([idx + 1, idx + segments, idx + segments + 1]);
            }
        }
        
        return faces;
    }
    
    console.log('3D Face Reconstruction App initialized');
    console.log('Use generateDemoFace() to see a demo');
});
