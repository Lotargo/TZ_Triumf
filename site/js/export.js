/**
 * Export functionality for documentation
 */

class DocExporter {
    constructor() {
        this.docs = [];
        this.loadDocsList();
    }
    
    async loadDocsList() {
        // List of all documentation files
        this.docs = [
            'docs/research/01-overview.md',
            'docs/research/02-3dmm.md',
            'docs/research/03-nerf.md',
            'docs/research/04-deep-learning.md',
            'docs/research/05-photogrammetry.md',
            'docs/research/06-domain-gap.md',
            'docs/research/07-comparison.md',
            'docs/practical/architecture.md',
            'docs/practical/setup.md'
        ];
    }
    
    async fetchDoc(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.text();
        } catch (error) {
            console.error(`Error fetching ${url}:`, error);
            return null;
        }
    }
    
    async downloadSingleMD(url, filename) {
        const content = await this.fetchDoc(url);
        if (content) {
            this.downloadFile(content, filename, 'text/markdown');
        }
    }
    
    async downloadAllMD() {
        let combined = '# 3D Face Reconstruction - Documentation\n\n';
        combined += '## Boiko Oleg | 2026\n\n';
        combined += '---\n\n';
        
        for (const doc of this.docs) {
            const content = await this.fetchDoc(doc);
            if (content) {
                combined += content + '\n\n---\n\n';
            }
        }
        
        this.downloadFile(combined, 'documentation.md', 'text/markdown');
    }
    
    async generatePDF() {
        // Simple PDF generation using browser print
        // For production, use jsPDF or similar library
        
        let content = `
            <html>
            <head>
                <title>3D Face Reconstruction - Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #2563eb; }
                    h2 { color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; }
                    h3 { color: #64748b; }
                    code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
                    pre { background: #1e293b; color: #e2e8f0; padding: 15px; border-radius: 8px; overflow-x: auto; }
                    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                    th, td { border: 1px solid #e2e8f0; padding: 10px; text-align: left; }
                    th { background: #f8fafc; }
                </style>
            </head>
            <body>
        `;
        
        for (const doc of this.docs) {
            const markdown = await this.fetchDoc(doc);
            if (markdown) {
                content += this.markdownToHTML(markdown);
            }
        }
        
        content += `
            </body>
            </html>
        `;
        
        // Open in new window for printing
        const printWindow = window.open('', '_blank');
        printWindow.document.write(content);
        printWindow.document.close();
        printWindow.print();
    }
    
    markdownToHTML(markdown) {
        // Simple markdown to HTML conversion
        let html = markdown
            // Headers
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks
            .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`(.*?)`/g, '<code>$1</code>')
            // Links
            .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2">$1</a>')
            // Lists
            .replace(/^\- (.*$)/gim, '<li>$1</li>')
            // Paragraphs
            .replace(/\n\n/g, '</p><p>')
            // Line breaks
            .replace(/\n/g, '<br>');
        
        return `<div class="content">${html}</div>`;
    }
    
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Export
window.DocExporter = DocExporter;
