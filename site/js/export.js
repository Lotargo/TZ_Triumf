class DocExporter {
    constructor() {
        this.docs = [
            "docs/report.md",
            "docs/research/01-overview.md",
            "docs/research/02-3dmm.md",
            "docs/research/03-nerf.md",
            "docs/research/04-deep-learning.md",
            "docs/research/05-photogrammetry.md",
            "docs/research/06-domain-gap.md",
            "docs/research/07-comparison.md",
            "docs/research/08-comfyui-masks-meshes.md",
            "docs/research/09-game-face-textures-masks.md",
            "docs/practical/architecture.md",
        ];
    }

    async fetchDoc(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Не удалось загрузить ${url}: ${response.status}`);
        }
        return response.text();
    }

    async downloadAllMD() {
        const parts = [];

        for (const doc of this.docs) {
            try {
                parts.push(await this.fetchDoc(doc));
            } catch (error) {
                console.error(error);
            }
        }

        this.downloadFile(parts.join("\n\n---\n\n"), "3d-face-reconstruction-docs.md", "text/markdown;charset=utf-8");
    }

    async generatePDF() {
        try {
            const markdown = await this.fetchDoc("docs/report.md");
            const printWindow = window.open("", "_blank");
            printWindow.document.write(this.wrapPrintableHtml(this.markdownToHTML(markdown)));
            printWindow.document.close();
            printWindow.focus();
            printWindow.print();
        } catch (error) {
            console.error(error);
            window.print();
        }
    }

    wrapPrintableHtml(content) {
        return `<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>3D-реконструкция лица по фото</title>
<style>
body{max-width:820px;margin:36px auto;color:#24313f;font:16px/1.62 Arial,sans-serif}
h1{font-size:34px;line-height:1.12}h2{margin-top:30px;border-bottom:1px solid #dfe8e7;padding-bottom:8px}
table{width:100%;border-collapse:collapse;margin:18px 0}td,th{border:1px solid #dfe8e7;padding:8px;text-align:left;vertical-align:top}
th{background:#eef7f5}code{background:#f4f7f7;padding:2px 5px;border-radius:4px}pre{white-space:pre-wrap;background:#f4f7f7;padding:14px;border-radius:8px}
@page{margin:18mm}
</style>
</head>
<body>${content}</body>
</html>`;
    }

    markdownToHTML(markdown) {
        const escaped = markdown
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        return escaped
            .replace(/^### (.*)$/gim, "<h3>$1</h3>")
            .replace(/^## (.*)$/gim, "<h2>$1</h2>")
            .replace(/^# (.*)$/gim, "<h1>$1</h1>")
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/`([^`]+)`/g, "<code>$1</code>")
            .replace(/^\|(.+)\|$/gim, "<pre>|$1|</pre>")
            .replace(/^- (.*)$/gim, "<p>• $1</p>")
            .replace(/\n{2,}/g, "</p><p>")
            .replace(/^/, "<p>")
            .replace(/$/, "</p>");
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }
}

window.DocExporter = DocExporter;
