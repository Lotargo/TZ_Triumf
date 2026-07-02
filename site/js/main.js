document.addEventListener("DOMContentLoaded", () => {
    const exporter = new DocExporter();
    const viewer = new DocViewer(exporter);

    document.getElementById("btn-pdf")?.addEventListener("click", () => {
        exporter.generatePDF();
    });

    document.getElementById("btn-all-md")?.addEventListener("click", () => {
        exporter.downloadAllMD();
    });

    document.querySelectorAll(".btn-view-doc").forEach((button) => {
        button.addEventListener("click", () => {
            viewer.open(button.dataset.docUrl, button.dataset.docTitle);
        });
    });
});

class DocViewer {
    constructor(exporter) {
        this.exporter = exporter;
        this.modal = document.getElementById("doc-modal");
        this.title = document.getElementById("doc-modal-title");
        this.content = document.getElementById("doc-modal-content");
        this.download = document.getElementById("doc-modal-download");
        this.lastActiveElement = null;

        this.modal.querySelectorAll("[data-doc-close]").forEach((closeControl) => {
            closeControl.addEventListener("click", () => this.close());
        });

        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && this.modal.getAttribute("aria-hidden") === "false") {
                this.close();
            }
        });
    }

    async open(url, title) {
        if (!url) {
            return;
        }

        this.lastActiveElement = document.activeElement;
        this.title.textContent = title || "Документ";
        this.download.href = url;
        this.download.setAttribute("download", url.split("/").pop() || "document.md");
        this.content.innerHTML = '<div class="doc-loading">Загрузка документа...</div>';
        document.body.classList.add("modal-open");
        this.modal.classList.add("is-open");
        this.modal.setAttribute("aria-hidden", "false");

        try {
            const markdown = await this.exporter.fetchDoc(url);
            this.content.innerHTML = this.renderMarkdown(markdown);
            this.renderMath();
        } catch (error) {
            console.error(error);
            this.content.innerHTML = '<div class="doc-error">Не удалось загрузить документ.</div>';
        }

        this.content.focus({ preventScroll: true });
    }

    close() {
        this.modal.setAttribute("aria-hidden", "true");
        this.modal.classList.remove("is-open");
        document.body.classList.remove("modal-open");
        this.lastActiveElement?.focus?.({ preventScroll: true });
    }

    renderMarkdown(markdown) {
        if (window.marked) {
            marked.setOptions({
                breaks: false,
                gfm: true,
                headerIds: false,
                mangle: false,
            });

            const html = marked.parse(markdown);
            return window.DOMPurify ? DOMPurify.sanitize(html) : html;
        }

        return this.basicMarkdownToHTML(markdown);
    }

    renderMath() {
        if (!window.renderMathInElement) {
            return;
        }

        renderMathInElement(this.content, {
            delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "\\[", right: "\\]", display: true },
                { left: "\\(", right: "\\)", display: false },
                { left: "$", right: "$", display: false },
            ],
            throwOnError: false,
        });
    }

    basicMarkdownToHTML(markdown) {
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
            .replace(/^- (.*)$/gim, "<p>• $1</p>")
            .replace(/\n{2,}/g, "</p><p>")
            .replace(/^/, "<p>")
            .replace(/$/, "</p>");
    }
}
