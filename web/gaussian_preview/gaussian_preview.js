/**
 * ComfyUI GeomPack - Gaussian Splat Preview Widget
 * Interactive 3D Gaussian Splatting viewer using gsplat.js
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name (handles ComfyUI-GeometryPack or comfyui-geometrypack)
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI_VNCCS";
})();

console.log("[VNCCS.GaussianPreview] Loading extension...");

app.registerExtension({
    name: "vnccs.gaussianpreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VNCCS_BackgroundPreview") {
            console.log("[VNCCS.GaussianPreview] Registering Preview Gaussian node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#1a1a1a";
                container.style.overflow = "hidden";

                // Create iframe for gsplat.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#1a1a1a";
                iframe.allowFullscreen = true;
                iframe.setAttribute("allow", "fullscreen; clipboard-write");

                // Point to gsplat.js HTML viewer (with cache buster)
                iframe.src = `/extensions/${EXTENSION_FOLDER}/gaussian_preview/static/viewer_gaussian.html?v=` + Date.now();

                // Create info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#1a1a1a";
                infoPanel.style.borderTop = "1px solid #444";
                infoPanel.style.padding = "6px 12px";
                infoPanel.style.fontSize = "10px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.3";
                infoPanel.style.flexShrink = "0";
                infoPanel.style.overflow = "hidden";
                infoPanel.innerHTML = '<span style="color: #888;">Gaussian splat info will appear here after execution</span>';

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget with required options
                const widget = this.addDOMWidget("preview_gaussian", "GAUSSIAN_PREVIEW", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // Store reference to node for dynamic widget sizing. The node
                // size is the source of truth; resizing the node reveals more or
                // less of the 3D viewport without changing scene framing.
                const node = this;
                widget.computeSize = function () {
                    const isNodeResizeClamp = app.canvas?.resizing_node === node;
                    const width = Math.max(240, node.size[0] - 20);
                    if (isNodeResizeClamp) {
                        return [width, 80];
                    }
                    const top = this.last_y ?? 120;
                    const height = Math.max(80, node.size[1] - top - 8);
                    return [width, height];
                };

                // Store references
                this.gaussianViewerIframe = iframe;
                this.gaussianInfoPanel = infoPanel;

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                });

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Handle screenshot messages
                    if (event.data.type === 'SCREENSHOT' && event.data.image) {
                        try {
                            // Convert base64 data URL to blob
                            const base64Data = event.data.image.split(',')[1];
                            const byteString = atob(base64Data);
                            const arrayBuffer = new ArrayBuffer(byteString.length);
                            const uint8Array = new Uint8Array(arrayBuffer);

                            for (let i = 0; i < byteString.length; i++) {
                                uint8Array[i] = byteString.charCodeAt(i);
                            }

                            const blob = new Blob([uint8Array], { type: 'image/png' });

                            // Generate filename with timestamp
                            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                            const filename = `gaussian-screenshot-${timestamp}.png`;

                            // Create FormData for upload
                            const formData = new FormData();
                            formData.append('image', blob, filename);
                            formData.append('type', 'output');
                            formData.append('subfolder', '');

                            // Upload to ComfyUI backend
                            const response = await fetch('/upload/image', {
                                method: 'POST',
                                body: formData
                            });

                            if (response.ok) {
                                const result = await response.json();
                            } else {
                                throw new Error(`Upload failed: ${response.status}`);
                            }

                        } catch (error) {
                            console.error('[GeomPack Gaussian] Error saving screenshot:', error);
                        }
                    }
                    // Handle copy image to clipboard messages
                    else if (event.data.type === 'COPY_IMAGE' && event.data.success) {
                    }
                    else if (event.data.type === 'COPY_IMAGE' && !event.data.success) {
                    }
                    // Handle error messages from iframe
                    else if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        if (infoPanel) {
                            infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${event.data.error}</div>`;
                        }
                    }
                });

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function (message) {
                    onExecuted?.apply(this, arguments);

                    // Check for errors
                    if (message?.error && message.error[0]) {
                        infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${message.error[0]}</div>`;
                        return;
                    }

                    // The message IS the UI data (not message.ui)
                    if (message?.ply_path && message.ply_path[0]) {
                        const filename = message.filename?.[0];
                        const fileSizeMb = message.file_size_mb?.[0] || 'N/A';
                        const subfolder = message.subfolder?.[0] || "";
                        const type = message.type?.[0] || "output";
                        const previewFilename = message.preview_filename?.[0] || filename;
                        const previewSubfolder = message.preview_subfolder?.[0] || subfolder;
                        const previewType = message.preview_type?.[0] || type;
                        const previewSizeMb = message.preview_file_size_mb?.[0] || fileSizeMb;
                        const previewFormat = message.preview_format?.[0] || "ply";
                        const coordinateBasis = message.coordinate_basis?.[0] || "worldmirror";

                        // Extract camera parameters if provided
                        const extrinsics = message.extrinsics?.[0] || null;
                        const intrinsics = message.intrinsics?.[0] || null;

                        // Update info panel
                        infoPanel.innerHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                                <span style="color: #888;">File:</span>
                                <span style="color: #6cc;">${filename}</span>
                                <span style="color: #888;">Size:</span>
                                <span>${fileSizeMb} MB</span>
                                <span style="color: #888;">Preview:</span>
                                <span>${previewFormat.toUpperCase()} · ${previewSizeMb} MB</span>
                            </div>
                        `;

                        // ComfyUI serves output files via /view API endpoint
                        const filepath = `/view?filename=${encodeURIComponent(previewFilename)}&type=${encodeURIComponent(previewType)}&subfolder=${encodeURIComponent(previewSubfolder)}`;

                        // Function to fetch and send data to iframe
                        const fetchAndSend = async () => {
                            if (!iframe.contentWindow) {
                                return;
                            }

                            try {
                                // Fetch the preview file from parent context (authenticated)
                                const response = await fetch(filepath);
                                if (!response.ok) {
                                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                                }
                                const arrayBuffer = await response.arrayBuffer();

                                // Send the data to iframe with camera parameters
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH_DATA",
                                    data: arrayBuffer,
                                    filename: previewFilename,
                                    sourceFilename: filename,
                                    format: previewFormat,
                                    extrinsics: extrinsics,
                                    intrinsics: intrinsics,
                                    coordinateBasis: coordinateBasis,
                                    timestamp: Date.now()
                                }, "*", [arrayBuffer]);
                            } catch (error) {
                                console.error("[VNCCS.GaussianPreview] Error fetching preview data:", error);
                                infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error loading preview: ${error.message}</div>`;
                            }
                        };

                        // Fetch and send when iframe is ready
                        if (iframeLoaded) {
                            fetchAndSend();
                        } else {
                            setTimeout(fetchAndSend, 500);
                        }
                    }
                };

                return r;
            };
        }
    }
});
