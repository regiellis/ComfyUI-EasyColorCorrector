import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

/**
 * Frontend logic for the EasyColorCorrection node.
 * This script provides a dynamic UI for three powerful modes with optional real-time preview:
 * 1.  Auto: 🤖 AI-POWERED ENHANCEMENT - Face detection, scene analysis, perceptual color science
 * 2.  Preset: 🎨 STYLE-BASED PRESETS - Curated looks with intelligent variation
 * 3.  Manual: 🎛️  PROFESSIONAL GRADING - Full manual control with advanced tools
 */
app.registerExtension({
	name: "comfyui-easycolorcorrection",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "EasyColorCorrection") {

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				if (onNodeCreated) {
					onNodeCreated.apply(this, arguments);
				}

				const node = this;
				const modeWidget = node.widgets.find(w => w.name === "mode");
				const realtimeWidget = node.widgets.find(w => w.name === "realtime_preview");

                // A clear map of all widgets we need to control
                const allWidgets = {
                    reference_strength: node.widgets.find(w => w.name === "reference_strength"),
                    extract_palette: node.widgets.find(w => w.name === "extract_palette"),
                    realtime_preview: node.widgets.find(w => w.name === "realtime_preview"),
                    ai_analysis: node.widgets.find(w => w.name === "ai_analysis"),
                    adjust_for_skin_tone: node.widgets.find(w => w.name === "adjust_for_skin_tone"),
                    white_balance_strength: node.widgets.find(w => w.name === "white_balance_strength"),
                    enhancement_strength: node.widgets.find(w => w.name === "enhancement_strength"),
                    pop_factor: node.widgets.find(w => w.name === "pop_factor"),
                    warmth: node.widgets.find(w => w.name === "warmth"),
                    vibrancy: node.widgets.find(w => w.name === "vibrancy"),
                    contrast: node.widgets.find(w => w.name === "contrast"),
                    brightness: node.widgets.find(w => w.name === "brightness"),
                    preset: node.widgets.find(w => w.name === "preset"),
                    variation: node.widgets.find(w => w.name === "variation"),
                    lift: node.widgets.find(w => w.name === "lift"),
                    gamma: node.widgets.find(w => w.name === "gamma"),
                    gain: node.widgets.find(w => w.name === "gain"),
                    noise: node.widgets.find(w => w.name === "noise"),
                    effect_strength: node.widgets.find(w => w.name === "effect_strength"),
                };

                // Mode-specific AI analysis tooltips to clarify how it works in each mode
                const aiAnalysisTooltips = {
                    "Auto": "🤖 AI drives the entire enhancement: scene detection, lighting analysis, face detection",
                    "Preset": "🎨 AI adapts presets: automatically adjusts based on detected content type (anime/concept art/portrait)",
                    "Manual": "🎛️ AI provides guidance: enhanced masking, stronger artistic controls, better curve mapping"
                };

                // Configuration defining which widgets are visible in each mode
                const visibilityConfig = {
                    "Auto": ["reference_strength", "extract_palette", "realtime_preview", "ai_analysis", "adjust_for_skin_tone", "white_balance_strength", "enhancement_strength", "pop_factor", "effect_strength"],
                    "Preset": ["reference_strength", "extract_palette", "realtime_preview", "ai_analysis", "adjust_for_skin_tone", "white_balance_strength", "enhancement_strength", "preset", "variation", "effect_strength"],
                    "Manual": ["reference_strength", "extract_palette", "realtime_preview", "ai_analysis", "adjust_for_skin_tone", "white_balance_strength", "enhancement_strength", "warmth", "vibrancy", "contrast", "brightness", "lift", "gamma", "gain", "noise", "effect_strength"],
                };

                // Real-time preview functionality
                let realtimeTimeout = null;
                const REALTIME_DELAY = 150; // ms debounce

                function scheduleRealtimeUpdate() {
                    if (!realtimeWidget || !realtimeWidget.value) return;
                    
                    // Clear existing timeout
                    if (realtimeTimeout) {
                        clearTimeout(realtimeTimeout);
                    }
                    
                    // Schedule new update
                    realtimeTimeout = setTimeout(() => {
                        triggerNodeExecution();
                    }, REALTIME_DELAY);
                }

                async function triggerNodeExecution() {
                    try {
                        // Check if the node has inputs connected
                        if (!node.inputs || !node.inputs[0] || !node.inputs[0].link) {
                            return; // No image input connected
                        }

                        // Build selective workflow - only this node and its dependencies
                        const workflow = app.graph.serialize();
                        const executionNodes = buildSelectiveWorkflow(workflow, node.id);
                        
                        if (Object.keys(executionNodes).length > 0) {
                            await api.queuePrompt(0, executionNodes);
                        }
                    } catch (error) {
                        console.warn("Real-time preview error:", error);
                    }
                }

                function buildSelectiveWorkflow(workflow, targetNodeId) {
                    try {
                        // Convert to ComfyUI prompt format
                        const fullPrompt = app.graphToPrompt();
                        if (!fullPrompt || !fullPrompt.workflow) {
                            console.warn("Could not build workflow prompt");
                            return {};
                        }
                        
                        const allNodes = fullPrompt.workflow;
                        const selectedNodes = {};
                        const visited = new Set();
                        
                        // Recursively collect dependencies
                        function collectDependencies(nodeId) {
                            const nodeIdStr = String(nodeId);
                            
                            if (visited.has(nodeIdStr) || !allNodes[nodeIdStr]) {
                                return;
                            }
                            
                            visited.add(nodeIdStr);
                            const nodeData = allNodes[nodeIdStr];
                            
                            // Add this node to selection
                            selectedNodes[nodeIdStr] = { ...nodeData };
                            
                            // Find input dependencies
                            if (nodeData.inputs) {
                                Object.values(nodeData.inputs).forEach(input => {
                                    if (Array.isArray(input) && input.length >= 2) {
                                        // This is a node connection [nodeId, outputIndex]
                                        const dependencyNodeId = input[0];
                                        collectDependencies(dependencyNodeId);
                                    }
                                });
                            }
                        }
                        
                        // Start from target node
                        collectDependencies(targetNodeId);
                        
                        console.log(`🚀 AI Color Correction: executing ${Object.keys(selectedNodes).length} nodes instead of ${Object.keys(allNodes).length}`);
                        console.log(`🤖 AI Analysis will detect: faces, scene type, lighting conditions, dominant colors`);
                        return selectedNodes;
                        
                    } catch (error) {
                        console.warn("Error building selective workflow:", error);
                        return {};
                    }
                }

                // Widgets that should trigger real-time updates when changed
                const realtimeWidgets = [
                    "reference_strength", "white_balance_strength", "enhancement_strength", "pop_factor", "effect_strength",
                    "warmth", "vibrancy", "contrast", "brightness", "preset", "variation",
                    "lift", "gamma", "gain", "noise"
                ];

                // === HISTOGRAM FUNCTIONALITY ===
                function generateHistogramData(imageData) {
                    const data = imageData.data;
                    const rHist = new Array(256).fill(0);
                    const gHist = new Array(256).fill(0);
                    const bHist = new Array(256).fill(0);
                    
                    for (let i = 0; i < data.length; i += 4) {
                        const r = Math.floor(data[i]);
                        const g = Math.floor(data[i + 1]);
                        const b = Math.floor(data[i + 2]);
                        
                        rHist[r]++;
                        gHist[g]++;
                        bHist[b]++;
                    }
                    
                    return { red: rHist, green: gHist, blue: bHist };
                }

                function createHistogramWidget() {
                    // Create proper LiteGraph widget
                    const histogramWidget = {
                        type: "histogram_display",
                        name: "📊 RGB Histogram",
                        value: null,
                        size: [280, 120],
                        
                        draw: function(ctx, nodeWidth, widgetY, height) {
                            if (!this.value) return;
                            
                            const margin = 10;
                            const w = Math.min(280, nodeWidth - margin * 2);
                            const h = 100;
                            const x = margin;
                            const y = widgetY;
                            
                            // Background
                            ctx.fillStyle = "#1a1a1a";
                            ctx.fillRect(x, y, w, h);
                            ctx.strokeStyle = "#444";
                            ctx.strokeRect(x, y, w, h);
                            
                            if (this.value && this.value.red) {
                                const histData = this.value;
                                const maxVal = Math.max(
                                    Math.max(...histData.red),
                                    Math.max(...histData.green),
                                    Math.max(...histData.blue)
                                );
                                
                                if (maxVal > 0) {
                                    const scaleX = w / 256;
                                    const scaleY = h / maxVal;
                                    
                                    // Red channel
                                    ctx.strokeStyle = 'rgba(255, 100, 100, 0.8)';
                                    ctx.lineWidth = 1;
                                    ctx.beginPath();
                                    for (let i = 0; i < 256; i++) {
                                        const plotX = x + i * scaleX;
                                        const plotY = y + h - (histData.red[i] * scaleY);
                                        if (i === 0) ctx.moveTo(plotX, plotY);
                                        else ctx.lineTo(plotX, plotY);
                                    }
                                    ctx.stroke();
                                    
                                    // Green channel
                                    ctx.strokeStyle = 'rgba(100, 255, 100, 0.8)';
                                    ctx.beginPath();
                                    for (let i = 0; i < 256; i++) {
                                        const plotX = x + i * scaleX;
                                        const plotY = y + h - (histData.green[i] * scaleY);
                                        if (i === 0) ctx.moveTo(plotX, plotY);
                                        else ctx.lineTo(plotX, plotY);
                                    }
                                    ctx.stroke();
                                    
                                    // Blue channel
                                    ctx.strokeStyle = 'rgba(100, 100, 255, 0.8)';
                                    ctx.beginPath();
                                    for (let i = 0; i < 256; i++) {
                                        const plotX = x + i * scaleX;
                                        const plotY = y + h - (histData.blue[i] * scaleY);
                                        if (i === 0) ctx.moveTo(plotX, plotY);
                                        else ctx.lineTo(plotX, plotY);
                                    }
                                    ctx.stroke();
                                }
                            }
                        },
                        
                        updateHistogram: function(imageElement) {
                            if (!imageElement) return;
                            
                            const tempCanvas = document.createElement('canvas');
                            const tempCtx = tempCanvas.getContext('2d');
                            tempCanvas.width = Math.min(imageElement.width, 512); // Limit size for performance
                            tempCanvas.height = Math.min(imageElement.height, 512);
                            
                            try {
                                tempCtx.drawImage(imageElement, 0, 0, tempCanvas.width, tempCanvas.height);
                                const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                                this.value = generateHistogramData(imageData);
                            } catch (error) {
                                console.log("Histogram update failed:", error);
                                this.value = null;
                            }
                        }
                    };
                    
                    return histogramWidget;
                }

                // === COLOR PALETTE DISPLAY ===
                function displayColorPalette(paletteString) {
                    if (!paletteString) return;
                    
                    const colors = paletteString.split(',');
                    console.log('🎨 Color Palette:', colors.map(c => `%c●%c ${c}`).join(' '), 
                               ...colors.flatMap(c => [`color: ${c}; font-size: 16px`, 'color: inherit']));
                }

                function createColorPaletteWidget() {
                    // Create proper LiteGraph widget
                    const paletteWidget = {
                        type: "color_palette_display",
                        name: "🎨 Color Palette",
                        value: [],
                        size: [320, 80],
                        
                        draw: function(ctx, nodeWidth, widgetY, height) {
                            if (!this.value || this.value.length === 0) return;
                            
                            const margin = 10;
                            const w = Math.min(320, nodeWidth - margin * 2);
                            const h = 60;
                            const x = margin;
                            const y = widgetY;
                            
                            // Background
                            ctx.fillStyle = "#1a1a1a";
                            ctx.fillRect(x, y, w, h);
                            ctx.strokeStyle = "#444";
                            ctx.strokeRect(x, y, w, h);
                            
                            // Draw color swatches
                            const swatchWidth = w / this.value.length;
                            const swatchHeight = h - 20;
                            
                            this.value.forEach((color, index) => {
                                const swatchX = x + (index * swatchWidth) + 2;
                                const swatchY = y + 2;
                                
                                // Draw color swatch
                                ctx.fillStyle = color;
                                ctx.fillRect(swatchX, swatchY, swatchWidth - 4, swatchHeight);
                                
                                // Draw subtle border
                                ctx.strokeStyle = '#666';
                                ctx.lineWidth = 1;
                                ctx.strokeRect(swatchX, swatchY, swatchWidth - 4, swatchHeight);
                                
                                // Draw hex label
                                ctx.fillStyle = '#ffffff';
                                ctx.font = '9px monospace';
                                ctx.textAlign = 'center';
                                const textX = swatchX + (swatchWidth - 4) / 2;
                                const textY = y + h - 5;
                                ctx.fillText(color.toUpperCase(), textX, textY);
                            });
                        },
                        
                        updatePalette: function(colorsArray) {
                            this.value = colorsArray || [];
                        }
                    };
                    
                    return paletteWidget;
                }

				function updateWidgetVisibility() {
					if (!modeWidget) return;
					const currentMode = modeWidget.value;
                    const visibleWidgets = visibilityConfig[currentMode] || [];

                    // Use ComfyUI's built-in 'hidden' property
                    for (const name in allWidgets) {
                        const widget = allWidgets[name];
                        if (widget) {
                            widget.hidden = !visibleWidgets.includes(name);
                        }
                    }

                    // Update AI analysis tooltip based on current mode
                    if (allWidgets.ai_analysis && aiAnalysisTooltips[currentMode]) {
                        allWidgets.ai_analysis.tooltip = aiAnalysisTooltips[currentMode];
                        
                        // Helpful console message to clarify AI analysis works in all modes
                        if (allWidgets.ai_analysis.value) {
                            console.log(`🎯 ${currentMode} Mode: AI Analysis enhances this mode! ${aiAnalysisTooltips[currentMode]}`);
                        }
                    }
					
                    // Trigger a redraw to ensure the node resizes correctly
					node.computeSize();
					node.setDirtyCanvas(true, true);
				}

                function attachRealtimeCallbacks() {
                    // Attach real-time callbacks to relevant widgets
                    for (const widgetName of realtimeWidgets) {
                        const widget = allWidgets[widgetName];
                        if (widget && widget.type === "number") {
                            const originalCallback = widget.callback;
                            widget.callback = function() {
                                if (originalCallback) originalCallback.apply(this, arguments);
                                scheduleRealtimeUpdate();
                            };
                        }
                    }
                }

				// --- Attach Callback to Mode Widget ---
				if (modeWidget) {
					const originalModeCallback = modeWidget.callback;
					modeWidget.callback = function() {
						if (originalModeCallback) originalModeCallback.apply(this, arguments);
						updateWidgetVisibility();
					};
				}

                // === ADVANCED FEATURES SETUP ===
                let histogramWidget = null;
                let paletteWidget = null;
                
                // Add widgets if extract_palette is enabled
                function updateAdvancedFeatures() {
                    const extractPalette = allWidgets.extract_palette;
                    
                    if (extractPalette && extractPalette.value && !histogramWidget) {
                        // Add histogram widget
                        histogramWidget = createHistogramWidget();
                        node.widgets.push(histogramWidget);
                        console.log("📊 Histogram display enabled");
                        
                        // Add color palette widget
                        paletteWidget = createColorPaletteWidget();
                        node.widgets.push(paletteWidget);
                        console.log("🎨 Color palette display enabled");
                        
                        // Try to get current image and update histogram
                        setTimeout(() => {
                            const imageInput = node.inputs && node.inputs[0];
                            if (imageInput && imageInput.link) {
                                // Find connected image node and update histogram
                                const connectedNode = app.graph.getNodeById(imageInput.link.origin_id);
                                if (connectedNode && connectedNode.imgs && connectedNode.imgs[0]) {
                                    histogramWidget.updateHistogram(connectedNode.imgs[0]);
                                }
                            }
                        }, 100);
                        
                    } else if (extractPalette && !extractPalette.value && histogramWidget) {
                        // Remove histogram widget
                        const histogramIndex = node.widgets.indexOf(histogramWidget);
                        if (histogramIndex !== -1) {
                            node.widgets.splice(histogramIndex, 1);
                        }
                        histogramWidget = null;
                        console.log("📊 Histogram display disabled");
                        
                        // Remove palette widget
                        const paletteIndex = node.widgets.indexOf(paletteWidget);
                        if (paletteIndex !== -1) {
                            node.widgets.splice(paletteIndex, 1);
                        }
                        paletteWidget = null;
                        console.log("🎨 Color palette display disabled");
                    }
                    
                    node.computeSize();
                    node.setDirtyCanvas(true, true);
                }

                // Monitor extract_palette changes
                if (allWidgets.extract_palette) {
                    const originalCallback = allWidgets.extract_palette.callback;
                    allWidgets.extract_palette.callback = function() {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        setTimeout(updateAdvancedFeatures, 10);
                    };
                }

                // Listen for execution completion to update histogram and display palette
                const originalOnExecuted = node.onExecuted;
                node.onExecuted = function(message) {
                    if (originalOnExecuted) originalOnExecuted.call(this, message);
                    
                    // Display color palette if extracted
                    if (message.palette_data && message.palette_data.length > 0) {
                        displayColorPalette(message.palette_data);
                        
                        // Update palette widget if enabled
                        if (paletteWidget && message.palette_data) {
                            const colors = message.palette_data.split(',');
                            paletteWidget.updatePalette(colors);
                            node.setDirtyCanvas(true, true);
                        }
                    }
                    
                    // Update histogram if enabled
                    if (histogramWidget && node.imgs && node.imgs[0]) {
                        histogramWidget.updateHistogram(node.imgs[0]);
                    }
                    
                    // Enhanced image preview integration for outputs
                    // Check if we have histogram and palette image outputs
                    if (extract_palette && extractPalette && extractPalette.value) {
                        try {
                            // Try to create preview for histogram and palette images if available
                            // This leverages ComfyUI's built-in image preview system
                            const outputImages = [];
                            
                            // Add main processed image
                            if (node.imgs && node.imgs[0]) {
                                outputImages.push(node.imgs[0]);
                            }
                            
                            // Note: Histogram and palette images are now available as separate outputs
                            // Users can connect them to Preview Image nodes for visualization
                            console.log("🖼️ Histogram and palette images available as separate node outputs");
                            
                        } catch (error) {
                            console.log("Preview integration note:", error);
                        }
                    }
                };

				// --- Initial UI Setup ---
				setTimeout(() => {
                    updateWidgetVisibility();
                    attachRealtimeCallbacks();
                    updateAdvancedFeatures();
                }, 10);
			};
		}
	},
});
