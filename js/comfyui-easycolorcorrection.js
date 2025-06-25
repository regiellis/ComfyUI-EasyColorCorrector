import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

/**
 * Frontend logic for the EasyColorCorrection node.
 * This script provides a dynamic UI for three powerful modes with optional real-time preview:
 * 1.  Auto: 🤖 AI-POWERED ENHANCEMENT - Face detection, scene analysis, perceptual color science
 * 2.  Preset: 🎨 STYLE-BASED PRESETS - Curated looks with intelligent variation
 * 3.  Manual: 🎛️  PROFESSIONAL GRADING - Full manual control with advanced tools
 */

// EasyColorCorrection Global Settings
const SETTINGS = {
	NOTIFICATIONS: {
		NONE: "none",
		MINIMAL: "minimal", 
		FULL: "full"
	},
	PREVIEW: {
		AUTO_ENABLE: "auto_enable_preview",
		QUALITY: "preview_quality"
	},
	PROCESSING: {
		AUTO_GPU: "auto_gpu_detection",
		PREFER_CPU: "prefer_cpu_processing",
		BATCH_SIZE: "batch_processing_size"
	}
};

// Helper function to show notifications based on settings
function showNotification(message, type = "info") {
	const notificationLevel = app.ui.settings.getSettingValue("EasyColorCorrection.notifications", SETTINGS.NOTIFICATIONS.FULL);
	
	switch (notificationLevel) {
		case SETTINGS.NOTIFICATIONS.NONE:
			// No notifications
			break;
		case SETTINGS.NOTIFICATIONS.MINIMAL:
			// Only show errors and important messages
			if (type === "error" || type === "important") {
				app.ui.dialog.show(message);
			}
			break;
		case SETTINGS.NOTIFICATIONS.FULL:
		default:
			// Show all notifications
			app.ui.dialog.show(message);
			break;
	}
	
	// Always log to console for debugging
	debugLog(message);
}

// Helper function for debug logging
function debugLog(message, level = "info") {
	const debugMode = app.ui.settings.getSettingValue("EasyColorCorrection.debug_mode", false);
	
	if (debugMode || level === "error") {
		const timestamp = new Date().toISOString().substr(11, 8);
		console.log(`[${timestamp}] [EasyColorCorrection] ${message}`);
	}
}

// Helper function to check batch size warnings
function checkBatchSizeWarning(frameCount) {
	const threshold = app.ui.settings.getSettingValue("EasyColorCorrection.batch_processing_size", 100);
	
	if (frameCount > threshold) {
		showNotification(
			`⚠️ Large batch detected: ${frameCount} frames (>${threshold}). Consider CPU processing to save VRAM.`, 
			"important"
		);
		return true;
	}
	return false;
}

app.registerExtension({
	name: "comfyui-easycolorcorrection",
	
	async setup() {
		// Register settings when extension loads
		app.ui.settings.addSetting({
			id: "EasyColorCorrection.notifications",
			name: "🎨 Easy Color Correction: Notification Level",
			type: "combo",
			options: [
				{ value: SETTINGS.NOTIFICATIONS.NONE, text: "None - No notifications" },
				{ value: SETTINGS.NOTIFICATIONS.MINIMAL, text: "Minimal - Errors and important only" },
				{ value: SETTINGS.NOTIFICATIONS.FULL, text: "Full - All notifications (default)" }
			],
			defaultValue: SETTINGS.NOTIFICATIONS.FULL,
			tooltip: "Control how many notifications Easy Color Correction shows"
		});
		
		app.ui.settings.addSetting({
			id: "EasyColorCorrection.auto_enable_preview",
			name: "🎨 Easy Color Correction: Auto-Enable Preview",
			type: "boolean",
			defaultValue: true,
			tooltip: "Automatically enable real-time preview for new nodes"
		});
		
		app.ui.settings.addSetting({
			id: "EasyColorCorrection.preview_quality",
			name: "🎨 Easy Color Correction: Preview Quality",
			type: "combo",
			options: [
				{ value: "low", text: "Low - Faster preview" },
				{ value: "medium", text: "Medium - Balanced (default)" },
				{ value: "high", text: "High - Best quality" }
			],
			defaultValue: "medium",
			tooltip: "Quality level for real-time preview rendering"
		});
		
		app.ui.settings.addSetting({
			id: "EasyColorCorrection.auto_gpu_detection",
			name: "🎨 Easy Color Correction: Auto GPU Detection",
			type: "boolean",
			defaultValue: true,
			tooltip: "Automatically detect and suggest GPU usage for batch processing"
		});
		
		app.ui.settings.addSetting({
			id: "EasyColorCorrection.prefer_cpu_processing",
			name: "🎨 Easy Color Correction: Prefer CPU Processing",
			type: "boolean",
			defaultValue: false,
			tooltip: "Default to CPU processing for new batch nodes (saves VRAM)"
		});
		
		app.ui.settings.addSetting({
			id: "EasyColorCorrection.batch_processing_size",
			name: "🎨 Easy Color Correction: Batch Size Warning Threshold",
			type: "number",
			defaultValue: 100,
			min: 10,
			max: 1000,
			step: 10,
			tooltip: "Show VRAM warning when batch size exceeds this number of frames"
		});
		
		app.ui.settings.addSetting({
			id: "EasyColorCorrection.show_processing_time",
			name: "🎨 Easy Color Correction: Show Processing Time",
			type: "boolean",
			defaultValue: true,
			tooltip: "Display processing time in notifications"
		});
		
		app.ui.settings.addSetting({
			id: "EasyColorCorrection.debug_mode",
			name: "🎨 Easy Color Correction: Debug Mode",
			type: "boolean",
			defaultValue: false,
			tooltip: "Enable detailed console logging for debugging"
		});
	},

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "ColorCorrectionViewer") {
			// Handle ColorCorrectionViewer node using proper ComfyUI patterns
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				if (onNodeCreated) {
					onNodeCreated.apply(this, arguments);
				}

				const node = this;
				
				// Add simple info display widget
				const infoWidget = node.addWidget("text", "📊 Video Info", "Waiting for data...", function(v) {}, {
					multiline: false,
					readonly: true
				});
				
				// Add playback controls using standard widgets
				const playButton = node.addWidget("button", "▶ Play/Pause", null, function() {
					const videoData = node.videoData;
					if (videoData && videoData.frames) {
						videoData.isPlaying = !videoData.isPlaying;
						playButton.name = videoData.isPlaying ? "⏸ Pause" : "▶ Play";
						
						if (videoData.isPlaying) {
							startPlayback(node);
						} else {
							stopPlayback(node);
						}
					}
				});
				
				const frameWidget = node.addWidget("number", "Current Frame", 0, function(v) {
					const videoData = node.videoData;
					if (videoData && videoData.frames) {
						videoData.currentFrame = Math.max(0, Math.min(v, videoData.totalFrames - 1));
						updateFrameDisplay(node);
					}
				}, {
					min: 0,
					max: 100,
					step: 1
				});
				
				// Store references
				node.infoWidget = infoWidget;
				node.playButton = playButton;
				node.frameWidget = frameWidget;
				node.videoData = {
					isPlaying: false,
					currentFrame: 0,
					totalFrames: 0,
					frames: null,
					fps: 24.0,
					playbackInterval: null
				};
				
				// Frame display area (custom drawing)
				const displayWidget = {
					type: "custom_display",
					name: "Frame Display",
					draw: function(ctx, nodeWidth, widgetY, height) {
						const margin = 10;
						const w = Math.min(400, nodeWidth - margin * 2);
						const h = 300;
						const x = margin;
						const y = widgetY;
						
						// Background
						ctx.fillStyle = "#1a1a1a";
						ctx.fillRect(x, y, w, h);
						ctx.strokeStyle = "#444";
						ctx.strokeRect(x, y, w, h);
						
						// Display current frame or message
						ctx.fillStyle = "#ffffff";
						ctx.font = "14px Arial";
						ctx.textAlign = "center";
						
						const videoData = node.videoData;
						if (videoData && videoData.frames && videoData.totalFrames > 0) {
							const frameText = `Frame ${videoData.currentFrame + 1} of ${videoData.totalFrames}`;
							ctx.fillText(frameText, x + w/2, y + h/2);
							
							// Progress indicator
							const progress = videoData.currentFrame / (videoData.totalFrames - 1);
							const progressWidth = w - 20;
							const progressHeight = 4;
							const progressX = x + 10;
							const progressY = y + h - 20;
							
							ctx.fillStyle = "#333";
							ctx.fillRect(progressX, progressY, progressWidth, progressHeight);
							ctx.fillStyle = "#51cf66";
							ctx.fillRect(progressX, progressY, progressWidth * progress, progressHeight);
						} else {
							ctx.fillText("No video data", x + w/2, y + h/2);
						}
					}
				};
				
				node.addCustomWidget(displayWidget);
				
				// Playback functions
				function startPlayback(node) {
					const videoData = node.videoData;
					if (videoData.playbackInterval) {
						clearInterval(videoData.playbackInterval);
					}
					
					const frameDelay = 1000 / videoData.fps;
					videoData.playbackInterval = setInterval(() => {
						videoData.currentFrame++;
						if (videoData.currentFrame >= videoData.totalFrames) {
							videoData.currentFrame = 0; // Loop
						}
						
						node.frameWidget.value = videoData.currentFrame;
						updateFrameDisplay(node);
						node.setDirtyCanvas(true, true);
					}, frameDelay);
				}
				
				function stopPlayback(node) {
					const videoData = node.videoData;
					if (videoData.playbackInterval) {
						clearInterval(videoData.playbackInterval);
						videoData.playbackInterval = null;
					}
				}
				
				function updateFrameDisplay(node) {
					// This would update the actual frame display
					// For now, just trigger a redraw
					node.setDirtyCanvas(true, true);
				}
			};
			
			// Override onExecuted to receive data properly
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function(message) {
				if (onExecuted) {
					onExecuted.call(this, message);
				}
				
				debugLog("ColorCorrectionViewer onExecuted: " + JSON.stringify(message));
				
				// Update video data from the execution message
				if (message && typeof message === 'object') {
					const videoData = this.videoData;
					
					// Look for frame count in various possible locations
					const frameCount = message.frame_count || message.frames || message.total_frames;
					
					if (frameCount && Array.isArray(frameCount) && frameCount.length > 0) {
						videoData.totalFrames = frameCount[0];
						videoData.currentFrame = 0;
						videoData.fps = this.widgets.find(w => w.name === "fps")?.value || 24.0;
						
						// Update UI
						this.infoWidget.value = `${videoData.totalFrames} frames @ ${videoData.fps} FPS`;
						this.frameWidget.options.max = videoData.totalFrames - 1;
						this.frameWidget.value = 0;
						
						// Check for batch size warnings
						checkBatchSizeWarning(videoData.totalFrames);
						
						// Show success notification
						showNotification(`📹 Video loaded: ${videoData.totalFrames} frames`, "info");
						
						debugLog(`Video data updated: ${videoData.totalFrames} frames @ ${videoData.fps} FPS`);
						this.setDirtyCanvas(true, true);
					} else {
						debugLog("No frame count found in message: " + Object.keys(message).join(", "), "error");
						showNotification("❌ No video data received", "error");
					}
				}
			};
			
			// Also listen for API execution events as backup
			const onExecutionStart = nodeType.prototype.onExecutionStart;
			nodeType.prototype.onExecutionStart = function() {
				if (onExecutionStart) {
					onExecutionStart.call(this);
				}
				
				// Reset UI state
				this.infoWidget.value = "Processing...";
				debugLog("ColorCorrectionViewer execution started");
			};
			
			// Listen for node property changes (outputs)
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function(type, slotIndex, isConnected, link_info, slot_info) {
				if (onConnectionsChange) {
					onConnectionsChange.call(this, type, slotIndex, isConnected, link_info, slot_info);
				}
				
				// Check if we can access outputs directly
				if (this.outputs && this.outputs[1] && this.outputs[1].value !== undefined) {
					const frameCount = this.outputs[1].value;
					if (frameCount && frameCount !== this.videoData.totalFrames) {
						this.videoData.totalFrames = frameCount;
						this.videoData.currentFrame = 0;
						this.infoWidget.value = `${frameCount} frames @ ${this.videoData.fps} FPS`;
						this.frameWidget.options.max = frameCount - 1;
						this.frameWidget.value = 0;
						console.log("Got frame count from outputs:", frameCount);
						this.setDirtyCanvas(true, true);
					}
				}
			};
		}
		
		else if (nodeData.name === "EasyColorCorrection") {

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				if (onNodeCreated) {
					onNodeCreated.apply(this, arguments);
				}

				const node = this;
				const modeWidget = node.widgets.find(w => w.name === "mode");
				const realtimeWidget = node.widgets.find(w => w.name === "realtime_preview");
				const useGpuWidget = node.widgets.find(w => w.name === "use_gpu");
				
				// Apply global settings to new nodes
				if (realtimeWidget) {
					const autoEnablePreview = app.ui.settings.getSettingValue("EasyColorCorrection.auto_enable_preview", true);
					if (autoEnablePreview) {
						realtimeWidget.value = true;
					}
				}
				
				if (useGpuWidget) {
					const preferCpu = app.ui.settings.getSettingValue("EasyColorCorrection.prefer_cpu_processing", false);
					if (preferCpu) {
						useGpuWidget.value = false;
					}
				}

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
                    tint: node.widgets.find(w => w.name === "tint"),
                    preset: node.widgets.find(w => w.name === "preset"),
                    variation: node.widgets.find(w => w.name === "variation"),
                    lift: node.widgets.find(w => w.name === "lift"),
                    gamma: node.widgets.find(w => w.name === "gamma"),
                    gain: node.widgets.find(w => w.name === "gain"),
                    noise: node.widgets.find(w => w.name === "noise"),
                    effect_strength: node.widgets.find(w => w.name === "effect_strength"),
                    // Colorize mode widgets
                    colorize_strength: node.widgets.find(w => w.name === "colorize_strength"),
                    skin_warmth: node.widgets.find(w => w.name === "skin_warmth"),
                    sky_saturation: node.widgets.find(w => w.name === "sky_saturation"),
                    vegetation_green: node.widgets.find(w => w.name === "vegetation_green"),
                    sepia_tone: node.widgets.find(w => w.name === "sepia_tone"),
                    colorize_mode: node.widgets.find(w => w.name === "colorize_mode"),
                    use_gpu: node.widgets.find(w => w.name === "use_gpu"),
                };

                // Mode-specific AI analysis tooltips to clarify how it works in each mode
                const aiAnalysisTooltips = {
                    "Auto": "🚀 GPU-accelerated AI: tensor-based scene detection, lighting analysis, edge detection",
                    "Preset": "🎨 GPU-optimized presets: fast content type detection (anime/concept art/portrait)",
                    "Manual": "🎛️ GPU-enhanced guidance: advanced masking and controls with tensor operations",
                    "Colorize": "🎨 Smart colorization: automatically skips art/anime, only colorizes photos and realistic content"
                };

                // Configuration defining which widgets are visible in each mode
                const visibilityConfig = {
                    "Auto": ["reference_strength", "extract_palette", "realtime_preview", "ai_analysis", "adjust_for_skin_tone", "white_balance_strength", "enhancement_strength", "pop_factor", "effect_strength", "use_gpu"],
                    "Preset": ["reference_strength", "extract_palette", "realtime_preview", "ai_analysis", "adjust_for_skin_tone", "white_balance_strength", "enhancement_strength", "preset", "variation", "effect_strength", "use_gpu"],
                    "Manual": ["reference_strength", "extract_palette", "realtime_preview", "ai_analysis", "adjust_for_skin_tone", "white_balance_strength", "enhancement_strength", "warmth", "vibrancy", "contrast", "brightness", "tint", "lift", "gamma", "gain", "noise", "effect_strength", "use_gpu"],
                    "Colorize": ["reference_strength", "extract_palette", "realtime_preview", "ai_analysis", "adjust_for_skin_tone", "colorize_strength", "skin_warmth", "sky_saturation", "vegetation_green", "sepia_tone", "colorize_mode", "effect_strength", "use_gpu"],
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
                    "warmth", "vibrancy", "contrast", "brightness", "tint", "preset", "variation",
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
                    const aiAnalysisEnabled = allWidgets.ai_analysis ? allWidgets.ai_analysis.value : true;

                    // Use ComfyUI's built-in 'hidden' property
                    for (const name in allWidgets) {
                        const widget = allWidgets[name];
                        if (widget) {
                            let shouldShow = visibleWidgets.includes(name);
                            
                            // Hide enhancement_strength when AI analysis is disabled
                            if (name === "enhancement_strength" && !aiAnalysisEnabled) {
                                shouldShow = false;
                            }
                            
                            widget.hidden = !shouldShow;
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
						
						// Auto-disable AI analysis when switching to Manual mode
						if (this.value === "Manual" && allWidgets.ai_analysis) {
							allWidgets.ai_analysis.value = false;
							console.log("🎛️ Manual Mode: Auto-disabled AI analysis for pure manual control");
						}
						
						updateWidgetVisibility();
					};
				}

				// --- Attach Callback to Preset Widget ---
				if (allWidgets.preset) {
					const originalPresetCallback = allWidgets.preset.callback;
					allWidgets.preset.callback = function() {
						if (originalPresetCallback) originalPresetCallback.apply(this, arguments);
						updateSlidersFromPreset(this.value);
					};
				}

				// --- Attach Callback to AI Analysis Widget ---
				if (allWidgets.ai_analysis) {
					const originalAiCallback = allWidgets.ai_analysis.callback;
					allWidgets.ai_analysis.callback = function() {
						if (originalAiCallback) originalAiCallback.apply(this, arguments);
						updateWidgetVisibility(); // Update visibility when AI analysis changes
					};
				}

                // === PRESET VALUE DEFINITIONS ===
                const PRESETS = {
                    "Natural Portrait": { warmth: 0.08, vibrancy: 0.12, contrast: 0.08, brightness: 0.03 },
                    "Warm Portrait": { warmth: 0.18, vibrancy: 0.15, contrast: 0.06, brightness: 0.05 },
                    "Cool Portrait": { warmth: -0.12, vibrancy: 0.08, contrast: 0.10, brightness: 0.02 },
                    "High Key Portrait": { warmth: 0.05, vibrancy: 0.08, contrast: -0.05, brightness: 0.20 },
                    "Dramatic Portrait": { warmth: 0.02, vibrancy: 0.20, contrast: 0.25, brightness: -0.05 },
                    "Epic Fantasy": { warmth: 0.1, vibrancy: 0.4, contrast: 0.3, brightness: 0.05 },
                    "Sci-Fi Chrome": { warmth: -0.2, vibrancy: 0.3, contrast: 0.35, brightness: 0.1 },
                    "Dark Fantasy": { warmth: -0.1, vibrancy: 0.25, contrast: 0.4, brightness: -0.15 },
                    "Vibrant Concept": { warmth: 0.05, vibrancy: 0.5, contrast: 0.25, brightness: 0.08 },
                    "Matte Painting": { warmth: 0.08, vibrancy: 0.3, contrast: 0.2, brightness: 0.03 },
                    "Digital Art": { warmth: 0.0, vibrancy: 0.45, contrast: 0.28, brightness: 0.05 },
                    "Anime Bright": { warmth: 0.12, vibrancy: 0.45, contrast: 0.2, brightness: 0.12 },
                    "Anime Moody": { warmth: -0.05, vibrancy: 0.35, contrast: 0.25, brightness: -0.05 },
                    "Cyberpunk": { warmth: -0.15, vibrancy: 0.45, contrast: 0.25, brightness: -0.03 },
                    "Pastel Dreams": { warmth: 0.12, vibrancy: -0.08, contrast: -0.08, brightness: 0.12 },
                    "Neon Nights": { warmth: -0.18, vibrancy: 0.40, contrast: 0.20, brightness: -0.05 },
                    "Comic Book": { warmth: 0.05, vibrancy: 0.5, contrast: 0.35, brightness: 0.08 },
                    "Cinematic": { warmth: 0.12, vibrancy: 0.15, contrast: 0.18, brightness: 0.02 },
                    "Teal & Orange": { warmth: -0.08, vibrancy: 0.25, contrast: 0.15, brightness: 0.0 },
                    "Film Noir": { warmth: -0.05, vibrancy: -0.80, contrast: 0.35, brightness: -0.08 },
                    "Vintage Film": { warmth: 0.15, vibrancy: -0.10, contrast: 0.12, brightness: 0.03 },
                    "Bleach Bypass": { warmth: -0.02, vibrancy: -0.25, contrast: 0.30, brightness: 0.05 },
                    "Golden Hour": { warmth: 0.25, vibrancy: 0.18, contrast: 0.08, brightness: 0.08 },
                    "Blue Hour": { warmth: -0.20, vibrancy: 0.15, contrast: 0.12, brightness: 0.02 },
                    "Sunny Day": { warmth: 0.15, vibrancy: 0.20, contrast: 0.10, brightness: 0.08 },
                    "Overcast": { warmth: -0.08, vibrancy: 0.05, contrast: 0.08, brightness: 0.05 },
                    "Sepia": { warmth: 0.30, vibrancy: -0.35, contrast: 0.08, brightness: 0.03 },
                    "Black & White": { warmth: 0.0, vibrancy: -1.0, contrast: 0.15, brightness: 0.0 },
                    "Faded": { warmth: 0.05, vibrancy: -0.15, contrast: -0.12, brightness: 0.08 },
                    "Moody": { warmth: -0.08, vibrancy: 0.12, contrast: 0.20, brightness: -0.08 }
                };

                // Function to update sliders when preset changes
                function updateSlidersFromPreset(presetName) {
                    const presetValues = PRESETS[presetName];
                    if (!presetValues) return;
                    
                    // Update warmth, vibrancy, contrast, brightness sliders
                    if (allWidgets.warmth && allWidgets.warmth.type === "number") {
                        allWidgets.warmth.value = presetValues.warmth;
                    }
                    if (allWidgets.vibrancy && allWidgets.vibrancy.type === "number") {
                        allWidgets.vibrancy.value = presetValues.vibrancy;
                    }
                    if (allWidgets.contrast && allWidgets.contrast.type === "number") {
                        allWidgets.contrast.value = presetValues.contrast;
                    }
                    if (allWidgets.brightness && allWidgets.brightness.type === "number") {
                        allWidgets.brightness.value = presetValues.brightness;
                    }
                    
                    // Trigger realtime update if enabled
                    scheduleRealtimeUpdate();
                    
                    // Force node redraw to show updated slider values
                    node.setDirtyCanvas(true, true);
                    
                    console.log(`🎨 Applied preset "${presetName}":`, presetValues);
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

		// === BATCH COLOR CORRECTION NODE ===
		if (nodeData.name === "BatchColorCorrection") {

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				if (onNodeCreated) {
					onNodeCreated.apply(this, arguments);
				}

				const node = this;
				const modeWidget = node.widgets.find(w => w.name === "mode");
				const framesPerBatchWidget = node.widgets.find(w => w.name === "frames_per_batch");
				const useGpuWidget = node.widgets.find(w => w.name === "use_gpu");

                // A clear map of all widgets we need to control for batch processing
                const allWidgets = {
                    frames_per_batch: node.widgets.find(w => w.name === "frames_per_batch"),
                    use_gpu: node.widgets.find(w => w.name === "use_gpu"),
                    reference_strength: node.widgets.find(w => w.name === "reference_strength"),
                    extract_palette: node.widgets.find(w => w.name === "extract_palette"),
                    ai_analysis: node.widgets.find(w => w.name === "ai_analysis"),
                    adjust_for_skin_tone: node.widgets.find(w => w.name === "adjust_for_skin_tone"),
                    white_balance_strength: node.widgets.find(w => w.name === "white_balance_strength"),
                    enhancement_strength: node.widgets.find(w => w.name === "enhancement_strength"),
                    warmth: node.widgets.find(w => w.name === "warmth"),
                    vibrancy: node.widgets.find(w => w.name === "vibrancy"),
                    contrast: node.widgets.find(w => w.name === "contrast"),
                    brightness: node.widgets.find(w => w.name === "brightness"),
                    preset: node.widgets.find(w => w.name === "preset"),
                    lift: node.widgets.find(w => w.name === "lift"),
                    gamma: node.widgets.find(w => w.name === "gamma"),
                    gain: node.widgets.find(w => w.name === "gain"),
                    noise: node.widgets.find(w => w.name === "noise"),
                    effect_strength: node.widgets.find(w => w.name === "effect_strength"),
                };
				
				// Apply global settings to new BatchColorCorrection nodes
				if (useGpuWidget) {
					const preferCpu = app.ui.settings.getSettingValue("EasyColorCorrection.prefer_cpu_processing", false);
					if (preferCpu) {
						useGpuWidget.value = false;
						console.log("🎬 BatchColorCorrection: Applied CPU preference setting - GPU disabled by default");
					}
				}

                // Mode-specific tooltips for batch processing
                const aiAnalysisTooltips = {
                    "Auto": "🚀 GPU-accelerated AI analysis: fast tensor-based scene detection, lighting analysis, edge detection",
                    "Preset": "🎨 GPU-optimized presets: consistent style across frames with efficient tensor operations",
                    "Manual": "🎛️ GPU-enhanced controls: advanced color grading with full GPU acceleration"
                };

                // Batch-specific configuration defining which widgets are visible in each mode
                const visibilityConfig = {
                    "Auto": ["frames_per_batch", "use_gpu", "reference_strength", "extract_palette", "ai_analysis", "adjust_for_skin_tone", "white_balance_strength", "enhancement_strength", "effect_strength"],
                    "Preset": ["frames_per_batch", "use_gpu", "reference_strength", "extract_palette", "ai_analysis", "adjust_for_skin_tone", "white_balance_strength", "enhancement_strength", "preset", "effect_strength"],
                    "Manual": ["frames_per_batch", "use_gpu", "reference_strength", "extract_palette", "ai_analysis", "adjust_for_skin_tone", "white_balance_strength", "enhancement_strength", "warmth", "vibrancy", "contrast", "brightness", "lift", "gamma", "gain", "noise", "effect_strength"],
                };

                // Progress tracking for batch processing
                let batchProgress = {
                    currentBatch: 0,
                    totalBatches: 0,
                    totalFrames: 0,
                    isProcessing: false
                };

                function createBatchProgressWidget() {
                    const progressWidget = {
                        type: "batch_progress_display",
                        name: "🎬 Batch Progress",
                        value: batchProgress,
                        size: [320, 60],
                        
                        draw: function(ctx, nodeWidth, widgetY, height) {
                            const margin = 10;
                            const w = Math.min(320, nodeWidth - margin * 2);
                            const h = 50;
                            const x = margin;
                            const y = widgetY;
                            
                            // Background
                            ctx.fillStyle = "#1a1a1a";
                            ctx.fillRect(x, y, w, h);
                            ctx.strokeStyle = "#444";
                            ctx.strokeRect(x, y, w, h);
                            
                            if (this.value.totalFrames > 0) {
                                // Progress text
                                ctx.fillStyle = "#ffffff";
                                ctx.font = "12px Arial";
                                ctx.textAlign = "left";
                                
                                const progressText = this.value.isProcessing 
                                    ? `🚀 GPU Processing: Batch ${this.value.currentBatch}/${this.value.totalBatches} • ${this.value.totalFrames} frames`
                                    : `🚀 GPU Ready: ${this.value.totalFrames} frames • ${this.value.totalBatches} batches`;
                                    
                                ctx.fillText(progressText, x + 5, y + 20);
                                
                                // Progress bar
                                if (this.value.isProcessing && this.value.totalBatches > 0) {
                                    const progressWidth = w - 10;
                                    const progressHeight = 8;
                                    const progressX = x + 5;
                                    const progressY = y + 30;
                                    
                                    // Background bar
                                    ctx.fillStyle = "#333";
                                    ctx.fillRect(progressX, progressY, progressWidth, progressHeight);
                                    
                                    // Progress bar
                                    const progress = this.value.currentBatch / this.value.totalBatches;
                                    ctx.fillStyle = "#4CAF50";
                                    ctx.fillRect(progressX, progressY, progressWidth * progress, progressHeight);
                                    
                                    // Border
                                    ctx.strokeStyle = "#666";
                                    ctx.strokeRect(progressX, progressY, progressWidth, progressHeight);
                                }
                            }
                        },
                        
                        updateProgress: function(current, total, frames, isProcessing = false) {
                            this.value.currentBatch = current;
                            this.value.totalBatches = total;
                            this.value.totalFrames = frames;
                            this.value.isProcessing = isProcessing;
                        }
                    };
                    
                    return progressWidget;
                }

                // Frame count estimation based on frames_per_batch
                function updateBatchEstimation() {
                    if (!framesPerBatchWidget) return;
                    
                    const framesPerBatch = framesPerBatchWidget.value || 16;
                    
                    // Try to estimate frame count from connected input
                    if (node.inputs && node.inputs[0] && node.inputs[0].link) {
                        const connectedNode = app.graph.getNodeById(node.inputs[0].link.origin_id);
                        if (connectedNode && connectedNode.imgs) {
                            const estimatedFrames = connectedNode.imgs.length || 1;
                            const estimatedBatches = Math.ceil(estimatedFrames / framesPerBatch);
                            
                            if (progressWidget) {
                                progressWidget.updateProgress(0, estimatedBatches, estimatedFrames, false);
                                node.setDirtyCanvas(true, true);
                            }
                        }
                    }
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
                        
                        // Helpful console message for batch processing
                        if (allWidgets.ai_analysis.value) {
                            console.log(`🎬 Batch ${currentMode} Mode: AI Analysis optimized for video sequences! ${aiAnalysisTooltips[currentMode]}`);
                        }
                    }
					
                    // Trigger a redraw to ensure the node resizes correctly
					node.computeSize();
					node.setDirtyCanvas(true, true);
				}

				// --- Attach Callback to Mode Widget ---
				if (modeWidget) {
					const originalModeCallback = modeWidget.callback;
					modeWidget.callback = function() {
						if (originalModeCallback) originalModeCallback.apply(this, arguments);
						updateWidgetVisibility();
					};
				}

                // --- Attach Callback to Frames Per Batch Widget ---
                if (framesPerBatchWidget) {
                    const originalFramesCallback = framesPerBatchWidget.callback;
                    framesPerBatchWidget.callback = function() {
                        if (originalFramesCallback) originalFramesCallback.apply(this, arguments);
                        updateBatchEstimation();
                        
                        // Console guidance for batch sizing
                        const frames = this.value;
                        if (frames <= 8) {
                            console.log(`📊 Batch Size: ${frames} frames - High quality, slower processing`);
                        } else if (frames <= 32) {
                            console.log(`⚖️ Batch Size: ${frames} frames - Balanced quality and speed`);
                        } else {
                            console.log(`⚡ Batch Size: ${frames} frames - Fast processing, high memory usage`);
                        }
                    };
                }

                // === BATCH PROGRESS WIDGET SETUP ===
                let progressWidget = null;
                
                function addProgressWidget() {
                    if (!progressWidget) {
                        progressWidget = createBatchProgressWidget();
                        node.widgets.push(progressWidget);
                        console.log("🎬 Batch progress display enabled");
                        updateBatchEstimation();
                    }
                }

                // === BATCH PREVIEW WIDGET ===
                let previewWidget = null;
                
                function createBatchPreviewWidget() {
                    const previewWidget = {
                        type: "batch_preview_display",
                        name: "🎞️ Frame Preview",
                        value: {
                            frames: [],
                            currentFrame: 0,
                            isPlaying: false,
                            frameRate: 4, // fps for preview playback
                            lastUpdate: 0
                        },
                        size: [400, 320],
                        
                        draw: function(ctx, nodeWidth, widgetY, height) {
                            const margin = 10;
                            const w = Math.min(400, nodeWidth - margin * 2);
                            const h = 300;
                            const x = margin;
                            const y = widgetY;
                            
                            // Main preview area background
                            ctx.fillStyle = "#1a1a1a";
                            ctx.fillRect(x, y, w, h);
                            ctx.strokeStyle = "#444";
                            ctx.strokeRect(x, y, w, h);
                            
                            if (this.value.frames.length > 0) {
                                const currentFrame = this.value.frames[this.value.currentFrame];
                                if (currentFrame) {
                                    // Calculate aspect ratio and fit image
                                    const imgAspect = currentFrame.width / currentFrame.height;
                                    const previewAspect = w / (h - 80); // Leave space for controls
                                    
                                    let drawWidth, drawHeight, drawX, drawY;
                                    if (imgAspect > previewAspect) {
                                        drawWidth = w - 20;
                                        drawHeight = drawWidth / imgAspect;
                                        drawX = x + 10;
                                        drawY = y + 10 + (h - 80 - drawHeight) / 2;
                                    } else {
                                        drawHeight = h - 100;
                                        drawWidth = drawHeight * imgAspect;
                                        drawX = x + (w - drawWidth) / 2;
                                        drawY = y + 10;
                                    }
                                    
                                    // Draw current frame
                                    try {
                                        ctx.drawImage(currentFrame, drawX, drawY, drawWidth, drawHeight);
                                        
                                        // Frame border
                                        ctx.strokeStyle = "#666";
                                        ctx.lineWidth = 1;
                                        ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);
                                    } catch (error) {
                                        // Fallback if image can't be drawn
                                        ctx.fillStyle = "#333";
                                        ctx.fillRect(drawX, drawY, drawWidth, drawHeight);
                                        ctx.fillStyle = "#999";
                                        ctx.font = "14px Arial";
                                        ctx.textAlign = "center";
                                        ctx.fillText("Frame Preview", drawX + drawWidth/2, drawY + drawHeight/2);
                                    }
                                }
                                
                                // Controls area
                                const controlsY = y + h - 70;
                                
                                // Progress bar
                                const progressWidth = w - 120;
                                const progressHeight = 6;
                                const progressX = x + 10;
                                const progressBarY = controlsY + 5;
                                
                                // Progress background
                                ctx.fillStyle = "#333";
                                ctx.fillRect(progressX, progressBarY, progressWidth, progressHeight);
                                
                                // Progress fill
                                if (this.value.frames.length > 1) {
                                    const progress = this.value.currentFrame / (this.value.frames.length - 1);
                                    ctx.fillStyle = "#4CAF50";
                                    ctx.fillRect(progressX, progressBarY, progressWidth * progress, progressHeight);
                                }
                                
                                // Progress border
                                ctx.strokeStyle = "#666";
                                ctx.strokeRect(progressX, progressBarY, progressWidth, progressHeight);
                                
                                // Frame counter
                                ctx.fillStyle = "#ffffff";
                                ctx.font = "12px Arial";
                                ctx.textAlign = "right";
                                ctx.fillText(`${this.value.currentFrame + 1}/${this.value.frames.length}`, x + w - 10, controlsY + 15);
                                
                                // Help text (small)
                                ctx.fillStyle = "#888";
                                ctx.font = "10px Arial";
                                ctx.textAlign = "left";
                                ctx.fillText("Space: play/pause, ←→: frame, Del: clear, +/-: speed", x + 10, controlsY + 35);
                                
                                // Play/Pause button area
                                const buttonSize = 30;
                                const buttonX = x + progressWidth + 20;
                                const buttonY = controlsY;
                                
                                ctx.fillStyle = this.value.isPlaying ? "#ff6b6b" : "#4CAF50";
                                ctx.fillRect(buttonX, buttonY, buttonSize, buttonSize);
                                ctx.strokeStyle = "#fff";
                                ctx.strokeRect(buttonX, buttonY, buttonSize, buttonSize);
                                
                                // Play/Pause icon
                                ctx.fillStyle = "#fff";
                                ctx.font = "16px Arial";
                                ctx.textAlign = "center";
                                const iconText = this.value.isPlaying ? "⏸" : "▶";
                                ctx.fillText(iconText, buttonX + buttonSize/2, buttonY + buttonSize/2 + 5);
                                
                                // Auto-play animation
                                if (this.value.isPlaying && this.value.frames.length > 1) {
                                    const now = Date.now();
                                    if (now - this.value.lastUpdate > (1000 / this.value.frameRate)) {
                                        this.value.currentFrame = (this.value.currentFrame + 1) % this.value.frames.length;
                                        this.value.lastUpdate = now;
                                        node.setDirtyCanvas(true, true);
                                    }
                                }
                                
                            } else {
                                // No frames message
                                ctx.fillStyle = "#666";
                                ctx.font = "16px Arial";
                                ctx.textAlign = "center";
                                ctx.fillText("No frames to preview", x + w/2, y + h/2);
                                ctx.font = "12px Arial";
                                ctx.fillText("Process some frames to see preview", x + w/2, y + h/2 + 25);
                            }
                        },
                        
                        mouse: function(event, pos, node) {
                            if (event.type === "pointerdown" && this.value.frames.length > 0) {
                                const margin = 10;
                                const w = Math.min(400, node.size[0] - margin * 2);
                                const h = 300;
                                const x = margin;
                                const y = this.widgetY || 0;
                                const controlsY = y + h - 70;
                                
                                // Check if clicked on progress bar
                                const progressWidth = w - 120;
                                const progressX = x + 10;
                                const progressBarY = controlsY + 5;
                                
                                if (pos[0] >= progressX && pos[0] <= progressX + progressWidth &&
                                    pos[1] >= progressBarY && pos[1] <= progressBarY + 16) {
                                    // Clicked on progress bar - seek to position
                                    const clickProgress = (pos[0] - progressX) / progressWidth;
                                    this.value.currentFrame = Math.floor(clickProgress * (this.value.frames.length - 1));
                                    this.value.currentFrame = Math.max(0, Math.min(this.value.currentFrame, this.value.frames.length - 1));
                                    node.setDirtyCanvas(true, true);
                                    console.log(`🎞️ Seeked to frame ${this.value.currentFrame + 1}/${this.value.frames.length}`);
                                    return true;
                                }
                                
                                // Check if clicked on play/pause button
                                const buttonSize = 30;
                                const buttonX = x + progressWidth + 20;
                                const buttonY = controlsY;
                                
                                if (pos[0] >= buttonX && pos[0] <= buttonX + buttonSize &&
                                    pos[1] >= buttonY && pos[1] <= buttonY + buttonSize) {
                                    // Toggle play/pause
                                    this.value.isPlaying = !this.value.isPlaying;
                                    if (this.value.isPlaying) {
                                        this.value.lastUpdate = Date.now();
                                        console.log("🎞️ Preview playback started");
                                    } else {
                                        console.log("🎞️ Preview playback paused");
                                    }
                                    node.setDirtyCanvas(true, true);
                                    return true;
                                }
                                
                                // Double-click on preview area to clear frames
                                if (pos[1] < controlsY - 20) {
                                    const now = Date.now();
                                    if (this.lastClickTime && (now - this.lastClickTime) < 300) {
                                        // Double click - clear frames
                                        this.clearFrames();
                                        node.setDirtyCanvas(true, true);
                                        console.log("🎞️ Preview cleared (double-click)");
                                        return true;
                                    }
                                    this.lastClickTime = now;
                                }
                            }
                            return false;
                        },
                        
                        onKeyDown: function(event, node) {
                            if (this.value.frames.length === 0) return false;
                            
                            switch(event.key) {
                                case ' ': // Spacebar - play/pause
                                    this.value.isPlaying = !this.value.isPlaying;
                                    if (this.value.isPlaying) {
                                        this.value.lastUpdate = Date.now();
                                    }
                                    node.setDirtyCanvas(true, true);
                                    return true;
                                    
                                case 'ArrowLeft': // Previous frame
                                    this.value.currentFrame = Math.max(0, this.value.currentFrame - 1);
                                    this.value.isPlaying = false;
                                    node.setDirtyCanvas(true, true);
                                    return true;
                                    
                                case 'ArrowRight': // Next frame
                                    this.value.currentFrame = Math.min(this.value.frames.length - 1, this.value.currentFrame + 1);
                                    this.value.isPlaying = false;
                                    node.setDirtyCanvas(true, true);
                                    return true;
                                    
                                case 'Home': // First frame
                                    this.value.currentFrame = 0;
                                    this.value.isPlaying = false;
                                    node.setDirtyCanvas(true, true);
                                    return true;
                                    
                                case 'End': // Last frame
                                    this.value.currentFrame = this.value.frames.length - 1;
                                    this.value.isPlaying = false;
                                    node.setDirtyCanvas(true, true);
                                    return true;
                                    
                                case 'Delete': // Clear frames
                                case 'Backspace':
                                    this.clearFrames();
                                    node.setDirtyCanvas(true, true);
                                    console.log("🎞️ Preview cleared (keyboard)");
                                    return true;
                                    
                                case '+': // Increase frame rate
                                case '=':
                                    this.value.frameRate = Math.min(30, this.value.frameRate + 1);
                                    console.log(`🎞️ Frame rate: ${this.value.frameRate} fps`);
                                    return true;
                                    
                                case '-': // Decrease frame rate
                                    this.value.frameRate = Math.max(1, this.value.frameRate - 1);
                                    console.log(`🎞️ Frame rate: ${this.value.frameRate} fps`);
                                    return true;
                            }
                            return false;
                        },
                        
                        addFrame: function(imageElement) {
                            if (imageElement && this.value.frames.length < 50) { // Limit to 50 frames for performance
                                this.value.frames.push(imageElement);
                                console.log(`🎞️ Added frame ${this.value.frames.length} to preview`);
                                
                                // Auto-advance to newest frame if not playing
                                if (!this.value.isPlaying) {
                                    this.value.currentFrame = this.value.frames.length - 1;
                                }
                            }
                        },
                        
                        clearFrames: function() {
                            this.value.frames = [];
                            this.value.currentFrame = 0;
                            this.value.isPlaying = false;
                            console.log("🎞️ Preview frames cleared");
                        },
                        
                        setFrameRate: function(fps) {
                            this.value.frameRate = Math.max(1, Math.min(fps, 30));
                        }
                    };
                    
                    return previewWidget;
                }
                
                function addPreviewWidget() {
                    if (!previewWidget) {
                        previewWidget = createBatchPreviewWidget();
                        node.widgets.push(previewWidget);
                        console.log("🎞️ Batch preview display enabled");
                    }
                }

                // Enhanced execution tracking for batch processing
                const originalOnExecuted = node.onExecuted;
                node.onExecuted = function(message) {
                    if (originalOnExecuted) originalOnExecuted.call(this, message);
                    
                    // Update progress when execution completes
                    if (progressWidget && message.frame_count) {
                        const frameCount = parseInt(message.frame_count);
                        const framesPerBatch = framesPerBatchWidget ? framesPerBatchWidget.value : 16;
                        const totalBatches = Math.ceil(frameCount / framesPerBatch);
                        
                        progressWidget.updateProgress(totalBatches, totalBatches, frameCount, false);
                        node.setDirtyCanvas(true, true);
                        
                        console.log(`✅ Batch processing complete: ${frameCount} frames processed in ${totalBatches} batches`);
                    }
                    
                    // Update preview widget with processed frames
                    if (previewWidget && node.imgs && node.imgs.length > 0) {
                        // Clear previous frames
                        previewWidget.clearFrames();
                        
                        // Add new frames (limit to 50 for performance)
                        const maxFrames = Math.min(50, node.imgs.length);
                        const step = Math.max(1, Math.floor(node.imgs.length / maxFrames));
                        
                        for (let i = 0; i < node.imgs.length; i += step) {
                            if (previewWidget.value.frames.length < maxFrames) {
                                previewWidget.addFrame(node.imgs[i]);
                            }
                        }
                        
                        console.log(`🎞️ Preview updated with ${previewWidget.value.frames.length} frames`);
                        node.setDirtyCanvas(true, true);
                    }
                    
                    // Display representative color palette for video
                    if (message.palette_data && message.palette_data.length > 0) {
                        console.log('🎨 Representative Video Palette:', message.palette_data.split(',').map(c => `%c●%c ${c}`).join(' '), 
                                   ...message.palette_data.split(',').flatMap(c => [`color: ${c}; font-size: 16px`, 'color: inherit']));
                    }
                };

                // Monitor input connections for frame estimation
                const originalOnConnectionsChange = node.onConnectionsChange;
                node.onConnectionsChange = function(type, index, connected, link_info) {
                    if (originalOnConnectionsChange) originalOnConnectionsChange.call(this, type, index, connected, link_info);
                    
                    if (type === 1 && index === 0) { // Input connection change on first input
                        setTimeout(updateBatchEstimation, 100);
                    }
                };

				// --- Initial UI Setup ---
				setTimeout(() => {
                    addProgressWidget();
                    addPreviewWidget();
                    updateWidgetVisibility();
                    updateBatchEstimation();
                }, 10);
			};
		}
	},
});
