/**
 * EXAMPLES: Proper ComfyUI Custom Widget Implementation
 * Based on analysis of working extensions in the ComfyUI ecosystem
 */

// Example 1: Histogram Widget (Fixed Implementation)
function createHistogramWidget() {
    const widget = {
        type: "HISTOGRAM_DISPLAY",
        name: "histogram_display", 
        size: [320, 120],
        value: null, // Store histogram data here
        
        // Main drawing function - ComfyUI calls this during node rendering
        draw: function(ctx, nodeWidth, widgetY, height) {
            const margin = 15;
            const widgetWidth = nodeWidth - (margin * 2);
            const widgetHeight = height || 100;
            
            // Draw widget background
            ctx.save();
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || '#1a1a1a';
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || '#444';
            ctx.lineWidth = 1;
            
            ctx.beginPath();
            ctx.roundRect(margin, widgetY, widgetWidth, widgetHeight, [4]);
            ctx.fill();
            ctx.stroke();
            
            // Draw histogram if data exists
            if (this.value && this.value.red) {
                this.drawHistogramData(ctx, margin + 5, widgetY + 5, widgetWidth - 10, widgetHeight - 10);
            } else {
                // Draw placeholder text
                ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || '#888';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('No image data', margin + widgetWidth/2, widgetY + widgetHeight/2);
            }
            
            ctx.restore();
        },
        
        // Helper function to draw histogram data
        drawHistogramData: function(ctx, x, y, width, height) {
            if (!this.value) return;
            
            const { red, green, blue } = this.value;
            const maxVal = Math.max(
                Math.max(...red),
                Math.max(...green), 
                Math.max(...blue)
            );
            
            if (maxVal === 0) return;
            
            ctx.save();
            ctx.globalCompositeOperation = 'screen';
            
            // Draw RGB channels
            const channels = [
                { data: red, color: 'rgba(255, 100, 100, 0.8)' },
                { data: green, color: 'rgba(100, 255, 100, 0.8)' },
                { data: blue, color: 'rgba(100, 100, 255, 0.8)' }
            ];
            
            channels.forEach(channel => {
                ctx.strokeStyle = channel.color;
                ctx.lineWidth = 1;
                ctx.beginPath();
                
                for (let i = 0; i < 256; i++) {
                    const px = x + (i / 255) * width;
                    const py = y + height - (channel.data[i] / maxVal) * height;
                    
                    if (i === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.stroke();
            });
            
            ctx.restore();
        },
        
        // Update histogram data
        updateHistogram: function(imageData) {
            if (!imageData) return;
            
            this.value = this.generateHistogramData(imageData);
        },
        
        // Generate histogram from image data
        generateHistogramData: function(imageData) {
            const data = imageData.data;
            const rHist = new Array(256).fill(0);
            const gHist = new Array(256).fill(0); 
            const bHist = new Array(256).fill(0);
            
            for (let i = 0; i < data.length; i += 4) {
                rHist[Math.floor(data[i])]++;
                gHist[Math.floor(data[i + 1])]++;
                bHist[Math.floor(data[i + 2])]++;
            }
            
            return { red: rHist, green: gHist, blue: bHist };
        },
        
        // Optional: Handle mouse events
        mouse: function(event, pos, node) {
            // Return true if event was handled, false otherwise
            return false;
        },
        
        // Optional: Compute widget size
        computeSize: function(width) {
            return [width, 120]; // [width, height]
        }
    };
    
    return widget;
}

// Example 2: Color Palette Widget (Fixed Implementation)
function createColorPaletteWidget() {
    const widget = {
        type: "COLOR_PALETTE_DISPLAY",
        name: "color_palette_display",
        size: [320, 80],
        value: [], // Store color array here
        
        draw: function(ctx, nodeWidth, widgetY, height) {
            const margin = 15;
            const widgetWidth = nodeWidth - (margin * 2);
            const widgetHeight = height || 60;
            
            // Draw widget background
            ctx.save();
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || '#1a1a1a';
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || '#444';
            ctx.lineWidth = 1;
            
            ctx.beginPath();
            ctx.roundRect(margin, widgetY, widgetWidth, widgetHeight, [4]);
            ctx.fill();
            ctx.stroke();
            
            // Draw color palette if data exists
            if (this.value && this.value.length > 0) {
                this.drawColorPalette(ctx, margin + 5, widgetY + 5, widgetWidth - 10, widgetHeight - 10);
            } else {
                // Draw placeholder
                ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || '#888';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('No palette extracted', margin + widgetWidth/2, widgetY + widgetHeight/2);
            }
            
            ctx.restore();
        },
        
        drawColorPalette: function(ctx, x, y, width, height) {
            if (!this.value || this.value.length === 0) return;
            
            const swatchWidth = width / this.value.length;
            const swatchHeight = height - 20; // Leave space for labels
            
            ctx.save();
            
            this.value.forEach((color, index) => {
                const swatchX = x + index * swatchWidth;
                const swatchY = y;
                
                // Draw color swatch
                ctx.fillStyle = color;
                ctx.fillRect(swatchX + 1, swatchY, swatchWidth - 2, swatchHeight);
                
                // Draw border
                ctx.strokeStyle = '#666';
                ctx.lineWidth = 1;
                ctx.strokeRect(swatchX + 1, swatchY, swatchWidth - 2, swatchHeight);
                
                // Draw label if space allows
                if (swatchWidth > 30) {
                    ctx.fillStyle = '#fff';
                    ctx.font = '8px monospace';
                    ctx.textAlign = 'center';
                    ctx.fillText(
                        color.substring(0, 7), // Show #RRGGBB
                        swatchX + swatchWidth/2,
                        y + height - 5
                    );
                }
            });
            
            ctx.restore();
        },
        
        updatePalette: function(colorsArray) {
            this.value = colorsArray || [];
        }
    };
    
    return widget;
}

// Example 3: Interactive Button Widget (Based on rgthree-comfy)
function createButtonWidget(label, callback) {
    const widget = {
        type: "CUSTOM_BUTTON",
        name: "custom_button",
        size: [200, 30],
        value: label,
        isPressed: false,
        
        draw: function(ctx, nodeWidth, widgetY, height) {
            const margin = 15;
            const buttonWidth = Math.min(200, nodeWidth - margin * 2);
            const buttonHeight = height || 25;
            const buttonX = margin + (nodeWidth - margin * 2 - buttonWidth) / 2; // Center button
            
            ctx.save();
            
            // Button background
            ctx.fillStyle = this.isPressed ? '#555' : LiteGraph.WIDGET_BGCOLOR || '#333';
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || '#666';
            ctx.lineWidth = 1;
            
            ctx.beginPath();
            ctx.roundRect(buttonX, widgetY, buttonWidth, buttonHeight, [buttonHeight * 0.2]);
            ctx.fill();
            ctx.stroke();
            
            // Button text
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || '#fff';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(this.value, buttonX + buttonWidth/2, widgetY + buttonHeight/2);
            
            ctx.restore();
            
            // Store button bounds for mouse detection
            this._bounds = [buttonX, widgetY, buttonWidth, buttonHeight];
        },
        
        mouse: function(event, pos, node) {
            if (!this._bounds) return false;
            
            const [x, y, width, height] = this._bounds;
            const isInside = pos[0] >= x && pos[0] <= x + width && 
                           pos[1] >= y && pos[1] <= y + height;
            
            if (event.type === "pointerdown" && isInside) {
                this.isPressed = true;
                node.setDirtyCanvas(true, true);
                return true;
            }
            
            if (event.type === "pointerup") {
                if (this.isPressed && isInside && callback) {
                    callback();
                }
                this.isPressed = false;
                node.setDirtyCanvas(true, true);
                return this.isPressed;
            }
            
            return false;
        }
    };
    
    return widget;
}

// Example 4: Progress Bar Widget (Based on ComfyUI-Crystools)
function createProgressBarWidget() {
    const widget = {
        type: "PROGRESS_BAR",
        name: "progress_bar",
        size: [300, 25],
        value: 0, // Progress value 0-100
        
        draw: function(ctx, nodeWidth, widgetY, height) {
            const margin = 15;
            const barWidth = nodeWidth - margin * 2;
            const barHeight = height || 20;
            
            ctx.save();
            
            // Background
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || '#1a1a1a';
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || '#444';
            ctx.lineWidth = 1;
            
            ctx.beginPath();
            ctx.roundRect(margin, widgetY, barWidth, barHeight, [barHeight * 0.3]);
            ctx.fill();
            ctx.stroke();
            
            // Progress fill
            if (this.value > 0) {
                const fillWidth = (barWidth - 4) * (this.value / 100);
                ctx.fillStyle = '#4a9eff';
                ctx.beginPath();
                ctx.roundRect(margin + 2, widgetY + 2, fillWidth, barHeight - 4, [barHeight * 0.2]);
                ctx.fill();
            }
            
            // Progress text
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || '#fff';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`${Math.round(this.value)}%`, margin + barWidth/2, widgetY + barHeight/2);
            
            ctx.restore();
        },
        
        updateProgress: function(percentage) {
            this.value = Math.max(0, Math.min(100, percentage));
        }
    };
    
    return widget;
}

// Usage Examples:
/*
// In your node's onNodeCreated function:

// 1. Add histogram widget
const histogramWidget = createHistogramWidget();
node.addCustomWidget(histogramWidget);

// 2. Add color palette widget  
const paletteWidget = createColorPaletteWidget();
node.addCustomWidget(paletteWidget);

// 3. Add button widget
const buttonWidget = createButtonWidget("Process Image", () => {
    console.log("Button clicked!");
});
node.addCustomWidget(buttonWidget);

// 4. Add progress bar
const progressWidget = createProgressBarWidget();
node.addCustomWidget(progressWidget);

// Update widgets:
histogramWidget.updateHistogram(imageData);
paletteWidget.updatePalette(['#ff0000', '#00ff00', '#0000ff']);
progressWidget.updateProgress(75);
*/