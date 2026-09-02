document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generateBtn');
    const imageUpload = document.getElementById('imageUpload');
    const rowsInput = document.getElementById('rows');
    const colsInput = document.getElementById('cols');
    const loading = document.getElementById('loading');
    const canvasContainer = document.getElementById('canvas-container');
    const instructions = document.getElementById('instructions');
    const referencePanel = document.getElementById('reference-panel');
    const referenceImage = document.getElementById('referenceImage');
    const lightboxOverlay = document.getElementById('lightbox-overlay');
    const lightboxImage = document.getElementById('lightboxImage');
    const evaluateBtn = document.getElementById('evaluateBtn');
    const resultOverlay = document.getElementById('result-overlay');
    const resultEmoji = document.getElementById('result-emoji');
    const resultTitle = document.getElementById('result-title');
    const resultScore = document.getElementById('result-score');
    const resultMessage = document.getElementById('result-message');
    const closeResultBtn = document.getElementById('closeResultBtn');

    let canvas = null;
    let referenceUrl = null;
    let totalPieces = 0;
    let placedPieces = new Set();
    let startTime = null;

    const ENCOURAGEMENTS = [
        { min: 80, lines: ['Almost there, just a few pieces left. Keep going!', 'So close to finishing, hang in there!'] },
        { min: 50, lines: ["You're over halfway there, keep it up!", 'Good progress, keep checking the reference image!'] },
        { min: 20, lines: ['Great start, take your time!', "You're getting somewhere, keep going!"] },
        { min: 0,  lines: ['Just getting started! Check the reference image in the corner for edge pieces.', 'Every puzzle starts somewhere. Try the four corners first!'] }
    ];

    const CONGRATS = [
        { min: 95, emoji: '🏆', lines: ["Perfect! Fast and precise, you're a puzzle master!", 'Textbook-level performance, flawless!'] },
        { min: 85, emoji: '🎉', lines: ['Awesome! Fast and accurate, really impressive!', 'Great speed and sharp eyes, nicely done!'] },
        { min: 70, emoji: '👏', lines: ['Well done! You put the whole picture together!', 'Solid finish, keep up the great work!'] },
        { min: 0,  emoji: '😊', lines: ['Congrats on finishing! Slow and steady wins the race!', 'You did it! The process matters more than speed!'] }
    ];

    function pickLine(tiers, value) {
        const tier = tiers.find(t => value >= t.min);
        const lines = tier.lines;
        return { emoji: tier.emoji, text: lines[Math.floor(Math.random() * lines.length)] };
    }

    function computeScore(elapsedSec, pieceCount) {
        const idealSeconds = pieceCount * 6;
        const ratio = idealSeconds / Math.max(elapsedSec, idealSeconds * 0.3);
        return Math.max(0, Math.min(100, Math.round(60 + ratio * 40)));
    }

    function showResult({ emoji, title, score, message }) {
        resultEmoji.textContent = emoji;
        resultTitle.textContent = title;
        if (score === null) {
            resultScore.style.display = 'none';
        } else {
            resultScore.style.display = 'block';
            resultScore.textContent = `${score} pts`;
        }
        resultMessage.textContent = message;
        resultOverlay.classList.add('visible');
    }

    closeResultBtn.addEventListener('click', () => {
        resultOverlay.classList.remove('visible');
    });

    evaluateBtn.addEventListener('click', () => {
        if (totalPieces === 0) return;
        const percent = Math.round((placedPieces.size / totalPieces) * 100);

        if (placedPieces.size < totalPieces) {
            const { emoji, text } = pickLine(ENCOURAGEMENTS, percent);
            showResult({
                emoji,
                title: `${percent}% Complete`,
                score: null,
                message: text
            });
            return;
        }

        const elapsedSec = Math.round((Date.now() - startTime) / 1000);
        const score = computeScore(elapsedSec, totalPieces);
        const { emoji, text } = pickLine(CONGRATS, score);
        const minutes = Math.floor(elapsedSec / 60);
        const seconds = elapsedSec % 60;
        const timeStr = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;

        showResult({
            emoji,
            title: 'Puzzle Complete!',
            score,
            message: `Time: ${timeStr}. ${text}`
        });
    });

    referenceImage.addEventListener('click', () => {
        if (!referenceImage.src) return;
        lightboxImage.src = referenceImage.src;
        lightboxOverlay.classList.add('visible');
    });

    lightboxOverlay.addEventListener('click', () => {
        lightboxOverlay.classList.remove('visible');
    });

    generateBtn.addEventListener('click', async () => {
        if (!imageUpload.files[0]) {
            alert('Please upload an image first.');
            return;
        }

        const file = imageUpload.files[0];
        const rows = rowsInput.value;
        const cols = colsInput.value;

        const formData = new FormData();
        formData.append('image', file);
        formData.append('rows', rows);
        formData.append('cols', cols);

        generateBtn.disabled = true;
        loading.style.display = 'block';
        canvasContainer.style.display = 'none';
        instructions.style.display = 'none';

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to generate puzzle');
            }

            const data = await response.json();

            // Show the uploaded image as a reference panel so it can be used
            // as a guide while solving the puzzle.
            if (referenceUrl) {
                URL.revokeObjectURL(referenceUrl);
            }
            referenceUrl = URL.createObjectURL(file);
            referenceImage.src = referenceUrl;
            referencePanel.style.display = 'flex';

            initCanvas(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while generating the puzzle.');
        } finally {
            generateBtn.disabled = false;
            loading.style.display = 'none';
        }
    });

    function initCanvas(data) {
        // Show canvas container
        canvasContainer.style.display = 'block';
        instructions.style.display = 'block';
        evaluateBtn.style.display = 'inline-block';
        resultOverlay.classList.remove('visible');

        // Reset progress tracking for the new puzzle
        totalPieces = data.pieces.length;
        placedPieces = new Set();
        startTime = Date.now();

        // Initialize Fabric.js canvas
        if (canvas) {
            canvas.dispose();
        }

        // Build the layout from the frame outward: figure out how big the
        // frame needs to be, surround it with a generous scatter margin on
        // every side, then fit the whole thing to the viewport as one unit.
        const baseFrameThickness = 32;
        const baseMatThickness = 20;
        const basePuzzleWidth = data.original_width * 2;
        const basePuzzleHeight = data.original_height * 2;
        const baseOuterWidth = basePuzzleWidth + 2 * (baseFrameThickness + baseMatThickness);
        const baseOuterHeight = basePuzzleHeight + 2 * (baseFrameThickness + baseMatThickness);

        // Scatter margin: half the puzzle's own size on each side
        const baseMarginX = basePuzzleWidth * 0.5;
        const baseMarginY = basePuzzleHeight * 0.5;

        const baseCw = baseOuterWidth + 2 * baseMarginX;
        const baseCh = baseOuterHeight + 2 * baseMarginY;

        const maxW = window.innerWidth * 0.9;
        const maxH = window.innerHeight * 0.8;
        const fitScale = Math.min(1, maxW / baseCw, maxH / baseCh);

        const cw = baseCw * fitScale;
        const ch = baseCh * fitScale;

        canvas = new fabric.Canvas('puzzleCanvas', {
            width: cw,
            height: ch,
            backgroundColor: '#f0f0f0'
        });

        // The assembled puzzle (and therefore each piece) renders at twice
        // the fitted scale, so it reads clearly inside the frame.
        const pieceScale = fitScale * 2;

        const puzzleWidth = data.original_width * pieceScale;
        const puzzleHeight = data.original_height * pieceScale;
        const puzzleLeft = (cw - puzzleWidth) / 2;
        const puzzleTop = (ch - puzzleHeight) / 2;

        // Draw a picture-frame around the puzzle area
        const frameThickness = baseFrameThickness * fitScale;
        const matThickness = baseMatThickness * fitScale;

        const outerWidth = puzzleWidth + 2 * (frameThickness + matThickness);
        const outerHeight = puzzleHeight + 2 * (frameThickness + matThickness);
        const frameOuterLeft = puzzleLeft - frameThickness - matThickness;
        const frameOuterTop = puzzleTop - frameThickness - matThickness;

        const frameOuter = new fabric.Rect({
            left: frameOuterLeft,
            top: frameOuterTop,
            width: outerWidth,
            height: outerHeight,
            rx: 6,
            ry: 6,
            fill: new fabric.Gradient({
                type: 'linear',
                coords: { x1: 0, y1: 0, x2: outerWidth, y2: outerHeight },
                colorStops: [
                    { offset: 0, color: '#a9784a' },
                    { offset: 0.5, color: '#7c4f2a' },
                    { offset: 1, color: '#a9784a' }
                ]
            }),
            shadow: new fabric.Shadow({
                color: 'rgba(0,0,0,0.3)',
                blur: 14,
                offsetX: 0,
                offsetY: 6
            }),
            selectable: false,
            evented: false
        });
        canvas.add(frameOuter);

        const frameMat = new fabric.Rect({
            left: puzzleLeft - matThickness,
            top: puzzleTop - matThickness,
            width: puzzleWidth + 2 * matThickness,
            height: puzzleHeight + 2 * matThickness,
            fill: '#faf7f0',
            selectable: false,
            evented: false
        });
        canvas.add(frameMat);

        // Dashed guide marking the exact area pieces need to snap into
        const targetArea = new fabric.Rect({
            left: puzzleLeft,
            top: puzzleTop,
            width: puzzleWidth,
            height: puzzleHeight,
            fill: 'transparent',
            stroke: 'rgba(0,0,0,0.2)',
            strokeWidth: 1,
            strokeDashArray: [6, 4],
            selectable: false,
            evented: false
        });
        canvas.add(targetArea);
        const backgroundLayers = canvas.getObjects().length; // frameOuter, frameMat, targetArea

        // Scatter pieces in the margin surrounding the frame, rather than
        // anywhere on the canvas, so they stay out of the frame itself.
        const frameOuterRight = frameOuterLeft + outerWidth;
        const frameOuterBottom = frameOuterTop + outerHeight;

        function getPerimeterPosition(pieceW, pieceH) {
            const half = (Math.sqrt(pieceW * pieceW + pieceH * pieceH) * pieceScale) / 2;
            const margin = 6;

            const topSpace = frameOuterTop - half - margin;
            const bottomSpace = ch - frameOuterBottom - half - margin;
            const leftSpace = frameOuterLeft - half - margin;
            const rightSpace = cw - frameOuterRight - half - margin;

            const sides = [];
            if (topSpace > 0) sides.push({ name: 'top', weight: cw });
            if (bottomSpace > 0) sides.push({ name: 'bottom', weight: cw });
            if (leftSpace > 0) sides.push({ name: 'left', weight: outerHeight });
            if (rightSpace > 0) sides.push({ name: 'right', weight: outerHeight });

            if (sides.length === 0) {
                // Frame fills the canvas: fall back to a random spot anywhere
                return {
                    x: half + margin + Math.random() * Math.max(1, cw - 2 * (half + margin)),
                    y: half + margin + Math.random() * Math.max(1, ch - 2 * (half + margin))
                };
            }

            const totalWeight = sides.reduce((sum, side) => sum + side.weight, 0);
            let r = Math.random() * totalWeight;
            let chosen = sides[sides.length - 1];
            for (const side of sides) {
                if (r < side.weight) { chosen = side; break; }
                r -= side.weight;
            }

            switch (chosen.name) {
                case 'top':
                    return {
                        x: half + margin + Math.random() * Math.max(1, cw - 2 * (half + margin)),
                        y: half + margin + Math.random() * Math.max(1, topSpace)
                    };
                case 'bottom':
                    return {
                        x: half + margin + Math.random() * Math.max(1, cw - 2 * (half + margin)),
                        y: ch - half - margin - Math.random() * Math.max(1, bottomSpace)
                    };
                case 'left':
                    return {
                        x: half + margin + Math.random() * Math.max(1, leftSpace),
                        y: frameOuterTop + half + margin + Math.random() * Math.max(1, outerHeight - 2 * (half + margin))
                    };
                case 'right':
                    return {
                        x: cw - half - margin - Math.random() * Math.max(1, rightSpace),
                        y: frameOuterTop + half + margin + Math.random() * Math.max(1, outerHeight - 2 * (half + margin))
                    };
            }
        }

        // Load and add pieces
        let loadedCount = 0;

        data.pieces.forEach(pieceData => {
            fabric.Image.fromURL(pieceData.url, (img) => {
                // Render each piece at the larger puzzle scale
                img.scale(pieceScale);

                const scatterPos = getPerimeterPosition(pieceData.width, pieceData.height);

                // Set initial scattered position and rotation
                img.set({
                    left: scatterPos.x,
                    top: scatterPos.y,
                    angle: pieceData.scatter_angle,
                    originX: 'center',
                    originY: 'center',
                    hasControls: true, // Enable rotation and scaling controls
                    hasBorders: true,
                    transparentCorners: false,
                    cornerColor: 'blue',
                    cornerSize: 10,
                    padding: 5,
                    // Store original coordinates for snapping logic later
                    targetX: pieceData.original_x * pieceScale + puzzleLeft + (img.width * pieceScale) / 2,
                    targetY: pieceData.original_y * pieceScale + puzzleTop + (img.height * pieceScale) / 2,
                    id: pieceData.id
                });

                // Customize controls: only allow rotation, disable scaling
                img.setControlsVisibility({
                    mt: false, mb: false, ml: false, mr: false,
                    bl: false, br: false, tl: false, tr: false,
                    mtr: true // mtr is the rotation control
                });

                canvas.add(img);
                
                loadedCount++;
                if (loadedCount === data.pieces.length) {
                    canvas.renderAll();
                }
            });
        });

        // Optional: Snapping logic
        canvas.on('object:modified', function(options) {
            const obj = options.target;
            if (!obj || obj.type !== 'image') return;

            // Normalize angle to 0-360
            let angle = obj.angle % 360;
            if (angle < 0) angle += 360;

            // Snap rotation if close to 0 (upright)
            if (angle < 10 || angle > 350) {
                obj.set({ angle: 0 });
            }

            // Snap position if close to target and upright
            let isCorrect = false;
            if (obj.angle === 0) {
                const dx = obj.left - obj.targetX;
                const dy = obj.top - obj.targetY;
                const distance = Math.sqrt(dx*dx + dy*dy);

                if (distance < 20) {
                    obj.set({
                        left: obj.targetX,
                        top: obj.targetY
                    });
                    // Bring to bottom so it doesn't cover active pieces, but above the frame/mat/target area
                    obj.moveTo(backgroundLayers);
                    isCorrect = true;
                }
            }

            if (isCorrect) {
                placedPieces.add(obj.id);
            } else {
                placedPieces.delete(obj.id);
            }

            canvas.renderAll();
        });
    }
});
