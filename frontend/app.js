document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const dropzone = document.getElementById('dropzone');
    const canvasWrapper = document.getElementById('canvasWrapper');
    const slideCanvas = document.getElementById('slideCanvas');
    const ctx = slideCanvas.getContext('2d');
    
    const sliderGroup = document.getElementById('sliderGroup');
    const opacitySlider = document.getElementById('opacitySlider');
    const opacityVal = document.getElementById('opacityVal');
    const actionBar = document.getElementById('actionBar');
    const resetBtn = document.getElementById('resetBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    const emptyResults = document.getElementById('emptyResults');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsContent = document.getElementById('resultsContent');
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    
    // Diagnostic Metric Elements
    const gradeBadge = document.getElementById('gradeBadge');
    const gradeTitle = document.getElementById('gradeTitle');
    const riskTitle = document.getElementById('riskTitle');
    const metricBurden = document.getElementById('metricBurden');
    const metricTumorArea = document.getElementById('metricTumorArea');
    const metricTumorUm = document.getElementById('metricTumorUm');
    const metricTotalArea = document.getElementById('metricTotalArea');
    const metricScale = document.getElementById('metricScale');
    const recommendationText = document.getElementById('recommendationText');
    
    let currentFile = null;
    let originalImage = null;
    let maskImage = null;
    
    // Drag & Drop event listeners
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = '#3B82F6';
    });
    
    dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'rgba(59, 130, 246, 0.4)';
    });
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'rgba(59, 130, 246, 0.4)';
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    function handleFileSelect(file) {
        const validExtensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp', '.svs'];
        const fileName = (file.name || '').toLowerCase();
        const hasValidExt = validExtensions.some(ext => fileName.endsWith(ext));
        
        if (!file.type.startsWith('image/') && !hasValidExt) {
            alert('Please select a valid microscopic biopsy slide (PNG, JPG, TIFF, BMP, WEBP, SVS).');
            return;
        }
        
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (event) => {
            originalImage = new Image();
            originalImage.onload = () => {
                maskImage = null;
                renderCanvas();
                dropzone.style.display = 'none';
                canvasWrapper.style.display = 'flex';
                actionBar.style.display = 'flex';
                sliderGroup.style.display = 'none';
                
                emptyResults.style.display = 'block';
                resultsContent.style.display = 'none';
                loadingSpinner.style.display = 'none';
            };
            originalImage.onerror = () => {
                // Browser cannot decode TIFF/SVS natively, but backend will decode it!
                maskImage = null;
                dropzone.style.display = 'none';
                canvasWrapper.style.display = 'flex';
                actionBar.style.display = 'flex';
                sliderGroup.style.display = 'none';
                
                slideCanvas.width = 512;
                slideCanvas.height = 384;
                ctx.fillStyle = '#0f172a';
                ctx.fillRect(0, 0, 512, 384);
                ctx.fillStyle = '#60a5fa';
                ctx.font = 'bold 16px Outfit, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(`Slide Loaded: ${file.name}`, 256, 180);
                ctx.fillStyle = '#94a3b8';
                ctx.font = '13px Outfit, sans-serif';
                ctx.fillText('Click "Run Tumor Segmentation" to analyze', 256, 210);
                
                emptyResults.style.display = 'block';
                resultsContent.style.display = 'none';
                loadingSpinner.style.display = 'none';
            };
            originalImage.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    function renderCanvas() {
        if (!originalImage || !originalImage.complete || originalImage.naturalWidth === 0) return;
        
        slideCanvas.width = originalImage.width;
        slideCanvas.height = originalImage.height;
        
        ctx.clearRect(0, 0, slideCanvas.width, slideCanvas.height);
        ctx.drawImage(originalImage, 0, 0);
        
        if (maskImage) {
            const opacity = opacitySlider.value / 100;
            ctx.save();
            ctx.globalAlpha = opacity;
            ctx.drawImage(maskImage, 0, 0);
            ctx.restore();
        }
    }
    
    opacitySlider.addEventListener('input', () => {
        opacityVal.textContent = `${opacitySlider.value}%`;
        renderCanvas();
    });
    
    resetBtn.addEventListener('click', () => {
        currentFile = null;
        originalImage = null;
        maskImage = null;
        
        dropzone.style.display = 'block';
        canvasWrapper.style.display = 'none';
        actionBar.style.display = 'none';
        sliderGroup.style.display = 'none';
        
        emptyResults.style.display = 'block';
        resultsContent.style.display = 'none';
        loadingSpinner.style.display = 'none';
        fileInput.value = '';
    });
    
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        
        emptyResults.style.display = 'none';
        resultsContent.style.display = 'none';
        loadingSpinner.style.display = 'block';
        analyzeBtn.disabled = true;
        
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('pixel_scale_um', '0.5');
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server returned status ${response.status}`);
            }
            
            const data = await response.json();
            
            // If original image wasn't decoded by browser (e.g. TIFF/BMP), use backend original_base64
            const finishRender = () => {
                maskImage = new Image();
                maskImage.onload = () => {
                    renderCanvas();
                    sliderGroup.style.display = 'flex';
                    loadingSpinner.style.display = 'none';
                    resultsContent.style.display = 'flex';
                    analyzeBtn.disabled = false;
                };
                maskImage.src = data.overlay_base64;
            };

            if (data.original_base64 && (!originalImage || !originalImage.complete || originalImage.naturalWidth === 0)) {
                originalImage = new Image();
                originalImage.onload = finishRender;
                originalImage.src = data.original_base64;
            } else {
                finishRender();
            }
            
            // Update Dashboard Metrics
            const m = data.metrics;
            const g = data.grading;
            
            gradeBadge.textContent = g.grade;
            gradeTitle.textContent = g.description;
            riskTitle.textContent = g.risk_category;
            
            metricBurden.textContent = `${m.tumor_burden_percent.toFixed(2)} %`;
            metricTumorArea.textContent = `${m.tumor_area_mm2.toFixed(4)} mm²`;
            metricTumorUm.textContent = `${m.tumor_area_um2.toFixed(0)} μm²`;
            metricTotalArea.textContent = `${m.total_area_mm2.toFixed(4)} mm²`;
            metricScale.textContent = `${m.pixel_scale_um} μm / px`;
            
            recommendationText.innerHTML = `<b>Clinical Protocol</b>: ${g.clinical_recommendation}`;
            
        } catch (err) {
            console.error(err);
            alert(`Analysis failed: ${err.message}`);
            loadingSpinner.style.display = 'none';
            emptyResults.style.display = 'block';
            analyzeBtn.disabled = false;
        }
    });
    
    downloadReportBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        
        downloadReportBtn.disabled = true;
        downloadReportBtn.textContent = '⏳ Generating Clinical PDF Report...';
        
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('patient_id', 'PAT-89210');
        formData.append('biopsy_site', 'Breast Tissue / Lymph Node');
        formData.append('pixel_scale_um', '0.5');
        
        try {
            const response = await fetch('/api/generate-report', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('PDF Generation failed');
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Pathology_Report_PAT-89210.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            
        } catch (err) {
            alert(`Error downloading report: ${err.message}`);
        } finally {
            downloadReportBtn.disabled = false;
            downloadReportBtn.textContent = '📥 Download Clinical Pathology PDF Report';
        }
    });
});
