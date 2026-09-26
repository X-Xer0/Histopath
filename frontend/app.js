/* ============================================================
   HistologyAI · Nucleus Segmentation & Morphometry
   Talks to the FastAPI backend in ../backend/app
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    const el = (id) => document.getElementById(id);

    const fileInput = el('fileInput');
    const dropzone = el('dropzone');
    const canvasWrapper = el('canvasWrapper');
    const slideCanvas = el('slideCanvas');
    const ctx = slideCanvas.getContext('2d');

    const viewToggles = el('viewToggles');
    const viewCaption = el('viewCaption');
    const sliderGroup = el('sliderGroup');
    const opacitySlider = el('opacitySlider');
    const opacityVal = el('opacityVal');
    const actionBar = el('actionBar');
    const resetBtn = el('resetBtn');
    const analyzeBtn = el('analyzeBtn');
    const preciseToggle = el('preciseToggle');
    const engineBadge = el('engineBadge');
    const scaleBadge = el('scaleBadge');

    const emptyResults = el('emptyResults');
    const loadingSpinner = el('loadingSpinner');
    const loadingText = el('loadingText');
    const resultsContent = el('resultsContent');
    const downloadCsvBtn = el('downloadCsvBtn');
    const downloadReportBtn = el('downloadReportBtn');

    // state
    let currentFile = null;
    let views = { original: null, mask: null, overlay: null, instances: null };
    let activeView = 'overlay';

    const VIEW_CAPTIONS = {
        original: 'Uploaded slide, unmodified',
        mask: 'Binary nuclear mask produced by the model',
        overlay: 'Detected nuclear regions outlined on the slide',
        instances: 'Each detected nucleus coloured separately',
    };

    /* ---------------- backend status ---------------- */
    fetch('/api/health')
        .then((r) => r.json())
        .then((d) => {
            engineBadge.textContent = `● ${d.engine}`;
            scaleBadge.textContent = `${Number(d.default_pixel_scale_um).toFixed(2)} µm / px`;
            if (d.engine.includes('Fallback')) {
                engineBadge.style.color = '#F5B461';
            }
        })
        .catch(() => {
            engineBadge.textContent = '● Backend unreachable';
            engineBadge.style.color = '#F4739B';
        });

    /* ---------------- file handling ---------------- */
    ['dragover', 'dragenter'].forEach((evt) =>
        dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.style.borderColor = '#4C8DF6'; })
    );
    ['dragleave', 'dragend'].forEach((evt) =>
        dropzone.addEventListener(evt, () => { dropzone.style.borderColor = 'rgba(76,141,246,.45)'; })
    );
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'rgba(76,141,246,.45)';
        if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileSelect(e.target.files[0]);
    });

    function handleFileSelect(file) {
        currentFile = file;
        views = { original: null, mask: null, overlay: null, instances: null };
        activeView = 'overlay';
        setActiveToggle('overlay');

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                views.original = img;
                showViewer();
                renderCanvas();
            };
            img.onerror = () => {
                // TIFF / BMP that the browser cannot decode - the backend will supply the image.
                views.original = null;
                showViewer();
                drawPlaceholder(`Slide loaded: ${file.name}`, 'Click "Run nucleus segmentation" to analyse');
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    function showViewer() {
        dropzone.style.display = 'none';
        canvasWrapper.style.display = 'flex';
        actionBar.style.display = 'flex';
        viewToggles.style.display = 'flex';
        sliderGroup.style.display = views.overlay ? 'flex' : 'none';
        emptyResults.style.display = 'block';
        resultsContent.style.display = 'none';
        loadingSpinner.style.display = 'none';
        viewCaption.textContent = VIEW_CAPTIONS[activeView] || '';
    }

    function drawPlaceholder(line1, line2) {
        slideCanvas.width = 640;
        slideCanvas.height = 420;
        ctx.fillStyle = '#05080F';
        ctx.fillRect(0, 0, slideCanvas.width, slideCanvas.height);
        ctx.fillStyle = '#4C8DF6';
        ctx.font = '600 16px Outfit, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(line1, 320, 200);
        ctx.fillStyle = '#8492A8';
        ctx.font = '13px Outfit, sans-serif';
        ctx.fillText(line2, 320, 228);
    }

    /* ---------------- rendering ---------------- */
    function renderCanvas() {
        const base = views.original;
        if (!base || !base.complete || base.naturalWidth === 0) return;

        slideCanvas.width = base.naturalWidth;
        slideCanvas.height = base.naturalHeight;
        ctx.clearRect(0, 0, slideCanvas.width, slideCanvas.height);
        ctx.drawImage(base, 0, 0);

        const layer = views[activeView];
        if (layer && activeView !== 'original') {
            ctx.save();
            ctx.globalAlpha = opacitySlider.value / 100;
            ctx.drawImage(layer, 0, 0);
            ctx.restore();
        }
        viewCaption.textContent = VIEW_CAPTIONS[activeView] || '';
    }

    function setActiveToggle(view) {
        viewToggles.querySelectorAll('.vt-btn').forEach((b) =>
            b.classList.toggle('is-active', b.dataset.view === view)
        );
    }

    viewToggles.querySelectorAll('.vt-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            activeView = btn.dataset.view;
            setActiveToggle(activeView);
            renderCanvas();
        });
    });

    opacitySlider.addEventListener('input', () => {
        opacityVal.textContent = `${opacitySlider.value}%`;
        renderCanvas();
    });

    resetBtn.addEventListener('click', () => {
        currentFile = null;
        views = { original: null, mask: null, overlay: null, instances: null };
        dropzone.style.display = 'block';
        canvasWrapper.style.display = 'none';
        actionBar.style.display = 'none';
        viewToggles.style.display = 'none';
        sliderGroup.style.display = 'none';
        emptyResults.style.display = 'block';
        resultsContent.style.display = 'none';
        loadingSpinner.style.display = 'none';
        fileInput.value = '';
    });

    /* ---------------- analysis ---------------- */
    function buildForm() {
        const fd = new FormData();
        fd.append('file', currentFile);
        fd.append('pixel_scale_um', '0.5');
        fd.append('precise_mode', preciseToggle.checked ? 'true' : 'false');
        return fd;
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        emptyResults.style.display = 'none';
        resultsContent.style.display = 'none';
        loadingSpinner.style.display = 'block';
        loadingText.textContent = preciseToggle.checked
            ? 'Segmenting nuclei with 8× test-time augmentation… (slower)'
            : 'Segmenting nuclei…';
        analyzeBtn.disabled = true;

        try {
            const res = await fetch('/api/predict', { method: 'POST', body: buildForm() });
            if (!res.ok) {
                const detail = await res.json().catch(() => ({}));
                throw new Error(detail.detail || `Server returned ${res.status}`);
            }
            const data = await res.json();

            await loadImages(data);
            renderMetrics(data.metrics, data.engine, data.precise_mode);
            drawHistogram(data.metrics.size_distribution);

            sliderGroup.style.display = 'flex';
            loadingSpinner.style.display = 'none';
            resultsContent.style.display = 'flex';
        } catch (err) {
            console.error(err);
            loadingSpinner.style.display = 'none';
            emptyResults.style.display = 'block';
            alert(`Analysis failed: ${err.message}`);
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    function loadImages(data) {
        const sources = {
            original: data.original_base64,
            mask: data.mask_base64,
            overlay: data.overlay_base64,
            instances: data.instance_base64,
        };
        const load = (src) =>
            new Promise((resolve) => {
                if (!src) return resolve(null);
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => resolve(null);
                img.src = src;
            });

        return Promise.all(Object.values(sources).map(load)).then((loaded) => {
            const keys = Object.keys(sources);
            keys.forEach((k, i) => { if (loaded[i]) views[k] = loaded[i]; });
            renderCanvas();
        });
    }

    function renderMetrics(m, engine, precise) {
        el('metricCount').textContent = Number(m.nuclei_count).toLocaleString();
        el('metricPerMm2').textContent = `${Number(m.nuclei_per_mm2).toLocaleString()} nuclei per mm²`;

        el('metricDensity').textContent = `${m.nuclear_density_percent.toFixed(2)} %`;
        el('metricDiameter').textContent = `${m.mean_equivalent_diameter_um.toFixed(2)} µm`;
        el('metricDiameterRange').textContent =
            `range ${m.min_equivalent_diameter_um.toFixed(1)} – ${m.max_equivalent_diameter_um.toFixed(1)} µm`;
        el('metricArea').textContent = `${m.mean_nuclear_area_um2.toFixed(2)} µm²`;
        el('metricAreaMedian').textContent = `median ${m.median_nuclear_area_um2.toFixed(2)} µm²`;
        el('metricCv').textContent = `${m.size_variability_cv_percent.toFixed(1)} %`;
        el('metricNuclearArea').textContent = `${m.nuclear_area_mm2.toFixed(6)} mm²`;
        el('metricNuclearAreaUm').textContent = `${Number(m.nuclear_area_um2).toLocaleString()} µm²`;
        el('metricTissueArea').textContent = `${m.tissue_area_mm2.toFixed(6)} mm²`;
        el('metricEngine').textContent = precise ? `${engine} · precise` : `${engine} · standard`;

        const chip = el('cellularityChip');
        el('cellularityValue').textContent = m.cellularity ? m.cellularity.category : '—';
        chip.title = m.cellularity
            ? `${m.cellularity.detail}. Bands: ${m.cellularity.basis}.`
            : '';
    }

    /* ---------------- size histogram ---------------- */
    function drawHistogram(dist) {
        const canvas = el('sizeHistogram');
        const dpr = window.devicePixelRatio || 1;
        const cssW = canvas.clientWidth || 640;
        const cssH = 170;
        canvas.width = cssW * dpr;
        canvas.height = cssH * dpr;
        const c = canvas.getContext('2d');
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
        c.clearRect(0, 0, cssW, cssH);

        const counts = (dist && dist.counts) || [];
        const edges = (dist && dist.bin_edges_um) || [];

        const padL = 34, padR = 10, padT = 12, padB = 26;
        const plotW = cssW - padL - padR;
        const plotH = cssH - padT - padB;

        // axes
        c.strokeStyle = 'rgba(255,255,255,.14)';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(padL, padT);
        c.lineTo(padL, padT + plotH);
        c.lineTo(padL + plotW, padT + plotH);
        c.stroke();

        if (!counts.length || Math.max(...counts) === 0) {
            c.fillStyle = '#5B687C';
            c.font = '12px Outfit, sans-serif';
            c.textAlign = 'center';
            c.fillText('no nuclei detected', padL + plotW / 2, padT + plotH / 2);
            return;
        }

        const peak = Math.max(...counts);
        const barW = plotW / counts.length;

        // y gridlines
        c.strokeStyle = 'rgba(255,255,255,.055)';
        c.fillStyle = '#5B687C';
        c.font = '10px Outfit, sans-serif';
        c.textAlign = 'right';
        for (let g = 0; g <= 2; g++) {
            const y = padT + plotH - (plotH * g) / 2;
            c.beginPath(); c.moveTo(padL, y); c.lineTo(padL + plotW, y); c.stroke();
            c.fillText(String(Math.round((peak * g) / 2)), padL - 7, y + 3);
        }

        counts.forEach((count, i) => {
            const h = (count / peak) * plotH;
            const x = padL + i * barW + barW * 0.14;
            const w = barW * 0.72;

            const grad = c.createLinearGradient(0, padT + plotH - h, 0, padT + plotH);
            grad.addColorStop(0, '#5FD4E8');
            grad.addColorStop(1, 'rgba(76,141,246,.55)');
            c.fillStyle = grad;
            c.beginPath();
            c.roundRect(x, padT + plotH - h, w, h, 3);
            c.fill();

            c.fillStyle = '#5B687C';
            c.textAlign = 'center';
            c.font = '10px Outfit, sans-serif';
            if (edges[i] !== undefined) c.fillText(String(Math.round(edges[i])), x + w / 2, padT + plotH + 15);
        });

        c.fillStyle = '#8492A8';
        c.textAlign = 'center';
        c.font = '10px Outfit, sans-serif';
        c.fillText('equivalent nuclear diameter (µm)', padL + plotW / 2, cssH - 4);
    }

    /* ---------------- downloads ---------------- */
    async function downloadFrom(endpoint, fallbackName) {
        if (!currentFile) return;
        const res = await fetch(endpoint, { method: 'POST', body: buildForm() });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const blob = await res.blob();

        const disposition = res.headers.get('Content-Disposition') || '';
        const match = disposition.match(/filename=([^;]+)/);
        const filename = match ? match[1].trim() : fallbackName;

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    }

    downloadCsvBtn.addEventListener('click', async () => {
        const original = downloadCsvBtn.textContent;
        downloadCsvBtn.disabled = true;
        downloadCsvBtn.textContent = 'Generating CSV…';
        try {
            await downloadFrom('/api/export-csv', 'nuclei.csv');
        } catch (err) {
            alert(`CSV export failed: ${err.message}`);
        } finally {
            downloadCsvBtn.disabled = false;
            downloadCsvBtn.textContent = original;
        }
    });

    downloadReportBtn.addEventListener('click', async () => {
        const original = downloadReportBtn.textContent;
        downloadReportBtn.disabled = true;
        downloadReportBtn.textContent = 'Generating report…';
        try {
            await downloadFrom('/api/generate-report', 'Nuclei_Report.pdf');
        } catch (err) {
            alert(`Report generation failed: ${err.message}`);
        } finally {
            downloadReportBtn.disabled = false;
            downloadReportBtn.textContent = original;
        }
    });

    window.addEventListener('resize', () => {
        if (resultsContent.style.display !== 'none') renderCanvas();
    });
});
