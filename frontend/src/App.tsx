import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from 'react'
import { Link, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { Corruption, getHealth, getSample, getSamples, HealthResponse, imageUrl, makeSketch, restore, RestoreMode, RestoreResponse, Sample, Severity, SketchResponse } from './api'
import { wandbProjects, workspaces } from './config'

const descriptions: Record<string, string> = {
  '/universal': 'Restore pet photos across clean, noisy, blurred, and occluded inputs with one model.',
  '/hard': 'Classify the input degradation, then route pet photos to a dedicated specialist.',
  '/soft': 'Blend restoration branches using the model’s learned mixture weights.',
  '/sketch': 'Generate a styled sketch from a face photo.',
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [online, setOnline] = useState(false)
  const [dark, setDark] = useState(() => localStorage.getItem('restorelab-theme') === 'dark')
  const [toast, setToast] = useState('')
  const location = useLocation()
  const refreshHealth = useCallback(async () => {
    try { const data = await getHealth(); setHealth(data); setOnline(true) }
    catch { setOnline(false); setHealth(null) }
  }, [])
  useEffect(() => { void refreshHealth(); const id = window.setInterval(refreshHealth, 15000); return () => window.clearInterval(id) }, [refreshHealth])
  useEffect(() => { document.documentElement.classList.toggle('dark', dark); localStorage.setItem('restorelab-theme', dark ? 'dark' : 'light') }, [dark])
  useEffect(() => { if (!toast) return; const id = window.setTimeout(() => setToast(''), 4500); return () => window.clearTimeout(id) }, [toast])
  const title = location.pathname === '/' ? 'Overview & Suite Hub' : workspaces.find(item => item.path === location.pathname)?.name || 'RestoreLab'
  return <div className="min-h-screen bg-canvas text-ink">
    <header className="topbar">
      <Link className="brand" to="/"><span className="brand-mark">R</span><span>RestoreLab</span><span className="version">v2</span></Link>
      <div className="top-actions">
        <span className={`status-pill ${online ? 'is-online' : 'is-offline'}`}><i />{online ? 'Backend online' : 'Backend offline'}</span>
        <button className="icon-btn" aria-label="Toggle dark mode" onClick={() => setDark(value => !value)}>{dark ? '☼' : '☾'}</button>
      </div>
    </header>
    <aside className="sidebar">
      <p className="eyebrow side-label">Workspaces</p>
      <nav className="nav-list">
        <NavLink end to="/" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}><span className="nav-icon">⌂</span><span><b>Overview & Suite Hub</b><small>Suite central connector</small></span></NavLink>
        {workspaces.map(item => <NavLink key={item.path} to={item.path} className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}><span className="nav-icon">{item.icon}</span><span><b>{item.name}</b><small>{item.subtitle}</small></span></NavLink>)}
      </nav>
      <div className="sidebar-foot"><span className="tiny-dot" /> Local ONNX · CPU</div>
    </aside>
    <main className="main-content">
      {!online && <div className="offline-banner"><span>!</span><div><strong>Backend offline</strong><small>Can’t reach the API right now. Check that the backend is running.</small></div><button onClick={() => void refreshHealth()}>Retry</button></div>}
      <Routes>
        <Route path="/" element={<Overview health={health} online={online} />} />
        <Route path="/universal" element={<RestoreWorkspace kind="universal" health={health} notify={setToast} />} />
        <Route path="/hard" element={<RestoreWorkspace kind="hard" health={health} notify={setToast} />} />
        <Route path="/soft" element={<RestoreWorkspace kind="soft" health={health} notify={setToast} />} />
        <Route path="/sketch" element={<SketchWorkspace health={health} notify={setToast} />} />
        <Route path="*" element={<Overview health={health} online={online} />} />
      </Routes>
      {title && <span className="sr-only">{title}</span>}
    </main>
    {toast && <div className="toast" role="alert"><span>!</span><p>{toast}</p><button aria-label="Dismiss" onClick={() => setToast('')}>×</button></div>}
  </div>
}

function Overview({ health, online }: { health: HealthResponse | null; online: boolean }) {
  const descriptionsByPath: Record<string, string> = {
    '/universal': 'One restoration network for clean and corrupted pet photos.',
    '/hard': 'A classifier selects the matching specialist restoration model.',
    '/soft': 'Learned branch weights combine restoration paths continuously.',
    '/sketch': 'Turn face photos into sketches in three selectable styles.',
  }
  return <div className="page-wrap">
    <div className="page-heading overview-heading"><div><div className="heading-line"><h1>RestoreLab AI Suite</h1><span className="runtime-chip"><i /> ONNX Runtime · CPU · 128 × 128</span></div><p>Choose a workspace to restore pet photos or create a face sketch.</p></div></div>
    <section className="overview-status panel"><div><span className={`status-pip ${online ? 'good' : 'bad'}`} /><div><strong>{online ? 'Backend reachable' : 'Backend offline'}</strong><small>{online ? `${health?.loaded_models.length ?? 0} of ${Object.keys(health?.models ?? {}).length} models loaded` : 'Waiting for the API server'}</small></div></div><div className="model-status-list">{health ? Object.entries(health.models).map(([name, info]) => <span key={name} className={info.loaded ? 'model-loaded' : 'model-missing'} title={info.error || (info.loaded ? 'Loaded' : 'Not loaded')}><i />{name}</span>) : <span className="muted-copy">Model status will appear when the backend is online.</span>}</div></section>
    <div className="section-title"><div><span className="eyebrow">Workspaces</span><h2>Pick up where you left off</h2></div><span className="count-chip">4 workspaces</span></div>
    <div className="workspace-grid">{workspaces.map((item, index) => { const loaded = workspaceLoaded(health, item.path); return <article className="workspace-card panel" key={item.path}><div className="card-topline"><span className="workspace-icon">{item.icon}</span><span className="card-index">0{index + 1}</span></div><span className="card-kicker">{item.subtitle}</span><h3>{item.name}</h3><p>{descriptionsByPath[item.path]}</p><div className="card-status"><span className={`status-pip ${loaded ? 'good' : online ? 'bad' : 'neutral'}`} />{loaded ? 'Model loaded' : online ? 'Model unavailable' : 'Status unavailable'}</div><Link className="primary-btn card-link" to={item.path}>Open workspace <span>→</span></Link></article> })}</div>
    <section className="wandb-panel panel"><div><span className="eyebrow">Experiment tracking</span><h2>Training projects</h2><p>Open a project in Weights & Biases.</p></div><div className="wandb-links">{wandbProjects.map(project => <a href={project.url} target="_blank" rel="noreferrer" key={project.name}>{project.name}<span>↗</span></a>)}</div></section>
  </div>
}

type WorkspaceKind = 'universal' | 'hard' | 'soft'
type PickedFile = { file: File; url: string }
const corruptionNames: Record<Corruption, string> = { clean: 'Clean', salt: 'Salt & pepper', blur: 'Gaussian blur', occlusion: 'Occlusion' }
const severities: Severity[] = ['low', 'medium', 'high']
const severityCaption: Record<Corruption, Record<Severity, string>> = {
  clean: { low: 'No change', medium: 'No change', high: 'No change' },
  salt: { low: 'p = 0.03', medium: 'p = 0.08', high: 'p = 0.15' },
  blur: { low: '(3, 0.7)', medium: '(5, 1.5)', high: '(7, 2.5)' },
  occlusion: { low: '1 rect · 10%', medium: '2 rects · 20%', high: '3 rects · 35%' },
}

function RestoreWorkspace({ kind, health, notify }: { kind: WorkspaceKind; health: HealthResponse | null; notify: (message: string) => void }) {
  const meta = workspaces.find(item => item.path === `/${kind}`)!
  const [picked, setPicked] = useState<PickedFile | null>(null)
  const [samples, setSamples] = useState<Sample[]>([])
  const [mode, setMode] = useState<RestoreMode>('apply')
  const [corruption, setCorruption] = useState<Corruption>('salt')
  const [severity, setSeverity] = useState<Severity>('medium')
  const [routing, setRouting] = useState<'predicted' | 'oracle'>('predicted')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<RestoreResponse | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  useEffect(() => { getSamples().then(result => setSamples(result.samples)).catch(error => notify(error.message)) }, [notify])
  useEffect(() => () => { if (picked?.url.startsWith('blob:')) URL.revokeObjectURL(picked.url) }, [picked])
  const acceptFile = (file?: File) => {
    if (!file) return
    const validType = ['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || /\.(jpe?g|png|webp)$/i.test(file.name)
    if (!validType) { notify('Choose a JPEG, PNG, or WEBP image.'); return }
    if (file.size > 10 * 1024 * 1024) { notify('Image exceeds the 10 MB limit.'); return }
    const url = URL.createObjectURL(file)
    setPicked({ file, url }); setResult(null)
  }
  const handleDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files[0]) }
  const loadSample = async (sample: Sample) => {
    setBusy(true)
    try { const uri = await getSample(sample.name); const response = await fetch(uri); const blob = await response.blob(); const file = new File([blob], sample.name, { type: 'image/png' }); acceptFile(file) }
    catch (error) { notify(error instanceof Error ? error.message : 'Could not load sample image.') }
    finally { setBusy(false) }
  }
  const changeMode = (next: RestoreMode) => {
    setMode(next)
    if (kind === 'hard' && next === 'already_corrupted' && routing === 'oracle') setRouting('predicted')
  }
  const runRestore = async () => {
    if (!picked) { notify('Add a pet photo before restoring.'); return }
    setBusy(true); setResult(null)
    try { setResult(await restore(kind, { file: picked.file, mode, corruption, severity, routing: kind === 'hard' ? routing : undefined })) }
    catch (error) { notify(error instanceof Error ? error.message : 'Restoration failed.') }
    finally { setBusy(false) }
  }
  const status = { loaded: workspaceLoaded(health, `/${kind}`) }
  const resultInput = result ? imageUrl(result.input_b64) : ''
  const resultOutput = result ? imageUrl(result.output_b64) : ''
  return <div className="page-wrap">
    <div className="page-heading"><div><div className="heading-line"><h1>{meta.name}</h1><span className="runtime-chip"><i /> ONNX Runtime · CPU · input 128 × 128</span></div><p>{descriptions[`/${kind}`]}</p></div><span className={`model-badge ${status?.loaded ? 'loaded' : 'missing'}`}><i />{status?.loaded ? 'Model ready' : 'Model unavailable'}</span></div>
    <div className="workspace-layout">
      <aside className="control-panel panel">
        <div className="panel-heading"><span className="eyebrow">Input configuration</span><span className="tune-icon">☷</span></div>
        <div className={`dropzone ${dragging ? 'dragging' : ''}`} onClick={() => fileInput.current?.click()} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={handleDrop} role="button" tabIndex={0} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') fileInput.current?.click() }}><input ref={fileInput} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={(event: ChangeEvent<HTMLInputElement>) => acceptFile(event.target.files?.[0])}/><span className="upload-icon">↑</span><strong>Drop pet photo</strong><span>or <u>browse files</u></span><small>JPEG, PNG, WEBP · up to 10 MB</small></div>
        {picked && <div className="selected-file"><img src={picked.url} alt="Selected pet photo"/><span title={picked.file.name}>{picked.file.name}</span><button aria-label="Remove image" onClick={() => { setPicked(null); setResult(null) }}>×</button></div>}
        <label className="field-label" htmlFor={`samples-${kind}`}>Or use a sample</label><select id={`samples-${kind}`} className="select-field" value="" disabled={busy || !samples.length} onChange={event => { const sample = samples.find(item => item.name === event.target.value); if (sample) void loadSample(sample) }}><option value="">{samples.length ? 'Choose a pet photo…' : 'No samples available'}</option>{samples.map(sample => <option key={sample.name} value={sample.name}>{sample.name}</option>)}</select>
        <div className="control-divider"/>
        <div className="field-group"><span className="field-label">Input pipeline mode</span><div className="segmented"><button className={mode === 'apply' ? 'selected' : ''} onClick={() => changeMode('apply')}>Apply corruption</button><button className={mode === 'already_corrupted' ? 'selected' : ''} onClick={() => changeMode('already_corrupted')}>Already corrupted</button></div></div>
        {kind === 'hard' && <div className="field-group"><span className="field-label">Routing</span><div className="segmented"><button className={routing === 'predicted' ? 'selected' : ''} onClick={() => setRouting('predicted')}>Predicted</button><button disabled={mode === 'already_corrupted'} title={mode === 'already_corrupted' ? 'Oracle routing needs a known applied corruption.' : ''} className={routing === 'oracle' ? 'selected' : ''} onClick={() => setRouting('oracle')}>Oracle</button></div></div>}
        <div className="field-group"><div className="label-row"><span className="field-label">Corruption type</span><span className="field-value">{corruptionNames[corruption]}</span></div><div className="corruption-grid">{(Object.keys(corruptionNames) as Corruption[]).map(value => <button key={value} className={`choice-btn ${corruption === value ? 'selected' : ''}`} onClick={() => setCorruption(value)}>{corruptionNames[value]}</button>)}</div></div>
        <div className="field-group"><span className="field-label">Severity · parameters</span><div className="severity-grid">{severities.map(value => <button key={value} className={severity === value ? 'selected' : ''} onClick={() => setSeverity(value)}><b>{value}</b><small>{severityCaption[corruption][value]}</small></button>)}</div></div>
        <button className="primary-btn run-btn" disabled={busy || !picked || !status?.loaded} onClick={() => void runRestore()}>{busy ? <><Spinner/> Processing…</> : <>Restore image <span>→</span></>}</button>
        {!status?.loaded && <small className="helper-text">This model is not loaded. Check the backend model mount.</small>}
      </aside>
      <div className="results-column">
        {!result ? <div className="empty-result panel"><div className="empty-symbol">{picked ? '◌' : '▧'}</div><h2>{busy ? 'Restoration in progress' : picked ? 'Ready to restore' : 'Your result will appear here'}</h2><p>{busy ? 'The model is processing your pet photo.' : picked ? 'Choose the corruption settings, then run restoration.' : 'Upload a pet photo or choose one of the bundled samples to get started.'}</p>{busy && <Spinner/>}</div> : <>
          <div className={`image-grid ${kind === 'universal' && result.original_b64 ? 'three-up' : ''}`}>
            {kind === 'universal' && result.original_b64 && <ImagePanel title="Original" src={imageUrl(result.original_b64)} />}
            <ImagePanel title={kind === 'universal' ? (result.original_b64 ? 'Corrupted input' : 'Input image') : 'Input'} src={resultInput}/>
            <ImagePanel title="Restored output" src={resultOutput} accent/>
          </div>
          {kind === 'universal' && <div className="detail-panel panel"><div className="detail-head"><h3>Run details</h3><DownloadButton src={resultOutput} filename="restorelab-restored.png"/></div><div className="detail-grid"><Detail label="Corruption" value={`${result.corruption_settings.type} · ${result.corruption_settings.severity}`}/><Detail label="Parameters" value={formatParams(result.corruption_settings.params)}/><Detail label="Inference" value={`${formatMs(result.inference_ms)} ms`}/>{result.psnr !== undefined && <Detail label="PSNR" value={`${formatMetric(result.psnr)} dB`} />}{result.ssim !== undefined && <Detail label="SSIM" value={formatMetric(result.ssim)} />}</div></div>}
          {kind === 'hard' && <div className="detail-panel panel"><div className="detail-head"><div><span className="eyebrow">Classifier telemetry</span><h3>Routing decision</h3></div><DownloadButton src={resultOutput} filename="restorelab-hard-restored.png"/></div><div className="telemetry-layout"><div className="prob-list">{[['clean', 'Clean'], ['salt', 'Salt & Pepper'], ['blur', 'Gaussian Blur'], ['occlusion', 'Occlusion']].map(([key, label]) => <ProbabilityBar key={key} label={label} value={result.probs?.[key] ?? 0}/>)}</div><div className="routing-stats"><Detail label="Predicted class" value={prettyClass(result.predicted_class)}/><Detail label="Selected expert" value={result.selected_expert === 'identity' ? 'Identity bypass' : prettyClass(result.selected_expert)}/><Detail label="Classifier" value={`${formatMs(result.classifier_ms ?? 0)} ms`}/><Detail label="Expert" value={`${formatMs(result.expert_ms ?? 0)} ms`}/><Detail label="Total" value={`${formatMs(result.total_ms ?? result.inference_ms)} ms`}/><Detail label="Routing mode" value={result.routing_mode === 'oracle' ? 'Oracle' : 'Predicted'}/></div></div><div className="detail-foot">{result.corruption_settings.type} · {formatParams(result.corruption_settings.params)}</div></div>}
          {kind === 'soft' && <div className="detail-panel panel"><div className="detail-head"><div><span className="eyebrow">Mixture telemetry</span><h3>Branch contributions</h3></div><div className="head-tools"><span className="tau-chip">Temperature τ = {result.tau?.toFixed(2) ?? '1.26'}</span><DownloadButton src={resultOutput} filename="restorelab-soft-restored.png"/></div></div><div className="weight-list">{[['identity', 'Clean / Identity'], ['salt', 'Salt'], ['blur', 'Blur'], ['occlusion', 'Occlusion']].map(([key, label]) => <ProbabilityBar key={key} label={label} value={result.weights?.[key] ?? 0} highlight={topWeights(result.weights)[key]}/>)}</div><div className="soft-foot"><span>Dominant branch · {prettyClass(result.dominant_branch)}</span><span>Inference · {formatMs(result.inference_ms)} ms</span></div><div className="detail-foot">{result.corruption_settings.type} · {formatParams(result.corruption_settings.params)}</div></div>}
        </>}
      </div>
    </div>
  </div>
}

function SketchWorkspace({ health, notify }: { health: HealthResponse | null; notify: (message: string) => void }) {
  const [picked, setPicked] = useState<PickedFile | null>(null)
  const [style, setStyle] = useState(1)
  const [result, setResult] = useState<SketchResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [camera, setCamera] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const loaded = health?.models.sketch?.loaded
  useEffect(() => () => { if (picked?.url.startsWith('blob:')) URL.revokeObjectURL(picked.url); streamRef.current?.getTracks().forEach(track => track.stop()) }, [picked])
  const chooseFile = (file?: File) => {
    if (!file) return
    const validType = ['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || /\.(jpe?g|png|webp)$/i.test(file.name)
    if (!validType) { notify('Choose a JPEG, PNG, or WEBP image.'); return }
    if (file.size > 10 * 1024 * 1024) { notify('Image exceeds the 10 MB limit.'); return }
    if (camera) stopCamera()
    setPicked({ file, url: URL.createObjectURL(file) }); setResult(null)
  }
  async function startCamera() {
    setCameraError('')
    if (!navigator.mediaDevices?.getUserMedia) { setCameraError('Webcam access is unavailable in this browser or page context.'); return }
    try { const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false }); streamRef.current = stream; setCamera(true); requestAnimationFrame(() => { if (videoRef.current) videoRef.current.srcObject = stream }) }
    catch { setCameraError('Could not access the webcam. Check browser camera permissions.') }
  }
  function stopCamera() { streamRef.current?.getTracks().forEach(track => track.stop()); streamRef.current = null; setCamera(false) }
  function capture() { const video = videoRef.current; if (!video || !video.videoWidth) return; const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight; const context = canvas.getContext('2d'); if (!context) return; context.drawImage(video, 0, 0); canvas.toBlob(blob => { if (blob) chooseFile(new File([blob], 'webcam-face.png', { type: 'image/png' })) }, 'image/png') }
  async function generate() {
    if (!picked) { notify('Add a face photo before generating a sketch.'); return }
    setBusy(true); setResult(null)
    try { setResult(await makeSketch(picked.file, style)) } catch (error) { notify(error instanceof Error ? error.message : 'Sketch generation failed.') } finally { setBusy(false) }
  }
  const handleDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]) }
  return <div className="page-wrap">
    <div className="page-heading"><div><div className="heading-line"><h1>Face-to-Sketch Generator</h1><span className="runtime-chip"><i /> ONNX Runtime · CPU · 128 × 128</span></div><p>Transform a face photo into a sketch in your selected style.</p></div><span className={`model-badge ${loaded ? 'loaded' : 'missing'}`}><i />{loaded ? 'Model ready' : 'Model unavailable'}</span></div>
    <div className="sketch-layout"><section className="control-panel panel"><div className="panel-heading"><span className="eyebrow">Portrait input</span><span className="tune-icon">☷</span></div>
      {camera ? <div className="camera-box"><video ref={videoRef} autoPlay playsInline muted/><div className="camera-actions"><button className="secondary-btn" onClick={stopCamera}>Retake / cancel</button><button className="primary-btn" onClick={capture}>Capture photo</button></div></div> : <div className={`dropzone face-drop ${dragging ? 'dragging' : ''}`} onClick={() => inputRef.current?.click()} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={handleDrop}><input ref={inputRef} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={event => chooseFile(event.target.files?.[0])}/><span className="upload-icon">↑</span><strong>Drop face photo</strong><span>or <u>browse files</u></span><small>JPEG, PNG, WEBP · up to 10 MB</small></div>}
      {cameraError && <p className="camera-error">{cameraError}</p>}{picked && <div className="selected-file"><img src={picked.url} alt="Selected face"/><span title={picked.file.name}>{picked.file.name}</span><button aria-label="Remove image" onClick={() => { setPicked(null); setResult(null) }}>×</button></div>}
      {!camera && <button className="secondary-btn camera-button" onClick={() => void startCamera()}><span>◉</span> Use webcam</button>}
      <div className="control-divider"/><div className="field-group"><span className="field-label">Sketch style</span><div className="style-list">{[1, 2, 3].map(value => <button key={value} className={style === value ? 'selected' : ''} onClick={() => setStyle(value)}><span className="style-swatch">{value}</span><span>Style {value}</span>{style === value && <span className="check">✓</span>}</button>)}</div></div>
      <button className="primary-btn run-btn" disabled={busy || !picked || !loaded} onClick={() => void generate()}>{busy ? <><Spinner/> Generating…</> : <>Generate sketch <span>→</span></>}</button>{!loaded && <small className="helper-text">Sketch model is not loaded.</small>}
    </section><section className="sketch-result">{result ? <><div className="image-grid two-up"><ImagePanel title="Input photo" src={imageUrl(result.input_b64)}/><ImagePanel title={`Style ${result.style} sketch`} src={imageUrl(result.sketch_b64)} accent/></div><div className="detail-panel panel"><div className="detail-head"><div><span className="eyebrow">Generation complete</span><h3>Style {result.style}</h3></div><DownloadButton src={imageUrl(result.sketch_b64)} filename={`restorelab-sketch-style-${result.style}.png`}/></div><div className="detail-grid"><Detail label="Style" value={`Style ${result.style}`}/><Detail label="Inference" value={`${formatMs(result.inference_ms)} ms`}/></div></div></> : <div className="empty-result panel"><div className="empty-symbol">▧</div><h2>{busy ? 'Creating your sketch' : 'Your sketch will appear here'}</h2><p>{busy ? 'The model is drawing your portrait.' : 'Add a face photo, choose a style, then generate your sketch.'}</p>{busy && <Spinner/>}</div>}</section></div>
  </div>
}

function ImagePanel({ title, src, accent = false }: { title: string; src: string; accent?: boolean }) { return <div className={`image-panel panel ${accent ? 'image-accent' : ''}`}><div className="image-panel-head"><span>{title}</span><span className="image-dim">128 × 128</span></div><div className="image-frame">{src ? <img src={src} alt={title}/> : <div className="image-placeholder">Image unavailable</div>}</div></div> }
function Detail({ label, value }: { label: string; value: string }) { return <div className="detail-item"><span>{label}</span><strong>{value}</strong></div> }
function Spinner() { return <span className="spinner" aria-label="Loading"/> }
function DownloadButton({ src, filename }: { src: string; filename: string }) { return <a className="secondary-btn download-btn" href={src} download={filename}>↓ <span>Download</span></a> }
function ProbabilityBar({ label, value, highlight = false }: { label: string; value: number; highlight?: boolean }) { const percent = Math.max(0, Math.min(100, value * 100)); return <div className={`prob-row ${highlight ? 'highlight' : ''}`}><div className="prob-label"><span>{label}</span><strong>{percent.toFixed(1)}%</strong></div><div className="prob-track"><i style={{ width: `${percent}%` }}/></div></div> }
function formatMs(value: number) { return Number.isFinite(value) ? value.toFixed(2) : '—' }
function formatMetric(value: number) { return Number.isFinite(value) ? value.toFixed(4) : '∞' }
function prettyClass(value?: string) { if (!value) return '—'; const names: Record<string, string> = { clean: 'Clean', identity: 'Clean / Identity', salt: 'Salt & Pepper', blur: 'Gaussian Blur', occlusion: 'Occlusion' }; return names[value] || value }
function formatParams(params: Record<string, unknown>) { const pairs = Object.entries(params); return pairs.length ? pairs.map(([key, value]) => `${key.replace(/_/g, ' ')}: ${typeof value === 'number' ? Number.isInteger(value) ? value : value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '') : String(value)}`).join(' · ') : 'None' }
function topWeights(weights?: Record<string, number>) { const entries = Object.entries(weights || {}).sort((a, b) => b[1] - a[1]); return Object.fromEntries(entries.slice(0, 2).map(([key]) => [key, true])) }
function workspaceLoaded(health: HealthResponse | null, path: string) {
  if (!health) return false
  const required = path === '/hard' ? ['classifier', 'salt', 'blur', 'occlusion'] : [workspaces.find(item => item.path === path)?.model || '']
  return required.every(name => !!health.models[name]?.loaded)
}

export default App
