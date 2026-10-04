export type ModelStatus = { loaded: boolean; error?: string }
export type HealthResponse = { status: string; models: Record<string, ModelStatus>; loaded_models: string[] }
export type Sample = { name: string; url: string }
export type SampleList = { samples: Sample[] }
export type RestoreMode = 'apply' | 'already_corrupted'
export type Corruption = 'clean' | 'salt' | 'blur' | 'occlusion'
export type Severity = 'low' | 'medium' | 'high'
export type RestoreResponse = {
  input_b64: string; output_b64: string; original_b64?: string; inference_ms: number
  corruption_settings: { type: string; severity: string; params: Record<string, unknown> }
  psnr?: number; ssim?: number; probs?: Record<string, number>; weights?: Record<string, number>
  dominant_branch?: string; predicted_class?: string; selected_expert?: string
  routing_mode?: string; classifier_ms?: number; expert_ms?: number; total_ms?: number; tau?: number
}
export type SketchResponse = { input_b64: string; sketch_b64: string; style: number; inference_ms: number }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try { response = await fetch(path, init) } catch { throw new Error('Backend offline. Start the API server and try again.') }
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = await response.json()
      const detail = body.detail
      message = typeof detail === 'string' ? detail : detail?.reason || detail?.error || JSON.stringify(detail) || message
    } catch { /* keep fallback */ }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const getHealth = () => request<HealthResponse>('/api/health')
export const getSamples = () => request<SampleList>('/api/samples')
export const getSample = async (name: string) => {
  const encoded = await request<{ image_b64: string }>(`/api/samples/${encodeURIComponent(name)}`)
  return `data:image/png;base64,${encoded.image_b64}`
}

export type RestoreOptions = { file: File; mode: RestoreMode; corruption: Corruption; severity: Severity; seed?: string; routing?: 'predicted' | 'oracle' }
export async function restore(kind: 'universal' | 'hard' | 'soft', options: RestoreOptions) {
  const body = new FormData()
  body.set('image', options.file)
  body.set('mode', options.mode)
  body.set('corruption', options.corruption)
  body.set('severity', options.severity)
  if (options.seed) body.set('seed', options.seed)
  if (options.routing) body.set('routing', options.routing)
  return request<RestoreResponse>(`/api/restore/${kind}`, { method: 'POST', body })
}
export async function makeSketch(file: File, style: number) {
  const body = new FormData()
  body.set('image', file)
  body.set('style', String(style))
  return request<SketchResponse>('/api/sketch', { method: 'POST', body })
}

export function imageUrl(base64: string) { return `data:image/png;base64,${base64}` }
