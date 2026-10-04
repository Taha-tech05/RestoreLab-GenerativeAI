export const workspaces = [
  { path: '/universal', name: 'Universal Restoration', subtitle: 'Single autoencoder baseline', icon: '◈', model: 'universal' },
  { path: '/hard', name: 'Hard-Routed Restoration', subtitle: 'Discrete classifier routing', icon: '⌘', model: 'classifier' },
  { path: '/soft', name: 'Soft Mixture-of-Experts', subtitle: 'Continuous latent blending', icon: '⠿', model: 'soft' },
  { path: '/sketch', name: 'Face-to-Sketch Generator', subtitle: 'Conditional portrait synthesis', icon: '▧', model: 'sketch' },
] as const

// Replace these placeholders with the team's W&B project URLs.
export const wandbProjects = [
  { name: 'Task 1 · Universal restoration', url: 'https://wandb.ai/your-team/your-universal-project' },
  { name: 'Task 2 · Hard-routed experts', url: 'https://wandb.ai/your-team/your-hard-routing-project' },
  { name: 'Task 3 · Soft mixture of experts', url: 'https://wandb.ai/your-team/your-soft-moe-project' },
  { name: 'Task 4 · Face to sketch', url: 'https://wandb.ai/your-team/your-sketch-project' },
]
