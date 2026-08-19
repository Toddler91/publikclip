import { useState } from 'react'
import { open } from '@tauri-apps/plugin-dialog'
import type { JobSummary, PartialResult, SessionState } from '../types'
import KeyModal from './KeyModal'

const STAGE_ORDER = [
  'ingest', 'asr', 'diarize', 'events', 'candidates', 'score', 'camera', 'render'
]

// Captioning runs a short prefix of the same pipeline, so it gets its own
// order rather than showing five stages that will never light up.
const CAPTION_STAGE_ORDER = ['ffmpeg', 'ingest', 'asr', 'events', 'captions']

const STAGE_LABELS: Record<string, string> = {
  ffmpeg: 'FFMPEG',
  ingest: 'INGEST',
  asr: 'TRANSCRIBE',
  diarize: 'SPEAKERS',
  events: 'LISTEN',
  candidates: 'SCAN',
  score: 'JUDGE',
  camera: 'DIRECT',
  render: 'RENDER',
  captions: 'CAPTIONS'
}

const CAPTION_PRESETS = ['classic', 'beast', 'hormozi', 'minimal', 'karaoke-pop']
const VIDEO_EXTS = ['mp4', 'mkv', 'mov', 'avi', 'webm', 'm4v', 'ts', 'flv', 'wmv']

type Mode = 'clip' | 'caption'

interface Props {
  jobs: JobSummary[]
  sessions: Record<string, SessionState>
  onControlSession: (id: string, action: 'pause' | 'unpause' | 'stop') => void
  running: boolean
  stages: Record<string, { fraction: number; message: string }>
  error: string | null
  onRun: (source: string, llm: string, captions: string) => void
  onCaption: (source: string, preset: string, output: string | undefined, tags: boolean) => void
  onDelete: (id: string) => void
  onOpenLoop: () => void
  onOpenJob: (id: string) => void
  onResume: (id: string, llm?: string, partialOk?: boolean) => void
  partial: PartialResult | null
  activeJob: string | null
}

export default function Studio({
  jobs, sessions, onControlSession, running, stages, error, partial, activeJob,
  onRun, onCaption, onDelete, onOpenLoop, onOpenJob, onResume
}: Props) {
  const [mode, setMode] = useState<Mode>('clip')
  const [source, setSource] = useState('')
  const [llm, setLlm] = useState('gemini')
  const [captions, setCaptions] = useState('classic')
  const [showKey, setShowKey] = useState(false)

  const [capSource, setCapSource] = useState('')
  const [capPreset, setCapPreset] = useState('classic')
  const [capOut, setCapOut] = useState('')
  const [capTags, setCapTags] = useState(false)

  const clipJobs = jobs.filter((j) => j.kind !== 'caption')
  const captionJobs = jobs.filter((j) => j.kind === 'caption')
  const deck = mode === 'caption' ? CAPTION_STAGE_ORDER : STAGE_ORDER

  const pickVideo = async () => {
    const picked = await open({
      multiple: false,
      filters: [{ name: 'Video', extensions: VIDEO_EXTS }]
    })
    if (typeof picked === 'string') setCapSource(picked)
  }

  const pickFolder = async () => {
    const picked = await open({ directory: true })
    if (typeof picked === 'string') setCapOut(picked)
  }

  const confirmDelete = (job: JobSummary) => {
    const name = job.title ?? job.id
    if (window.confirm('Delete "' + name + '" and everything it produced? This cannot be undone.')) {
      onDelete(job.id)
    }
  }

  const renderJobRow = (job: JobSummary) => {
    const live = sessions[job.id]
    const isCaption = job.kind === 'caption'
    // A caption job never renders clips, so "open results" is meaningless for
    // it; finished is the whole story.
    const finished = isCaption ? job.ingested : job.rendered
    const led = live
      ? live.state === 'paused' ? 'led-paused' : 'led-live'
      : finished ? 'led-on' : 'led-half'
    return (
      <div key={job.id} className={`rail-job-row ${live ? 'is-live' : ''}`}>
        <button
          className={`rail-job ${finished ? '' : 'partial'}`}
          onClick={() => {
            if (isCaption) return
            if (job.rendered) onOpenJob(job.id)
            else onResume(job.id)
          }}
          // A live session must not be resumed again — that is exactly
          // how two processes end up on one job.
          disabled={running || Boolean(live) || isCaption}
          title={
            live ? `${live.state} — ${live.stage ?? 'working'} (pid ${live.pid})`
              : isCaption ? 'captioned video'
                : job.rendered ? 'open results' : 'resume from checkpoint'
          }
        >
          <span className={`led ${led}`} />
          <span className="rail-job-title">{job.title ?? job.id}</span>
          <span className="rail-job-hint">
            {live ? (live.state === 'paused' ? 'paused' : live.stage ?? 'running')
              : isCaption ? 'captioned' : job.rendered ? 'open' : 'resume'}
          </span>
        </button>
        <div className="rail-job-controls">
          {live ? (
            live.controllable ? (
              <>
                <button
                  className="ctl"
                  title={live.state === 'paused' ? 'continue' : 'pause'}
                  onClick={() =>
                    onControlSession(job.id, live.state === 'paused' ? 'unpause' : 'pause')
                  }
                >
                  {live.state === 'paused' ? '▶' : '❚❚'}
                </button>
                <button
                  className="ctl ctl-stop"
                  title="stop — keeps finished chunks, resume picks up there"
                  onClick={() => onControlSession(job.id, 'stop')}
                >
                  ■
                </button>
              </>
            ) : (
              // Started by an earlier app instance: visible, but this
              // process has no handle on it to signal.
              <span className="ctl-note" title={`pid ${live.pid} — not started by this app window`}>
                external
              </span>
            )
          ) : (
            // The pipeline refuses to delete a job a live process owns, so the
            // button is simply absent while one runs rather than failing.
            <button
              className="ctl ctl-del"
              title="delete this session and its files"
              onClick={() => confirmDelete(job)}
              disabled={running}
            >
              ✕
            </button>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="studio">
      <div className="grain" />
      {showKey && <KeyModal onClose={() => setShowKey(false)} />}
      <aside className="rail">
        <header className="rail-brand">
          <span className="rail-logo">publikclip</span>
          <span className="rail-sub">the clipper that shows its work</span>
        </header>
        <div className="rail-jobs">
          <p className="rail-label">SESSIONS</p>
          {clipJobs.length === 0 && <p className="rail-empty">nothing yet</p>}
          {clipJobs.map(renderJobRow)}

          {captionJobs.length > 0 && (
            <>
              <p className="rail-label rail-label-2nd">CAPTIONS</p>
              {captionJobs.map(renderJobRow)}
            </>
          )}
        </div>
        <footer className="rail-foot">
          <button className="btn-ghost" onClick={() => setShowKey(true)}>
            ◈ gemini key
          </button>
          <button className="btn-ghost" onClick={onOpenLoop}>
            ⟳ instagram loop
          </button>
        </footer>
      </aside>

      <main className="stage-area">
        <section className="input-block">
          <div className="mode-switch">
            <button
              className={`mode ${mode === 'clip' ? 'mode-on' : ''}`}
              onClick={() => setMode('clip')}
              disabled={running}
            >
              CLIP IT
            </button>
            <button
              className={`mode ${mode === 'caption' ? 'mode-on' : ''}`}
              onClick={() => setMode('caption')}
              disabled={running}
            >
              JUST CAPTION
            </button>
          </div>

          {mode === 'clip' ? (
            <>
              <h1 className="input-heading">
                FEED IT<span className="amber"> AN HOUR.</span>
              </h1>
              <div className="input-row">
                <input
                  value={source}
                  onChange={(e) => setSource(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && source.trim() && !running && onRun(source.trim(), llm, captions)}
                  placeholder="YouTube URL or a path to a video file"
                  disabled={running}
                />
                <button
                  className="btn-primary"
                  onClick={() => onRun(source.trim(), llm, captions)}
                  disabled={running || !source.trim()}
                >
                  {running ? 'WORKING' : 'CUT IT'}
                </button>
              </div>
              <div className="run-options">
                <div className="opt-group">
                  <span className="opt-label">brain</span>
                  {['gemini', 'ollama'].map((m) => (
                    <button
                      key={m}
                      className={`opt ${llm === m ? 'opt-on' : ''}`}
                      onClick={() => setLlm(m)}
                      disabled={running}
                    >
                      {m}
                    </button>
                  ))}
                </div>
                <div className="opt-group">
                  <span className="opt-label">captions</span>
                  {CAPTION_PRESETS.map((preset) => (
                    <button
                      key={preset}
                      className={`opt ${captions === preset ? 'opt-on' : ''}`}
                      onClick={() => setCaptions(preset)}
                      disabled={running}
                    >
                      {preset}
                    </button>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <>
              <h1 className="input-heading">
                JUST THE<span className="amber"> WORDS.</span>
              </h1>
              <div className="input-row">
                <button className="btn-file" onClick={pickVideo} disabled={running}>
                  {capSource ? capSource.split(/[\\/]/).pop() : 'CHOOSE A VIDEO…'}
                </button>
                <button
                  className="btn-primary"
                  onClick={() => onCaption(capSource, capPreset, capOut || undefined, capTags)}
                  disabled={running || !capSource}
                >
                  {running ? 'WORKING' : 'CAPTION IT'}
                </button>
              </div>
              <div className="run-options">
                <div className="opt-group">
                  <span className="opt-label">style</span>
                  {CAPTION_PRESETS.map((preset) => (
                    <button
                      key={preset}
                      className={`opt ${capPreset === preset ? 'opt-on' : ''}`}
                      onClick={() => setCapPreset(preset)}
                      disabled={running}
                    >
                      {preset}
                    </button>
                  ))}
                </div>
                <div className="opt-group">
                  <span className="opt-label">save to</span>
                  <button className="opt" onClick={pickFolder} disabled={running}>
                    {capOut || 'beside the source'}
                  </button>
                  {capOut && (
                    <button className="opt" onClick={() => setCapOut('')} disabled={running}>
                      reset
                    </button>
                  )}
                </div>
                <div className="opt-group">
                  <span className="opt-label">tags</span>
                  <button
                    className={`opt ${capTags ? 'opt-on' : ''}`}
                    onClick={() => setCapTags(!capTags)}
                    disabled={running}
                    title="also detect laughter and gasps for [laughs] tags — slower"
                  >
                    {capTags ? '[laughs] on' : '[laughs] off'}
                  </button>
                </div>
              </div>
            </>
          )}
        </section>

        {(running || Object.keys(stages).length > 0) && (
          <section className="deck">
            {deck.filter((s) => stages[s] || running).map((name, i) => {
              const st = stages[name]
              const state = !st ? 'idle' : st.fraction >= 1 ? 'done' : 'live'
              return (
                <div className={`deck-row ${state}`} key={name} style={{ animationDelay: `${i * 40}ms` }}>
                  <span className="deck-name mono">{STAGE_LABELS[name] ?? name.toUpperCase()}</span>
                  <div className="deck-bar">
                    <div
                      className={`deck-fill ${st && st.fraction < 0 ? 'indeterminate' : ''}`}
                      style={st && st.fraction >= 0 ? { width: `${Math.min(100, st.fraction * 100)}%` } : undefined}
                    />
                  </div>
                  <span className="deck-msg">{st?.message ?? ''}</span>
                </div>
              )
            })}
          </section>
        )}

        {error && (
          <section className="error-block">
            <span className="led led-err" />
            <span className="error-text">{error}</span>
            {partial && (
              <div className="error-actions">
                <button
                  className="btn-continue"
                  disabled={running || !activeJob}
                  onClick={() => activeJob && onResume(activeJob, undefined, true)}
                  title={`Skip the rest of ${partial.stage} and continue with what finished`}
                >
                  CONTINUE WITH {partial.done}/{partial.total}
                </button>
                <span className="error-note">
                  keeps the {partial.done} already scored and moves on to the next stage
                </span>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  )
}
