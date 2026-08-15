import { useState } from 'react'
import type { JobSummary, PartialResult, SessionState } from '../types'
import KeyModal from './KeyModal'

const STAGE_ORDER = [
  'ingest', 'asr', 'diarize', 'events', 'candidates', 'score', 'camera', 'render'
]

const STAGE_LABELS: Record<string, string> = {
  ingest: 'INGEST',
  asr: 'TRANSCRIBE',
  diarize: 'SPEAKERS',
  events: 'LISTEN',
  candidates: 'SCAN',
  score: 'JUDGE',
  camera: 'DIRECT',
  render: 'RENDER'
}

const CAPTION_PRESETS = ['classic', 'beast', 'hormozi', 'minimal', 'karaoke-pop']

interface Props {
  jobs: JobSummary[]
  sessions: Record<string, SessionState>
  onControlSession: (id: string, action: 'pause' | 'unpause' | 'stop') => void
  running: boolean
  stages: Record<string, { fraction: number; message: string }>
  error: string | null
  onRun: (source: string, llm: string, captions: string) => void
  onOpenLoop: () => void
  onOpenJob: (id: string) => void
  onResume: (id: string, llm?: string, partialOk?: boolean) => void
  partial: PartialResult | null
  activeJob: string | null
}

export default function Studio({
  jobs, sessions, onControlSession, running, stages, error, partial, activeJob,
  onRun, onOpenLoop, onOpenJob, onResume
}: Props) {
  const [source, setSource] = useState('')
  const [llm, setLlm] = useState('gemini')
  const [captions, setCaptions] = useState('classic')
  const [showKey, setShowKey] = useState(false)

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
          {jobs.length === 0 && <p className="rail-empty">nothing yet</p>}
          {jobs.map((job) => {
            const live = sessions[job.id]
            const led = live
              ? live.state === 'paused' ? 'led-paused' : 'led-live'
              : job.rendered ? 'led-on' : 'led-half'
            return (
              <div key={job.id} className={`rail-job-row ${live ? 'is-live' : ''}`}>
                <button
                  className={`rail-job ${job.rendered ? '' : 'partial'}`}
                  onClick={() => (job.rendered ? onOpenJob(job.id) : onResume(job.id))}
                  // A live session must not be resumed again — that is exactly
                  // how two processes end up on one job.
                  disabled={running || Boolean(live)}
                  title={
                    live ? `${live.state} — ${live.stage ?? 'working'} (pid ${live.pid})`
                      : job.rendered ? 'open results' : 'resume from checkpoint'
                  }
                >
                  <span className={`led ${led}`} />
                  <span className="rail-job-title">{job.title ?? job.id}</span>
                  <span className="rail-job-hint">
                    {live ? (live.state === 'paused' ? 'paused' : live.stage ?? 'running')
                      : job.rendered ? 'open' : 'resume'}
                  </span>
                </button>
                {live && (
                  <div className="rail-job-controls">
                    {live.controllable ? (
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
                    )}
                  </div>
                )}
              </div>
            )
          })}
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
              {['gemini', 'ollama'].map((mode) => (
                <button
                  key={mode}
                  className={`opt ${llm === mode ? 'opt-on' : ''}`}
                  onClick={() => setLlm(mode)}
                  disabled={running}
                >
                  {mode}
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
        </section>

        {(running || Object.keys(stages).length > 0) && (
          <section className="deck">
            {STAGE_ORDER.filter((s) => stages[s] || running).map((name, i) => {
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
