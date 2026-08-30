import { useCallback, useEffect, useRef, useState } from 'react'
import { listen } from '@tauri-apps/api/event'
import { api } from './api'
import type { JobResults, JobSummary, PartialResult, PipelineEvent, SessionState, SetupState } from './types'
import Onboarding from './components/Onboarding'
import Studio from './components/Studio'
import Review from './components/Review'
import Loop from './components/Loop'
import './styles.css'

type View = 'boot' | 'onboarding' | 'studio' | 'review' | 'loop'

export default function App() {
  const [view, setView] = useState<View>('boot')
  const [setup, setSetup] = useState<SetupState | null>(null)
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [activeJob, setActiveJob] = useState<string | null>(null)
  const [results, setResults] = useState<JobResults | null>(null)
  const [stages, setStages] = useState<Record<string, { fraction: number; message: string }>>({})
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [sessions, setSessions] = useState<Record<string, SessionState>>({})
  // Set when a stage stopped early but kept usable work, so the UI can offer
  // "continue with what you have" instead of only a retry that will fail again.
  const [partial, setPartial] = useState<PartialResult | null>(null)
  const unlistenRef = useRef<(() => void) | null>(null)
  const activeJobRef = useRef<string | null>(null)
  activeJobRef.current = activeJob

  const refreshJobs = useCallback(() => {
    api.listJobs().then(setJobs).catch(() => setJobs([]))
  }, [])

  // Live process state per job. Polled rather than pushed: a session can be
  // paused, killed, or left over from a previous app instance, and none of
  // those produce a pipeline event.
  useEffect(() => {
    const poll = () => {
      api.sessionStates().then(setSessions).catch(() => setSessions({}))
    }
    poll()
    const timer = window.setInterval(poll, 2000)
    return () => window.clearInterval(timer)
  }, [])

  const controlSession = useCallback(async (id: string, action: 'pause' | 'unpause' | 'stop') => {
    try {
      if (action === 'pause') await api.pauseJob(id)
      else if (action === 'unpause') await api.unpauseJob(id)
      else await api.stopJob(id)
    } catch (err) {
      setRunError(String(err))
    }
    api.sessionStates().then(setSessions).catch(() => undefined)
  }, [])

  useEffect(() => {
    api.setupState().then((s) => {
      setSetup(s)
      setView(s.onboarded ? 'studio' : 'onboarding')
    })
    refreshJobs()
  }, [refreshJobs])

  // Instagram loop: opportunistic sync on launch + hourly while open
  // (decision #12 — no background process, the app's own uptime is the
  // schedule). Fire-and-forget; the Loop screen re-reads on entry.
  useEffect(() => {
    const kick = () => {
      api
        .igStatus()
        .then((s) => (s.connected ? api.igSync() : null))
        .catch(() => null)
    }
    kick()
    const timer = window.setInterval(kick, 60 * 60 * 1000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    let disposed = false
    listen<PipelineEvent>('pipeline-event', ({ payload }) => {
      if (payload.event === 'job' && payload.job_id) {
        setActiveJob(payload.job_id)
        setResults(null)
      } else if (payload.event === 'progress' && payload.stage) {
        setStages((prev) => ({
          ...prev,
          [payload.stage!]: {
            fraction: payload.fraction ?? -1,
            message: payload.message ?? ''
          }
        }))
      } else if (payload.event === 'result') {
        setRunning(false)
        refreshJobs()
        setPartial(payload.ok ? null : payload.partial ?? null)
        if (!payload.ok) {
          setRunError(String(payload.error ?? 'Pipeline failed'))
        } else if (payload.kind === 'caption') {
          // One finished video, not a set of clips — there is no review to
          // open, and asking for clip results would fail.
          setActiveJob(null)
        } else if (activeJobRef.current) {
          api.jobResults(activeJobRef.current).then((r) => {
            setResults(r)
            setView('review')
          })
        }
      } else if (payload.event === 'exited') {
        setRunning(false)
        setRunError('The pipeline exited unexpectedly. Resume the job to continue from its last checkpoint.')
      }
    }).then((un) => {
      if (disposed) un()
      else unlistenRef.current = un
    })
    return () => {
      disposed = true
      unlistenRef.current?.()
    }
  }, [refreshJobs])

  const startRun = useCallback(
    async (source: string, llm: string, captions: string) => {
      setRunning(true)
      setRunError(null)
      setPartial(null)
      setStages({})
      setResults(null)
      setActiveJob(null)
      await api.runJob(source, llm, captions)
    },
    []
  )

  const openJob = useCallback(async (jobId: string) => {
    const r = await api.jobResults(jobId)
    setActiveJob(jobId)
    setResults(r)
    if (r.render?.outputs?.length) setView('review')
  }, [])

  if (view === 'boot') return <div className="boot" />

  if (view === 'onboarding' && setup) {
    return (
      <Onboarding
        onDone={() => {
          api.markOnboarded()
          setSetup({ ...setup, onboarded: true })
          setView('studio')
        }}
      />
    )
  }

  if (view === 'loop') {
    return <Loop onBack={() => setView('studio')} />
  }

  if (view === 'review' && results) {
    return (
      <Review
        results={results}
        onBack={() => {
          setView('studio')
          refreshJobs()
        }}
        onRestyle={(captions, camera) => {
          setRunning(true)
          setRunError(null)
          setStages({})
          setActiveJob(results.job_id)
          setView('studio')
          api.resumeJob(results.job_id, undefined, captions, camera)
        }}
      />
    )
  }

  return (
    <Studio
      jobs={jobs}
      sessions={sessions}
      onControlSession={controlSession}
      running={running}
      stages={stages}
      error={runError}
      onRun={startRun}
      onCaption={(src, preset, output, tags) => {
        setRunning(true)
        setRunError(null)
        setPartial(null)
        setStages({})
        setActiveJob(null)
        api.runCaption(src, preset, output, tags)
      }}
      onDelete={async (id) => {
        try {
          await api.deleteJob(id)
        } catch (e) {
          setRunError(String(e))
        }
        // The rail is built from a directory scan, so it only reflects the
        // deletion once we ask again.
        refreshJobs()
      }}
      onOpenLoop={() => setView('loop')}
      onOpenJob={openJob}
      partial={partial}
      activeJob={activeJob}
      onResume={(id, llm, partialOk) => {
        setRunning(true)
        setRunError(null)
        setPartial(null)
        setStages({})
        setActiveJob(id)
        api.resumeJob(id, llm, undefined, undefined, partialOk)
      }}
    />
  )
}
