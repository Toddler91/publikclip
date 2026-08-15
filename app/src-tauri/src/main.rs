// publikclip desktop shell. The pipeline is a Python sidecar speaking JSONL
// on stdout (`publikclip --jsonl ...`); this shell spawns it, forwards every
// event to the frontend, and exposes small filesystem/settings commands.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, RunEvent};

/// Pipeline children we spawned, job id -> pid.
///
/// Without this the app has no handle on its own sidecars: quitting left a
/// transcription running forever, reparented to pid 1, and the next Resume
/// started a second copy of the same job on top of it.
fn children() -> &'static Mutex<HashMap<String, u32>> {
    static CHILDREN: OnceLock<Mutex<HashMap<String, u32>>> = OnceLock::new();
    CHILDREN.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_child(job_id: &str, pid: u32) {
    if let Ok(mut map) = children().lock() {
        map.insert(job_id.to_string(), pid);
    }
}

fn unregister_child(pid: u32) {
    if let Ok(mut map) = children().lock() {
        map.retain(|_, v| *v != pid);
    }
}

fn pid_for_job(job_id: &str) -> Option<u32> {
    children().lock().ok()?.get(job_id).copied()
}

/// Send a signal to a pipeline process tree.
///
/// Signals go through `kill(1)` rather than adding a libc dependency, and
/// target the process group first (negative pid): the real work runs in a
/// python child of the `uv` wrapper we spawned, so signalling only the wrapper
/// would leave the transcription running.
#[cfg(unix)]
fn signal_process(pid: u32, sig: &str) -> Result<(), String> {
    let group = Command::new("kill")
        .args([format!("-{sig}"), format!("-{pid}")])
        .status();
    if matches!(group, Ok(s) if s.success()) {
        return Ok(());
    }
    // Not a process group leader — fall back to the pid itself.
    match Command::new("kill").args([format!("-{sig}"), pid.to_string()]).status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("kill -{sig} {pid} exited with {status}")),
        Err(err) => Err(format!("could not signal {pid}: {err}")),
    }
}

#[cfg(windows)]
fn signal_process(pid: u32, sig: &str) -> Result<(), String> {
    // Windows has no SIGSTOP/SIGCONT; only termination is expressible.
    if sig != "TERM" && sig != "KILL" {
        return Err("pausing is not supported on Windows".to_string());
    }
    match Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("taskkill exited with {status}")),
        Err(err) => Err(format!("could not stop {pid}: {err}")),
    }
}

fn stop_all_children() {
    let pids: Vec<u32> = match children().lock() {
        Ok(map) => map.values().copied().collect(),
        Err(_) => return,
    };
    for pid in pids {
        let _ = signal_process(pid, "TERM");
    }
}

fn home_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("PUBLIKCLIP_HOME") {
        return PathBuf::from(custom);
    }
    dirs_home().join(".publikclip")
}

fn dirs_home() -> PathBuf {
    // HOME on Unix; Windows services and some launch paths only set USERPROFILE.
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Command that never flashes a console window on Windows (CREATE_NO_WINDOW).
/// Every pipeline/tool spawn goes through this — a GUI app popping cmd.exe
/// windows for each sidecar call reads as malware to most people.
fn quiet_command(program: &str) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

/// Where the Python pipeline lives and how to invoke it.
/// Dev builds call `uv run` against the repo's pipeline/ directory. Packaged
/// builds invoke the bundled python env (M6); resolution stays in one place.
fn pipeline_invocation() -> (String, Vec<String>) {
    if cfg!(debug_assertions) {
        let pipeline_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../pipeline")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from("../pipeline"));
        (
            "uv".to_string(),
            vec![
                "--directory".to_string(),
                pipeline_dir.to_string_lossy().to_string(),
                "run".to_string(),
                "publikclip".to_string(),
            ],
        )
    } else {
        // Packaged: bundled uv + pipeline source under the platform's
        // resource layout — macOS keeps them in the .app's Resources dir,
        // Windows (NSIS) lands them in resources\ next to the exe. The venv
        // bootstraps into PUBLIKCLIP_HOME on first run (uv handles Python
        // 3.12 download + deps; the onboarding screen owns expectations).
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        let resources = if cfg!(target_os = "macos") {
            exe_dir.join("../Resources/resources")
        } else {
            exe_dir.join("resources")
        };
        let uv = if cfg!(target_os = "windows") { "bin/uv.exe" } else { "bin/uv" };
        (
            resources.join(uv).to_string_lossy().to_string(),
            vec![
                "--directory".to_string(),
                resources.join("pipeline").to_string_lossy().to_string(),
                "run".to_string(),
                "publikclip".to_string(),
            ],
        )
    }
}

#[tauri::command]
fn run_job(app: AppHandle, source: String, llm: Option<String>, captions: Option<String>) -> Result<(), String> {
    let (program, base_args) = pipeline_invocation();
    std::thread::spawn(move || {
        let mut args = base_args.clone();
        args.push("--jsonl".to_string());
        args.push("run".to_string());
        args.push(source);
        if let Some(mode) = llm {
            args.push("--llm".to_string());
            args.push(mode);
        }
        if let Some(preset) = captions {
            args.push("--captions".to_string());
            args.push(preset);
        }
        stream_pipeline(&app, &program, &args, None);
    });
    Ok(())
}

#[tauri::command]
fn resume_job(
    app: AppHandle,
    job_id: String,
    llm: Option<String>,
    captions: Option<String>,
    camera: Option<String>,
) -> Result<(), String> {
    let (program, base_args) = pipeline_invocation();
    let tracked = job_id.clone();
    std::thread::spawn(move || {
        let mut args = base_args.clone();
        args.push("--jsonl".to_string());
        args.push("resume".to_string());
        args.push(job_id);
        if let Some(mode) = llm {
            args.push("--llm".to_string());
            args.push(mode);
        }
        if let Some(preset) = captions {
            args.push("--captions".to_string());
            args.push(preset);
        }
        if let Some(cam) = camera {
            args.push("--camera".to_string());
            args.push(cam);
        }
        stream_pipeline(&app, &program, &args, Some(tracked));
    });
    Ok(())
}

fn stream_pipeline(app: &AppHandle, program: &str, args: &[String], job_id: Option<String>) {
    let mut command = quiet_command(program);
    command.args(args).stdout(Stdio::piped()).stderr(Stdio::null());
    // Own process group, so pause/stop can signal the whole tree with a
    // negative pid. The work runs in a python grandchild of this `uv` wrapper.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(err) => {
            let _ = app.emit(
                "pipeline-event",
                json!({"event": "result", "ok": false, "error": format!("could not start pipeline: {err}")}),
            );
            return;
        }
    };

    let pid = child.id();
    if let Some(id) = job_id.as_ref() {
        register_child(id, pid);
    }

    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(value) = serde_json::from_str::<Value>(&line) {
                // `run` only learns its job id once the pipeline mints it.
                if value.get("event").and_then(Value::as_str) == Some("job") {
                    if let Some(id) = value.get("job_id").and_then(Value::as_str) {
                        register_child(id, pid);
                    }
                }
                // Heartbeats exist so the pipeline notices a dead app; the UI
                // has no use for them.
                if value.get("event").and_then(Value::as_str) == Some("heartbeat") {
                    continue;
                }
                let _ = app.emit("pipeline-event", value);
            }
        }
    }
    let status = child.wait();
    unregister_child(pid);
    if let Ok(status) = status {
        if !status.success() {
            let _ = app.emit("pipeline-event", json!({"event": "exited", "code": status.code()}));
        }
    }
}

/// Live state of every job, read from the run locks the pipeline writes.
///
/// The DB records intent and cannot be trusted for this: a job killed
/// mid-stage still says "running" forever, because nothing survived to write
/// otherwise. The lock file plus the process table are the truth.
#[tauri::command]
fn session_states() -> Result<Value, String> {
    let jobs_dir = home_dir().join("jobs");
    let entries = match fs::read_dir(&jobs_dir) {
        Ok(e) => e,
        Err(_) => return Ok(json!({})),
    };

    let mut locks: Vec<(String, u32, Option<String>)> = Vec::new();
    for entry in entries.flatten() {
        let job_id = entry.file_name().to_string_lossy().to_string();
        let Ok(text) = fs::read_to_string(entry.path().join("run.lock")) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let Some(pid) = value.get("pid").and_then(Value::as_u64) else {
            continue;
        };
        let stage = value.get("stage").and_then(Value::as_str).map(str::to_string);
        locks.push((job_id, pid as u32, stage));
    }
    if locks.is_empty() {
        return Ok(json!({}));
    }

    // One `ps` for every pid at once: state letter T means SIGSTOP-suspended,
    // and a pid missing from the output is a stale lock whose owner is gone.
    let mut states: HashMap<u32, String> = HashMap::new();
    #[cfg(unix)]
    {
        let pid_list: Vec<String> = locks.iter().map(|(_, p, _)| p.to_string()).collect();
        if let Ok(out) = Command::new("ps").arg("-o").arg("pid=,stat=").arg("-p").arg(pid_list.join(",")).output() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let mut parts = line.split_whitespace();
                if let (Some(pid), Some(stat)) = (parts.next(), parts.next()) {
                    if let Ok(pid) = pid.parse::<u32>() {
                        states.insert(pid, stat.to_string());
                    }
                }
            }
        }
    }

    let mut out = serde_json::Map::new();
    for (job_id, pid, stage) in locks {
        let Some(stat) = states.get(&pid) else {
            continue; // owner died without cleaning up; not a live session
        };
        let state = if stat.starts_with('T') { "paused" } else { "running" };
        let ours = pid_for_job(&job_id).is_some();
        out.insert(job_id, json!({"pid": pid, "state": state, "stage": stage, "controllable": ours}));
    }
    Ok(Value::Object(out))
}

#[tauri::command]
fn pause_job(job_id: String) -> Result<(), String> {
    let pid = pid_for_job(&job_id).ok_or("this job is not running in this app")?;
    signal_process(pid, "STOP")
}

#[tauri::command]
fn unpause_job(job_id: String) -> Result<(), String> {
    let pid = pid_for_job(&job_id).ok_or("this job is not running in this app")?;
    signal_process(pid, "CONT")
}

#[tauri::command]
fn stop_job(job_id: String) -> Result<(), String> {
    let pid = pid_for_job(&job_id).ok_or("this job is not running in this app")?;
    // A suspended process cannot act on SIGTERM until it is resumed, so wake
    // it first — otherwise "stop" silently does nothing to a paused job.
    let _ = signal_process(pid, "CONT");
    signal_process(pid, "TERM")
}

/// Everything the review UI needs for one job, read straight off the job
/// dir's checkpoint files (artifacts are the truth).
#[tauri::command]
fn job_results(job_id: String) -> Result<Value, String> {
    let dir = home_dir().join("jobs").join(&job_id);
    if !dir.exists() {
        return Err(format!("no job dir for {job_id}"));
    }
    let read_stage = |name: &str| -> Value {
        fs::read_to_string(dir.join(format!("{name}.json")))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| v.get("data").cloned())
            .unwrap_or(Value::Null)
    };
    Ok(json!({
        "job_id": job_id,
        "dir": dir.to_string_lossy(),
        "ingest": read_stage("ingest"),
        "score": read_stage("score"),
        "camera": read_stage("camera"),
        "render": read_stage("render"),
        "events": read_stage("events"),
        "candidates": read_stage("candidates"),
    }))
}

#[tauri::command]
fn list_job_dirs() -> Result<Vec<Value>, String> {
    let jobs_dir = home_dir().join("jobs");
    let mut out = vec![];
    if let Ok(entries) = fs::read_dir(&jobs_dir) {
        for entry in entries.flatten() {
            let id = entry.file_name().to_string_lossy().to_string();
            let dir = entry.path();
            let has_render = dir.join("render.json").exists();
            let has_ingest = dir.join("ingest.json").exists();
            let title = fs::read_to_string(dir.join("ingest.json"))
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                .and_then(|v| v["data"]["title"].as_str().map(String::from));
            out.push(json!({
                "id": id, "title": title,
                "ingested": has_ingest, "rendered": has_render,
            }));
        }
    }
    out.sort_by(|a, b| b["id"].as_str().cmp(&a["id"].as_str()));
    Ok(out)
}

#[tauri::command]
fn save_gemini_key(key: String) -> Result<bool, String> {
    let home = home_dir();
    fs::create_dir_all(&home).map_err(|e| e.to_string())?;
    let path = home.join("secrets.json");
    let mut current: Value = fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    current["gemini_api_key"] = json!(key.trim());
    fs::write(&path, serde_json::to_string_pretty(&current).unwrap()).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
    }
    Ok(true)
}

#[tauri::command]
fn get_setup_state() -> Result<Value, String> {
    let secrets = home_dir().join("secrets.json");
    let has_key = fs::read_to_string(&secrets)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .map(|v| v["gemini_api_key"].as_str().map(|k| !k.is_empty()).unwrap_or(false))
        .unwrap_or(false);
    let onboarded = home_dir().join("onboarded").exists();
    Ok(json!({"has_gemini_key": has_key, "onboarded": onboarded}))
}

#[tauri::command]
fn mark_onboarded() -> Result<(), String> {
    let home = home_dir();
    fs::create_dir_all(&home).map_err(|e| e.to_string())?;
    fs::write(home.join("onboarded"), "1").map_err(|e| e.to_string())
}

#[tauri::command]
async fn check_ollama() -> Result<Value, String> {
    let out = quiet_command("curl")
        .args(["-s", "-m", "3", "http://localhost:11434/api/tags"])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Ok(json!({"running": false, "models": []}));
    }
    let parsed: Value = serde_json::from_slice(&out.stdout).unwrap_or(json!({}));
    let models: Vec<String> = parsed["models"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|m| m["name"].as_str().map(String::from)).collect())
        .unwrap_or_default();
    Ok(json!({"running": true, "models": models}))
}

/// Sync pipeline call that returns one JSON blob (edit context, visual
/// suggestions). Long-running render-clip goes through run_edit_render
/// instead so progress streams.
#[tauri::command]
async fn edit_tool(args: Vec<String>) -> Result<Value, String> {
    let (program, base_args) = pipeline_invocation();
    let mut full = base_args;
    full.push("edit".to_string());
    full.extend(args);
    let out = quiet_command(&program)
        .args(&full)
        .output()
        .map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    // last JSON line is the payload (progress lines may precede it)
    let line = stdout.lines().rev().find(|l| l.trim_start().starts_with('{'));
    match line.and_then(|l| serde_json::from_str::<Value>(l).ok()) {
        Some(v) => Ok(v),
        None => Err(format!(
            "edit tool produced no JSON: {}",
            String::from_utf8_lossy(&out.stderr).chars().take(400).collect::<String>()
        )),
    }
}

#[tauri::command]
fn run_edit_render(app: AppHandle, job_id: String, clip: u32) -> Result<(), String> {
    let (program, base_args) = pipeline_invocation();
    std::thread::spawn(move || {
        let mut args = base_args.clone();
        args.push("--jsonl".to_string());
        args.push("edit".to_string());
        args.push("render-clip".to_string());
        args.push(job_id);
        args.push(clip.to_string());
        stream_pipeline(&app, &program, &args, None);
    });
    Ok(())
}

#[tauri::command]
fn save_clip_edits(job_id: String, edits: Value) -> Result<(), String> {
    let path = home_dir().join("jobs").join(&job_id).join("clip_edits.json");
    // Merge: the app sends one clip's state at a time; other clips' edits
    // must survive.
    let mut current: Value = fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    if let (Some(obj), Some(new)) = (current.as_object_mut(), edits.as_object()) {
        for (k, v) in new {
            obj.insert(k.clone(), v.clone());
        }
    }
    fs::write(&path, serde_json::to_string_pretty(&current).unwrap()).map_err(|e| e.to_string())
}

#[tauri::command]
fn save_pexels_key(key: String) -> Result<bool, String> {
    let home = home_dir();
    fs::create_dir_all(&home).map_err(|e| e.to_string())?;
    let path = home.join("secrets.json");
    let mut current: Value = fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    current["pexels_api_key"] = json!(key.trim());
    fs::write(&path, serde_json::to_string_pretty(&current).unwrap()).map_err(|e| e.to_string())?;
    Ok(true)
}

#[tauri::command]
fn ig_status() -> Result<Value, String> {
    let path = home_dir().join("instagram.json");
    let connected = fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok());
    match connected {
        Some(v) => Ok(json!({
            "connected": true,
            "username": v["username"],
            "obtained_at": v["token_obtained_at"],
        })),
        None => Ok(json!({"connected": false})),
    }
}

/// Runs the CLI's OAuth dance (it opens the browser + catches the localhost
/// callback). Blocking by design — the frontend shows a "finish in your
/// browser" state until this returns.
#[tauri::command]
async fn ig_connect(app_id: String, app_secret: String) -> Result<String, String> {
    let (program, base_args) = pipeline_invocation();
    let mut args = base_args;
    args.extend([
        "ig".into(), "connect".into(),
        "--app-id".into(), app_id,
        "--app-secret".into(), app_secret,
    ]);
    let out = quiet_command(&program)
        .args(&args)
        .output()
        .map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
    if out.status.success() {
        Ok(stdout)
    } else {
        Err(if stderr.is_empty() { stdout } else { stderr })
    }
}

/// One-shot `publikclip ig <args...>` call returning the CLI's JSON line
/// (sync / overview / link / unlink / reject — same contract as edit_tool).
#[tauri::command]
async fn ig_tool(args: Vec<String>) -> Result<Value, String> {
    let (program, base_args) = pipeline_invocation();
    let mut full = base_args;
    full.push("ig".to_string());
    full.extend(args);
    let out = quiet_command(&program)
        .args(&full)
        .output()
        .map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().rev().find(|l| l.trim_start().starts_with('{'));
    match line.and_then(|l| serde_json::from_str::<Value>(l).ok()) {
        Some(v) => Ok(v),
        None => Err(format!(
            "ig tool produced no JSON: {}",
            String::from_utf8_lossy(&out.stderr).chars().take(400).collect::<String>()
        )),
    }
}

#[tauri::command]
fn export_clip(path: String, title: Option<String>) -> Result<String, String> {
    let src = PathBuf::from(&path);
    if !src.exists() {
        return Err("clip file missing".into());
    }
    let downloads = dirs_home().join("Downloads");
    let stem = title.unwrap_or_else(|| "publikclip".into());
    let safe: String = stem
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' { c } else { '_' })
        .collect::<String>()
        .trim()
        .replace(' ', "-")
        .chars()
        .take(60)
        .collect();
    let mut dest = downloads.join(format!("{safe}.mp4"));
    let mut n = 1;
    while dest.exists() {
        dest = downloads.join(format!("{safe}-{n}.mp4"));
        n += 1;
    }
    fs::copy(&src, &dest).map_err(|e| e.to_string())?;
    Ok(dest.to_string_lossy().to_string())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            run_job,
            resume_job,
            job_results,
            list_job_dirs,
            save_gemini_key,
            get_setup_state,
            mark_onboarded,
            check_ollama,
            ig_status,
            ig_connect,
            ig_tool,
            edit_tool,
            run_edit_render,
            save_clip_edits,
            save_pexels_key,
            export_clip,
            session_states,
            pause_job,
            unpause_job,
            stop_job
        ])
        .setup(|app| {
            let _ = app.get_webview_window("main");
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while running publikclip")
        .run(|_app, event| {
            // Take the pipeline down with us. Left alone these children keep
            // running after the app quits, reparented to pid 1, and the next
            // launch starts a second copy of the same job on top of them.
            if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
                stop_all_children();
            }
        });
}
