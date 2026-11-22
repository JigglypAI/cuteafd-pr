//! Client for spark-attic's `atticd`, the GB10 display-carveout lender (#2546).
//!
//! The stock interface only: the socket `/run/spark-attic/attic.sock` and key
//! `/etc/spark-attic/key`, overridden by the project's own `SPARK_ATTIC_SOCK`
//! and `SPARK_ATTIC_KEY_FILE`. One flat JSON object per `SOCK_SEQPACKET`
//! message; an `alloc` reply carries the lent slice's POSIX fd as
//! `SCM_RIGHTS`. Every release v0.1.0..v0.1.7 speaks this protocol (`info`
//! gained `pages` in 0.1.3); `status` carries the version, `info` does not.
//!
//! Every failure is a typed [`AtticError`], and callers fail open: no daemon,
//! a root-only stock install, a refusing or unknown daemon all mean "run
//! without the carveout" with one log line naming the reason.
use serde_json::Value;
use std::{
    io,
    os::fd::{FromRawFd, OwnedFd, RawFd},
    path::{Path, PathBuf},
    time::Duration,
};

pub const DEFAULT_SOCKET: &str = "/run/spark-attic/attic.sock";
pub const DEFAULT_KEY_FILE: &str = "/etc/spark-attic/key";
pub const SLICE_BYTES: u64 = 2 << 20;
pub const KEY_MAX: usize = 127;
pub const REPLY_MAX: usize = 4096;
/// atticd's default per-client lend count.
pub const MAX_LENDS: usize = 64;
/// Released protocol range (major, minor).
pub const SUPPORTED: &[(u64, u64)] = &[(0, 1)];
/// The documented non-root step (spark-attic INSTALL.md §6), named in the refusal line.
pub const NON_ROOT_STEP: &str = "give atticd --socket-group <group> (systemctl edit attic.service: ExecStart) \
    and chgrp <group> /etc/spark-attic/key; chmod 0640 /etc/spark-attic/key (spark-attic INSTALL.md §6)";

#[derive(Debug, thiserror::Error)]
pub enum AtticError {
    #[error("spark-attic not installed (no {0})")]
    NotInstalled(PathBuf),
    #[error("atticd at {path} refuses this non-root process ({source}); {NON_ROOT_STEP}")]
    NoAccess { path: PathBuf, source: io::Error },
    #[error("spark-attic key {path}: {source}")]
    Key { path: PathBuf, source: io::Error },
    #[error("spark-attic key is not 1..={KEY_MAX} printable bytes without space, quote or backslash")]
    InvalidKey,
    #[error("atticd: {0}")]
    Io(#[from] io::Error),
    #[error("atticd reply: {0}")]
    Protocol(String),
    #[error("atticd refused: {0}")]
    Refused(String),
    #[error("spark-attic {version:?} is outside the supported protocol range 0.1.x")]
    UnsupportedVersion { version: String },
    #[error("spark-attic {version} is {state}: {why}")]
    NotReady { version: String, state: String, why: String },
}

pub type Result<T> = std::result::Result<T, AtticError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    pub socket: PathBuf,
    pub key_file: PathBuf,
    pub timeout: Duration,
}

impl Settings {
    /// The stock paths, or the project's documented overrides.
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from);
        Self { socket: var("SPARK_ATTIC_SOCK").unwrap_or_else(|| DEFAULT_SOCKET.into()),
            key_file: var("SPARK_ATTIC_KEY_FILE").unwrap_or_else(|| DEFAULT_KEY_FILE.into()),
            timeout: Duration::from_secs(5) }
    }
}

pub fn key_ok(key: &str) -> bool {
    (1..=KEY_MAX).contains(&key.len()) && key.bytes().all(|b| (0x21..=0x7e).contains(&b) && b != b'"' && b != b'\\')
}

fn permission(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(libc::EACCES) | Some(libc::EPERM))
}

pub fn read_key(path: &Path) -> Result<String> {
    let text = std::fs::read_to_string(path).map_err(|source| {
        if permission(&source) {
            AtticError::NoAccess { path: path.into(), source }
        } else if source.kind() == io::ErrorKind::NotFound {
            AtticError::NotInstalled(path.into())
        } else {
            AtticError::Key { path: path.into(), source }
        }
    })?;
    let key = text.lines().next().unwrap_or("").trim().to_owned();
    if !key_ok(&key) { return Err(AtticError::InvalidKey); }
    Ok(key)
}

pub fn encode(key: &str, op: &str, size: Option<u64>) -> Result<Vec<u8>> {
    if !key_ok(key) { return Err(AtticError::InvalidKey); }
    let mut request = serde_json::json!({"key": key, "op": op});
    if let Some(size) = size { request["size"] = size.into(); }
    Ok(serde_json::to_vec(&request).expect("JSON request"))
}

/// One validated reply. Received fds are owned, so a refusal closes them.
pub fn decode(message: &[u8], fds: Vec<OwnedFd>, truncated: bool, want_fd: bool)
    -> Result<(serde_json::Map<String, Value>, Option<OwnedFd>)> {
    if message.is_empty() { return Err(AtticError::Protocol("atticd closed the connection".into())); }
    if truncated { return Err(AtticError::Protocol("reply truncated".into())); }
    let reply: Value = serde_json::from_slice(message)
        .map_err(|_| AtticError::Protocol("reply is not JSON".into()))?;
    let Value::Object(reply) = reply else { return Err(AtticError::Protocol("reply is not an object".into())) };
    if reply.get("ok") != Some(&Value::Bool(true)) {
        return Err(AtticError::Refused(reply.get("error").and_then(Value::as_str)
            .unwrap_or("no reason given").into()));
    }
    let mut fds = fds;
    if fds.len() != usize::from(want_fd) {
        return Err(AtticError::Protocol(format!("{} descriptors with the reply", fds.len())));
    }
    Ok((reply, fds.pop()))
}

/// The version from `status`; refuses unknown ranges and a daemon not lending.
pub fn check_status(status: &serde_json::Map<String, Value>) -> Result<String> {
    let text = |key: &str| status.get(key).and_then(Value::as_str).unwrap_or("").to_owned();
    if text("name") != "spark-attic" {
        return Err(AtticError::Protocol(format!("daemon is not spark-attic (name {:?})", text("name"))));
    }
    let version = text("version");
    let mut parts = version.split('.').map(|part| part.parse::<u64>());
    let supported = match (parts.next(), parts.next()) {
        (Some(Ok(major)), Some(Ok(minor))) => SUPPORTED.contains(&(major, minor)),
        _ => false,
    };
    if !supported { return Err(AtticError::UnsupportedVersion { version }); }
    if text("state") != "ready" {
        return Err(AtticError::NotReady { version, state: text("state"), why: text("why") });
    }
    Ok(version)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Info { pub size: u64, pub free: u64, pub largest: u64 }

/// One atticd connection; lent slices stay lent until it drops.
pub struct Client { socket: OwnedFd, key: String }

impl Client {
    pub fn connect(settings: &Settings) -> Result<Self> {
        if !settings.socket.exists() { return Err(AtticError::NotInstalled(settings.socket.clone())); }
        let key = read_key(&settings.key_file)?;
        // SAFETY: plain socket(2); the returned fd is owned immediately.
        let raw = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0) };
        if raw < 0 { return Err(io::Error::last_os_error().into()); }
        // SAFETY: raw is a fresh, valid descriptor owned by nothing else.
        let socket = unsafe { OwnedFd::from_raw_fd(raw) };
        let path = settings.socket.as_os_str().as_encoded_bytes();
        // SAFETY: sockaddr_un is plain old data; zeroed is a valid value.
        let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
        if path.len() >= address.sun_path.len() {
            return Err(AtticError::Protocol("socket path too long".into()));
        }
        address.sun_family = libc::AF_UNIX as libc::sa_family_t;
        for (slot, byte) in address.sun_path.iter_mut().zip(path) { *slot = *byte as libc::c_char; }
        let timeout = libc::timeval { tv_sec: settings.timeout.as_secs() as libc::time_t,
            tv_usec: settings.timeout.subsec_micros() as libc::suseconds_t };
        for option in [libc::SO_RCVTIMEO, libc::SO_SNDTIMEO] {
            // SAFETY: the option value points to a live timeval of the size passed.
            unsafe { libc::setsockopt(raw, libc::SOL_SOCKET, option, (&timeout as *const libc::timeval).cast(),
                std::mem::size_of::<libc::timeval>() as libc::socklen_t) };
        }
        // SAFETY: address is initialized above and outlives the call.
        let connected = unsafe { libc::connect(raw, (&address as *const libc::sockaddr_un).cast(),
            std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t) };
        if connected != 0 {
            let source = io::Error::last_os_error();
            return Err(if permission(&source) { AtticError::NoAccess { path: settings.socket.clone(), source } }
                else { AtticError::Io(source) });
        }
        Ok(Self { socket, key })
    }

    /// A connected peer (tests, or a socket opened elsewhere).
    pub fn from_socket(socket: OwnedFd, key: String) -> Result<Self> {
        if !key_ok(&key) { return Err(AtticError::InvalidKey); }
        Ok(Self { socket, key })
    }

    fn ask(&self, request: &[u8], want_fd: bool) -> Result<(serde_json::Map<String, Value>, Option<OwnedFd>)> {
        use std::os::fd::AsRawFd;
        let raw = self.socket.as_raw_fd();
        // SAFETY: request is a live byte slice; send(2) reads at most its length.
        let sent = unsafe { libc::send(raw, request.as_ptr().cast(), request.len(), libc::MSG_NOSIGNAL) };
        if sent < 0 { return Err(io::Error::last_os_error().into()); }
        let (message, fds, truncated) = receive(raw)?;
        decode(&message, fds, truncated, want_fd)
    }

    pub fn status(&self) -> Result<serde_json::Map<String, Value>> {
        Ok(self.ask(&encode(&self.key, "status", None)?, false)?.0)
    }

    /// Checked version: supported range and `state == ready`.
    pub fn version(&self) -> Result<String> { check_status(&self.status()?) }

    pub fn info(&self) -> Result<Info> {
        let (reply, _) = self.ask(&encode(&self.key, "info", None)?, false)?;
        let field = |name: &str| reply.get(name).and_then(Value::as_u64)
            .ok_or_else(|| AtticError::Protocol(format!("info lacks {name}")));
        Ok(Info { size: field("size")?, free: field("free")?, largest: field("largest")? })
    }

    /// Borrow whole 2 MiB slices as one lend.
    pub fn alloc(&self, bytes: u64) -> Result<OwnedFd> {
        if bytes == 0 || bytes % SLICE_BYTES != 0 {
            return Err(AtticError::Protocol(format!("{bytes} bytes is not whole 2 MiB slices")));
        }
        let (reply, fd) = self.ask(&encode(&self.key, "alloc", Some(bytes))?, true)?;
        if reply.get("size").and_then(Value::as_u64) != Some(bytes) {
            return Err(AtticError::Protocol(format!("lent {:?} bytes, asked for {bytes}", reply.get("size"))));
        }
        fd.ok_or_else(|| AtticError::Protocol("alloc reply without descriptor".into()))
    }
}

/// Carveout bytes to lend for an arena: never above `largest` (atticd checks
/// `free` but not `largest` before asking RM), whole slices, one lend.
pub fn plan_lend(arena_bytes: u64, largest: u64) -> u64 {
    let wanted = arena_bytes - arena_bytes % SLICE_BYTES;
    wanted.min(largest - largest % SLICE_BYTES)
}

fn receive(raw: RawFd) -> Result<(Vec<u8>, Vec<OwnedFd>, bool)> {
    let mut buffer = vec![0u8; REPLY_MAX];
    // Room for a few descriptors, so an over-sending peer's extras are still
    // received (and closed) rather than leaked as truncated control data.
    let mut control = [0u8; 128];
    let mut iov = libc::iovec { iov_base: buffer.as_mut_ptr().cast(), iov_len: buffer.len() };
    // SAFETY: msghdr is plain old data; zeroed is valid before the fields are set.
    let mut header: libc::msghdr = unsafe { std::mem::zeroed() };
    header.msg_iov = &mut iov;
    header.msg_iovlen = 1;
    header.msg_control = control.as_mut_ptr().cast();
    header.msg_controllen = control.len() as _;
    // SAFETY: header points at live buffers of the declared lengths.
    let received = unsafe { libc::recvmsg(raw, &mut header, libc::MSG_CMSG_CLOEXEC) };
    if received < 0 { return Err(io::Error::last_os_error().into()); }
    let mut fds = Vec::new();
    // SAFETY: CMSG_* walk the control buffer recvmsg filled, bounded by msg_controllen.
    unsafe {
        let mut cmsg = libc::CMSG_FIRSTHDR(&header);
        while !cmsg.is_null() {
            if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                let data = libc::CMSG_DATA(cmsg) as *const RawFd;
                let count = ((*cmsg).cmsg_len as usize - libc::CMSG_LEN(0) as usize) / std::mem::size_of::<RawFd>();
                for index in 0..count { fds.push(OwnedFd::from_raw_fd(data.add(index).read_unaligned())); }
            }
            cmsg = libc::CMSG_NXTHDR(&header, cmsg);
        }
    }
    let truncated = header.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0;
    buffer.truncate(received as usize);
    Ok((buffer, fds, truncated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    const KEY: &str = "attic_0123456789abcdef01234567";

    fn pair() -> (OwnedFd, OwnedFd) {
        let mut fds = [0; 2];
        // SAFETY: socketpair writes two fresh descriptors into the array.
        assert_eq!(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_SEQPACKET, 0, fds.as_mut_ptr()) }, 0);
        // SAFETY: both descriptors are fresh and owned here.
        unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) }
    }

    fn reply(server: &OwnedFd, body: &Value, fds: &[RawFd]) {
        let text = serde_json::to_vec(body).unwrap();
        let mut iov = libc::iovec { iov_base: text.as_ptr() as *mut _, iov_len: text.len() };
        let mut control = [0u8; 64];
        // SAFETY: test-only sendmsg with live buffers; CMSG macros stay inside control.
        unsafe {
            let mut header: libc::msghdr = std::mem::zeroed();
            header.msg_iov = &mut iov;
            header.msg_iovlen = 1;
            if !fds.is_empty() {
                header.msg_control = control.as_mut_ptr().cast();
                header.msg_controllen = libc::CMSG_SPACE((fds.len() * 4) as u32) as _;
                let cmsg = libc::CMSG_FIRSTHDR(&header);
                (*cmsg).cmsg_level = libc::SOL_SOCKET;
                (*cmsg).cmsg_type = libc::SCM_RIGHTS;
                (*cmsg).cmsg_len = libc::CMSG_LEN((fds.len() * 4) as u32) as _;
                for (index, fd) in fds.iter().enumerate() {
                    (libc::CMSG_DATA(cmsg) as *mut RawFd).add(index).write_unaligned(*fd);
                }
            }
            assert!(libc::sendmsg(server.as_raw_fd(), &header, 0) >= 0);
        }
    }

    fn request(server: &OwnedFd) -> Value {
        let mut buffer = [0u8; 1024];
        // SAFETY: recv into a live stack buffer.
        let n = unsafe { libc::recv(server.as_raw_fd(), buffer.as_mut_ptr().cast(), buffer.len(), 0) };
        serde_json::from_slice(&buffer[..n as usize]).unwrap()
    }

    fn open_fds() -> usize { std::fs::read_dir("/proc/self/fd").unwrap().count() }

    fn status(version: &str, state: &str) -> Value {
        serde_json::json!({"ok": true, "name": "spark-attic", "version": version, "state": state, "why": "x",
            "size": 2143289344u64, "free": 2143289344u64, "largest": 2143289344u64, "lent": 0, "draining": 0})
    }

    #[test]
    fn status_info_and_alloc_round_trip_flat_json_and_one_descriptor() {
        let (client, server) = pair();
        let client = Client::from_socket(client, KEY.into()).unwrap();
        let worker = std::thread::spawn(move || {
            let mut seen = vec![request(&server)];
            reply(&server, &status("0.1.6", "ready"), &[]);
            seen.push(request(&server));
            reply(&server, &serde_json::json!({"ok": true, "size": 2143289344u64, "free": 2143289344u64,
                "largest": 2143289344u64, "pages": "2MB pages"}), &[]);
            seen.push(request(&server));
            let file = std::fs::File::open("/dev/null").unwrap();
            reply(&server, &serde_json::json!({"ok": true, "size": 1u64 << 30}), &[file.as_raw_fd()]);
            seen
        });
        assert_eq!(client.version().unwrap(), "0.1.6");
        assert_eq!(client.info().unwrap(), Info { size: 2143289344, free: 2143289344, largest: 2143289344 });
        let fd = client.alloc(1 << 30).unwrap();
        assert!(fd.as_raw_fd() >= 0);
        assert_eq!(worker.join().unwrap(), vec![
            serde_json::json!({"key": KEY, "op": "status"}), serde_json::json!({"key": KEY, "op": "info"}),
            serde_json::json!({"key": KEY, "op": "alloc", "size": 1u64 << 30})]);
    }

    #[test]
    fn released_versions_pass_and_others_fail_open_with_typed_errors() {
        for version in ["0.1.0", "0.1.3", "0.1.6", "0.1.7", "0.1.12"] {
            let Value::Object(map) = status(version, "ready") else { unreachable!() };
            assert_eq!(check_status(&map).unwrap(), version);
        }
        for version in ["0.2.0", "1.0.0", "dev", ""] {
            let Value::Object(map) = status(version, "ready") else { unreachable!() };
            assert!(matches!(check_status(&map), Err(AtticError::UnsupportedVersion { .. })), "{version}");
        }
        let Value::Object(map) = status("0.1.6", "refusing") else { unreachable!() };
        assert!(matches!(check_status(&map), Err(AtticError::NotReady { .. })));
        let Value::Object(mut map) = status("0.1.6", "ready") else { unreachable!() };
        map.insert("name".into(), "other".into());
        assert!(matches!(check_status(&map), Err(AtticError::Protocol(_))));
    }

    #[test]
    fn refusals_and_extra_descriptors_close_every_received_fd() {
        let (client, server) = pair();
        let client = Client::from_socket(client, KEY.into()).unwrap();
        let before = open_fds();
        let worker = std::thread::spawn(move || {
            let file = std::fs::File::open("/dev/null").unwrap();
            request(&server);
            reply(&server, &serde_json::json!({"ok": false, "error": "only 0 MiB free"}), &[file.as_raw_fd()]);
            request(&server);
            reply(&server, &serde_json::json!({"ok": true, "size": 2u64 << 20}),
                &[file.as_raw_fd(), file.as_raw_fd()]);
            request(&server);
            reply(&server, &serde_json::json!({"ok": true, "size": 4u64 << 20}), &[file.as_raw_fd()]);
            request(&server);
            reply(&server, &serde_json::json!({"ok": true, "size": 2u64 << 20}), &[]);
        });
        assert!(matches!(client.alloc(2 << 20), Err(AtticError::Refused(reason)) if reason == "only 0 MiB free"));
        assert!(matches!(client.alloc(2 << 20), Err(AtticError::Protocol(_))));
        assert!(matches!(client.alloc(2 << 20), Err(AtticError::Protocol(_))));
        assert!(matches!(client.alloc(2 << 20), Err(AtticError::Protocol(_))));
        worker.join().unwrap();
        assert_eq!(open_fds(), before);
        assert!(client.alloc(3).is_err());
    }

    #[test]
    fn old_info_without_pages_and_closed_peer() {
        let (client, server) = pair();
        let client = Client::from_socket(client, KEY.into()).unwrap();
        let worker = std::thread::spawn(move || {
            request(&server);
            reply(&server, &serde_json::json!({"ok": true, "size": 10u64, "free": 8u64, "largest": 6u64}), &[]);
            request(&server);
        });
        assert_eq!(client.info().unwrap(), Info { size: 10, free: 8, largest: 6 });
        worker.join().unwrap();
        assert!(client.info().is_err());
    }

    #[test]
    fn missing_socket_unreadable_key_and_bad_key_are_typed() {
        let dir = std::env::temp_dir().join(format!("attic-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let settings = Settings { socket: dir.join("absent.sock"), key_file: dir.join("key"),
            timeout: Duration::from_secs(1) };
        assert!(matches!(Client::connect(&settings), Err(AtticError::NotInstalled(_))));
        std::fs::write(dir.join("key"), "has space\n").unwrap();
        assert!(matches!(read_key(&dir.join("key")), Err(AtticError::InvalidKey)));
        std::fs::write(dir.join("key"), format!("{KEY}\n")).unwrap();
        assert_eq!(read_key(&dir.join("key")).unwrap(), KEY);
        assert!(matches!(read_key(&dir.join("absent")), Err(AtticError::NotInstalled(_))));
        assert!(Client::from_socket(pair().0, "bad\"key".into()).is_err());
        let refusal = AtticError::NoAccess { path: DEFAULT_SOCKET.into(),
            source: io::Error::from_raw_os_error(libc::EACCES) }.to_string();
        assert!(refusal.contains("--socket-group") && refusal.contains("0640") && refusal.contains("INSTALL.md"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn lend_plan_is_slice_aligned_and_bounded_by_largest() {
        assert_eq!(plan_lend(115_133_644_800 / 90, 2143289344), 1279262720 - 1279262720 % SLICE_BYTES);
        assert_eq!(plan_lend(64 << 30, 2143289344), 2143289344);
        assert_eq!(plan_lend(64 << 30, (3 << 20) + 5), 2 << 20);
        assert_eq!(plan_lend(1 << 20, 2143289344), 0);
        assert_eq!(plan_lend(64 << 30, 0), 0);
    }

    #[test]
    fn settings_use_the_projects_own_overrides() {
        let settings = Settings::from_env();
        if std::env::var_os("SPARK_ATTIC_SOCK").is_none() { assert_eq!(settings.socket, Path::new(DEFAULT_SOCKET)); }
        if std::env::var_os("SPARK_ATTIC_KEY_FILE").is_none() { assert_eq!(settings.key_file, Path::new(DEFAULT_KEY_FILE)); }
    }
}

/// What the process borrowed: the connection (lends stay alive with it) and
/// the lent descriptors, until the first expert weight arena imports them.
pub struct Lease {
    _client: Client,
    pub version: String,
    pub parts: Vec<(OwnedFd, u64)>,
}

impl Lease {
    pub fn bytes(&self) -> u64 { self.parts.iter().map(|(_, bytes)| bytes).sum() }
}

/// Borrow up to `wanted` bytes (whole slices, at most `info.largest`).
pub fn borrow(settings: &Settings, wanted: u64) -> Result<Option<Lease>> {
    let client = Client::connect(settings)?;
    let version = client.version()?;
    let info = client.info()?;
    let bytes = plan_lend(wanted, info.largest);
    if bytes == 0 { return Ok(None); }
    let fd = client.alloc(bytes)?;
    Ok(Some(Lease { _client: client, version, parts: vec![(fd, bytes)] }))
}

static ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// `--spark-carveout off` disables every connection and credit in this process.
pub fn set_enabled(enabled: bool) { ENABLED.store(enabled, std::sync::atomic::Ordering::SeqCst); }
pub fn enabled() -> bool { ENABLED.load(std::sync::atomic::Ordering::SeqCst) }

static PENDING: std::sync::Mutex<Option<Lease>> = std::sync::Mutex::new(None);
/// Connections of consumed leases stay open for the process: atticd counts a
/// slice lent until its connection closes and the last mapping goes away.
static HELD: std::sync::Mutex<Vec<Client>> = std::sync::Mutex::new(Vec::new());

/// Install a borrowed lease for the next expert weight arena (one per process).
pub fn install(lease: Lease) {
    *PENDING.lock().unwrap_or_else(|error| error.into_inner()) = Some(lease);
}

/// Bytes borrowed and not yet mapped: device admission credits them, because
/// the carveout is outside `MemAvailable` and only the expert arena takes it.
pub fn pending_bytes() -> u64 {
    PENDING.lock().unwrap_or_else(|error| error.into_inner()).as_ref().map_or(0, Lease::bytes)
}

/// Take the pending lease when `bytes` can hold all of it; the caller maps it
/// and must call [`retain`] with the connection after a successful map.
pub fn take_for(bytes: u64) -> Option<Lease> {
    let mut pending = PENDING.lock().unwrap_or_else(|error| error.into_inner());
    if pending.as_ref().is_some_and(|lease| lease.bytes() <= bytes - bytes % SLICE_BYTES) { pending.take() } else { None }
}

/// Keep a mapped lease's connection alive for the process lifetime.
pub fn retain(lease: Lease) {
    HELD.lock().unwrap_or_else(|error| error.into_inner()).push(lease._client);
}

/// Give a lease back unmapped (the connection closes; atticd frees the slices).
pub fn release(lease: Lease) { drop(lease); }
