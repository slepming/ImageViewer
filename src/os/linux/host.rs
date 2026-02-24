#[cfg(target_os = "linux")]
pub fn gethostname() -> String {
    let raw_str = rustix::system::uname().nodename().to_bytes().to_vec();
    String::from_utf8(raw_str).unwrap_or_default()
}
