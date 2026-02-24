mod window;

use std::{env, process::exit};

use anyhow::Ok;
use log::{info, warn};
use window::App;
use winit::event_loop::EventLoop;

use crate::os::window::WindowManager;
use log::debug;
mod os;
mod shaders;

#[cfg(feature = "winit")]
fn main() -> Result<(), anyhow::Error> {
    use crate::os::window::WinitBackend;

    pretty_env_logger::init_timed();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        warn!("No image path provided");
        exit(1);
    }

    let image_path = &args[1];

    let mut backend: WinitBackend = WinitBackend::init();
    let window = backend.create_window();

    let mut app = App::new(image_path, backend.event_loop);

    Ok(())
}
