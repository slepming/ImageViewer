mod app;

use std::{env, process::exit};

use anyhow::Ok;
use app::App;
use log::{info, warn};
use winit::event_loop::EventLoop;
mod shaders;

fn main() -> Result<(), anyhow::Error> {
    pretty_env_logger::init_timed();
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        warn!("app haven't argument path to image. Please, set up argument path");
        exit(-1);
    }
    info!("{}", args[1]);
    let event_loop = EventLoop::new()?;
    let mut app = App::new(&args[1], &event_loop);
    event_loop.run_app(&mut app)?;
    Ok(())
}
