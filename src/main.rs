mod window;
use std::{env, process::exit};

use anyhow::Ok;
use log::{info, warn};
use tracy_client::Client;
use window::App;
use winit::event_loop::EventLoop;
mod os;
mod shaders;

#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<std::alloc::System> =
    tracy_client::ProfiledAllocator::new(std::alloc::System, 100);

fn main() -> Result<(), anyhow::Error> {
    pretty_env_logger::init_timed();
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        warn!("app haven't argument path to image. Please, set up argument path");
        exit(-1);
    }

    info!("{}", args[1]);
    let image_path = &args[1];
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App::new(image_path, &event_loop);
    event_loop.run_app(&mut app)?;
    Ok(())
}
