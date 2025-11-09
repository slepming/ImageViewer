mod app;

use std::{env, process::exit};

use anyhow::Ok;
use app::App;
use log::{Level, Log, SetLoggerError, info, warn};
use winit::event_loop::EventLoop;

struct SimpleLogger;

impl Log for SimpleLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= Level::Info
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

pub fn init() -> Result<(), SetLoggerError> {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Info))
}

static LOGGER: SimpleLogger = SimpleLogger;

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
