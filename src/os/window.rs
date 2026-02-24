use std::sync::Arc;

#[cfg(feature = "winit")]
use winit::event_loop::EventLoop;
use winit::{
    dpi::{PhysicalSize, Size},
    event_loop::ActiveEventLoop,
    monitor::MonitorHandle,
    platform::wayland::WindowAttributesExtWayland,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

#[cfg(target_os = "linux")]
use crate::os::linux::host::gethostname;

pub struct WindowManager<B: Backend> {
    pub monitor: MonitorHandle,
    pub inner_size: PhysicalSize<u32>,
    pub window: Arc<B::Window>,
    pub hostname: Option<String>,
}

pub trait Backend {
    type Window: HasWindowHandle + HasDisplayHandle + Send + Sync;

    fn init() -> Self;
    fn create_window(&mut self) -> Arc<Self::Window>;
    fn monitor(&mut self) -> MonitorHandle;
}

#[cfg(feature = "winit")]
pub struct WinitBackend {
    pub event_loop: ActiveEventLoop,
}

#[cfg(feature = "winit")]
impl Backend for WinitBackend {
    type Window = Window;

    fn init() -> Self {
        todo!()
    }

    fn create_window(&mut self) -> Arc<Self::Window> {
        #[cfg(target_os = "linux")]
        let window = self
            .event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(format!("{} image viewer", hostname.as_ref().unwrap()))
                    .with_name("viewer", format!("{} viewer", hostname.as_ref().unwrap()))
                    .with_transparent(true)
                    .with_blur(true)
                    .with_min_inner_size(Size::Physical(PhysicalSize {
                        width: 640,
                        height: 480,
                    }))
                    .with_max_inner_size(Size::Physical(PhysicalSize {
                        width: monitor.size().width,
                        height: monitor.size().height,
                    })),
            )
            .unwrap();

        #[cfg(target_os = "windows")]
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Image Viewer")
                    .with_min_inner_size(Size::Physical(PhysicalSize {
                        width: 640,
                        height: 480,
                    }))
                    .with_max_inner_size(Size::Physical(PhysicalSize {
                        width: monitor.size().width,
                        height: monitor.size().height,
                    })),
            )
            .unwrap();

        Arc::new(window)
    }

    fn monitor(&mut self) -> MonitorHandle {
        self.event_loop
            .available_monitors()
            .next()
            .expect("can't get monitor")
    }
}
