use std::sync::Arc;

#[cfg(target_os = "linux")]
use winit::platform::wayland::WindowAttributesExtWayland;

use winit::dpi::{PhysicalSize, Size};
use winit::raw_window_handle::HasWindowHandle;
use winit::{event_loop::ActiveEventLoop, monitor::MonitorHandle, window::Window};

#[cfg(target_os = "linux")]
use crate::os::linux::host::gethostname;

pub struct WindowManager {
    pub monitor: MonitorHandle,
    pub window: Arc<Window>,
    pub hostname: Option<String>,
}

impl WindowManager {
    pub fn new(event_loop: &ActiveEventLoop) -> Arc<Self> {
        let hostname: Option<String>;
        #[cfg(target_os = "linux")]
        {
            hostname = Some(gethostname());
        }

        #[cfg(target_os = "windows")]
        {
            hostname = None;
        }

        let monitor = event_loop
            .available_monitors()
            .next()
            .expect("can't get monitor");

        #[cfg(target_os = "linux")]
        let window = Arc::new(
            event_loop
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
                .unwrap(),
        );

        #[cfg(target_os = "windows")]
        let window = Arc::new(
            event_loop
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
                .unwrap(),
        );
        Arc::new(WindowManager {
            monitor,
            window,
            hostname: hostname,
        })
    }
}
