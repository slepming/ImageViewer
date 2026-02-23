use std::sync::Arc;

use winit::{event_loop::ActiveEventLoop, monitor::MonitorHandle, window::Window};

pub struct WindowManager {
    monitor: MonitorHandle,
    window: Window,
}

impl WindowManager {
    pub fn new(event_loop: &ActiveEventLoop) -> Arc<Self> {
        let monitor = event_loop
            .available_monitors()
            .next()
            .expect("can't get monitor");
        #[cfg(target_os = "linux")]
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title(format!("{} image viewer", self.app_data.hostname))
                        .with_name("viewer", format!("{} viewer", self.app_data.hostname))
                        .with_transparent(true)
                        .with_blur(true)
                        .with_min_inner_size(Size::Physical(PhysicalSize {
                            width: 640,
                            height: 480,
                        }))
                        .with_max_inner_size(Size::Physical(PhysicalSize {
                            width: first_monitor.size().width,
                            height: first_monitor.size().height,
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
                            width: first_monitor.size().width,
                            height: first_monitor.size().height,
                        })),
                )
                .unwrap(),
        );
        Arc::new(WindowManager { monitor, window })
    }
}
