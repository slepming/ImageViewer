// https://github.com/vulkano-rs/vulkano/blob/master/examples/image/main.rs
use std::{ops::RangeInclusive, process::exit, result::Result::Ok, sync::Arc};

use crate::shaders::rectangle::{frag_rect, vert_rect};
use image::{GenericImageView, ImageReader};
use log::{debug, error, info};
use png::Compression;
use vulkano::{
    DeviceSize, Validated, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract, RenderPassBeginInfo,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags},
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions, debug},
    memory::{
        self,
        allocator::{
            AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
            StandardMemoryAllocator,
        },
    },
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    single_pass_renderpass,
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalSize, Size},
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    platform::{
        modifier_supplement::KeyEventExtModifierSupplement, wayland::WindowAttributesExtWayland,
    },
    window::Window,
};

pub struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory: MemoryApp,
    texture: Arc<ImageView>,
    sampler: Arc<Sampler>,
    current_image: String,
    render: Option<RenderContext>,
}

pub struct MemoryApp {
    vertex_buffer: Arc<Subbuffer<[ImagePos]>>,
    memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    command_buffer_allocate: Arc<StandardCommandBufferAllocator>,
    descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
}

pub struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    descriptor_set: Arc<DescriptorSet>,
    recreate_swapchain: bool,
    surface: Arc<Surface>,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    pub fn new(image: &str, e: &EventLoop<()>) -> App {
        let library = VulkanLibrary::new().expect("no local Vulkan library/dll");
        let required_extensions = Surface::required_extensions(&e).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: InstanceExtensions {
                    khr_wayland_surface: true,
                    khr_display: true,
                    ext_swapchain_colorspace: true,
                    ..required_extensions
                },
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let physical_device = instance
            .enumerate_physical_devices()
            .expect("could not enumerate physical devices")
            .next()
            .expect("No Device available");

        for family in physical_device.queue_family_properties() {
            info!("Found available devices: {}", family.queue_count);
        }

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
            })
            .expect("Could not find a graphical device") as u32;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions {
                    khr_swapchain: true,
                    ..Default::default()
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();

        println!(
            "Founded device {:?}, Queue: {:?}",
            device.physical_device().properties().device_name,
            queue
        );

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocate = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let cube = [
            ImagePos {
                position: [-1.0, -1.0],
                zoom: 0.0,
            },
            ImagePos {
                position: [-1.0, 1.0],
                zoom: 0.0,
            },
            ImagePos {
                position: [1.0, -1.0],
                zoom: 0.0,
            },
            ImagePos {
                position: [1.0, 1.0],
                zoom: 0.0,
            },
        ];
        let vertex_buffer = Arc::new(
            Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                cube,
            )
            .expect("Creating vertex buffer failed"),
        );
        debug!("vertex_buffer: {}", vertex_buffer.len());

        let mut uploads = AutoCommandBufferBuilder::primary(
            command_buffer_allocate.clone(),
            queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let texture = {
            let reader = ImageReader::open(image).unwrap();
            let decode = reader.decode().expect("error decoding image");
            let info = decode.dimensions();
            info!("{:?}", info);
            let extent = [info.0, info.1, 1];

            let upload_buffer: Subbuffer<[u8]> = Buffer::new_slice(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                (info.0 * info.1 * 4) as DeviceSize,
            )
            .unwrap();

            upload_buffer
                .write()
                .unwrap()
                .copy_from_slice(&decode.to_rgba8().into_raw());

            let image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent,
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .expect("error creating image");

            uploads
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    upload_buffer,
                    image.clone(),
                ))
                .unwrap();

            ImageView::new_default(image.clone()).unwrap()
        };

        let _ = uploads.build().unwrap().execute(queue.clone()).unwrap();

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                border_color: vulkano::image::sampler::BorderColor::FloatTransparentBlack,
                ..Default::default()
            },
        )
        .unwrap();

        info!(
            "Device color gamut: {:?}",
            device.physical_device().display_properties().unwrap()
        );

        App {
            instance: instance,
            device: device,
            queue: queue,
            memory: MemoryApp {
                vertex_buffer: vertex_buffer,
                memory_allocator: memory_allocator,
                command_buffer_allocate: command_buffer_allocate,
                descriptor_allocator: descriptor_set_allocator,
            },
            current_image: image.to_string(),
            sampler: sampler,
            texture: texture,
            render: None,
        }
    }

    pub fn zoom_image(&mut self, cube: [ImagePos; 4]) {
        let buffer = Buffer::from_iter(
            self.memory.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            cube,
        ); // mb i need create new AutoCommandBufferBuilder?
        debug!("vertex_buffer: {}", self.memory.vertex_buffer.len());
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let first_monitor = event_loop
            .available_monitors()
            .next()
            .expect("can't get monitor");
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Slepming Image Viewer")
                        .with_name("viewer", "slepming viewer")
                        .with_transparent(true)
                        .with_blur(true)
                        .with_min_inner_size(Size::Physical(PhysicalSize {
                            height: 480,
                            width: 640,
                        }))
                        .with_max_inner_size(Size::Physical(PhysicalSize {
                            height: first_monitor.size().height,
                            width: first_monitor.size().width,
                        })),
                )
                .unwrap(),
        );

        debug!("Resolution {:?}", window.inner_size());

        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            Swapchain::new(
                self.device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: vulkano::swapchain::CompositeAlpha::PreMultiplied,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let render_pass = single_pass_renderpass!(self.device.clone(), attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        )
        .unwrap();

        let framebuffers = window_size_dependent_setup(&images, &render_pass);

        let pipeline = {
            let vs = vert_rect::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = frag_rect::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = ImagePos::per_vertex().definition(&vs).unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];
            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
                    .unwrap(),
            )
            .unwrap();
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(self.device.clone(), None, GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState {
                    topology: vulkano::pipeline::graphics::input_assembly::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::alpha()),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some((subpass.clone()).into()),
                ..GraphicsPipelineCreateInfo::layout(layout.clone())
            }).unwrap()
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: RangeInclusive::new(0.0, 1.0),
        };

        let layout = &pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            self.memory.descriptor_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::sampler(0, self.sampler.clone()),
                WriteDescriptorSet::image_view(1, self.texture.clone()),
            ],
            [],
        )
        .unwrap();

        let previous_frame_end = Some(vulkano::sync::now(self.device.clone()).boxed());

        self.render = Some(RenderContext {
            previous_frame_end: previous_frame_end,
            descriptor_set: descriptor_set,
            framebuffers: framebuffers,
            pipeline: pipeline,
            recreate_swapchain: false,
            render_pass: render_pass,
            viewport: viewport,
            surface: surface,
            window: window,
            swapchain: swapchain,
        });
    }
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.render.as_mut().unwrap();

        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed && !event.repeat {
                    match event.key_without_modifiers().as_ref() {
                        Key::Named(NamedKey::Escape) => {
                            // can I call CloseRequest?
                            exit(0);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                MouseScrollDelta::LineDelta(y, ..) => {
                    let new_pos = [
                        ImagePos {
                            position: [-1.0, -1.0],
                            zoom: y,
                        },
                        ImagePos {
                            position: [-1.0, 1.0],
                            zoom: y,
                        },
                        ImagePos {
                            position: [1.0, -1.0],
                            zoom: y,
                        },
                        ImagePos {
                            position: [1.0, 1.0],
                            zoom: y,
                        },
                    ];
                    self.zoom_image(&new_pos);
                }
                _ => {}
            },
            WindowEvent::CloseRequested => {
                debug!("exiting..");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .unwrap();

                    rcx.swapchain = new_swapchain;
                    rcx.framebuffers = window_size_dependent_setup(&new_images, &rcx.render_pass);
                    rcx.viewport.extent = window_size.into();
                    rcx.recreate_swapchain = false;
                    debug!(
                        "Window changed size to: {:?}; swapchain size image extent {:?}",
                        window_size,
                        rcx.swapchain.create_info().image_extent
                    )
                }

                let (image_index, suboptimal, acqure_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.memory.command_buffer_allocate.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 0.0, 0.0].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[image_index as usize].clone(),
                            )
                        },
                        Default::default(),
                    )
                    .unwrap()
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(rcx.pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        rcx.pipeline.layout().clone(),
                        0,
                        rcx.descriptor_set.clone(),
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, self.memory.vertex_buffer.as_ref().clone())
                    .unwrap();

                unsafe {
                    builder
                        .draw(self.memory.vertex_buffer.len() as u32, 1, 0, 0)
                        .unwrap();
                }

                builder.end_render_pass(Default::default()).unwrap();

                let command_buffer = builder.build().unwrap();
                let future = rcx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acqure_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rcx.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        error!("failed to flush future: {e}");
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                }

                if rcx.window.inner_size() != rcx.swapchain.image_extent().into() {
                    rcx.recreate_swapchain = true;
                }
            }
            _ => (),
        }
    }
    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let rcx = self.render.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
pub struct ImagePos {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32_SFLOAT)]
    zoom: f32,
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect()
}
