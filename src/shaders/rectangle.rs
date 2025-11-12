pub mod vert_rect {
    vulkano_shaders::shader! {
        bytes: "shaders/rect.vert.spv"
    }
}

pub mod frag_rect {
    vulkano_shaders::shader! {
        bytes: "shaders/rect.frag.spv"
    }
}
