//! Focus Desktop Simulator - A high-performance desktop simulator
//!
//! A Rust implementation of the Focus Desktop Simulator with an isometric 3D desk
//! and interactive objects. Uses wgpu for GPU rendering.

mod camera;
mod config;
mod desk_object;
mod mesh;
mod physics;
mod state;

use camera::Camera;
use config::{hex_to_rgb, hex_to_rgba, CONFIG};
use desk_object::{DeskObject, ObjectType};
use mesh::{generate_object_mesh, MeshData, Vertex};
use physics::PhysicsEngine;
use state::AppState;

use glam::{Mat4, Quat, Vec3};
use log::info;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

/// Camera uniform buffer data
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    position: [f32; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            position: [0.0; 4],
        }
    }

    fn update(&mut self, camera: &Camera) {
        self.view_proj = camera.view_projection_matrix().to_cols_array_2d();
        self.position = [
            camera.position.x,
            camera.position.y,
            camera.position.z,
            1.0,
        ];
    }
}

/// Model uniform buffer data for per-object transforms
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelUniform {
    model: [[f32; 4]; 4],
}

impl ModelUniform {
    fn new() -> Self {
        Self {
            model: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    fn from_transform(position: Vec3, rotation: Quat, scale: f32) -> Self {
        let model = Mat4::from_scale_rotation_translation(Vec3::splat(scale), rotation, position);
        Self {
            model: model.to_cols_array_2d(),
        }
    }
}

/// GPU mesh handle
struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl GpuMesh {
    fn from_mesh_data(device: &wgpu::Device, data: &MeshData) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Object Vertex Buffer"),
            contents: bytemuck::cast_slice(&data.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Object Index Buffer"),
            contents: bytemuck::cast_slice(&data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            num_indices: data.indices.len() as u32,
        }
    }
}

/// Main application state
struct App {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    model_bind_group_layout: wgpu::BindGroupLayout,
    depth_texture: wgpu::TextureView,
    desk_mesh: GpuMesh,
    floor_mesh: GpuMesh,
    object_meshes: HashMap<u64, (GpuMesh, wgpu::Buffer, wgpu::BindGroup)>,
    camera: Camera,
    state: AppState,
    physics: PhysicsEngine,
    mouse_position: (f32, f32),
    left_mouse_down: bool,
    dragging_object_id: Option<u64>,
    last_frame_time: Instant,
    shift_pressed: bool,
    current_object_type_index: usize,
}

impl App {
    async fn new(window: Arc<Window>) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();
        let aspect = size.width as f32 / size.height as f32;

        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface
        let surface = instance.create_surface(window.clone())?;

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find an appropriate adapter")?;

        // Create device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Create camera uniform buffer
        let camera_uniform = CameraUniform::new();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create camera bind group layout
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Create model bind group layout for per-object transforms
        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("model_bind_group_layout"),
            });

        // Create render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &model_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create depth texture
        let depth_texture = Self::create_depth_texture(&device, &config);

        // Create static meshes
        let desk_mesh = Self::create_desk_mesh(&device);
        let floor_mesh = Self::create_floor_mesh(&device);

        // Create camera
        let camera = Camera::new(aspect);

        // Load state
        let app_state = AppState::load();
        let mut physics = PhysicsEngine::new();
        physics.collision_radius_multiplier = app_state.collision_radius_multiplier;

        let mut app = Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            camera_buffer,
            camera_bind_group,
            model_bind_group_layout,
            depth_texture,
            desk_mesh,
            floor_mesh,
            object_meshes: HashMap::new(),
            camera,
            state: app_state,
            physics,
            mouse_position: (0.0, 0.0),
            left_mouse_down: false,
            dragging_object_id: None,
            last_frame_time: Instant::now(),
            shift_pressed: false,
            current_object_type_index: 0,
        };

        // Create meshes for existing objects
        app.rebuild_object_meshes();

        Ok(app)
    }

    fn rebuild_object_meshes(&mut self) {
        self.object_meshes.clear();
        let objects: Vec<DeskObject> = self.state.objects.clone();
        for obj in objects {
            self.create_object_mesh_from_data(
                obj.id,
                obj.object_type,
                obj.color,
                obj.accent_color,
                obj.position,
                obj.rotation,
                obj.scale,
            );
        }
    }

    fn create_object_mesh(&mut self, obj: &DeskObject) {
        self.create_object_mesh_from_data(
            obj.id,
            obj.object_type,
            obj.color,
            obj.accent_color,
            obj.position,
            obj.rotation,
            obj.scale,
        );
    }

    fn create_object_mesh_from_data(
        &mut self,
        id: u64,
        object_type: ObjectType,
        color: u32,
        accent_color: u32,
        position: Vec3,
        rotation: Quat,
        scale: f32,
    ) {
        let mesh_data = generate_object_mesh(object_type, color, accent_color);
        let gpu_mesh = GpuMesh::from_mesh_data(&self.device, &mesh_data);

        let model_uniform = ModelUniform::from_transform(position, rotation, scale);
        let model_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Model Buffer"),
                contents: bytemuck::cast_slice(&[model_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let model_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.model_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: model_buffer.as_entire_binding(),
            }],
            label: Some("model_bind_group"),
        });

        self.object_meshes
            .insert(id, (gpu_mesh, model_buffer, model_bind_group));
    }

    fn update_object_transform(&mut self, id: u64) {
        if let Some(obj) = self.state.get_object(id) {
            if let Some((_, buffer, _)) = self.object_meshes.get(&id) {
                let model_uniform =
                    ModelUniform::from_transform(obj.position, obj.rotation, obj.scale);
                self.queue
                    .write_buffer(buffer, 0, bytemuck::cast_slice(&[model_uniform]));
            }
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config);
            self.camera
                .set_aspect(new_size.width as f32 / new_size.height as f32);
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let _dt = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        // Update physics for dropping objects
        let objects_clone: Vec<DeskObject> = self.state.objects.clone();
        let mut updated_ids: Vec<u64> = Vec::new();
        for obj in &mut self.state.objects {
            if !obj.is_dragging {
                if self
                    .physics
                    .update_dropping(obj, &objects_clone, CONFIG.physics.drop_speed)
                {
                    updated_ids.push(obj.id);
                }
            }
        }

        for id in updated_ids {
            self.update_object_transform(id);
        }

        // Update camera uniform
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update(&self.camera);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
    }

    fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Create identity model matrix for static meshes
        let identity_model = ModelUniform::new();
        let identity_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Identity Model Buffer"),
                contents: bytemuck::cast_slice(&[identity_model]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let identity_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.model_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: identity_buffer.as_entire_binding(),
            }],
            label: Some("identity_model_bind_group"),
        });

        {
            let bg_color = hex_to_rgba(CONFIG.colors.background);
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg_color[0] as f64,
                            g: bg_color[1] as f64,
                            b: bg_color[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &identity_bind_group, &[]);

            // Render floor
            render_pass.set_vertex_buffer(0, self.floor_mesh.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                self.floor_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(0..self.floor_mesh.num_indices, 0, 0..1);

            // Render desk
            render_pass.set_vertex_buffer(0, self.desk_mesh.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                self.desk_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(0..self.desk_mesh.num_indices, 0, 0..1);

            // Render objects with their transforms
            for obj in &self.state.objects {
                if let Some((mesh, _, bind_group)) = self.object_meshes.get(&obj.id) {
                    render_pass.set_bind_group(1, bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::MouseInput { button, state, .. } => {
                if *button == MouseButton::Left {
                    self.left_mouse_down = *state == ElementState::Pressed;
                    if !self.left_mouse_down {
                        // End drag
                        if let Some(id) = self.dragging_object_id.take() {
                            let objects_clone: Vec<DeskObject> = self.state.objects.clone();
                            if let Some(obj) = self.state.get_object_mut(id) {
                                self.physics.end_drag(obj, &objects_clone);
                                self.update_object_transform(id);
                            }
                        }
                    } else {
                        self.try_pick_object();
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_position = (position.x as f32, position.y as f32);
                if self.left_mouse_down && self.dragging_object_id.is_some() {
                    self.update_drag();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };
                if let Some(id) = self.dragging_object_id {
                    if self.shift_pressed {
                        if let Some(obj) = self.state.get_object_mut(id) {
                            obj.scale = (obj.scale + scroll * 0.1).clamp(0.3, 3.0);
                            self.update_object_transform(id);
                        }
                    } else if let Some(obj) = self.state.get_object_mut(id) {
                        obj.rotation = Quat::from_rotation_y(scroll * 0.2) * obj.rotation;
                        self.update_object_transform(id);
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match key {
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                            self.shift_pressed = event.state == ElementState::Pressed;
                        }
                        KeyCode::KeyA if event.state == ElementState::Pressed => {
                            // Add object of current type
                            let object_types = [
                                ObjectType::Clock,
                                ObjectType::Lamp,
                                ObjectType::Plant,
                                ObjectType::Coffee,
                                ObjectType::Laptop,
                                ObjectType::Notebook,
                                ObjectType::PenHolder,
                                ObjectType::Books,
                                ObjectType::PhotoFrame,
                                ObjectType::Globe,
                                ObjectType::Trophy,
                                ObjectType::Hourglass,
                                ObjectType::Metronome,
                                ObjectType::Paper,
                                ObjectType::Magazine,
                            ];
                            let obj_type = object_types[self.current_object_type_index];
                            self.add_object(obj_type);
                            info!(
                                "Added {} (Press T to cycle types, A to add)",
                                obj_type.display_name()
                            );
                        }
                        KeyCode::KeyT if event.state == ElementState::Pressed => {
                            // Cycle through object types
                            self.current_object_type_index =
                                (self.current_object_type_index + 1) % 15;
                            let object_types = [
                                ObjectType::Clock,
                                ObjectType::Lamp,
                                ObjectType::Plant,
                                ObjectType::Coffee,
                                ObjectType::Laptop,
                                ObjectType::Notebook,
                                ObjectType::PenHolder,
                                ObjectType::Books,
                                ObjectType::PhotoFrame,
                                ObjectType::Globe,
                                ObjectType::Trophy,
                                ObjectType::Hourglass,
                                ObjectType::Metronome,
                                ObjectType::Paper,
                                ObjectType::Magazine,
                            ];
                            info!(
                                "Selected: {} (Press A to add)",
                                object_types[self.current_object_type_index].display_name()
                            );
                        }
                        KeyCode::Delete if event.state == ElementState::Pressed => {
                            // Delete dragged object
                            if let Some(id) = self.dragging_object_id.take() {
                                self.state.remove_object(id);
                                self.object_meshes.remove(&id);
                                info!("Deleted object");
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn try_pick_object(&mut self) {
        let (mx, my) = self.mouse_position;
        let ndc_x = (2.0 * mx / self.size.width as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * my / self.size.height as f32);

        let inv_proj = self.camera.projection_matrix().inverse();
        let inv_view = self.camera.view_matrix().inverse();

        let ray_clip = glam::Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let ray_eye = inv_proj * ray_clip;
        let ray_eye = glam::Vec4::new(ray_eye.x, ray_eye.y, -1.0, 0.0);
        let ray_world = (inv_view * ray_eye).truncate().normalize();

        let ray_origin = self.camera.position;
        let mut best_id = None;
        let mut best_dist = f32::MAX;

        for obj in &self.state.objects {
            let to_obj = obj.position - ray_origin;
            let t = to_obj.dot(ray_world);
            if t < 0.0 {
                continue;
            }

            let closest = ray_origin + ray_world * t;
            let dist = (closest - obj.position).length();
            let radius = obj.collision_radius() * 1.5;

            if dist < radius && t < best_dist {
                best_dist = t;
                best_id = Some(obj.id);
            }
        }

        if let Some(id) = best_id {
            self.dragging_object_id = Some(id);
            if let Some(obj) = self.state.get_object_mut(id) {
                obj.is_dragging = true;
            }
        }
    }

    fn update_drag(&mut self) {
        let (mx, my) = self.mouse_position;
        let ndc_x = (2.0 * mx / self.size.width as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * my / self.size.height as f32);

        let inv_proj = self.camera.projection_matrix().inverse();
        let inv_view = self.camera.view_matrix().inverse();

        let ray_clip = glam::Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let ray_eye = inv_proj * ray_clip;
        let ray_eye = glam::Vec4::new(ray_eye.x, ray_eye.y, -1.0, 0.0);
        let ray_world = (inv_view * ray_eye).truncate().normalize();

        let desk_y = self.physics.desk_surface_y();
        let plane_y = desk_y + 0.5;

        if let Some(intersection) = physics::ray_plane_intersection(
            self.camera.position,
            ray_world,
            Vec3::new(0.0, plane_y, 0.0),
            Vec3::Y,
        ) {
            if let Some(id) = self.dragging_object_id {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.position.x = intersection.x.clamp(-4.5, 4.5);
                    obj.position.z = intersection.z.clamp(-3.0, 3.0);
                    obj.position.y = plane_y;
                    self.update_object_transform(id);
                }
            }
        }
    }

    fn add_object(&mut self, object_type: ObjectType) {
        let id = self.state.next_id();
        let desk_y = self.physics.desk_surface_y();
        let position = Vec3::new(
            rand::random::<f32>() * 4.0 - 2.0,
            desk_y,
            rand::random::<f32>() * 3.0 - 1.5,
        );
        let object = DeskObject::new(id, object_type, position);
        self.create_object_mesh(&object);
        self.state.add_object(object);
    }

    fn save_state(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.state.save()
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_desk_mesh(device: &wgpu::Device) -> GpuMesh {
        let (r, g, b) = hex_to_rgb(CONFIG.desk.color);
        let hw = CONFIG.desk.width / 2.0;
        let hd = CONFIG.desk.depth / 2.0;
        let h = CONFIG.desk.height;

        let vertices = vec![
            // Top
            Vertex {
                position: [-hw, h, -hd],
                normal: [0.0, 1.0, 0.0],
                color: [r, g, b, 1.0],
            },
            Vertex {
                position: [hw, h, -hd],
                normal: [0.0, 1.0, 0.0],
                color: [r, g, b, 1.0],
            },
            Vertex {
                position: [hw, h, hd],
                normal: [0.0, 1.0, 0.0],
                color: [r, g, b, 1.0],
            },
            Vertex {
                position: [-hw, h, hd],
                normal: [0.0, 1.0, 0.0],
                color: [r, g, b, 1.0],
            },
            // Front
            Vertex {
                position: [-hw, 0.0, hd],
                normal: [0.0, 0.0, 1.0],
                color: [r * 0.8, g * 0.8, b * 0.8, 1.0],
            },
            Vertex {
                position: [hw, 0.0, hd],
                normal: [0.0, 0.0, 1.0],
                color: [r * 0.8, g * 0.8, b * 0.8, 1.0],
            },
            Vertex {
                position: [hw, h, hd],
                normal: [0.0, 0.0, 1.0],
                color: [r * 0.8, g * 0.8, b * 0.8, 1.0],
            },
            Vertex {
                position: [-hw, h, hd],
                normal: [0.0, 0.0, 1.0],
                color: [r * 0.8, g * 0.8, b * 0.8, 1.0],
            },
        ];

        let indices: Vec<u16> = vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Desk Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Desk Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        GpuMesh {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
        }
    }

    fn create_floor_mesh(device: &wgpu::Device) -> GpuMesh {
        let (r, g, b) = hex_to_rgb(CONFIG.colors.ground);
        let s = 50.0;

        let vertices = vec![
            Vertex {
                position: [-s, 0.0, -s],
                normal: [0.0, 1.0, 0.0],
                color: [r, g, b, 1.0],
            },
            Vertex {
                position: [s, 0.0, -s],
                normal: [0.0, 1.0, 0.0],
                color: [r, g, b, 1.0],
            },
            Vertex {
                position: [s, 0.0, s],
                normal: [0.0, 1.0, 0.0],
                color: [r, g, b, 1.0],
            },
            Vertex {
                position: [-s, 0.0, s],
                normal: [0.0, 1.0, 0.0],
                color: [r, g, b, 1.0],
            },
        ];

        let indices: Vec<u16> = vec![0, 1, 2, 0, 2, 3];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Floor Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Floor Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        GpuMesh {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    info!("Starting Focus Desktop Simulator...");
    info!("Controls:");
    info!("  T - Cycle through object types");
    info!("  A - Add selected object");
    info!("  Delete - Delete dragged object");
    info!("  Click+Drag - Move object");
    info!("  Scroll - Rotate object");
    info!("  Shift+Scroll - Scale object");

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let window = WindowBuilder::new()
        .with_title("Focus Desktop Simulator")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)
        .expect("Failed to create window");

    let window = Arc::new(window);
    let mut app = pollster::block_on(App::new(window.clone())).expect("Failed to create app");

    info!("Application initialized");

    event_loop.set_control_flow(ControlFlow::Poll);

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    app.handle_event(&event);

                    match event {
                        WindowEvent::CloseRequested => {
                            info!("Saving state and exiting...");
                            let _ = app.save_state();
                            elwt.exit();
                        }
                        WindowEvent::Resized(size) => app.resize(size),
                        WindowEvent::RedrawRequested => {
                            app.update();
                            if let Err(e) = app.render() {
                                match e {
                                    wgpu::SurfaceError::Lost => app.resize(app.size),
                                    wgpu::SurfaceError::OutOfMemory => elwt.exit(),
                                    _ => log::error!("Render error: {:?}", e),
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .expect("Event loop error");
}
