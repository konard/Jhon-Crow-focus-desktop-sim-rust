//! Focus Desktop Simulator - A high-performance desktop simulator
//!
//! A Rust implementation of the Focus Desktop Simulator with an isometric 3D desk
//! and interactive objects. Uses wgpu for GPU rendering and egui for UI.

mod camera;
mod config;
mod desk_object;
mod mesh;
mod physics;
mod state;
mod ui;

use camera::Camera;
use config::{hex_to_rgb, hex_to_rgba, CONFIG};
use desk_object::{DeskObject, ObjectType};
use mesh::{generate_object_mesh, generate_object_mesh_with_state, MeshData, Vertex};
use physics::PhysicsEngine;
use state::AppState;
use ui::{render_left_sidebar, render_right_sidebar, render_crosshair, ObjectInfo, UiAction, UiState};

use egui_wgpu::ScreenDescriptor;
use glam::{Mat4, Quat, Vec3};
use log::info;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
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

/// Maximum number of point lights
const MAX_LIGHTS: usize = 8;

/// Lighting uniform buffer data for dynamic lighting
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightingUniform {
    /// Point light positions (xyz) and intensity (w)
    /// w = 0.0 means light is off, w > 0 means on with that intensity
    point_lights: [[f32; 4]; MAX_LIGHTS],
    /// Number of active lights
    num_lights: u32,
    /// Room darkness level (0.0 = bright, 1.0 = very dark)
    room_darkness: f32,
    /// Padding for alignment
    _padding: [f32; 2],
}

impl LightingUniform {
    fn new() -> Self {
        Self {
            point_lights: [[0.0; 4]; MAX_LIGHTS],
            num_lights: 0,
            room_darkness: 1.0, // Start with dark room
            _padding: [0.0; 2],
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
    lighting_buffer: wgpu::Buffer,
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
    // Egui integration
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    ui_state: UiState,
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
                    memory_hints: wgpu::MemoryHints::default(),
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

        // Create lighting uniform buffer
        let lighting_uniform = LightingUniform::new();
        let lighting_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lighting Buffer"),
            contents: bytemuck::cast_slice(&[lighting_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create camera/lighting bind group layout (group 0)
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lighting_buffer.as_entire_binding(),
                },
            ],
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
            cache: None,
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

        // Initialize egui
        let egui_ctx = egui::Context::default();

        // Set up dark theme for egui
        let mut style = egui::Style::default();
        style.visuals = egui::Visuals::dark();
        style.visuals.window_fill = egui::Color32::from_rgba_unmultiplied(26, 26, 46, 242);
        style.visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(26, 26, 46, 242);
        egui_ctx.set_style(style);

        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1, false);

        let ui_state = UiState::new();

        let mut app = Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            camera_buffer,
            lighting_buffer,
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
            egui_ctx,
            egui_state,
            egui_renderer,
            ui_state,
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
        let mesh_data = generate_object_mesh_with_state(
            obj.object_type,
            obj.color,
            obj.accent_color,
            Some(&obj.state),
        );
        self.create_object_mesh_from_mesh_data(
            obj.id,
            mesh_data,
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
        self.create_object_mesh_from_mesh_data(id, mesh_data, position, rotation, scale);
    }

    fn create_object_mesh_from_mesh_data(
        &mut self,
        id: u64,
        mesh_data: MeshData,
        position: Vec3,
        rotation: Quat,
        scale: f32,
    ) {
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
        let dt = (now - self.last_frame_time).as_secs_f32();
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

        // Update animated objects
        let mut animation_updates: Vec<(u64, Quat)> = Vec::new();
        let mut clock_updates: Vec<u64> = Vec::new();

        // Get current time for clock updates
        let current_time = chrono::Local::now();
        let hours = current_time.format("%H").to_string().parse::<f32>().unwrap_or(0.0);
        let minutes = current_time.format("%M").to_string().parse::<f32>().unwrap_or(0.0);
        let seconds = current_time.format("%S").to_string().parse::<f32>().unwrap_or(0.0);
        let millis = current_time.format("%3f").to_string().parse::<f32>().unwrap_or(0.0) / 1000.0;

        for obj in &mut self.state.objects {
            match obj.object_type {
                ObjectType::Clock => {
                    // Calculate clock hand angles based on real time
                    // Hours: 360 degrees / 12 hours = 30 degrees per hour + minute offset
                    let hour_12 = hours % 12.0;
                    let hour_angle = -(hour_12 + minutes / 60.0) * (std::f32::consts::TAU / 12.0);
                    // Minutes: 360 degrees / 60 minutes = 6 degrees per minute + second offset
                    let minute_angle = -(minutes + seconds / 60.0) * (std::f32::consts::TAU / 60.0);
                    // Seconds: 360 degrees / 60 seconds = 6 degrees per second (smooth with millis)
                    let second_angle = -(seconds + millis) * (std::f32::consts::TAU / 60.0);

                    // Check if angles changed significantly (avoid constant rebuilds)
                    let hour_changed = (obj.state.clock_hour_angle - hour_angle).abs() > 0.01;
                    let minute_changed = (obj.state.clock_minute_angle - minute_angle).abs() > 0.01;
                    let second_changed = (obj.state.clock_second_angle - second_angle).abs() > 0.01;

                    if hour_changed || minute_changed || second_changed {
                        obj.state.clock_hour_angle = hour_angle;
                        obj.state.clock_minute_angle = minute_angle;
                        obj.state.clock_second_angle = second_angle;
                        clock_updates.push(obj.id);
                    }
                }
                ObjectType::Globe if obj.state.globe_rotating => {
                    // Rotate globe around Y axis
                    obj.state.globe_angle += dt * 0.5; // Rotation speed
                    if obj.state.globe_angle > std::f32::consts::TAU {
                        obj.state.globe_angle -= std::f32::consts::TAU;
                    }
                    let new_rotation = Quat::from_rotation_y(obj.state.globe_angle);
                    animation_updates.push((obj.id, new_rotation));
                }
                ObjectType::Hourglass if obj.state.hourglass_flipping => {
                    // Animate hourglass flip (180 degrees around X axis)
                    obj.state.hourglass_flip_progress += dt * 1.5; // Flip speed
                    if obj.state.hourglass_flip_progress >= 1.0 {
                        obj.state.hourglass_flip_progress = 1.0;
                        obj.state.hourglass_flipping = false;
                    }
                    // Smooth ease-in-out animation
                    let t = obj.state.hourglass_flip_progress;
                    let eased = if t < 0.5 {
                        2.0 * t * t
                    } else {
                        1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                    };
                    let angle = eased * std::f32::consts::PI;
                    let new_rotation = Quat::from_rotation_x(angle);
                    animation_updates.push((obj.id, new_rotation));
                }
                _ => {}
            }
        }

        // Rebuild clock meshes with updated hand positions
        for id in clock_updates {
            if let Some(obj) = self.state.get_object(id).cloned() {
                self.object_meshes.remove(&id);
                self.create_object_mesh(&obj);
            }
        }

        // Apply rotation updates
        for (id, rotation) in animation_updates {
            if let Some(obj) = self.state.get_object_mut(id) {
                obj.rotation = rotation;
            }
            self.update_object_transform(id);
        }

        // Update camera uniform
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update(&self.camera);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // Update lighting uniform based on lamp states
        let mut lighting_uniform = LightingUniform::new();
        let mut light_count = 0u32;

        // Room is dark by default (darkness = 1.0)
        lighting_uniform.room_darkness = 1.0;

        // Find all lamps and add their lights
        for obj in &self.state.objects {
            if obj.object_type == ObjectType::Lamp && obj.state.lamp_on {
                if light_count < MAX_LIGHTS as u32 {
                    // Lamp light is at the lamp head position (approximately)
                    // The lamp is about 0.8 units tall, light is at head (~0.75)
                    let light_pos = Vec3::new(
                        obj.position.x,
                        obj.position.y + 0.75 * obj.scale,
                        obj.position.z,
                    );
                    lighting_uniform.point_lights[light_count as usize] = [
                        light_pos.x,
                        light_pos.y,
                        light_pos.z,
                        2.5, // Light intensity
                    ];
                    light_count += 1;
                }
            }
        }

        lighting_uniform.num_lights = light_count;

        self.queue
            .write_buffer(&self.lighting_buffer, 0, bytemuck::cast_slice(&[lighting_uniform]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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

        // Render egui UI
        // Note: We need to prepare UI data before running egui to avoid borrow issues
        let (object_name, object_info) = if let Some(id) = self.ui_state.selected_object_id {
            self.state.get_object(id).map(|obj| {
                (
                    obj.object_type.display_name().to_string(),
                    ObjectInfo {
                        object_type: obj.object_type,
                        lamp_on: obj.state.lamp_on,
                        globe_rotating: obj.state.globe_rotating,
                        metronome_running: obj.state.metronome_running,
                        metronome_bpm: obj.state.metronome_bpm,
                        music_playing: obj.state.music_playing,
                        drink_type: obj.state.drink_type,
                        fill_level: obj.state.fill_level,
                        is_hot: obj.state.is_hot,
                    },
                )
            }).map_or((None, None), |(name, info)| (Some(name), Some(info)))
        } else {
            (None, None)
        };

        let egui_input = self.egui_state.take_egui_input(&self.window);
        let egui_ctx = self.egui_ctx.clone();

        // Check if there's an object under the crosshair (for visual feedback)
        let crosshair_hovering = self.ui_state.crosshair_target_id.is_some();
        let pointer_locked = self.ui_state.pointer_locked;

        let mut ui_actions = Vec::new();
        let egui_output = egui_ctx.run(egui_input, |ctx| {
            // Render left sidebar (palette)
            let left_actions = render_left_sidebar(ctx, &mut self.ui_state);
            ui_actions.extend(left_actions);

            // Render right sidebar (customization)
            let right_actions = render_right_sidebar(ctx, &mut self.ui_state, object_name.as_deref(), object_info.as_ref());
            ui_actions.extend(right_actions);

            // Render crosshair (only in pointer lock mode)
            render_crosshair(ctx, pointer_locked, crosshair_hovering);
        });

        // Process UI actions after egui rendering
        for action in ui_actions {
            self.process_ui_action(action);
        }

        // Handle egui platform output
        self.egui_state.handle_platform_output(&self.window, egui_output.platform_output);

        // Render egui
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.size.width, self.size.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        let tris = self.egui_ctx.tessellate(egui_output.shapes, egui_output.pixels_per_point);
        for (id, image_delta) in &egui_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &tris, &screen_descriptor);

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep previous content
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // Need to forget lifetime for egui-wgpu compatibility with wgpu 22
            let mut render_pass = render_pass.forget_lifetime();
            self.egui_renderer.render(&mut render_pass, &tris, &screen_descriptor);
        }

        // Free textures
        for id in &egui_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Process a UI action
    fn process_ui_action(&mut self, action: UiAction) {
        match action {
            UiAction::AddObject(object_type) => {
                self.add_object(object_type);
                info!("Added {} from UI", object_type.display_name());
            }
            UiAction::DeleteObject(id) => {
                self.state.remove_object(id);
                self.object_meshes.remove(&id);
                self.ui_state.close_customization();
                info!("Deleted object {} from UI", id);
            }
            UiAction::ChangeMainColor(id, color) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.color = color;
                }
                // Rebuild mesh with new color
                if let Some(obj) = self.state.get_object(id).cloned() {
                    self.object_meshes.remove(&id);
                    self.create_object_mesh(&obj);
                }
            }
            UiAction::ChangeAccentColor(id, color) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.accent_color = color;
                }
                // Rebuild mesh with new color
                if let Some(obj) = self.state.get_object(id).cloned() {
                    self.object_meshes.remove(&id);
                    self.create_object_mesh(&obj);
                }
            }
            UiAction::ClearAll => {
                self.state.objects.clear();
                self.object_meshes.clear();
                self.ui_state.close_customization();
                info!("Cleared all objects from UI");
            }
            UiAction::CloseCustomization => {
                self.ui_state.close_customization();
            }
            UiAction::ToggleLamp(id) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.state.lamp_on = !obj.state.lamp_on;
                    info!("Lamp {} is now {}", id, if obj.state.lamp_on { "ON" } else { "OFF" });
                    // Rebuild mesh to show light glow effect
                    let obj_clone = obj.clone();
                    self.object_meshes.remove(&id);
                    self.create_object_mesh(&obj_clone);
                }
            }
            UiAction::ToggleGlobeRotation(id) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.state.globe_rotating = !obj.state.globe_rotating;
                    info!("Globe {} rotation is now {}", id, if obj.state.globe_rotating { "ON" } else { "OFF" });
                }
            }
            UiAction::FlipHourglass(id) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    if !obj.state.hourglass_flipping {
                        obj.state.hourglass_flipping = true;
                        obj.state.hourglass_flip_progress = 0.0;
                        info!("Flipping hourglass {}", id);
                    }
                }
            }
            UiAction::ToggleMetronome(id) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.state.metronome_running = !obj.state.metronome_running;
                    info!("Metronome {} is now {}", id, if obj.state.metronome_running { "running" } else { "stopped" });
                }
            }
            UiAction::ChangeMetronomeBpm(id, bpm) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.state.metronome_bpm = bpm;
                    info!("Metronome {} BPM changed to {}", id, bpm);
                }
            }
            UiAction::ToggleMusicPlayer(id) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.state.music_playing = !obj.state.music_playing;
                    info!("Music player {} is now {}", id, if obj.state.music_playing { "playing" } else { "stopped" });
                }
            }
            UiAction::SelectPhoto(id) => {
                // Open file dialog to select an image
                info!("Photo selection requested for frame {}", id);

                // Use rfd to open a file dialog
                let file = rfd::FileDialog::new()
                    .add_filter("Images", &["png", "jpg", "jpeg", "gif", "bmp", "webp"])
                    .set_title("Select Photo for Frame")
                    .pick_file();

                if let Some(path) = file {
                    let path_str = path.to_string_lossy().to_string();
                    info!("Selected photo: {}", path_str);

                    if let Some(obj) = self.state.get_object_mut(id) {
                        obj.state.photo_path = Some(path_str.clone());
                    }

                    // TODO: In the future, load the image texture and apply it to the photo frame mesh
                    // For now, we just store the path for persistence
                }
            }
            UiAction::ChangeDrinkType(id, drink_type) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.state.drink_type = drink_type;
                    info!("Coffee mug {} drink type changed to {:?}", id, drink_type);
                    // Rebuild mesh to show new drink color
                    let obj_clone = obj.clone();
                    self.object_meshes.remove(&id);
                    self.create_object_mesh(&obj_clone);
                }
            }
            UiAction::ChangeFillLevel(id, fill_level) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.state.fill_level = fill_level;
                    info!("Coffee mug {} fill level changed to {:.0}%", id, fill_level * 100.0);
                    // Rebuild mesh to show new fill level
                    let obj_clone = obj.clone();
                    self.object_meshes.remove(&id);
                    self.create_object_mesh(&obj_clone);
                }
            }
            UiAction::ToggleHot(id) => {
                if let Some(obj) = self.state.get_object_mut(id) {
                    obj.state.is_hot = !obj.state.is_hot;
                    info!("Coffee mug {} is now {}", id, if obj.state.is_hot { "hot" } else { "cold" });
                }
            }
            UiAction::None => {}
        }
    }

    /// Handle a window event, returning whether egui consumed it
    fn handle_event(&mut self, event: &WindowEvent) -> bool {
        // First pass event to egui
        let response = self.egui_state.on_window_event(&self.window, event);

        // If egui consumed the event, don't process it further
        if response.consumed {
            return true;
        }

        match event {
            WindowEvent::Focused(focused) => {
                // Release pointer lock when window loses focus
                if !focused && self.ui_state.pointer_locked {
                    self.release_pointer_lock();
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if *button == MouseButton::Left {
                    self.left_mouse_down = *state == ElementState::Pressed;

                    if *state == ElementState::Pressed {
                        // If not in pointer lock mode, clicking enters it (unless UI is consuming)
                        if !self.ui_state.pointer_locked && !self.ui_state.left_sidebar_open && !self.ui_state.right_sidebar_open {
                            self.request_pointer_lock();
                        } else if self.ui_state.pointer_locked {
                            // In pointer lock mode, click picks up object under crosshair
                            self.try_pick_object_crosshair();
                        } else {
                            // Not pointer locked but UI might be open, try normal pick
                            self.try_pick_object();
                        }
                    } else {
                        // Mouse released - end drag
                        if let Some(id) = self.dragging_object_id.take() {
                            let objects_clone: Vec<DeskObject> = self.state.objects.clone();
                            if let Some(obj) = self.state.get_object_mut(id) {
                                self.physics.end_drag(obj, &objects_clone);
                                self.update_object_transform(id);
                            }
                        }
                    }
                } else if *button == MouseButton::Right && *state == ElementState::Pressed {
                    if self.ui_state.pointer_locked {
                        // Right-click in pointer lock mode: open customization for crosshair target
                        if let Some(id) = self.find_object_at_crosshair() {
                            if let Some(obj) = self.state.get_object(id) {
                                self.ui_state.open_customization(id, obj.color, obj.accent_color);
                                // Exit pointer lock when opening customization
                                self.release_pointer_lock();
                            }
                        } else {
                            // Right-click on empty space toggles the left sidebar
                            self.release_pointer_lock();
                            self.ui_state.toggle_left_sidebar();
                        }
                    } else {
                        // Right-click to open customization panel for clicked object
                        if let Some(id) = self.find_object_at_cursor() {
                            if let Some(obj) = self.state.get_object(id) {
                                self.ui_state.open_customization(id, obj.color, obj.accent_color);
                            }
                        } else {
                            // Right-click on empty space toggles the left sidebar
                            self.ui_state.toggle_left_sidebar();
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_position = (position.x as f32, position.y as f32);

                // In normal mode, update drag based on cursor position
                if !self.ui_state.pointer_locked && self.left_mouse_down && self.dragging_object_id.is_some() {
                    self.update_drag();
                }

                // Update crosshair target (what's under the crosshair)
                if self.ui_state.pointer_locked {
                    self.ui_state.crosshair_target_id = self.find_object_at_crosshair();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };

                // Get the object ID to manipulate (either dragging or under crosshair)
                let target_id = self.dragging_object_id.or(
                    if self.ui_state.pointer_locked {
                        self.ui_state.crosshair_target_id
                    } else {
                        // Also support scroll on hovered object in non-pointer-lock mode
                        self.find_object_at_cursor()
                    }
                );

                if let Some(id) = target_id {
                    if self.shift_pressed {
                        // Shift+Scroll scales the object
                        let new_scale = if let Some(obj) = self.state.get_object_mut(id) {
                            // Use a larger multiplier for more noticeable scaling
                            obj.scale = (obj.scale + scroll * 0.15).clamp(0.3, 3.0);
                            Some(obj.scale)
                        } else {
                            None
                        };
                        self.update_object_transform(id);
                        if let Some(scale) = new_scale {
                            info!("Scaled object to {:.2}", scale);
                        }
                    } else {
                        // Scroll rotates the object
                        if let Some(obj) = self.state.get_object_mut(id) {
                            obj.rotation = Quat::from_rotation_y(scroll * 0.2) * obj.rotation;
                        }
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
                            let object_types = ObjectType::all();
                            let obj_type = object_types[self.current_object_type_index];
                            self.add_object(obj_type);
                            info!(
                                "Added {} (Press T to cycle types, A to add)",
                                obj_type.display_name()
                            );
                        }
                        KeyCode::KeyT if event.state == ElementState::Pressed => {
                            // Cycle through object types
                            let object_types = ObjectType::all();
                            self.current_object_type_index =
                                (self.current_object_type_index + 1) % object_types.len();
                            info!(
                                "Selected: {} (Press A to add)",
                                object_types[self.current_object_type_index].display_name()
                            );
                        }
                        KeyCode::Delete if event.state == ElementState::Pressed => {
                            // Delete dragged object or object under crosshair
                            let id_to_delete = self.dragging_object_id.take()
                                .or(self.ui_state.crosshair_target_id);
                            if let Some(id) = id_to_delete {
                                self.state.remove_object(id);
                                self.object_meshes.remove(&id);
                                self.ui_state.crosshair_target_id = None;
                                info!("Deleted object");
                            }
                        }
                        KeyCode::Escape if event.state == ElementState::Pressed => {
                            // First release pointer lock, then close panels
                            if self.ui_state.pointer_locked {
                                self.release_pointer_lock();
                            } else {
                                self.ui_state.close_customization();
                                self.ui_state.left_sidebar_open = false;
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        false
    }

    /// Request pointer lock (FPS mode)
    fn request_pointer_lock(&mut self) {
        // Set cursor grab mode to locked and hide cursor
        if let Err(e) = self.window.set_cursor_grab(winit::window::CursorGrabMode::Locked) {
            // Fall back to confined if locked isn't supported
            if let Err(e2) = self.window.set_cursor_grab(winit::window::CursorGrabMode::Confined) {
                log::warn!("Could not lock cursor: {:?} / {:?}", e, e2);
                return;
            }
        }
        self.window.set_cursor_visible(false);
        self.ui_state.pointer_locked = true;
        info!("Pointer locked - ESC to exit, mouse to look around");
    }

    /// Release pointer lock
    fn release_pointer_lock(&mut self) {
        let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
        self.window.set_cursor_visible(true);
        self.ui_state.pointer_locked = false;
        self.ui_state.crosshair_target_id = None;

        // Also end any drag in progress
        if let Some(id) = self.dragging_object_id.take() {
            let objects_clone: Vec<DeskObject> = self.state.objects.clone();
            if let Some(obj) = self.state.get_object_mut(id) {
                self.physics.end_drag(obj, &objects_clone);
                self.update_object_transform(id);
            }
        }
    }

    /// Find object at screen center (crosshair position)
    fn find_object_at_crosshair(&self) -> Option<u64> {
        // Raycast from screen center (0, 0 in NDC)
        let ndc_x = 0.0;
        let ndc_y = 0.0;

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

        best_id
    }

    /// Try to pick object at crosshair (for pointer lock mode)
    fn try_pick_object_crosshair(&mut self) {
        if let Some(id) = self.find_object_at_crosshair() {
            self.dragging_object_id = Some(id);
            if let Some(obj) = self.state.get_object_mut(id) {
                obj.is_dragging = true;
            }
        }
    }

    /// Find object at cursor position (without starting drag)
    fn find_object_at_cursor(&self) -> Option<u64> {
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

        best_id
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

    /// Update drag position based on crosshair (for pointer lock mode)
    fn update_drag_crosshair(&mut self) {
        // Raycast from screen center (0, 0 in NDC)
        let ndc_x = 0.0;
        let ndc_y = 0.0;

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

        // Top face: CCW winding when viewed from above (camera looking down)
        // v0=back-left, v1=back-right, v2=front-right, v3=front-left
        // CCW from above: 0->3->2 and 0->2->1 (reversed from CW)
        // Front face: CCW winding when viewed from front (camera in front)
        // v4=bottom-front-left, v5=bottom-front-right, v6=top-front-right, v7=top-front-left
        // CCW from front: 4->7->6 and 4->6->5 (reversed from CW)
        let indices: Vec<u16> = vec![0, 3, 2, 0, 2, 1, 4, 7, 6, 4, 6, 5];

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

        // Floor face: CCW winding when viewed from above
        // v0=back-left, v1=back-right, v2=front-right, v3=front-left
        // CCW from above: 0->3->2 and 0->2->1
        let indices: Vec<u16> = vec![0, 3, 2, 0, 2, 1];

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

/// Application wrapper for winit 0.30 ApplicationHandler
struct AppWrapper {
    app: Option<App>,
}

impl ApplicationHandler for AppWrapper {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_none() {
            let window_attrs = WindowAttributes::default()
                .with_title("Focus Desktop Simulator")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

            let window = Arc::new(
                event_loop
                    .create_window(window_attrs)
                    .expect("Failed to create window"),
            );

            self.app = Some(pollster::block_on(App::new(window)).expect("Failed to create app"));
            info!("Application initialized");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let Some(app) = &mut self.app else { return };

        let _egui_consumed = app.handle_event(&event);

        match event {
            WindowEvent::CloseRequested => {
                info!("Saving state and exiting...");
                let _ = app.save_state();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => app.resize(size),
            WindowEvent::RedrawRequested => {
                app.update();
                if let Err(e) = app.render() {
                    match e {
                        wgpu::SurfaceError::Lost => app.resize(app.size),
                        wgpu::SurfaceError::OutOfMemory => event_loop.exit(),
                        _ => log::error!("Render error: {:?}", e),
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: winit::event::DeviceId, event: winit::event::DeviceEvent) {
        let Some(app) = &mut self.app else { return };

        // Handle mouse motion for camera rotation in pointer lock mode
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            if app.ui_state.pointer_locked {
                // Rotate camera based on mouse delta
                app.camera.rotate(delta.0 as f32, delta.1 as f32);

                // If dragging an object in pointer lock mode, update its position based on crosshair
                if app.dragging_object_id.is_some() {
                    app.update_drag_crosshair();
                }

                // Update crosshair target
                app.ui_state.crosshair_target_id = app.find_object_at_crosshair();
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(app) = &self.app {
            app.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    info!("Starting Focus Desktop Simulator...");
    info!("Controls:");
    info!("  Click on scene - Enter FPS mode (crosshair interaction)");
    info!("  ESC - Exit FPS mode");
    info!("  Mouse movement (FPS mode) - Look around");
    info!("  Click (FPS mode) - Pick up object under crosshair");
    info!("  Click Menu button (top-left) - Open object palette");
    info!("  Right-click on object - Open customization panel");
    info!("  Right-click on empty space - Toggle palette");
    info!("  Scroll - Rotate object (under crosshair or while dragging)");
    info!("  Shift+Scroll - Scale object");
    info!("  Delete - Delete object under crosshair");
    info!("  T - Cycle through object types (keyboard shortcut)");
    info!("  A - Add selected object (keyboard shortcut)");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app_wrapper = AppWrapper { app: None };
    event_loop.run_app(&mut app_wrapper).expect("Event loop error");
}
