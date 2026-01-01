//! Mesh generation module
//!
//! Creates 3D meshes for each object type with proper geometry.

use crate::config::hex_to_rgb;
use crate::desk_object::ObjectType;
use std::f32::consts::PI;

/// Vertex data structure for 3D rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

impl Vertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x4,
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Mesh data containing vertices and indices
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

impl MeshData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Add a quad (two triangles) with vertices in counter-clockwise order
    pub fn add_quad(&mut self, v0: Vertex, v1: Vertex, v2: Vertex, v3: Vertex) {
        let base = self.vertices.len() as u16;
        self.vertices.extend_from_slice(&[v0, v1, v2, v3]);
        // Two triangles: 0-1-2 and 0-2-3
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// Add a triangle
    pub fn add_triangle(&mut self, v0: Vertex, v1: Vertex, v2: Vertex) {
        let base = self.vertices.len() as u16;
        self.vertices.extend_from_slice(&[v0, v1, v2]);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2]);
    }

    /// Merge another mesh into this one
    pub fn merge(&mut self, other: MeshData) {
        let base = self.vertices.len() as u16;
        self.vertices.extend(other.vertices);
        self.indices
            .extend(other.indices.iter().map(|i| i + base));
    }
}

/// Create a cylinder mesh (open at top for objects like mugs)
pub fn create_cylinder(
    radius: f32,
    height: f32,
    segments: u32,
    color: [f32; 4],
    y_offset: f32,
    closed_bottom: bool,
    closed_top: bool,
) -> MeshData {
    let mut mesh = MeshData::new();

    for i in 0..segments {
        let angle0 = (i as f32 / segments as f32) * 2.0 * PI;
        let angle1 = ((i + 1) as f32 / segments as f32) * 2.0 * PI;

        let (x0, z0) = (angle0.cos() * radius, angle0.sin() * radius);
        let (x1, z1) = (angle1.cos() * radius, angle1.sin() * radius);

        // Side face normal (pointing outward)
        let nx0 = angle0.cos();
        let nz0 = angle0.sin();
        let nx1 = angle1.cos();
        let nz1 = angle1.sin();

        // Side quad
        mesh.add_quad(
            Vertex {
                position: [x0, y_offset, z0],
                normal: [nx0, 0.0, nz0],
                color,
            },
            Vertex {
                position: [x1, y_offset, z1],
                normal: [nx1, 0.0, nz1],
                color,
            },
            Vertex {
                position: [x1, y_offset + height, z1],
                normal: [nx1, 0.0, nz1],
                color,
            },
            Vertex {
                position: [x0, y_offset + height, z0],
                normal: [nx0, 0.0, nz0],
                color,
            },
        );

        // Bottom cap
        if closed_bottom {
            mesh.add_triangle(
                Vertex {
                    position: [0.0, y_offset, 0.0],
                    normal: [0.0, -1.0, 0.0],
                    color,
                },
                Vertex {
                    position: [x1, y_offset, z1],
                    normal: [0.0, -1.0, 0.0],
                    color,
                },
                Vertex {
                    position: [x0, y_offset, z0],
                    normal: [0.0, -1.0, 0.0],
                    color,
                },
            );
        }

        // Top cap
        if closed_top {
            mesh.add_triangle(
                Vertex {
                    position: [0.0, y_offset + height, 0.0],
                    normal: [0.0, 1.0, 0.0],
                    color,
                },
                Vertex {
                    position: [x0, y_offset + height, z0],
                    normal: [0.0, 1.0, 0.0],
                    color,
                },
                Vertex {
                    position: [x1, y_offset + height, z1],
                    normal: [0.0, 1.0, 0.0],
                    color,
                },
            );
        }
    }

    mesh
}

/// Create a box mesh
pub fn create_box(
    width: f32,
    height: f32,
    depth: f32,
    color: [f32; 4],
    y_offset: f32,
) -> MeshData {
    let mut mesh = MeshData::new();
    let hw = width / 2.0;
    let hd = depth / 2.0;

    // Front face (+Z)
    let front_color = [color[0] * 0.9, color[1] * 0.9, color[2] * 0.9, color[3]];
    mesh.add_quad(
        Vertex {
            position: [-hw, y_offset, hd],
            normal: [0.0, 0.0, 1.0],
            color: front_color,
        },
        Vertex {
            position: [hw, y_offset, hd],
            normal: [0.0, 0.0, 1.0],
            color: front_color,
        },
        Vertex {
            position: [hw, y_offset + height, hd],
            normal: [0.0, 0.0, 1.0],
            color: front_color,
        },
        Vertex {
            position: [-hw, y_offset + height, hd],
            normal: [0.0, 0.0, 1.0],
            color: front_color,
        },
    );

    // Back face (-Z)
    let back_color = [color[0] * 0.8, color[1] * 0.8, color[2] * 0.8, color[3]];
    mesh.add_quad(
        Vertex {
            position: [hw, y_offset, -hd],
            normal: [0.0, 0.0, -1.0],
            color: back_color,
        },
        Vertex {
            position: [-hw, y_offset, -hd],
            normal: [0.0, 0.0, -1.0],
            color: back_color,
        },
        Vertex {
            position: [-hw, y_offset + height, -hd],
            normal: [0.0, 0.0, -1.0],
            color: back_color,
        },
        Vertex {
            position: [hw, y_offset + height, -hd],
            normal: [0.0, 0.0, -1.0],
            color: back_color,
        },
    );

    // Right face (+X)
    let right_color = [color[0] * 0.85, color[1] * 0.85, color[2] * 0.85, color[3]];
    mesh.add_quad(
        Vertex {
            position: [hw, y_offset, hd],
            normal: [1.0, 0.0, 0.0],
            color: right_color,
        },
        Vertex {
            position: [hw, y_offset, -hd],
            normal: [1.0, 0.0, 0.0],
            color: right_color,
        },
        Vertex {
            position: [hw, y_offset + height, -hd],
            normal: [1.0, 0.0, 0.0],
            color: right_color,
        },
        Vertex {
            position: [hw, y_offset + height, hd],
            normal: [1.0, 0.0, 0.0],
            color: right_color,
        },
    );

    // Left face (-X)
    let left_color = [color[0] * 0.75, color[1] * 0.75, color[2] * 0.75, color[3]];
    mesh.add_quad(
        Vertex {
            position: [-hw, y_offset, -hd],
            normal: [-1.0, 0.0, 0.0],
            color: left_color,
        },
        Vertex {
            position: [-hw, y_offset, hd],
            normal: [-1.0, 0.0, 0.0],
            color: left_color,
        },
        Vertex {
            position: [-hw, y_offset + height, hd],
            normal: [-1.0, 0.0, 0.0],
            color: left_color,
        },
        Vertex {
            position: [-hw, y_offset + height, -hd],
            normal: [-1.0, 0.0, 0.0],
            color: left_color,
        },
    );

    // Top face (+Y)
    mesh.add_quad(
        Vertex {
            position: [-hw, y_offset + height, hd],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        Vertex {
            position: [hw, y_offset + height, hd],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        Vertex {
            position: [hw, y_offset + height, -hd],
            normal: [0.0, 1.0, 0.0],
            color,
        },
        Vertex {
            position: [-hw, y_offset + height, -hd],
            normal: [0.0, 1.0, 0.0],
            color,
        },
    );

    // Bottom face (-Y)
    let bottom_color = [color[0] * 0.7, color[1] * 0.7, color[2] * 0.7, color[3]];
    mesh.add_quad(
        Vertex {
            position: [-hw, y_offset, -hd],
            normal: [0.0, -1.0, 0.0],
            color: bottom_color,
        },
        Vertex {
            position: [hw, y_offset, -hd],
            normal: [0.0, -1.0, 0.0],
            color: bottom_color,
        },
        Vertex {
            position: [hw, y_offset, hd],
            normal: [0.0, -1.0, 0.0],
            color: bottom_color,
        },
        Vertex {
            position: [-hw, y_offset, hd],
            normal: [0.0, -1.0, 0.0],
            color: bottom_color,
        },
    );

    mesh
}

/// Create a sphere mesh (for globe, decorative elements)
pub fn create_sphere(
    radius: f32,
    h_segments: u32,
    v_segments: u32,
    color: [f32; 4],
    y_offset: f32,
) -> MeshData {
    let mut mesh = MeshData::new();

    for i in 0..h_segments {
        for j in 0..v_segments {
            let theta0 = (i as f32 / h_segments as f32) * 2.0 * PI;
            let theta1 = ((i + 1) as f32 / h_segments as f32) * 2.0 * PI;
            let phi0 = (j as f32 / v_segments as f32) * PI;
            let phi1 = ((j + 1) as f32 / v_segments as f32) * PI;

            let x00 = radius * phi0.sin() * theta0.cos();
            let y00 = radius * phi0.cos() + y_offset;
            let z00 = radius * phi0.sin() * theta0.sin();

            let x10 = radius * phi0.sin() * theta1.cos();
            let y10 = radius * phi0.cos() + y_offset;
            let z10 = radius * phi0.sin() * theta1.sin();

            let x01 = radius * phi1.sin() * theta0.cos();
            let y01 = radius * phi1.cos() + y_offset;
            let z01 = radius * phi1.sin() * theta0.sin();

            let x11 = radius * phi1.sin() * theta1.cos();
            let y11 = radius * phi1.cos() + y_offset;
            let z11 = radius * phi1.sin() * theta1.sin();

            // Normals point outward from center
            let n00 = [phi0.sin() * theta0.cos(), phi0.cos(), phi0.sin() * theta0.sin()];
            let n10 = [phi0.sin() * theta1.cos(), phi0.cos(), phi0.sin() * theta1.sin()];
            let n01 = [phi1.sin() * theta0.cos(), phi1.cos(), phi1.sin() * theta0.sin()];
            let n11 = [phi1.sin() * theta1.cos(), phi1.cos(), phi1.sin() * theta1.sin()];

            mesh.add_quad(
                Vertex {
                    position: [x00, y00, z00],
                    normal: n00,
                    color,
                },
                Vertex {
                    position: [x10, y10, z10],
                    normal: n10,
                    color,
                },
                Vertex {
                    position: [x11, y11, z11],
                    normal: n11,
                    color,
                },
                Vertex {
                    position: [x01, y01, z01],
                    normal: n01,
                    color,
                },
            );
        }
    }

    mesh
}

/// Create a clock mesh with frame, face, and markers
pub fn create_clock(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let frame_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let face_color = [ar, ag, ab, 1.0];

    // Clock frame (cylinder, thicker)
    mesh.merge(create_cylinder(0.25, 0.08, 24, frame_color, 0.32, true, true));

    // Clock face (flat disc)
    let face_mesh = create_cylinder(0.22, 0.01, 24, face_color, 0.40, true, true);
    mesh.merge(face_mesh);

    // Hour markers (small rectangles around the face)
    let marker_color = [r * 0.3, g * 0.3, b * 0.3, 1.0];
    for i in 0..12 {
        let angle = (i as f32 / 12.0) * 2.0 * PI - PI / 2.0;
        let cx = angle.cos() * 0.18;
        let cz = angle.sin() * 0.18;

        // Small box marker
        let mut marker = create_box(0.02, 0.005, 0.04, marker_color, 0.41);
        // Translate marker to position
        for v in &mut marker.vertices {
            let x = v.position[0];
            let z = v.position[2];
            v.position[0] = x * angle.cos() - z * angle.sin() + cx;
            v.position[2] = x * angle.sin() + z * angle.cos() + cz;
        }
        mesh.merge(marker);
    }

    mesh
}

/// Create a lamp mesh with base, arm, and head
pub fn create_lamp(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let lamp_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let glow_color = [ar, ag, ab, 1.0];

    // Base (cylinder)
    mesh.merge(create_cylinder(0.15, 0.04, 16, lamp_color, 0.0, true, true));

    // Stem (thin cylinder)
    mesh.merge(create_cylinder(0.02, 0.5, 8, lamp_color, 0.04, true, true));

    // Arm (angled box)
    let mut arm = create_box(0.02, 0.3, 0.02, lamp_color, 0.0);
    // Rotate arm 45 degrees
    for v in &mut arm.vertices {
        let y = v.position[1];
        let z = v.position[2];
        v.position[1] = y * 0.707 - z * 0.707 + 0.54;
        v.position[2] = y * 0.707 + z * 0.707;
    }
    mesh.merge(arm);

    // Lamp head (cone-like shape using cylinder with different radii)
    let head_y = 0.72;
    mesh.merge(create_cylinder(0.12, 0.08, 12, lamp_color, head_y, true, false));

    // Inner glow (smaller cylinder inside head)
    mesh.merge(create_cylinder(0.08, 0.02, 12, glow_color, head_y + 0.02, true, true));

    mesh
}

/// Create a plant mesh with pot, soil, and leaves
pub fn create_plant(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let pot_color = [ar, ag, ab, 1.0];
    let (r, g, b) = hex_to_rgb(main_color);
    let leaf_color = [r, g, b, 1.0];
    let soil_color = [0.25, 0.15, 0.1, 1.0];

    // Pot (tapered cylinder)
    mesh.merge(create_cylinder(0.12, 0.15, 12, pot_color, 0.0, true, false));
    mesh.merge(create_cylinder(0.10, 0.02, 12, pot_color, 0.15, false, false));

    // Soil (dark disc at top of pot)
    mesh.merge(create_cylinder(0.095, 0.02, 12, soil_color, 0.15, true, true));

    // Simple leaves (small spheres)
    let leaf_positions = [
        (0.0, 0.28, 0.0),
        (0.06, 0.24, 0.04),
        (-0.05, 0.25, 0.05),
        (0.04, 0.22, -0.05),
        (-0.04, 0.23, -0.04),
    ];

    for (x, y, z) in leaf_positions {
        let mut leaf = create_sphere(0.06, 8, 6, leaf_color, 0.0);
        for v in &mut leaf.vertices {
            v.position[0] += x;
            v.position[1] += y;
            v.position[2] += z;
        }
        mesh.merge(leaf);
    }

    mesh
}

/// Create a coffee mug mesh
pub fn create_coffee(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let mug_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let liquid_color = [ar, ag, ab, 1.0];

    // Mug body (open cylinder)
    mesh.merge(create_cylinder(0.08, 0.15, 16, mug_color, 0.0, true, false));

    // Liquid surface
    mesh.merge(create_cylinder(0.065, 0.01, 16, liquid_color, 0.12, true, true));

    // Handle (simplified as a small box on the side)
    let mut handle = create_box(0.03, 0.08, 0.02, mug_color, 0.04);
    for v in &mut handle.vertices {
        v.position[0] += 0.10;
    }
    mesh.merge(handle);

    mesh
}

/// Create a laptop mesh with base and screen
pub fn create_laptop(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let body_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let screen_color = [ar, ag, ab, 1.0];

    // Base (keyboard area)
    mesh.merge(create_box(0.4, 0.02, 0.28, body_color, 0.0));

    // Screen (angled)
    let mut screen = create_box(0.38, 0.25, 0.01, body_color, 0.0);
    // Rotate screen to be angled
    for v in &mut screen.vertices {
        let y = v.position[1];
        let z = v.position[2];
        // Rotate around X axis by about 70 degrees (0.94 rad)
        v.position[1] = y * 0.34 + z * 0.94 + 0.02;
        v.position[2] = -y * 0.94 + z * 0.34 - 0.14;
    }
    mesh.merge(screen);

    // Screen display (glowing part)
    let mut display = create_box(0.34, 0.20, 0.005, screen_color, 0.0);
    for v in &mut display.vertices {
        let y = v.position[1];
        let z = v.position[2];
        v.position[1] = y * 0.34 + z * 0.94 + 0.045;
        v.position[2] = -y * 0.94 + z * 0.34 - 0.13;
    }
    mesh.merge(display);

    mesh
}

/// Create a notebook mesh
pub fn create_notebook(main_color: u32, _accent_color: u32) -> MeshData {
    let (r, g, b) = hex_to_rgb(main_color);
    let color = [r, g, b, 1.0];

    // Simple flat box
    create_box(0.25, 0.03, 0.35, color, 0.0)
}

/// Create a pen holder mesh
pub fn create_pen_holder(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let holder_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let pen_color = [ar, ag, ab, 1.0];

    // Holder cup
    mesh.merge(create_cylinder(0.08, 0.15, 12, holder_color, 0.0, true, false));

    // A few pens sticking out
    for i in 0..3 {
        let angle = (i as f32 / 3.0) * 2.0 * PI + 0.3;
        let offset_x = angle.cos() * 0.03;
        let offset_z = angle.sin() * 0.03;
        let mut pen = create_cylinder(0.008, 0.2, 6, pen_color, 0.1, true, true);
        for v in &mut pen.vertices {
            v.position[0] += offset_x;
            v.position[2] += offset_z;
        }
        mesh.merge(pen);
    }

    mesh
}

/// Create a books mesh (stack of books)
pub fn create_books(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let book1_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let book2_color = [ar, ag, ab, 1.0];

    // Stack of 3 books
    mesh.merge(create_box(0.22, 0.035, 0.3, book1_color, 0.0));
    mesh.merge(create_box(0.24, 0.04, 0.28, book2_color, 0.035));
    mesh.merge(create_box(0.2, 0.03, 0.32, book1_color, 0.075));

    mesh
}

/// Create a photo frame mesh
pub fn create_photo_frame(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let frame_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let photo_color = [ar, ag, ab, 1.0];

    // Frame back
    mesh.merge(create_box(0.2, 0.25, 0.02, frame_color, 0.0));

    // Photo inside (slightly smaller, offset forward)
    let mut photo = create_box(0.16, 0.21, 0.005, photo_color, 0.02);
    for v in &mut photo.vertices {
        v.position[2] += 0.01;
    }
    mesh.merge(photo);

    // Stand (small triangle-ish support at back)
    let mut stand = create_box(0.02, 0.15, 0.08, frame_color, 0.0);
    for v in &mut stand.vertices {
        v.position[2] -= 0.05;
    }
    mesh.merge(stand);

    mesh
}

/// Create a globe mesh
pub fn create_globe(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let globe_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let stand_color = [ar, ag, ab, 1.0];

    // Stand base
    mesh.merge(create_cylinder(0.1, 0.02, 12, stand_color, 0.0, true, true));

    // Stand pole
    mesh.merge(create_cylinder(0.015, 0.15, 8, stand_color, 0.02, true, true));

    // Globe sphere
    mesh.merge(create_sphere(0.12, 16, 12, globe_color, 0.25));

    mesh
}

/// Create a trophy mesh
pub fn create_trophy(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let trophy_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let base_color = [ar, ag, ab, 1.0];

    // Base
    mesh.merge(create_box(0.12, 0.04, 0.12, base_color, 0.0));

    // Stem
    mesh.merge(create_cylinder(0.02, 0.1, 8, trophy_color, 0.04, true, true));

    // Cup (wider cylinder at top)
    mesh.merge(create_cylinder(0.08, 0.12, 12, trophy_color, 0.14, true, false));

    // Handles (simplified as small boxes on sides)
    let mut handle1 = create_box(0.04, 0.06, 0.015, trophy_color, 0.16);
    for v in &mut handle1.vertices {
        v.position[0] += 0.1;
    }
    mesh.merge(handle1);

    let mut handle2 = create_box(0.04, 0.06, 0.015, trophy_color, 0.16);
    for v in &mut handle2.vertices {
        v.position[0] -= 0.1;
    }
    mesh.merge(handle2);

    mesh
}

/// Create an hourglass mesh
pub fn create_hourglass(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let glass_color = [r, g, b, 0.8]; // Slightly transparent
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let frame_color = [ar, ag, ab, 1.0];

    // Top and bottom frames
    mesh.merge(create_box(0.1, 0.02, 0.1, frame_color, 0.0));
    mesh.merge(create_box(0.1, 0.02, 0.1, frame_color, 0.28));

    // Glass body (two cylinders meeting at center)
    mesh.merge(create_cylinder(0.06, 0.12, 12, glass_color, 0.02, true, false));
    mesh.merge(create_cylinder(0.06, 0.12, 12, glass_color, 0.16, false, true));

    // Center narrow part
    mesh.merge(create_cylinder(0.015, 0.04, 8, glass_color, 0.12, true, true));

    // Sand (simplified as small sphere in bottom)
    let sand_color = [0.9, 0.8, 0.5, 1.0];
    mesh.merge(create_sphere(0.04, 8, 6, sand_color, 0.06));

    mesh
}

/// Create a metronome mesh
pub fn create_metronome(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let body_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let arm_color = [ar, ag, ab, 1.0];

    // Body (tapered box)
    mesh.merge(create_box(0.12, 0.25, 0.1, body_color, 0.0));

    // Arm (thin box in center)
    mesh.merge(create_box(0.01, 0.2, 0.01, arm_color, 0.05));

    mesh
}

/// Create a paper mesh (flat sheet)
pub fn create_paper(main_color: u32, _accent_color: u32) -> MeshData {
    let (r, g, b) = hex_to_rgb(main_color);
    let color = [r, g, b, 1.0];

    create_box(0.21, 0.002, 0.297, color, 0.0) // A4 paper proportions scaled down
}

/// Create a magazine mesh
pub fn create_magazine(main_color: u32, accent_color: u32) -> MeshData {
    let mut mesh = MeshData::new();

    let (r, g, b) = hex_to_rgb(main_color);
    let cover_color = [r, g, b, 1.0];
    let (ar, ag, ab) = hex_to_rgb(accent_color);
    let title_color = [ar, ag, ab, 1.0];

    // Magazine body
    mesh.merge(create_box(0.22, 0.01, 0.3, cover_color, 0.0));

    // Title stripe
    let mut title = create_box(0.18, 0.002, 0.04, title_color, 0.01);
    for v in &mut title.vertices {
        v.position[2] -= 0.08;
    }
    mesh.merge(title);

    mesh
}

/// Generate mesh for a given object type
pub fn generate_object_mesh(object_type: ObjectType, main_color: u32, accent_color: u32) -> MeshData {
    match object_type {
        ObjectType::Clock => create_clock(main_color, accent_color),
        ObjectType::Lamp => create_lamp(main_color, accent_color),
        ObjectType::Plant => create_plant(main_color, accent_color),
        ObjectType::Coffee => create_coffee(main_color, accent_color),
        ObjectType::Laptop => create_laptop(main_color, accent_color),
        ObjectType::Notebook => create_notebook(main_color, accent_color),
        ObjectType::PenHolder => create_pen_holder(main_color, accent_color),
        ObjectType::Books => create_books(main_color, accent_color),
        ObjectType::PhotoFrame => create_photo_frame(main_color, accent_color),
        ObjectType::Globe => create_globe(main_color, accent_color),
        ObjectType::Trophy => create_trophy(main_color, accent_color),
        ObjectType::Hourglass => create_hourglass(main_color, accent_color),
        ObjectType::Metronome => create_metronome(main_color, accent_color),
        ObjectType::Paper => create_paper(main_color, accent_color),
        ObjectType::Magazine => create_magazine(main_color, accent_color),
    }
}
