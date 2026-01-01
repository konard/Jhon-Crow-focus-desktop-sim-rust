// Focus Desktop Simulator - Main Shader
// WGSL shader for 3D rendering with dynamic lighting and model transforms

// Maximum number of point lights (lamps)
const MAX_LIGHTS: u32 = 8u;

// Camera uniform buffer
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
}

// Scene lighting uniform buffer
struct LightingUniform {
    // Array of point light positions (xyz) and intensity (w)
    // w = 0.0 means light is off, w > 0 means on with that intensity
    point_lights: array<vec4<f32>, 8>,
    // Number of active lights
    num_lights: u32,
    // Room darkness level (0.0 = bright, 1.0 = very dark)
    room_darkness: f32,
    // Padding for alignment
    _padding: vec2<f32>,
}

// Model uniform buffer for per-object transforms
struct ModelUniform {
    model: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> lighting: LightingUniform;

@group(1) @binding(0)
var<uniform> model: ModelUniform;

// Vertex input
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

// Vertex shader
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Transform position by model matrix first, then by view-projection
    let world_pos = model.model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.view_proj * world_pos;

    // Pass through world position
    out.world_position = world_pos.xyz;

    // Transform normal by model matrix (ignoring translation)
    // For proper normal transformation we should use the inverse transpose,
    // but for uniform scaling this is equivalent
    let normal_transform = mat3x3<f32>(
        model.model[0].xyz,
        model.model[1].xyz,
        model.model[2].xyz
    );
    out.world_normal = normalize(normal_transform * in.normal);

    out.color = in.color;

    return out;
}

// Fragment shader with dynamic lighting from lamps
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);

    // Base ambient light - very dim when room is dark
    // When room_darkness = 1.0, ambient is very low (0.03)
    // When room_darkness = 0.0, ambient is normal (0.25)
    let base_ambient = 0.25;
    let dark_ambient = 0.03;
    let ambient_level = mix(base_ambient, dark_ambient, lighting.room_darkness);
    var total_light = vec3<f32>(ambient_level, ambient_level, ambient_level * 1.2);

    // Weak directional light from above (simulating very dim ceiling light)
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.2));
    let directional_intensity = mix(0.15, 0.02, lighting.room_darkness);
    let directional = max(dot(normal, light_dir), 0.0) * directional_intensity;
    total_light = total_light + vec3<f32>(directional, directional, directional * 0.9);

    // Calculate point light contributions from lamps
    for (var i: u32 = 0u; i < lighting.num_lights; i = i + 1u) {
        let light_pos = lighting.point_lights[i].xyz;
        let light_intensity = lighting.point_lights[i].w;

        if (light_intensity > 0.0) {
            // Direction from fragment to light
            let to_light = light_pos - in.world_position;
            let distance = length(to_light);
            let light_direction = normalize(to_light);

            // Attenuation (quadratic falloff with distance)
            let attenuation = light_intensity / (1.0 + 0.3 * distance + 0.1 * distance * distance);

            // Diffuse lighting
            let n_dot_l = max(dot(normal, light_direction), 0.0);

            // Warm lamp light color (yellowish)
            let lamp_color = vec3<f32>(1.0, 0.9, 0.7);

            // Add point light contribution
            total_light = total_light + lamp_color * n_dot_l * attenuation;

            // Add some ambient contribution from the lamp to nearby surfaces
            let ambient_boost = attenuation * 0.3;
            total_light = total_light + lamp_color * ambient_boost;
        }
    }

    // Apply lighting to base color
    let lit_color = in.color.rgb * total_light;

    // Simple fog effect based on distance from camera
    let dist = length(in.world_position - camera.position.xyz);
    let fog_factor = 1.0 - clamp((dist - 10.0) / 40.0, 0.0, 0.6);

    // Background/fog color - darker when room is dark
    let base_fog = vec3<f32>(0.1, 0.1, 0.18);
    let dark_fog = vec3<f32>(0.02, 0.02, 0.04);
    let fog_color = mix(base_fog, dark_fog, lighting.room_darkness);

    let final_color = mix(fog_color, lit_color, fog_factor);

    return vec4<f32>(final_color, in.color.a);
}
