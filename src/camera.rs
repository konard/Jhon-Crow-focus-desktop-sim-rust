//! Camera module for 3D rendering
//!
//! Implements a first-person style camera with yaw/pitch controls.

use glam::{Mat4, Vec3};
use crate::config::CONFIG;

/// Camera state and controls
pub struct Camera {
    /// Current position
    pub position: Vec3,
    /// Horizontal rotation (yaw) in radians
    pub yaw: f32,
    /// Vertical rotation (pitch) in radians
    pub pitch: f32,
    /// Field of view in radians
    pub fov: f32,
    /// Aspect ratio (width / height)
    pub aspect: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,
    /// Mouse sensitivity
    pub sensitivity: f32,
    /// Minimum pitch angle (looking down limit)
    pub min_pitch: f32,
    /// Maximum pitch angle (looking up limit)
    pub max_pitch: f32,
    /// Minimum yaw angle
    pub min_yaw: f32,
    /// Maximum yaw angle
    pub max_yaw: f32,
    /// Default yaw (for calculating limits)
    default_yaw: f32,
}

impl Camera {
    /// Create a new camera with default settings
    pub fn new(aspect: f32) -> Self {
        let config = &CONFIG.camera;

        // Calculate initial yaw and pitch from look_at target
        let (yaw, pitch) = Self::calculate_angles_from_look_at(
            config.position,
            config.look_at,
        );

        Self {
            position: config.position,
            yaw,
            pitch,
            fov: config.fov.to_radians(),
            aspect,
            near: config.near,
            far: config.far,
            sensitivity: 0.002,
            min_pitch: -1.55,  // ~89 degrees down
            max_pitch: 0.42,   // ~24 degrees up
            min_yaw: yaw - 1.40,  // ~80 degrees left
            max_yaw: yaw + 1.40,  // ~80 degrees right
            default_yaw: yaw,
        }
    }

    /// Calculate yaw and pitch angles from camera position and look-at point
    fn calculate_angles_from_look_at(camera_pos: Vec3, look_at: Vec3) -> (f32, f32) {
        let direction = look_at - camera_pos;
        let _horizontal_len = (direction.x * direction.x + direction.z * direction.z).sqrt();

        // Yaw: horizontal angle from positive Z axis
        let yaw = direction.x.atan2(direction.z);

        // Pitch: vertical angle
        let pitch = (direction.y / direction.length()).asin();

        (yaw, pitch)
    }

    /// Get the view matrix
    pub fn view_matrix(&self) -> Mat4 {
        // Calculate look direction from yaw and pitch
        let direction = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        );

        let target = self.position + direction;
        let up = Vec3::Y;

        Mat4::look_at_rh(self.position, target, up)
    }

    /// Get the projection matrix
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }

    /// Get the combined view-projection matrix
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Update aspect ratio (on window resize)
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    /// Rotate camera based on mouse movement (pointer lock mode)
    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        // Update yaw (horizontal) - clamped to limits
        // Moving mouse right (positive delta_x) should rotate camera right (decrease yaw)
        self.yaw = (self.yaw - delta_x * self.sensitivity)
            .clamp(self.min_yaw, self.max_yaw);

        // Update pitch (vertical) - clamped to limits
        self.pitch = (self.pitch - delta_y * self.sensitivity)
            .clamp(self.min_pitch, self.max_pitch);
    }

    /// Get the forward direction vector (ignoring pitch)
    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }

    /// Get the right direction vector
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    /// Get the look direction including pitch
    pub fn look_direction(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        ).normalize()
    }

    /// Reset camera to default position and orientation
    pub fn reset(&mut self) {
        let config = &CONFIG.camera;
        self.position = config.position;
        self.yaw = self.default_yaw;
        let (_, pitch) = Self::calculate_angles_from_look_at(config.position, config.look_at);
        self.pitch = pitch;
    }
}

/// Uniform buffer data for camera (GPU-compatible)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// View-projection matrix
    pub view_proj: [[f32; 4]; 4],
    /// Camera position in world space
    pub position: [f32; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            position: [0.0; 4],
        }
    }

    pub fn update(&mut self, camera: &Camera) {
        self.view_proj = camera.view_projection_matrix().to_cols_array_2d();
        self.position = [
            camera.position.x,
            camera.position.y,
            camera.position.z,
            1.0,
        ];
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}
