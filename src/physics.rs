//! Physics module for object interactions
//!
//! Handles collision detection, object dropping, and stacking.

use glam::Vec3;
use crate::config::CONFIG;
use crate::desk_object::DeskObject;

/// Physics state for an object
#[derive(Debug, Clone, Default)]
pub struct ObjectPhysicsState {
    /// Velocity vector (x, z movement on desk surface)
    pub velocity: Vec3,
    /// Angular velocity (rotation around Y axis)
    pub angular_velocity: f32,
    /// Tilt angles (rotation around X and Z axes)
    pub tilt: Vec3,
    /// Tilt velocity
    pub tilt_velocity: Vec3,
}

/// Physics engine for the desk simulation
pub struct PhysicsEngine {
    /// Global collision radius multiplier
    pub collision_radius_multiplier: f32,
    /// Global collision height multiplier
    pub collision_height_multiplier: f32,
    /// Friction coefficient
    pub friction: f32,
    /// Bounce factor for collisions
    pub bounce_factor: f32,
    /// Gravity constant
    pub gravity: f32,
    /// Desk bounds (min x, max x, min z, max z)
    pub desk_bounds: (f32, f32, f32, f32),
    /// Desk surface Y position
    pub desk_surface_y: f32,
}

impl Default for PhysicsEngine {
    fn default() -> Self {
        let config = &CONFIG;
        let half_width = config.desk.width / 2.0;
        let half_depth = config.desk.depth / 2.0;

        Self {
            collision_radius_multiplier: 1.0,
            collision_height_multiplier: 1.0,
            friction: config.physics.friction,
            bounce_factor: config.physics.bounce_factor,
            gravity: config.physics.gravity,
            desk_bounds: (-half_width, half_width, -half_depth, half_depth),
            desk_surface_y: config.desk.height,
        }
    }
}

impl PhysicsEngine {
    /// Create a new physics engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the Y position of the desk surface
    pub fn desk_surface_y(&self) -> f32 {
        self.desk_surface_y
    }

    /// Check if a position is within desk bounds
    pub fn is_on_desk(&self, position: Vec3) -> bool {
        position.x >= self.desk_bounds.0
            && position.x <= self.desk_bounds.1
            && position.z >= self.desk_bounds.2
            && position.z <= self.desk_bounds.3
    }

    /// Clamp a position to desk bounds
    pub fn clamp_to_desk(&self, position: Vec3, radius: f32) -> Vec3 {
        Vec3::new(
            position.x.clamp(self.desk_bounds.0 + radius, self.desk_bounds.1 - radius),
            position.y,
            position.z.clamp(self.desk_bounds.2 + radius, self.desk_bounds.3 - radius),
        )
    }

    /// Check collision between two objects
    pub fn check_collision(&self, obj1: &DeskObject, obj2: &DeskObject) -> bool {
        if obj1.id == obj2.id {
            return false;
        }

        let r1 = obj1.collision_radius() * self.collision_radius_multiplier;
        let r2 = obj2.collision_radius() * self.collision_radius_multiplier;
        let min_dist = r1 + r2;

        let dx = obj1.position.x - obj2.position.x;
        let dz = obj1.position.z - obj2.position.z;
        let dist_sq = dx * dx + dz * dz;

        dist_sq < min_dist * min_dist
    }

    /// Find the best position to place an object (avoiding collisions)
    pub fn find_valid_position(
        &self,
        target: Vec3,
        object: &DeskObject,
        other_objects: &[DeskObject],
    ) -> Vec3 {
        let radius = object.collision_radius() * self.collision_radius_multiplier;
        let mut position = self.clamp_to_desk(target, radius);

        // Check for collisions and push away
        for other in other_objects {
            if other.id == object.id {
                continue;
            }

            let other_radius = other.collision_radius() * self.collision_radius_multiplier;
            let min_dist = radius + other_radius;

            let dx = position.x - other.position.x;
            let dz = position.z - other.position.z;
            let dist = (dx * dx + dz * dz).sqrt();

            if dist < min_dist && dist > 0.001 {
                // Push away from collision
                let push_dist = min_dist - dist + 0.05;
                let push_x = (dx / dist) * push_dist;
                let push_z = (dz / dist) * push_dist;
                position.x += push_x;
                position.z += push_z;
            }
        }

        // Re-clamp to desk bounds
        self.clamp_to_desk(position, radius)
    }

    /// Calculate the resting Y position for an object (considering stacking)
    pub fn calculate_resting_y(
        &self,
        object: &DeskObject,
        other_objects: &[DeskObject],
    ) -> f32 {
        let radius = object.collision_radius() * self.collision_radius_multiplier;
        let physics = object.object_type.physics();
        let base_y = self.desk_surface_y + physics.base_offset * object.scale;

        let mut highest_y = base_y;

        // Check for objects we might be stacking on
        for other in other_objects {
            if other.id == object.id {
                continue;
            }

            let other_physics = other.object_type.physics();

            // Skip if object doesn't allow stacking on top
            if other_physics.no_stacking_on_top {
                continue;
            }

            let other_radius = other.collision_radius() * self.collision_radius_multiplier;
            let combined_radius = radius + other_radius;

            let dx = object.position.x - other.position.x;
            let dz = object.position.z - other.position.z;
            let dist_sq = dx * dx + dz * dz;

            // Check if we're above this object
            if dist_sq < combined_radius * combined_radius * 0.5 {
                let other_top = other.position.y + other.collision_height() * self.collision_height_multiplier;
                let stack_y = other_top + physics.base_offset * object.scale;

                if stack_y > highest_y {
                    highest_y = stack_y;
                }
            }
        }

        highest_y
    }

    /// Update object position during dragging
    pub fn update_dragging(
        &self,
        object: &mut DeskObject,
        target_xz: Vec3,
        lift_height: f32,
    ) {
        let radius = object.collision_radius();
        let target = self.clamp_to_desk(target_xz, radius);

        object.position.x = target.x;
        object.position.z = target.z;
        object.position.y = object.original_y + lift_height;
        object.is_dragging = true;
    }

    /// Update object position when dropping (smooth animation)
    pub fn update_dropping(
        &self,
        object: &mut DeskObject,
        _other_objects: &[DeskObject],
        drop_speed: f32,
    ) -> bool {
        if !object.is_dragging && (object.position.y - object.target_y).abs() > 0.001 {
            // Smoothly move toward target Y
            let diff = object.target_y - object.position.y;
            object.position.y += diff * drop_speed;

            if (object.position.y - object.target_y).abs() < 0.01 {
                object.position.y = object.target_y;
            }

            return true; // Still animating
        }

        false
    }

    /// End drag operation and calculate final position
    pub fn end_drag(&self, object: &mut DeskObject, other_objects: &[DeskObject]) {
        object.is_dragging = false;
        object.target_y = self.calculate_resting_y(object, other_objects);
        object.original_y = object.target_y;
    }
}

/// Ray-plane intersection for mouse picking
pub fn ray_plane_intersection(
    ray_origin: Vec3,
    ray_direction: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denom = plane_normal.dot(ray_direction);

    if denom.abs() < 0.0001 {
        return None; // Ray is parallel to plane
    }

    let t = (plane_point - ray_origin).dot(plane_normal) / denom;

    if t < 0.0 {
        return None; // Intersection is behind ray origin
    }

    Some(ray_origin + ray_direction * t)
}
