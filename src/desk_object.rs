//! Desk object module
//!
//! Defines the various objects that can be placed on the desk.

use glam::{Vec3, Quat};
use serde::{Deserialize, Serialize};

/// Type of desk object
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ObjectType {
    #[default]
    Clock,
    Lamp,
    Plant,
    Coffee,
    Laptop,
    Notebook,
    PenHolder,
    Books,
    PhotoFrame,
    Globe,
    Trophy,
    Hourglass,
    Metronome,
    Paper,
    Magazine,
    MusicPlayer,
    Pen,
}

/// Drink types for the coffee mug
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DrinkType {
    #[default]
    Coffee,
    Tea,
    HotChocolate,
    Water,
    Milk,
}

impl DrinkType {
    /// Get the color for this drink type (RGB hex)
    pub fn color(&self) -> u32 {
        match self {
            DrinkType::Coffee => 0x3d2314,      // Dark brown
            DrinkType::Tea => 0xc68b59,         // Amber/tea color
            DrinkType::HotChocolate => 0x4a3728, // Chocolate brown
            DrinkType::Water => 0x87ceeb,       // Light blue
            DrinkType::Milk => 0xfdfff5,        // Off-white
        }
    }

    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            DrinkType::Coffee => "Coffee",
            DrinkType::Tea => "Tea",
            DrinkType::HotChocolate => "Hot Chocolate",
            DrinkType::Water => "Water",
            DrinkType::Milk => "Milk",
        }
    }

    /// Get all drink types
    pub fn all() -> &'static [DrinkType] {
        &[
            DrinkType::Coffee,
            DrinkType::Tea,
            DrinkType::HotChocolate,
            DrinkType::Water,
            DrinkType::Milk,
        ]
    }
}

/// Object-specific interactive state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ObjectState {
    /// Lamp: Whether the light is on
    #[serde(default)]
    pub lamp_on: bool,

    /// Globe: Whether the globe is rotating
    #[serde(default)]
    pub globe_rotating: bool,

    /// Globe: Current rotation angle in radians
    #[serde(default)]
    pub globe_angle: f32,

    /// Hourglass: Whether currently flipping
    #[serde(default)]
    pub hourglass_flipping: bool,

    /// Hourglass: Flip animation progress (0.0 to 1.0)
    #[serde(default)]
    pub hourglass_flip_progress: f32,

    /// Photo Frame: Path to the photo file (optional)
    #[serde(default)]
    pub photo_path: Option<String>,

    /// Music Player: Whether playing
    #[serde(default)]
    pub music_playing: bool,

    /// Music Player: Path to music folder/file
    #[serde(default)]
    pub music_path: Option<String>,

    /// Music Player: Volume (0.0 to 1.0)
    #[serde(default = "default_volume")]
    pub music_volume: f32,

    /// Metronome: Whether running
    #[serde(default)]
    pub metronome_running: bool,

    /// Metronome: Beats per minute
    #[serde(default = "default_bpm")]
    pub metronome_bpm: u32,

    /// Coffee Mug: Drink type
    #[serde(default)]
    pub drink_type: DrinkType,

    /// Coffee Mug: Fill level (0.0 to 1.0)
    #[serde(default = "default_fill_level")]
    pub fill_level: f32,

    /// Coffee Mug: Whether the drink is hot (shows steam)
    #[serde(default)]
    pub is_hot: bool,

    /// Clock: Current hour angle (radians, calculated from real time)
    #[serde(skip)]
    pub clock_hour_angle: f32,

    /// Clock: Current minute angle (radians, calculated from real time)
    #[serde(skip)]
    pub clock_minute_angle: f32,

    /// Clock: Current second angle (radians, calculated from real time)
    #[serde(skip)]
    pub clock_second_angle: f32,
}

fn default_fill_level() -> f32 {
    0.8
}

fn default_volume() -> f32 {
    0.5
}

fn default_bpm() -> u32 {
    120
}

impl ObjectType {
    /// Get display name for the object type
    pub fn display_name(&self) -> &'static str {
        match self {
            ObjectType::Clock => "Clock",
            ObjectType::Lamp => "Desk Lamp",
            ObjectType::Plant => "Potted Plant",
            ObjectType::Coffee => "Coffee Mug",
            ObjectType::Laptop => "Laptop",
            ObjectType::Notebook => "Notebook",
            ObjectType::PenHolder => "Pen Holder",
            ObjectType::Books => "Books",
            ObjectType::PhotoFrame => "Photo Frame",
            ObjectType::Globe => "Globe",
            ObjectType::Trophy => "Trophy",
            ObjectType::Hourglass => "Hourglass",
            ObjectType::Metronome => "Metronome",
            ObjectType::Paper => "Paper",
            ObjectType::Magazine => "Magazine",
            ObjectType::MusicPlayer => "Music Player",
            ObjectType::Pen => "Pen",
        }
    }

    /// Get emoji icon for the object type
    pub fn icon(&self) -> &'static str {
        match self {
            ObjectType::Clock => "\u{1F550}", // Clock emoji
            ObjectType::Lamp => "\u{1F4A1}", // Lightbulb
            ObjectType::Plant => "\u{1FAB4}", // Potted plant
            ObjectType::Coffee => "\u{2615}", // Coffee
            ObjectType::Laptop => "\u{1F4BB}", // Laptop
            ObjectType::Notebook => "\u{1F4D3}", // Notebook
            ObjectType::PenHolder => "\u{270F}", // Pencil
            ObjectType::Books => "\u{1F4DA}", // Books
            ObjectType::PhotoFrame => "\u{1F5BC}", // Frame
            ObjectType::Globe => "\u{1F30D}", // Globe
            ObjectType::Trophy => "\u{1F3C6}", // Trophy
            ObjectType::Hourglass => "\u{23F3}", // Hourglass
            ObjectType::Metronome => "\u{1F3B5}", // Musical note
            ObjectType::Paper => "\u{1F4C4}", // Page
            ObjectType::Magazine => "\u{1F4F0}", // Newspaper
            ObjectType::MusicPlayer => "\u{1F3B6}", // Musical notes
            ObjectType::Pen => "\u{1F58A}", // Pen
        }
    }

    /// Get default color for the object type (hex RGB)
    pub fn default_color(&self) -> u32 {
        match self {
            ObjectType::Clock => 0x2d3748,
            ObjectType::Lamp => 0x4f46e5,
            ObjectType::Plant => 0x22c55e,
            ObjectType::Coffee => 0xfbbf24,
            ObjectType::Laptop => 0x64748b,
            ObjectType::Notebook => 0xef4444,
            ObjectType::PenHolder => 0x8b5cf6,
            ObjectType::Books => 0x0ea5e9,
            ObjectType::PhotoFrame => 0x78716c,
            ObjectType::Globe => 0x3b82f6,
            ObjectType::Trophy => 0xfbbf24,
            ObjectType::Hourglass => 0xfbbf24,
            ObjectType::Metronome => 0x78350f,
            ObjectType::Paper => 0xffffff,
            ObjectType::Magazine => 0xef4444,
            ObjectType::MusicPlayer => 0x1e293b,
            ObjectType::Pen => 0x3b82f6,
        }
    }

    /// Get default accent color for the object type (hex RGB)
    pub fn default_accent_color(&self) -> u32 {
        match self {
            ObjectType::Clock => 0xffffff,
            ObjectType::Lamp => 0xfef3c7,
            ObjectType::Plant => 0x78350f,
            ObjectType::Coffee => 0x78350f,
            ObjectType::Laptop => 0x1e293b,
            ObjectType::Notebook => 0x000000,
            ObjectType::PenHolder => 0x000000,
            ObjectType::Books => 0xfbbf24,
            ObjectType::PhotoFrame => 0xfbbf24,
            ObjectType::Globe => 0x22c55e,
            ObjectType::Trophy => 0x78350f,
            ObjectType::Hourglass => 0x78350f,
            ObjectType::Metronome => 0xfbbf24,
            ObjectType::Paper => 0x000000,
            ObjectType::Magazine => 0xffffff,
            ObjectType::MusicPlayer => 0x22c55e,
            ObjectType::Pen => 0x1e293b,
        }
    }

    /// Get the physics properties for the object type
    pub fn physics(&self) -> ObjectPhysics {
        match self {
            ObjectType::Clock => ObjectPhysics {
                weight: 0.5,
                stability: 0.5,
                height: 0.6,
                base_offset: 0.35,
                friction: 0.4,
                no_stacking_on_top: true,
            },
            ObjectType::Lamp => ObjectPhysics {
                weight: 1.2,
                stability: 0.85,
                height: 0.9,
                base_offset: 0.0,
                friction: 0.5,
                no_stacking_on_top: false,
            },
            ObjectType::Plant => ObjectPhysics {
                weight: 1.4,
                stability: 0.9,
                height: 0.5,
                base_offset: 0.0,
                friction: 0.6,
                no_stacking_on_top: false,
            },
            ObjectType::Coffee => ObjectPhysics {
                weight: 0.4,
                stability: 0.6,
                height: 0.3,
                base_offset: 0.0,
                friction: 0.5,
                no_stacking_on_top: false,
            },
            ObjectType::Laptop => ObjectPhysics {
                weight: 1.5,
                stability: 0.95,
                height: 0.3,
                base_offset: 0.0,
                friction: 0.6,
                no_stacking_on_top: false,
            },
            ObjectType::Notebook => ObjectPhysics {
                weight: 0.3,
                stability: 0.95,
                height: 0.1,
                base_offset: 0.0,
                friction: 0.7,
                no_stacking_on_top: false,
            },
            ObjectType::PenHolder => ObjectPhysics {
                weight: 0.6,
                stability: 0.6,
                height: 0.4,
                base_offset: 0.0,
                friction: 0.5,
                no_stacking_on_top: false,
            },
            ObjectType::Books => ObjectPhysics {
                weight: 0.8,
                stability: 0.9,
                height: 0.15,
                base_offset: 0.0,
                friction: 0.75,
                no_stacking_on_top: false,
            },
            ObjectType::PhotoFrame => ObjectPhysics {
                weight: 0.3,
                stability: 0.35,
                height: 0.5,
                base_offset: 0.25,
                friction: 0.4,
                no_stacking_on_top: true,
            },
            ObjectType::Globe => ObjectPhysics {
                weight: 1.0,
                stability: 0.7,
                height: 0.5,
                base_offset: 0.025,
                friction: 0.45,
                no_stacking_on_top: false,
            },
            ObjectType::Trophy => ObjectPhysics {
                weight: 0.9,
                stability: 0.6,
                height: 0.4,
                base_offset: 0.0,
                friction: 0.5,
                no_stacking_on_top: false,
            },
            ObjectType::Hourglass => ObjectPhysics {
                weight: 0.5,
                stability: 0.45,
                height: 0.35,
                base_offset: 0.015,
                friction: 0.4,
                no_stacking_on_top: false,
            },
            ObjectType::Metronome => ObjectPhysics {
                weight: 0.7,
                stability: 0.7,
                height: 0.45,
                base_offset: 0.0,
                friction: 0.55,
                no_stacking_on_top: false,
            },
            ObjectType::Paper => ObjectPhysics {
                weight: 0.05,
                stability: 0.98,
                height: 0.01,
                base_offset: 0.0,
                friction: 0.8,
                no_stacking_on_top: false,
            },
            ObjectType::Magazine => ObjectPhysics {
                weight: 0.3,
                stability: 0.95,
                height: 0.02,
                base_offset: 0.0,
                friction: 0.65,
                no_stacking_on_top: false,
            },
            ObjectType::MusicPlayer => ObjectPhysics {
                weight: 0.8,
                stability: 0.85,
                height: 0.15,
                base_offset: 0.0,
                friction: 0.55,
                no_stacking_on_top: false,
            },
            ObjectType::Pen => ObjectPhysics {
                weight: 0.05,
                stability: 0.3,
                height: 0.02,
                base_offset: 0.0,
                friction: 0.4,
                no_stacking_on_top: false,
            },
        }
    }

    /// Get all object types for the palette
    pub fn all() -> &'static [ObjectType] {
        &[
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
            ObjectType::MusicPlayer,
            ObjectType::Pen,
        ]
    }

    /// Check if this object type has interactive features
    pub fn is_interactive(&self) -> bool {
        matches!(
            self,
            ObjectType::Clock
                | ObjectType::Lamp
                | ObjectType::Globe
                | ObjectType::Hourglass
                | ObjectType::PhotoFrame
                | ObjectType::Metronome
                | ObjectType::MusicPlayer
                | ObjectType::Coffee
        )
    }
}

/// Physics properties for an object type
#[derive(Debug, Clone, Copy)]
pub struct ObjectPhysics {
    /// How heavy the object is (affects momentum)
    pub weight: f32,
    /// How stable the object is (resistance to tipping)
    pub stability: f32,
    /// Height of the object for stacking
    pub height: f32,
    /// Distance from origin to bottom (for Y correction when scaling)
    pub base_offset: f32,
    /// Surface friction coefficient (0-1)
    pub friction: f32,
    /// If true, objects cannot be stacked on top of this
    pub no_stacking_on_top: bool,
}

/// A desk object instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeskObject {
    /// Unique identifier
    #[serde(default = "default_id")]
    pub id: u64,
    /// Type of object
    #[serde(default)]
    pub object_type: ObjectType,
    /// Position in world space
    #[serde(with = "vec3_serde", default = "default_position")]
    pub position: Vec3,
    /// Rotation as quaternion
    #[serde(with = "quat_serde", default = "default_rotation")]
    pub rotation: Quat,
    /// Scale factor
    #[serde(default = "default_scale")]
    pub scale: f32,
    /// Main color (hex RGB)
    #[serde(default = "default_color")]
    pub color: u32,
    /// Accent color (hex RGB)
    #[serde(default = "default_accent_color")]
    pub accent_color: u32,
    /// Custom collision radius multiplier (1.0 = default)
    #[serde(default = "default_multiplier")]
    pub collision_radius_multiplier: f32,
    /// Custom collision height multiplier (1.0 = default)
    #[serde(default = "default_multiplier")]
    pub collision_height_multiplier: f32,
    /// Object-specific interactive state
    #[serde(default)]
    pub state: ObjectState,
    /// Whether the object is currently being dragged
    #[serde(skip)]
    pub is_dragging: bool,
    /// Target Y position for smooth dropping
    #[serde(skip)]
    pub target_y: f32,
    /// Original Y position (on desk surface)
    #[serde(skip)]
    pub original_y: f32,
}

// Default value functions for serde
fn default_id() -> u64 {
    1
}

fn default_position() -> Vec3 {
    Vec3::ZERO
}

fn default_rotation() -> Quat {
    Quat::IDENTITY
}

fn default_scale() -> f32 {
    1.0
}

fn default_color() -> u32 {
    0x808080
}

fn default_accent_color() -> u32 {
    0x404040
}

fn default_multiplier() -> f32 {
    1.0
}

impl DeskObject {
    /// Create a new desk object
    pub fn new(id: u64, object_type: ObjectType, position: Vec3) -> Self {
        let physics = object_type.physics();
        let y = position.y + physics.base_offset;

        Self {
            id,
            object_type,
            position: Vec3::new(position.x, y, position.z),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            color: object_type.default_color(),
            accent_color: object_type.default_accent_color(),
            collision_radius_multiplier: 1.0,
            collision_height_multiplier: 1.0,
            state: ObjectState::default(),
            is_dragging: false,
            target_y: y,
            original_y: y,
        }
    }

    /// Get the model matrix for this object
    pub fn model_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            self.rotation,
            self.position,
        )
    }

    /// Get the collision radius for this object
    pub fn collision_radius(&self) -> f32 {
        let _physics = self.object_type.physics();
        // Base radius based on object type, scaled
        let base_radius = match self.object_type {
            ObjectType::Lamp => 0.3,
            ObjectType::Plant => 0.25,
            ObjectType::Coffee => 0.15,
            ObjectType::Laptop => 0.5,
            ObjectType::Notebook => 0.35,
            ObjectType::Books => 0.3,
            ObjectType::Globe => 0.25,
            ObjectType::Trophy => 0.2,
            ObjectType::MusicPlayer => 0.25,
            _ => 0.2,
        };
        base_radius * self.scale * self.collision_radius_multiplier
    }

    /// Get the collision height for this object
    pub fn collision_height(&self) -> f32 {
        let physics = self.object_type.physics();
        physics.height * self.scale * self.collision_height_multiplier
    }

    /// Check if a point is inside the collision bounds
    pub fn contains_point(&self, point: Vec3) -> bool {
        let radius = self.collision_radius();
        let height = self.collision_height();

        let dx = point.x - self.position.x;
        let dz = point.z - self.position.z;
        let horizontal_dist_sq = dx * dx + dz * dz;

        let within_radius = horizontal_dist_sq <= radius * radius;
        let within_height = point.y >= self.position.y && point.y <= self.position.y + height;

        within_radius && within_height
    }
}

// Custom serialization for Vec3
mod vec3_serde {
    use glam::Vec3;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Vec3Repr {
        x: f32,
        y: f32,
        z: f32,
    }

    pub fn serialize<S>(vec: &Vec3, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Vec3Repr {
            x: vec.x,
            y: vec.y,
            z: vec.z,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec3, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = Vec3Repr::deserialize(deserializer)?;
        Ok(Vec3::new(repr.x, repr.y, repr.z))
    }
}

// Custom serialization for Quat
mod quat_serde {
    use glam::Quat;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct QuatRepr {
        x: f32,
        y: f32,
        z: f32,
        w: f32,
    }

    pub fn serialize<S>(quat: &Quat, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        QuatRepr {
            x: quat.x,
            y: quat.y,
            z: quat.z,
            w: quat.w,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Quat, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = QuatRepr::deserialize(deserializer)?;
        Ok(Quat::from_xyzw(repr.x, repr.y, repr.z, repr.w))
    }
}
