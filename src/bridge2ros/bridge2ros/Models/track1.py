import math

class TrackBuilder:
    def __init__(self, track_width, wall_height, wall_thickness, height=0.1):
        self.track_width = track_width
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.height = height
        self.segments = []

    # Function to create a straight track segment
    def create_straight_segment(self, name, segment_length, position):
        segment = self.create_box_segment(name, [segment_length, self.track_width, self.height], position)
        wall_left = self.create_wall_segment(f"{name}_wall_left", segment_length, position, offset=self.track_width / 2 + self.wall_thickness / 2)
        wall_right = self.create_wall_segment(f"{name}_wall_right", segment_length, position, offset=-(self.track_width / 2 + self.wall_thickness / 2))
        return segment + wall_left + wall_right

    # Function to create a left turn
    def create_left_turn(self, name, curve_radius, position, start_angle=0, end_angle=math.pi/2, resolution=36):
        segment = self.create_curve_segment(name, curve_radius, position, start_angle, end_angle, resolution)
        wall_outer = self.create_curve_wall(f"{name}_outer_wall", curve_radius + self.track_width / 2 + self.wall_thickness / 2, position, start_angle, end_angle, resolution)
        wall_inner = self.create_curve_wall(f"{name}_inner_wall", curve_radius - self.track_width / 2 - self.wall_thickness / 2, position, start_angle, end_angle, resolution)
        return segment + wall_outer + wall_inner

    # Function to create a right turn
    def create_right_turn(self, name, curve_radius, position, start_angle=0, end_angle=-math.pi/2, resolution=36):
        segment = self.create_curve_segment(name, curve_radius, position, start_angle, end_angle, resolution)
        wall_outer = self.create_curve_wall(f"{name}_outer_wall", curve_radius + self.track_width / 2 + self.wall_thickness / 2, position, start_angle, end_angle, resolution)
        wall_inner = self.create_curve_wall(f"{name}_inner_wall", curve_radius - self.track_width / 2 - self.wall_thickness / 2, position, start_angle, end_angle, resolution)
        return segment + wall_outer + wall_inner

    # Function to build the track by combining the components
    def build_track(self):
        track_sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="track">
    <static>true</static>
    <link name="track_link">
      {''.join(self.segments)}
    </link>
  </model>
</sdf>"""
        return track_sdf

    # Function to add straight segment to the track
    def add_straight(self, name, segment_length, position):
        self.segments.append(self.create_straight_segment(name, segment_length, position))

    # Function to add a left turn to the track
    def add_left_turn(self, name, curve_radius, position, start_angle=0, end_angle=math.pi/2):
        self.segments.append(self.create_left_turn(name, curve_radius, position, start_angle, end_angle))

    # Function to add a right turn to the track
    def add_right_turn(self, name, curve_radius, position, start_angle=0, end_angle=-math.pi/2):
        self.segments.append(self.create_right_turn(name, curve_radius, position, start_angle, end_angle))

    # Helper function to create box geometry (straight segment)
    def create_box_segment(self, name, size, pose):
        return f"""
        <collision name="collision_{name}">
          <geometry>
            <box>
              <size>{size[0]} {size[1]} {size[2]}</size>
            </box>
          </geometry>
          <pose>{pose[0]} {pose[1]} 0 0 0 0</pose>
        </collision>
        <visual name="visual_{name}">
          <geometry>
            <box>
              <size>{size[0]} {size[1]} {size[2]}</size>
            </box>
          </geometry>
          <pose>{pose[0]} {pose[1]} 0 0 0 0</pose>
        </visual>"""

    # Helper function to create a wall for straight segment
    def create_wall_segment(self, name, length, position, offset):
        return self.create_box_segment(name, [length, self.wall_thickness, self.wall_height], [position[0], position[1] + offset, self.wall_height / 2])

    # Helper function to create a curved segment (turn)
    def create_curve_segment(self, name, radius, position, start_angle, end_angle, resolution=36):
        segment = ""
        angle_step = (end_angle - start_angle) / resolution
        for i in range(resolution):
            angle = start_angle + i * angle_step
            x = position[0] + radius * math.cos(angle)
            y = position[1] + radius * math.sin(angle)
            segment += self.create_box_segment(
                f"{name}_{i}",
                [radius * 2 * math.pi / resolution, self.track_width, self.height],
                [x, y, self.height / 2]
            )
        return segment

    # Helper function to create a wall for curved segments
    def create_curve_wall(self, name, radius, position, start_angle, end_angle, resolution=36):
        wall = ""
        angle_step = (end_angle - start_angle) / resolution
        for i in range(resolution):
            angle = start_angle + i * angle_step
            x = position[0] + radius * math.cos(angle)
            y = position[1] + radius * math.sin(angle)
            wall += self.create_box_segment(
                f"{name}_{i}",
                [radius * 2 * math.pi / resolution, self.wall_thickness, self.wall_height],
                [x, y, self.wall_height / 2]
            )
        return wall

# Example Usage
track_width = 5.0    # Track width
wall_height = 2.0    # Wall height
wall_thickness = 0.1 # Wall thickness
segment_length = 10.0  # Length of straight segments
curve_radius = 3.0     # Radius of curved segments

# Create the TrackBuilder object
track = TrackBuilder(track_width, wall_height, wall_thickness)

# Add components to the track
track.add_straight("straight1", segment_length, [segment_length / 2, 0, 0])
track.add_left_turn("left_turn1", curve_radius, [segment_length, 0])
track.add_straight("straight2", segment_length, [segment_length + curve_radius, curve_radius])
track.add_right_turn("right_turn1", curve_radius, [segment_length, curve_radius])

# Build the final SDF track
sdf_data = track.build_track()

# Write to a file
with open("track.sdf", "w") as file:
    file.write(sdf_data)

print("OOP-style SDF file created: oop_track.sdf")