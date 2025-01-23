maze = [
    [3, 1, 1, 1, 1, 1, 1, 3],
    [2, 0, 2, 0, 2, 0, 0, 2],
    [2, 0, 0, 0, 2, 0, 0, 2],
    [2, 1, 0, 0, 2, 2, 0, 2],
    [2, 0, 0, 0, 2, 0, 0, 2],
    [2, 0, 0, 0, 2, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 1, 0, 2],
    [2, 0, 0, 1, 0, 0, 1, 2],
    [2, 0, 2, 0, 0, 0, 0, 2],
    [2, 0, 2, 0, 0, 0, 0, 2],
    [3, 1, 1, 1, 1, 1, 1, 3]
]

wall_sdf_template = """
<link name="wall_link_{x}_{y}">
  <pose>{x} {y} 1.25 0 0 {angle}</pose>
  <visual name="wall_visual_{x}_{y}">
    <geometry>
      <box>
        <size>5 0.1 2.5</size>
      </box>
    </geometry>
    <material>  
      <ambient>0.19225 0.19225 0.19225 1.0</ambient>  
      <diffuse>0.50754 0.50754 0.50754 1.0</diffuse>  
      <specular>0.508273 0.508273 0.508273 1.0</specular>  
      <emissive>0.0 0.0 0.0 0.0</emissive>  
    </material>   
  </visual>
</link>
"""

corner_wall_sdf_template = """
<link name="corner_wall_link_{x}_{y}_1">
  <pose>{x}+5 {y} 1.25 0 0 1.5708</pose>
  <visual name="corner_wall_visual_{x}_{y}_1">
    <geometry>
      <box>
        <size>5 0.1 2.5</size>
      </box>
    </geometry>
    <material>  
      <ambient>0.19225 0.19225 0.19225 1.0</ambient>  
      <diffuse>0.50754 0.50754 0.50754 1.0</diffuse>  
      <specular>0.508273 0.508273 0.508273 1.0</specular>  
      <emissive>0.0 0.0 0.0 0.0</emissive>  
    </material>  
  </visual>
</link>
<link name="corner_wall_link_{x}_{y}_2">
  <pose>{x} {y} 1.25 0 0 0</pose>
  <visual name="corner_wall_visual_{x}_{y}_2">
    <geometry>
      <box>
        <size>5 0.1 2.5</size>
      </box>
    </geometry>
    <material>  
      <ambient>0.19225 0.19225 0.19225 1.0</ambient>  
      <diffuse>0.50754 0.50754 0.50754 1.0</diffuse>  
      <specular>0.508273 0.508273 0.508273 1.0</specular>  
      <emissive>0.0 0.0 0.0 0.0</emissive>  
    </material>  
  </visual>
</link>
"""

sdf_content = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="maze">
    <static>true</static>
    <model name='ground_plane'>
      <static>true</static>
      <link name='link'>
        <visual name='visual'>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.800000012 0.800000012 0.800000012 1</ambient>
            <diffuse>0.800000012 0.800000012 0.800000012 1</diffuse>
            <specular>0.800000012 0.800000012 0.800000012 1</specular>
          </material>
        </visual>
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <pose>0 0 0 0 0 0</pose>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
      <pose>0 0 0 0 0 0</pose>
      <self_collide>false</self_collide>
    </model>
"""

for i in range(len(maze)):
    for j in range(len(maze[i])):
        if maze[i][j] == 1:
            angle = 1.5708  # 90 degrees in radians
            sdf_content += wall_sdf_template.format(x=i*5-5, y=j*5-5, angle=angle)
        elif maze[i][j] == 2:
                angle = 0
                sdf_content += wall_sdf_template.format(x=i*5-5, y=j*5-5, angle=angle)
        elif maze[i][j]== 3 :
                angle = 0
                sdf_content += corner_wall_sdf_template.format(x=i*5-5, y=j*5-5, angle=angle)

sdf_content += """
  </model>
</sdf>
"""

with open('maze.sdf', 'w') as f:
    f.write(sdf_content)
