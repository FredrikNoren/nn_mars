"use strict";

var fs = require('fs');
var pixelport = require('pixelport');
var Pon = pixelport.Pon;
var p = require('pixelport/pon_bindings');

var app = new pixelport.App(Pon.fromString(`app_options {
  //headless: true,
  tcp_interface: false
}`));

app.request(Pon.fromString(`append_entity {
  parent: root,
  type_name: 'Scene',
  properties: {
    view_projection: matrix_mul [
      projection { far: 100, aspect: @root.screen_aspect },
      lookat { eye: vec3 { x: -10, y: -10, z: 12 }, center: vec3 { z: 3 } }
    ],
    renderer: renderer {
      units: [
        render_unit {
          shader: shader_program [
            shader_from_file { filename: 'ground_vs.glsl', shader_type: 'vertex' },
            shader_from_file { filename: 'ground_fs.glsl', shader_type: 'fragment' }
          ]
        }
      ]
    }
  }
}`));

app.request(Pon.fromString(`append_entity {
  parent: root:Scene,
  type_name: 'Ground',
  properties: {
    width: 64,
    height: 64,
    mesh_resolution: 1,
    mesh: grid_mesh { n_vertices_width: div [@this.width, @this.mesh_resolution], n_vertices_height: div [@this.height, @this.mesh_resolution] },
    heightmap_filename: '../heightmap.dhm',
    heightmap_image: image_from_file {
      filename: @this.heightmap_filename,
      format: image_file_format 'dhm'
    },
    heightmap_texture: texture {
      image: @this.heightmap_image,
      internal_format: texture_internal_format 'red',
    },
    normalmap_texture: texture {
      image: normal_map_from_heightmap {
        heightmap: @this.heightmap_image,
        scale: 20
      },
      internal_format: texture_internal_format 'rgb',
    },
    heightmap: texture_unit { texture: @this.heightmap_texture, sampler: sampler { min_filter: min_mag_filter 'linear' } },
    normalmap: texture_unit { texture: @this.normalmap_texture, sampler: sampler { min_filter: min_mag_filter 'linear' } },
    rotation_z: 0,
    mesh_width: 10,
    mesh_height: mul [ @this.mesh_width, div [ @this.height, @this.width ]],
    world_transform: matrix_mul [
      rotate_z @this.rotation_z,
      scale vec3 { x: @this.mesh_width, y: @this.mesh_height, z: 1 },
      translate vec3 { x: -0.5, y: -0.5 }
    ],
    min_value: 0,
    max_value: 1,
    ambient: color_from_hex '694A20',
    diffuse: color_from_hex 'EDE855',
    animation: animation {
      track: key_frame_track {
        property: this.rotation_z,
        keys: [animation_key { time: 0, value: 0 }, animation_key { time: 1, value: mul [pi (), 2] }],
        loop_type: 'forever', curve_time: 'relative', duration: 40
      }
    }
  }
}`));

function renderHeightmap(i) {
  console.log('Rendering', i);
  if (i == -1) i = 49 //return app.shutdown();

  app.request(p.set_properties({
    entity: Pon.selector('root:Ground'),
    properties: {
      heightmap_filename: '../heightmaps/real2/heightmap-' + i + '.dhm'
    }
  }));

  setTimeout(() => renderHeightmap(i - 1), 1000);

  // app.request(p.await_all_resources({})).then(() => {
  //   setTimeout(() => {
  //     app.request(p.screenshot_to_file({ path: '../heightmaps-rendered/real2/heightmap-' + i + '.png' })).then(() => {
  //       renderHeightmap(i - 1);
  //     });
  //   }, 1000);
  // });

}

renderHeightmap(49)
