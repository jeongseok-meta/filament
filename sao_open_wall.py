"""Failing-first test for the Filament SAO fix: ambient occlusion must not
darken an open, flat wall, at any render size.

A box room (floor and four walls), one 90-degree camera. The middle of the far
wall is more than the AO radius (0.3 m) from every other surface, so its true
ambient occlusion is none: with SAO on it must read as with SAO off, at a small
and at a large render target. Filament 1.79 (MuJoCo 4.0.x) gives 11.0 vs 16.0
code levels at 256 px and 13.7 vs 16.0 at 1024 px.

Runs on whatever Vulkan device the loader picks (e.g. lavapipe with
VK_ICD_FILENAMES=.../lvp_icd.x86_64.json). Needs mujoco and mujoco-studio.
"""

import importlib
import unittest

import mujoco
import numpy as np

importlib.import_module("mujoco.studio")  # registers Filament's assets
mjrf = importlib.import_module("mujoco.mjrf")

XML = """
<mujoco>
  <custom><numeric name="filament.ao.enabled" data="{ao}"/></custom>
  <asset><material name="wall" rgba=".6 .6 .6 1"/></asset>
  <worldbody>
    <geom type="plane" size="3 3 .1" material="wall"/>
    <geom type="box" pos="0 3 1.5" size="3 .05 1.5" material="wall"/>
    <geom type="box" pos="0 -3 1.5" size="3 .05 1.5" material="wall"/>
    <geom type="box" pos="3 0 1.5" size=".05 3 1.5" material="wall"/>
    <geom type="box" pos="-3 0 1.5" size=".05 3 1.5" material="wall"/>
    <camera pos="0 -1.2 1.0" xyaxes="1 0 0 0 .5 .866" fovy="90"/>
  </worldbody>
</mujoco>
"""
VIEW = ("pos", "forward", "up", "frustum_center", "frustum_width",
        "frustum_bottom", "frustum_top", "frustum_near", "frustum_far",
        "orthographic")


def far_wall_mean(side: int, ao: bool) -> float:
  """Mean code level of the far wall's middle (|x| < 1.5, 0.9 < z < 2.6)."""
  model = mujoco.MjModel.from_xml_string(XML.format(ao=int(ao)))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  ctx = mjrf.Context(mjrf.ContextConfig(
      graphics_api=mjrf.GraphicsApi.GRAPHICS_API_VULKAN))
  resources = mjrf.ModelResources(ctx, model)
  scene = ctx.create_scene(mjrf.SceneParams())
  scene.configure_from_model(model)
  lights = mjrf.ModelLights(scene, resources)
  renderables = mjrf.ModelRenderables(scene, resources)
  renderables.set_options(mujoco.MjvOption())
  ctx.render([], [])
  lights.update(data)
  renderables.update(data)
  target = ctx.create_render_target(mjrf.RenderTargetConfig(
      width=side, height=side, color_format=mjrf.PixelFormat.PIXEL_FORMAT_RGB8))
  rgb = np.empty((side, side, 3), np.uint8)
  read = mjrf.ReadRequest(target=target)
  read.set_buffer(rgb)
  request = mjrf.RenderRequest(
      scene=scene, viewport=mjrf.Rect(0, 0, side, side), target=target,
      draw_mode=mjrf.DrawMode.DRAW_MODE_DEFAULT, post_processing=False,
      reflections=False, shadows=False, headlight=False)
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
  camera.fixedcamid = 0
  view = mujoco.mjv_camera2GLCamera(model, data, camera)
  request.camera = mjrf.Camera(**{f: getattr(view, f) for f in VIEW})
  ctx.wait_for_frame(ctx.render([request], [read]))

  # Pixels whose ray hits the far wall's inner face (y = 2.95) in the middle.
  pos = data.cam_xpos[0]
  rot = data.cam_xmat[0].reshape(3, 3)
  centers = np.arange(side) + 0.5
  x, y = np.meshgrid(centers, centers)
  rays = np.stack([(x - side / 2) / (side / 2), -(y - side / 2) / (side / 2),
                   -np.ones_like(x)], -1) @ rot.T
  t = (2.95 - pos[1]) / rays[..., 1]
  hit = pos + t[..., None] * rays
  mask = ((t > 0) & (np.abs(hit[..., 0]) < 1.5) & (hit[..., 2] > 0.9)
          & (hit[..., 2] < 2.6))
  value = float(rgb[mask].astype(float).mean())
  del read, request, target, renderables, lights, scene, resources, ctx
  return value


class SaoOpenWallTest(unittest.TestCase):

  def test_an_open_wall_gets_no_ambient_occlusion_at_any_size(self):
    for side in (256, 1024):
      with self.subTest(side=side):
        with_ao, without_ao = far_wall_mean(side, True), far_wall_mean(side, False)
        self.assertAlmostEqual(
            with_ao, without_ao, delta=0.5,
            msg=f"{side} px: open wall {with_ao:.2f} with SAO vs {without_ao:.2f} without")


if __name__ == "__main__":
  print("render size   SSAO on   SSAO off   (far wall, mean code level)")
  for side in (256, 512, 1024, 2048):
    print(f"{side:>11}   {far_wall_mean(side, True):7.2f}   {far_wall_mean(side, False):8.2f}")
  unittest.main()
