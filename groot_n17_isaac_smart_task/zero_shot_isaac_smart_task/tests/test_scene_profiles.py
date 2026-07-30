from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import scene_profiles  # noqa: E402


class SceneProfilesTest(unittest.TestCase):
    def test_required_scene_profiles_are_public_choices(self) -> None:
        self.assertEqual(
            scene_profiles.SCENE_PROFILE_CHOICES,
            (
                "task-default",
                "tray-red24",
                "tray-red24-d4-horizontal",
                "table-red24",
                "table-red24-d3-horizontal",
                "table-red24-d3-horizontal-right-front5mm",
                "table-red24-d3-horizontal-right10mm-front5mm",
                "multi-lego-tray",
                "multi-lego-tray-raw-a",
                "multi-lego-tray-real007-camera-a",
                "multi-lego-tray-real007-frame5",
                "multi-lego-tray-real007-frame5-forward2cm",
            ),
        )

    def test_single_pick_profiles_share_object_spec_and_differ_by_tray_activation(self) -> None:
        tray_profile = scene_profiles.get_scene_profile("tray-red24")
        table_profile = scene_profiles.get_scene_profile("table-red24")

        self.assertTrue(tray_profile.tray_active)
        self.assertFalse(table_profile.tray_active)
        self.assertEqual(tray_profile.object_keys, (scene_profiles.RED24_OBJECT_KEY,))
        self.assertEqual(table_profile.object_keys, (scene_profiles.RED24_OBJECT_KEY,))
        self.assertEqual(tray_profile.objects, table_profile.objects)
        self.assertEqual(tray_profile.default_target_key, table_profile.default_target_key)
        self.assertEqual(
            tray_profile.requires_explicit_instruction,
            table_profile.requires_explicit_instruction,
        )
        self.assertEqual(tray_profile.objects[0].pos, (0.0, 0.25, 0.025))
        self.assertEqual(
            tray_profile.objects[0].rot,
            (0.70710677, 0.0, 0.0, -0.70710677),
        )
        self.assertEqual(
            scene_profiles.resolve_scene_profile_target("tray-red24", "auto"),
            scene_profiles.RED24_OBJECT_KEY,
        )
        self.assertEqual(
            scene_profiles.resolve_scene_profile_target("table-red24", "auto"),
            scene_profiles.RED24_OBJECT_KEY,
        )

    def test_d4_horizontal_profile_changes_only_single_pick_yaw(self) -> None:
        baseline = scene_profiles.get_scene_profile("tray-red24")
        horizontal = scene_profiles.get_scene_profile("tray-red24-d4-horizontal")

        self.assertTrue(horizontal.tray_active)
        self.assertEqual(horizontal.object_keys, baseline.object_keys)
        self.assertEqual(horizontal.objects[0].pos, baseline.objects[0].pos)
        self.assertEqual(horizontal.objects[0].asset_filename, baseline.objects[0].asset_filename)
        self.assertEqual(horizontal.objects[0].rot, (1.0, 0.0, 0.0, 0.0))
        self.assertNotEqual(horizontal.objects[0].rot, baseline.objects[0].rot)
        self.assertEqual(horizontal.default_target_key, baseline.default_target_key)
        self.assertEqual(
            horizontal.requires_explicit_instruction,
            baseline.requires_explicit_instruction,
        )

    def test_d3_horizontal_profile_changes_only_table_pick_yaw(self) -> None:
        baseline = scene_profiles.get_scene_profile("table-red24")
        horizontal = scene_profiles.get_scene_profile("table-red24-d3-horizontal")

        self.assertFalse(horizontal.tray_active)
        self.assertEqual(horizontal.object_keys, baseline.object_keys)
        self.assertEqual(horizontal.objects[0].pos, baseline.objects[0].pos)
        self.assertEqual(horizontal.objects[0].asset_filename, baseline.objects[0].asset_filename)
        self.assertEqual(horizontal.objects[0].rot, (1.0, 0.0, 0.0, 0.0))
        self.assertNotEqual(horizontal.objects[0].rot, baseline.objects[0].rot)
        self.assertEqual(horizontal.default_target_key, baseline.default_target_key)
        self.assertEqual(
            horizontal.requires_explicit_instruction,
            baseline.requires_explicit_instruction,
        )

    def test_d3_right_front5mm_profile_changes_only_table_pick_xy(self) -> None:
        baseline = scene_profiles.get_scene_profile("table-red24-d3-horizontal")
        shifted = scene_profiles.get_scene_profile(
            "table-red24-d3-horizontal-right-front5mm"
        )

        self.assertFalse(shifted.tray_active)
        self.assertEqual(shifted.object_keys, baseline.object_keys)
        self.assertEqual(shifted.objects[0].pos, (0.005, 0.255, 0.025))
        self.assertAlmostEqual(
            shifted.objects[0].pos[0] - baseline.objects[0].pos[0],
            0.005,
        )
        self.assertAlmostEqual(
            shifted.objects[0].pos[1] - baseline.objects[0].pos[1],
            0.005,
        )
        self.assertEqual(shifted.objects[0].pos[2], baseline.objects[0].pos[2])
        self.assertEqual(
            shifted.objects[0].asset_filename,
            baseline.objects[0].asset_filename,
        )
        self.assertEqual(shifted.objects[0].rot, baseline.objects[0].rot)
        self.assertEqual(
            shifted.objects[0].diffuse_color,
            baseline.objects[0].diffuse_color,
        )
        self.assertEqual(shifted.objects[0].roughness, baseline.objects[0].roughness)
        self.assertEqual(shifted.default_target_key, baseline.default_target_key)
        self.assertEqual(
            shifted.requires_explicit_instruction,
            baseline.requires_explicit_instruction,
        )

    def test_d3_right10mm_front5mm_profile_changes_only_table_pick_xy(self) -> None:
        baseline = scene_profiles.get_scene_profile("table-red24-d3-horizontal")
        shifted = scene_profiles.get_scene_profile(
            "table-red24-d3-horizontal-right10mm-front5mm"
        )

        self.assertFalse(shifted.tray_active)
        self.assertEqual(shifted.object_keys, baseline.object_keys)
        self.assertEqual(shifted.objects[0].pos, (0.010, 0.255, 0.025))
        self.assertAlmostEqual(
            shifted.objects[0].pos[0] - baseline.objects[0].pos[0],
            0.010,
        )
        self.assertAlmostEqual(
            shifted.objects[0].pos[1] - baseline.objects[0].pos[1],
            0.005,
        )
        self.assertEqual(shifted.objects[0].pos[2], baseline.objects[0].pos[2])
        self.assertEqual(
            shifted.objects[0].asset_filename,
            baseline.objects[0].asset_filename,
        )
        self.assertEqual(shifted.objects[0].rot, baseline.objects[0].rot)
        self.assertEqual(
            shifted.objects[0].diffuse_color,
            baseline.objects[0].diffuse_color,
        )
        self.assertEqual(shifted.objects[0].roughness, baseline.objects[0].roughness)
        self.assertEqual(shifted.default_target_key, baseline.default_target_key)
        self.assertEqual(
            shifted.requires_explicit_instruction,
            baseline.requires_explicit_instruction,
        )

    def test_multi_lego_tray_follows_leisaac_authored_layout(self) -> None:
        profile = scene_profiles.get_scene_profile("multi-lego-tray")
        self.assertTrue(profile.tray_active)
        self.assertEqual(
            profile.object_keys,
            (
                scene_profiles.RED24_OBJECT_KEY,
                scene_profiles.RED22_OBJECT_KEY,
                scene_profiles.BLUE24_OBJECT_KEY,
            ),
        )
        positions = {obj.key: obj.pos for obj in profile.objects}
        self.assertEqual(
            positions,
            {
                scene_profiles.RED24_OBJECT_KEY: (0.126, 0.25, 0.015),
                scene_profiles.RED22_OBJECT_KEY: (0.126, 0.15, 0.015),
                scene_profiles.BLUE24_OBJECT_KEY: (0.186, 0.15, 0.015),
            },
        )
        blue = next(obj for obj in profile.objects if obj.key == scene_profiles.BLUE24_OBJECT_KEY)
        self.assertEqual(blue.asset_filename, scene_profiles.RED24_ASSET)
        self.assertIsNotNone(blue.diffuse_color)
        self.assertEqual(blue.diffuse_color, (0.0, 0.0, 1.0))

    def test_raw_a_multi_lego_layout_changes_only_fixed_object_poses(self) -> None:
        authored = scene_profiles.get_scene_profile("multi-lego-tray")
        raw_a = scene_profiles.get_scene_profile("multi-lego-tray-raw-a")

        self.assertTrue(raw_a.tray_active)
        self.assertTrue(raw_a.requires_explicit_instruction)
        self.assertEqual(raw_a.object_keys, authored.object_keys)
        self.assertEqual(
            {obj.key: obj.pos for obj in raw_a.objects},
            {
                scene_profiles.RED24_OBJECT_KEY: (0.126, 0.18, 0.015),
                scene_profiles.RED22_OBJECT_KEY: (0.156, 0.12, 0.015),
                scene_profiles.BLUE24_OBJECT_KEY: (0.186, 0.15, 0.015),
            },
        )
        for authored_obj, raw_a_obj in zip(authored.objects, raw_a.objects, strict=True):
            self.assertEqual(raw_a_obj.key, authored_obj.key)
            self.assertEqual(raw_a_obj.asset_filename, authored_obj.asset_filename)
            self.assertEqual(raw_a_obj.rot, authored_obj.rot)
            self.assertEqual(raw_a_obj.diffuse_color, authored_obj.diffuse_color)
            self.assertEqual(raw_a_obj.roughness, authored_obj.roughness)

    def test_real007_camera_a_changes_only_front_camera_pose_from_raw_a(self) -> None:
        raw_a = scene_profiles.get_scene_profile("multi-lego-tray-raw-a")
        camera_a = scene_profiles.get_scene_profile("multi-lego-tray-real007-camera-a")

        self.assertEqual(camera_a.tray_active, raw_a.tray_active)
        self.assertIs(camera_a.objects, raw_a.objects)
        self.assertEqual(camera_a.default_target_key, raw_a.default_target_key)
        self.assertEqual(
            camera_a.requires_explicit_instruction,
            raw_a.requires_explicit_instruction,
        )
        self.assertIsNone(raw_a.front_camera_pos)
        self.assertIsNone(raw_a.front_camera_rot)
        self.assertEqual(camera_a.front_camera_pos, (0.018, 0.34, 0.22))
        self.assertEqual(camera_a.front_camera_rot, (0.0, 0.0, 0.1891, 0.9820))

    def test_real007_frame5_profile_uses_calibrated_camera_and_frame_layout(self) -> None:
        profile = scene_profiles.get_scene_profile("multi-lego-tray-real007-frame5")

        self.assertEqual(profile.front_camera_pos, (0.018, 0.34, 0.22))
        self.assertEqual(profile.front_camera_rot, (0.0, 0.0, 0.1891, 0.9820))
        self.assertEqual(
            {obj.key: obj.pos for obj in profile.objects},
            {
                scene_profiles.RED24_OBJECT_KEY: (0.151951, 0.172605, 0.015),
                scene_profiles.RED22_OBJECT_KEY: (0.191199, 0.163356, 0.015),
                scene_profiles.BLUE24_OBJECT_KEY: (0.166582, 0.218731, 0.015),
            },
        )
        self.assertEqual(
            tuple(obj.rot for obj in profile.objects),
            (
                scene_profiles.RED24_ROT,
                scene_profiles.IDENTITY_ROT,
                scene_profiles.RED24_ROT,
            ),
        )

    def test_real007_frame5_forward2cm_changes_only_all_object_y_positions(self) -> None:
        baseline = scene_profiles.get_scene_profile("multi-lego-tray-real007-frame5")
        forward = scene_profiles.get_scene_profile(
            "multi-lego-tray-real007-frame5-forward2cm"
        )

        self.assertEqual(forward.tray_active, baseline.tray_active)
        self.assertEqual(forward.default_target_key, baseline.default_target_key)
        self.assertEqual(
            forward.requires_explicit_instruction,
            baseline.requires_explicit_instruction,
        )
        self.assertEqual(forward.front_camera_pos, baseline.front_camera_pos)
        self.assertEqual(forward.front_camera_rot, baseline.front_camera_rot)
        self.assertEqual(forward.object_keys, baseline.object_keys)
        for baseline_obj, forward_obj in zip(
            baseline.objects,
            forward.objects,
            strict=True,
        ):
            self.assertEqual(forward_obj.key, baseline_obj.key)
            self.assertEqual(forward_obj.asset_filename, baseline_obj.asset_filename)
            self.assertEqual(forward_obj.rot, baseline_obj.rot)
            self.assertEqual(forward_obj.diffuse_color, baseline_obj.diffuse_color)
            self.assertEqual(forward_obj.roughness, baseline_obj.roughness)
            self.assertEqual(forward_obj.pos[0], baseline_obj.pos[0])
            self.assertAlmostEqual(forward_obj.pos[1] - baseline_obj.pos[1], 0.02)
            self.assertEqual(forward_obj.pos[2], baseline_obj.pos[2])

    def test_multi_lego_target_must_be_explicit_and_belong_to_profile(self) -> None:
        for profile_name in (
            "multi-lego-tray",
            "multi-lego-tray-raw-a",
            "multi-lego-tray-real007-camera-a",
            "multi-lego-tray-real007-frame5",
            "multi-lego-tray-real007-frame5-forward2cm",
        ):
            with self.subTest(profile_name=profile_name):
                with self.assertRaisesRegex(ValueError, "contains multiple LEGO"):
                    scene_profiles.resolve_scene_profile_target(profile_name, "auto")
                self.assertEqual(
                    scene_profiles.resolve_scene_profile_target(
                        profile_name,
                        scene_profiles.BLUE24_OBJECT_KEY,
                    ),
                    scene_profiles.BLUE24_OBJECT_KEY,
                )
                with self.assertRaisesRegex(ValueError, "is not part of"):
                    scene_profiles.resolve_scene_profile_target(profile_name, "missing")

    def test_explicit_profiles_use_base_so101_task(self) -> None:
        scene_profiles.validate_scene_profile_task(
            scene_profiles.DEFAULT_SO101_TASK,
            "table-red24",
        )
        with self.assertRaisesRegex(ValueError, "selects the base environment"):
            scene_profiles.validate_scene_profile_task(
                "LeIsaac-SO101-SmartTask-Blue-v0",
                "table-red24",
            )

    def test_wrapper_deactivates_legacy_legos_and_controls_tray_collision(self) -> None:
        source = Path("/tmp/source_scene.usd")
        table_text = scene_profiles.build_scene_wrapper_text(source, "table-red24")
        tray_text = scene_profiles.build_scene_wrapper_text(source, "tray-red24")
        self.assertIn('over "tray" (\n        active = false', table_text)
        self.assertIn('over "tray" (\n        active = true', tray_text)
        for prim_name in (
            "red_2x4_lego_brick",
            "blue_2x4_lego_brick",
            "red_2x2_lego_brick",
            "red_2x4_lego_brick_pick",
        ):
            self.assertIn(f'over "{prim_name}" (\n        active = false', table_text)
            self.assertIn("payload = None", table_text.split(f'over "{prim_name}"', 1)[1].split(")", 1)[0])
        self.assertIn('over "camera_left_xform"', table_text)
        self.assertIn('over "camera_front_xform"', table_text)
        self.assertEqual(table_text.count("references = None"), 2)

    def test_wrapper_authors_front_camera_pose_only_for_camera_profile(self) -> None:
        source = Path("/tmp/source_scene.usd")
        raw_a_text = scene_profiles.build_scene_wrapper_text(
            source,
            "multi-lego-tray-raw-a",
        )
        camera_a_text = scene_profiles.build_scene_wrapper_text(
            source,
            "multi-lego-tray-real007-camera-a",
        )

        self.assertNotIn("xformOp:translate", raw_a_text)
        self.assertNotIn("xformOp:orient", raw_a_text)
        self.assertIn(
            "double3 xformOp:translate = (0.018, 0.34, 0.22)",
            camera_a_text,
        )
        self.assertIn(
            "quatd xformOp:orient = (0.0, 0.0, 0.1891, 0.982)",
            camera_a_text,
        )
        self.assertEqual(camera_a_text.count("xformOp:translate"), 1)
        self.assertEqual(camera_a_text.count("xformOp:orient"), 1)
        self.assertNotIn("xformOpOrder", camera_a_text)
        self.assertNotIn("xformOp:scale", camera_a_text)
        self.assertNotIn("xformOp:rotate", camera_a_text)
        self.assertNotIn("xformOp:transform", camera_a_text)


    def test_wrapper_is_file_backed_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "profiles"
            source = Path(tmp_dir) / "scene.usd"
            first = scene_profiles.write_scene_wrapper(source, "table-red24", output_dir)
            second = scene_profiles.write_scene_wrapper(source, "table-red24", output_dir)
            self.assertEqual(first, second)
            self.assertTrue(first.is_file())
            self.assertIn(str(source.resolve()), first.read_text(encoding="utf-8"))
    def test_managed_objects_use_complete_usd_assets_and_blue_material(self) -> None:
        class FakeCfg:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class FakeRigidObjectCfg(FakeCfg):
            class InitialStateCfg(FakeCfg):
                pass

        class FakeUsdFileCfg(FakeCfg):
            pass

        fake_isaaclab = ModuleType("isaaclab")
        fake_isaaclab.__path__ = []
        fake_sim = ModuleType("isaaclab.sim")
        fake_sim.RigidBodyPropertiesCfg = FakeCfg
        fake_sim.CollisionPropertiesCfg = FakeCfg
        fake_sim.PreviewSurfaceCfg = FakeCfg
        fake_sim.UsdFileCfg = FakeUsdFileCfg
        fake_assets = ModuleType("isaaclab.assets")
        fake_assets.RigidObjectCfg = FakeRigidObjectCfg
        fake_isaaclab.sim = fake_sim

        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_dir = Path(tmp_dir)
            with patch.dict(
                sys.modules,
                {
                    "isaaclab": fake_isaaclab,
                    "isaaclab.sim": fake_sim,
                    "isaaclab.assets": fake_assets,
                },
            ):
                with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                    scene_profiles.add_scene_profile_objects(
                        SimpleNamespace(scene=SimpleNamespace()),
                        "multi-lego-tray",
                        asset_dir,
                    )

                for filename in (scene_profiles.RED24_ASSET, scene_profiles.RED22_ASSET):
                    (asset_dir / filename).touch()

                env_cfg = SimpleNamespace(scene=SimpleNamespace())
                added = scene_profiles.add_scene_profile_objects(
                    env_cfg,
                    "multi-lego-tray",
                    asset_dir,
                )
                real_env_cfg = SimpleNamespace(scene=SimpleNamespace())
                scene_profiles.add_scene_profile_objects(
                    real_env_cfg,
                    "multi-lego-tray",
                    asset_dir,
                    red24_material=scene_profiles.RED24_MATERIAL_REAL_RED,
                )

        self.assertEqual(
            added,
            (
                scene_profiles.RED24_OBJECT_KEY,
                scene_profiles.RED22_OBJECT_KEY,
                scene_profiles.BLUE24_OBJECT_KEY,
            ),
        )
        red_spawn = getattr(env_cfg.scene, scene_profiles.RED24_OBJECT_KEY).spawn
        blue_spawn = getattr(env_cfg.scene, scene_profiles.BLUE24_OBJECT_KEY).spawn
        self.assertIsInstance(red_spawn, FakeUsdFileCfg)
        self.assertFalse(hasattr(red_spawn, "mass_props"))
        self.assertTrue(blue_spawn.usd_path.endswith(scene_profiles.RED24_ASSET))
        self.assertEqual(blue_spawn.visual_material.diffuse_color, (0.0, 0.0, 1.0))
        real_red_spawn = getattr(
            real_env_cfg.scene,
            scene_profiles.RED24_OBJECT_KEY,
        ).spawn
        self.assertEqual(
            real_red_spawn.visual_material.diffuse_color,
            scene_profiles.REAL_RED24_DIFFUSE_COLOR,
        )
        self.assertEqual(
            real_red_spawn.visual_material.roughness,
            scene_profiles.REAL_RED24_ROUGHNESS,
        )

    def test_multi_target_rebinds_terms_success_and_all_randomizers(self) -> None:
        class FakeSceneEntityCfg:
            def __init__(self, name):
                self.name = name

        fake_isaaclab = ModuleType("isaaclab")
        fake_isaaclab.__path__ = []
        fake_managers = ModuleType("isaaclab.managers")
        fake_managers.SceneEntityCfg = FakeSceneEntityCfg

        placed_on_tray = object()
        fake_leisaac = ModuleType("leisaac")
        fake_leisaac.__path__ = []
        fake_tasks = ModuleType("leisaac.tasks")
        fake_tasks.__path__ = []
        fake_smart_task = ModuleType("leisaac.tasks.smart_task")
        fake_smart_task.__path__ = []
        fake_smart_task.mdp = SimpleNamespace(object_placed_on_tray=placed_on_tray)
        fake_leisaac.tasks = fake_tasks
        fake_tasks.smart_task = fake_smart_task

        def term():
            return SimpleNamespace(params={"object_cfg": object()})

        success = SimpleNamespace(func=object(), params={"object_cfg": object()})
        env_cfg = SimpleNamespace(
            observations=SimpleNamespace(
                subtask_terms=SimpleNamespace(
                    lego_grasped=term(),
                    lego_lifted=term(),
                    lego_placed_on_tray=term(),
                )
            ),
            terminations=SimpleNamespace(success=success),
            events=SimpleNamespace(
                reset_all="keep",
                domain_randomize_0=object(),
                domain_randomize_7=object(),
            ),
        )

        with patch.dict(
            sys.modules,
            {
                "isaaclab": fake_isaaclab,
                "isaaclab.managers": fake_managers,
                "leisaac": fake_leisaac,
                "leisaac.tasks": fake_tasks,
                "leisaac.tasks.smart_task": fake_smart_task,
            },
        ):
            scene_profiles.apply_scene_profile_target(
                env_cfg,
                "multi-lego-tray-raw-a",
                scene_profiles.BLUE24_OBJECT_KEY,
            )

        self.assertEqual(env_cfg.target_object_name, scene_profiles.BLUE24_OBJECT_KEY)
        for term_name in ("lego_grasped", "lego_lifted", "lego_placed_on_tray"):
            object_cfg = getattr(env_cfg.observations.subtask_terms, term_name).params["object_cfg"]
            self.assertEqual(object_cfg.name, scene_profiles.BLUE24_OBJECT_KEY)
        self.assertEqual(success.params["object_cfg"].name, scene_profiles.BLUE24_OBJECT_KEY)
        self.assertIs(success.func, placed_on_tray)
        self.assertIsNone(env_cfg.events.domain_randomize_0)
        self.assertIsNone(env_cfg.events.domain_randomize_7)
        self.assertEqual(env_cfg.events.reset_all, "keep")


if __name__ == "__main__":
    unittest.main()
