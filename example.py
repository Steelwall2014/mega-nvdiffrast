import math
from typing import Any
import torch
from configs import Configuration
from obj import load_obj_material_to_module, load_obj_mesh_to_module
from modules.light import EnvironmentLight, DirectionalLight
from modules.material import MaterialModule
from modules.mesh import GeometryModule, IGeometryModule, VirtualGeometryModule
from modules.texture import TextureModule, VirtualTextureModule
from modules.renderer import vertex_shader, rasterize
from modules.light import create_trainable_env_rnd
from step import TwinFlowAdam
import util
from dataset_photo import DatasetPhoto
import nvdiffrast.torch as dr
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from timer import timers
import renderutils as ru
from renderutils.loss import _tonemap_srgb
import torch.multiprocessing as mp
import os
import torch.distributed as dist
from virtual_shadow_mapping import vsm_feedback_pass_distance, vsm_feedback_pass, vsm_rendering_pass, transform_points, compute_visibility

images_dir = "data/UrbanScene3D/Image/PolyTech/Smith_et_al/PolyTech_fine/undistorted"
csv_path = "data/UrbanScene3D/Image/PolyTech/Smith_et_al/PolyTech_fine/PolyTech.csv"
mask_dir = "data/UrbanScene3D/Image/PolyTech/Smith_et_al/PolyTech_fine/depth_mask"
base_mesh_path = "data/UrbanScene3D/Reconstructed/PolyTech/Smith_et_al/PolyTech_fine/polytech_5_percent.obj"
BATCH_SIZE = 1


def tonemap_srgb(f: torch.Tensor) -> torch.Tensor:
    if f.shape[-1] == 4:
        return torch.cat([_tonemap_srgb(torch.log(torch.clamp(f[..., :3], min=0, max=65535) + 1)), f[..., 3:4]], dim=-1)
    else:
        return _tonemap_srgb(f)

def fragment_shader(
        view_pos: torch.Tensor,     # shape: [num_views, 1, 1, 3]
        pos: torch.Tensor,          # shape: [num_vertices, 3]
        clip_pos: torch.Tensor,     # shape: [num_vertices, 4]
        tri_pos: torch.Tensor,      # shape: [num_triangles, 3]
        visibility: torch.Tensor,   # shape: [num_views, height, width, 1]
        rast: torch.Tensor,         # shape: [num_views, height, width, 4]
        gb_pos: torch.Tensor,       # shape: [num_views, height, width, 3]
        gb_normal: torch.Tensor,    # shape: [num_views, height, width, 4]
        gb_tangent: torch.Tensor,   # shape: [num_views, height, width, 4]
        gb_base_color: torch.Tensor,# shape: [num_views, height, width, 3]
        gb_arm: torch.Tensor,       # shape: [num_views, height, width, 3], attenuation, roughness, metallic
        perturbed_nrm: torch.Tensor,# shape: [num_views, height, width, 3], sampled normal 
        env_light: EnvironmentLight,
        directional_light: DirectionalLight = None):
    mask = rast[..., 3:4] > 0

    v0 = pos[tri_pos[:, 0], :]
    v1 = pos[tri_pos[:, 1], :]
    v2 = pos[tri_pos[:, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = dr.interpolate(face_normals[None, ...], rast, face_normal_indices.int())

    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal)
    shaded_color = env_light.shade(gb_pos, gb_normal, gb_base_color, gb_arm, view_pos)

    if directional_light is not None:
        light_direction = directional_light.light_direction
        light_color = directional_light.light_color
        timers("pbr_bsdf").start()
        light_direction = light_direction.view(1, 1, 1, 3)
        light_pos = gb_pos - light_direction # directional light
        direct_light_color = ru.pbr_bsdf(gb_base_color, gb_arm, gb_pos, gb_normal, view_pos, light_pos) * visibility
        direct_light_color *= light_color
        shaded_color += direct_light_color

    shaded_color = mask * torch.cat([shaded_color, torch.ones_like(shaded_color[..., 0:1])], dim=-1)
    shaded_color = dr.antialias(shaded_color, rast, clip_pos, tri_pos)
    shaded_color = tonemap_srgb(shaded_color)

    return shaded_color

class SharedResources:
    # 用于多进程共享的资源
    # 为了节省cpu内存，一些tensor都使用了共享内存
    def __init__(self) -> None:
        self.mesh: VirtualGeometryModule|GeometryModule = None
        self.material: MaterialModule = None
        self.optimizers: dict[Any, TwinFlowAdam] = None
        self.dataset: DatasetPhoto = None

def shading(ctx, resolution, 
            mesh: IGeometryModule, material: MaterialModule, 
            view_matrix: torch.Tensor,      # shape: [num_views, 4, 4]
            proj_matrix: torch.Tensor,      # shape: [num_views, 4, 4]
            view_pos: torch.Tensor,         # shape: [num_views, 1, 1, 3]
            env_light: EnvironmentLight,
            directional_light: DirectionalLight = None):
    """
    Returns:
        shaded_color: torch.Tensor, shape: [num_views, height, width, 4]
    """
    num_views = proj_matrix.shape[0]
    mvp = proj_matrix @ view_matrix

    timers("rasterize").start()
    cull_out = mesh.frustum_cull(mvp)
    vs_out = vertex_shader(mvp, cull_out)
    rast_out = rasterize(ctx, resolution, vs_out)
    timers("rasterize").stop()

    timers("sample").start()
    gb_base_color = material.BaseColor.sample(rast_out.gb_texc, rast_out.gb_texd)
    timers("sample").stop()
    timers("sample").start()
    perturbed_nrm = material.Normal.sample(rast_out.gb_texc, rast_out.gb_texd)
    timers("sample").stop()
    timers("sample").start()
    gb_arm = material.AttenuationRoughnessMetallic.sample(rast_out.gb_texc, rast_out.gb_texd)
    timers("sample").stop()

    visibility = 1
    if directional_light is not None and directional_light.cast_shadows:
        mask = rast_out.rast[..., 3:4] > 0
        vsm_resolution = directional_light.vsm_resolution
        half_frustum_width = directional_light.half_frustum_width
        vsm_page_size_x = directional_light.page_size_x
        vsm_page_size_y = directional_light.page_size_y
        first_level_extent = directional_light.first_level_extent

        up = torch.tensor(util.WORLD_UP, dtype=torch.float32, device="cuda")
        light_view_matrix = util.lookAt(directional_light.light_position, directional_light.light_position+directional_light.light_direction, up)
        vsm_mip_level_bias, vsm_uv_da = None, None

        # 用相机到像素的距离做feedback
        camera_pos = view_pos.view(-1, 3).repeat(num_views, 1)
        vsm_uv, vsm_mip_level_bias, feedback = vsm_feedback_pass_distance(
            camera_pos, rast_out.gb_pos, light_view_matrix, 
            half_frustum_width, mask=mask, 
            vsm_resolution=vsm_resolution, 
            page_size_x=vsm_page_size_x, 
            page_size_y=vsm_page_size_y, 
            first_level_extent=first_level_extent)

        # 用1场景像素对应1个vsm像素的方式做feedback
        # vsm_uv, vsm_uv_da, feedback = vsm_feedback_pass(
        #     vs_out.pos, rast_out.rast, vs_out.tri_pos, light_view_matrix, 
        #     half_frustum_width, 
        #     rast_db=rast_out.rast_db,
        #     mask=mask, 
        #     vsm_resolution=vsm_resolution, 
        #     page_size_x=vsm_page_size_x, 
        #     page_size_y=vsm_page_size_y)

        def get_vertices(mvp):
            cull_out = mesh.frustum_cull(mvp)
            vs_out = vertex_shader(mvp, cull_out)
            return vs_out
        m1_vsm, m2_vsm = vsm_rendering_pass(ctx, feedback, light_view_matrix, half_frustum_width, get_vertices, 
                                            vsm_resolution=vsm_resolution, page_size_x=vsm_page_size_x, page_size_y=vsm_page_size_y)

        depth_actual = transform_points(rast_out.gb_pos, light_view_matrix)
        depth_actual = -depth_actual[..., 2:3] # z轴正方向朝后，所以要加个负号
        visibility = compute_visibility(vsm_uv, depth_actual, m1_vsm, m2_vsm, mask=mask, vsm_uv_da=vsm_uv_da, vsm_mip_level_bias=vsm_mip_level_bias, 
                                            vsm_resolution=vsm_resolution, page_size_x=vsm_page_size_x, page_size_y=vsm_page_size_y)


    shaded_color = fragment_shader(view_pos, 
                                   vs_out.pos, vs_out.clip_pos, vs_out.tri_pos, visibility, 
                                   rast_out.rast, rast_out.gb_pos, rast_out.gb_normal, rast_out.gb_tangent, 
                                   gb_base_color, gb_arm, perturbed_nrm, env_light, directional_light)

    return shaded_color

def main(local_rank, FLAGS: Configuration, resources: SharedResources, use_vgvt: bool):
    torch.cuda.set_device(local_rank)
    util.seed_everything()

    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["NUM_GPUS"] = str(FLAGS.num_gpus)
    os.environ["MASTER_ADDR"] = FLAGS.master_addr
    os.environ["MASTER_PORT"] = FLAGS.master_port
    dist.init_process_group("nccl", rank=local_rank, world_size=FLAGS.world_size)

    ctx = dr.RasterizeCudaContext()
        
    mesh = resources.mesh
    material = resources.material
    optimizers = resources.optimizers
    dataset = resources.dataset
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)

    env_light = create_trainable_env_rnd(512, 0, 0.2).requires_grad_(False)
    env_light.build_mips()

    directional_light = DirectionalLight([-1, -1, -1], [5, 5, 5], mesh.get_AABB(), vsm_resolution=[8192, 8192], page_size_x=512, page_size_y=512, cast_shadows=True)

    resolution = FLAGS.resolution

    for itr, samples in enumerate(dataloader):
        torch.cuda.empty_cache()

        for b in range(len(samples)):
            sample = samples[b]

            image_tensor: torch.Tensor = sample["image_tensor"]
            image_name: str = sample["image_name"]
            print(image_name)
            focal_35mm: float = sample["focal35mm"]
            camera_position: torch.Tensor = sample["camera_position"].cuda()    # shape: [1, 3]
            camera_rotation: torch.Tensor = sample["camera_rotation"].cuda()    # quaterion, wxyz format, shape: [4]
            reference_mask: torch.Tensor = sample["mask"].cuda()
            undistort_mask: torch.Tensor = sample["undistort_mask"].cuda()
            image_height, image_width = image_tensor.shape[0], image_tensor.shape[1]
            render_height, render_width = resolution[0], resolution[1]
            
            image_height, image_width = int(image_height), int(image_width)
            image_height = math.ceil(image_height / 8) * 8      # 必须是8的倍数
            image_width = math.ceil(image_width / 8) * 8        # 必须是8的倍数
            image_tensor = util.scale_img_hwc(image_tensor, [image_height, image_width])
            render_height = min(render_height, image_height)
            render_width = min(render_width, image_width)

            with torch.no_grad():
                crop_method = FLAGS.crop_method
                tiles: list[torch.Tensor] = []
                tile_masks: list[torch.Tensor] = []
                tile_undistort_masks: list[torch.Tensor] = []
                crops = util.crop_image(image_height, image_width, tile_height=render_height, tile_width=render_width, method=crop_method)
                for start_x, end_x, start_y, end_y in crops:
                    tile = image_tensor[start_y:end_y, start_x:end_x, :]
                    tile_mask = reference_mask[start_y:end_y, start_x:end_x]
                    tile_undistort_mask = undistort_mask[start_y:end_y, start_x:end_x]
                    tiles.append(tile)
                    tile_masks.append(tile_mask)
                    tile_undistort_masks.append(tile_undistort_mask)
            
            tiles_per_wave = min(FLAGS.tiles_per_wave, len(tiles))
            for t in range(0, len(tiles), tiles_per_wave):

                wave_tiles: list[torch.Tensor] = tiles[t:t+tiles_per_wave]
                wave_masks: list[torch.Tensor] = tile_masks[t:t+tiles_per_wave]
                wave_undistort_masks: list[torch.Tensor] = tile_undistort_masks[t:t+tiles_per_wave]
                wave_crops: list[tuple] = crops[t:t+tiles_per_wave]
                Reference = torch.stack(wave_tiles, dim=0).cuda()
                Mask = torch.stack(wave_masks, dim=0).cuda()
                UndistortMask = torch.stack(wave_undistort_masks, dim=0).cuda()
                view = util.lookAt_quat(camera_position.T, camera_rotation)
                projection = util.prepare_projections(wave_crops, image_width, image_height, focal_35mm, device="cuda")
                projection = torch.stack(projection, dim=0)

                num_views = projection.shape[0]
                view = view[None, ...].repeat(num_views, 1, 1)
                view_pos = camera_position.reshape(-1, 1, 1, 3)

                timers("fwd+bwd").start()
                color = shading(ctx, [render_height, render_width], mesh, material, view, projection, view_pos, env_light, directional_light)
                loss = torch.nn.functional.smooth_l1_loss(color[..., :3]*Mask*UndistortMask, Reference*Mask*UndistortMask)
                loss.backward()

                mesh.offload()
                material.offload()
                timers("fwd+bwd").stop()

                util.save_image("test.jpg", color[0, ..., :3].detach().cpu().numpy())
        
        timers("step").start()
        def mesh_step():
            dist.barrier()
            mesh_gradients, uv_gradients = mesh.normal_tangent_bwd()
            mesh_gradients = util.all_reduce(mesh_gradients)
            optimizers[mesh].step(mesh_gradients)
            optimizers[mesh].zero_grad()
        def base_color_step():
            dist.barrier()
            baseColor_gradients = material.BaseColor.texture_mipmap_bwd()
            baseColor_gradients = util.all_reduce(baseColor_gradients)
            optimizers[material.BaseColor].step(baseColor_gradients)
            optimizers[material.BaseColor].zero_grad()
        def normal_step():
            dist.barrier()
            normal_gradients = material.Normal.texture_mipmap_bwd()
            normal_gradients = util.all_reduce(normal_gradients)
            optimizers[material.Normal].step(normal_gradients)
            optimizers[material.Normal].zero_grad()
        def arm_step():
            dist.barrier()
            arm_gradients = material.AttenuationRoughnessMetallic.texture_mipmap_bwd()
            arm_gradients = util.all_reduce(arm_gradients)
            optimizers[material.AttenuationRoughnessMetallic].step(arm_gradients)
            optimizers[material.AttenuationRoughnessMetallic].zero_grad()
        mesh_step()
        base_color_step()
        normal_step()
        arm_step()

        timers("step").stop()



if __name__ == "__main__":
    use_vgvt = True

    FLAGS = Configuration()
    FLAGS.mesh_lru_max_size = 200
    FLAGS.texture_lru_max_size = 1000
    FLAGS.resolution = [1024, 1024]
    FLAGS.default_roughness = 0.8
    FLAGS.default_metallic = 0.0
    FLAGS.texture_resolution = 8192
    FLAGS.texture_page_size = 256
    FLAGS.world_size = 1
    FLAGS.num_gpus = FLAGS.world_size
    FLAGS.tiles_per_wave = 2
    dataset = DatasetPhoto(images_dir, csv_path, mask_dir=mask_dir)

    mesh: IGeometryModule = load_obj_mesh_to_module(
        base_mesh_path, 
        use_vg=use_vgvt, 
        max_partition_size=FLAGS.max_partition_size, 
        lru_cache_max_size=FLAGS.mesh_lru_max_size)
    mesh.share_memory_()
    material: MaterialModule = load_obj_material_to_module(
        base_mesh_path, 
        FLAGS.texture_resolution,
        use_vt=use_vgvt,
        default_roughness=FLAGS.default_roughness,
        default_metallic=FLAGS.default_metallic,
        page_size_x=FLAGS.texture_page_size,
        page_size_y=FLAGS.texture_page_size,
        fp16_texture=False,
        lru_cache_max_size=FLAGS.texture_lru_max_size)[0]
    material.share_memory_()
    optimizers: dict[Any, TwinFlowAdam] = {}

    mesh.pin_memory_()
    material.BaseColor.pin_memory_()
    material.Normal.pin_memory_()
    material.AttenuationRoughnessMetallic.pin_memory_()

    optimizers[mesh] = TwinFlowAdam(mesh.get_trainable_params()[0], lr=0.001)
    optimizers[mesh].state_pin_memory_()
    optimizers[material.BaseColor] = TwinFlowAdam(material.BaseColor.get_trainable_params(), lr=0.001)
    optimizers[material.BaseColor].state_pin_memory_()
    optimizers[material.Normal] = TwinFlowAdam(material.Normal.get_trainable_params(), lr=0.001)
    optimizers[material.Normal].state_pin_memory_()
    optimizers[material.AttenuationRoughnessMetallic] = TwinFlowAdam(material.AttenuationRoughnessMetallic.get_trainable_params(), lr=0.001)
    optimizers[material.AttenuationRoughnessMetallic].state_pin_memory_()

    resources = SharedResources()
    resources.mesh = mesh
    resources.material = material
    resources.optimizers = optimizers
    resources.dataset = dataset        
    torch.cuda.empty_cache()
    mp.spawn(main, nprocs=FLAGS.num_gpus, args=(FLAGS, resources, use_vgvt,))

        