class Configuration:

    def __init__(self):
        self.num_gpus = 1
        self.world_size = 1
        self.master_addr = "localhost"
        self.master_port = "29501"

        self.tiles_per_wave: int = 1
        self.resolution: list = [1024, 1024]
        self.max_partition_size: int = 8192
        self.twin_flow_ratio: float = 0.0
        self.crop_method: str = "random"
        self.mesh_lru_max_size: int = 200
        self.texture_lru_max_size: int = 1000
        self.texture_resolution: int = 16384
        self.texture_page_size: int = 512
        self.default_roughness: float = 0.8
        self.default_metallic: float = 0.0
        self.spp: int = 1                       # samples per pixel
        self.msaa: bool = False                 # multi-sample anti-aliasing