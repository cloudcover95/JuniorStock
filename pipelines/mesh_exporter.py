# juniorstock/pipelines/mesh_exporter.py
import mlx.core as mx
import os

class MeshCLIExporter:
    """
    Serializes Omni Math compressed manifolds into standard .ply format.
    Optimized for downstream Blender CLI parsing and topological void detection.
    """
    def __init__(self, export_dir: str = "juniorstock/exports/"):
        self.export_dir = export_dir

    def export_manifold_ply(self, manifold: mx.array, filename: str = "market_mesh.ply"):
        """
        Dumps the b1.58 stabilized topological state into a .ply point cloud.
        Strictly avoids OS UI overhead.
        """
        path = os.path.join(self.export_dir, filename)
        
        # Verify path isolation protocol
        if "01_Legal" in os.path.abspath(path) or "02_Assets" in os.path.abspath(path):
            raise PermissionError("[SECURITY FAULT] Export target violates path isolation.")

        # Ensure tensor is 3D for spatial rendering (k=3 components extracted from SVD)
        if manifold.shape[-1] < 3:
            raise ValueError("[DIMENSION ERR] Manifold lacks sufficient k-components for 3D topological projection.")
        
        vertices = manifold[:, :3].tolist()
        num_vertices = len(vertices)

        with open(path, 'w') as f:
            # .ply header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # Write coordinate mesh
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
                
        print(f"[EXPORT SUCCESS] Topological mesh written to {path}. Ready for Metal rendering.")
